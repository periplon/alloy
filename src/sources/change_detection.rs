//! Change detection for incremental updates.
//!
//! Tracks document state to detect changes and enable incremental indexing,
//! avoiding re-processing of unchanged content.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;

use super::SourceItem;

/// Type of change detected for a document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeType {
    /// Document is new (not previously indexed).
    New,
    /// Document content has been modified.
    Modified,
    /// Document has been deleted.
    Deleted,
    /// Document is unchanged.
    Unchanged,
}

/// Change detection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeDetectionStrategy {
    /// Use content hash (SHA-256).
    Hash,
    /// Use modification time.
    Mtime,
    /// Use both hash and mtime.
    Both,
}

/// Metadata about an indexed document for change detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedDocumentMeta {
    /// Document ID.
    pub doc_id: String,
    /// File path or URI.
    pub path: String,
    /// Source ID.
    pub source_id: String,
    /// Content hash (SHA-256).
    pub content_hash: String,
    /// Last modification time.
    pub mtime: DateTime<Utc>,
    /// When the document was indexed.
    pub indexed_at: DateTime<Utc>,
    /// Number of chunks created.
    pub chunk_count: usize,
    /// File size in bytes.
    pub size: u64,
}

/// A detected change event.
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    /// The source item.
    pub item: SourceItem,
    /// Type of change detected.
    pub change_type: ChangeType,
    /// Previous metadata if available.
    pub previous: Option<IndexedDocumentMeta>,
}

/// Change detection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChangeDetectionConfig {
    /// Enable incremental indexing.
    pub enabled: bool,
    /// Detection strategy.
    pub strategy: ChangeDetectionStrategy,
    /// Whether to verify hash on mtime match (more accurate but slower).
    pub verify_hash: bool,
}

impl Default for ChangeDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: ChangeDetectionStrategy::Both,
            verify_hash: false,
        }
    }
}

/// Change detector for tracking document state.
pub struct ChangeDetector {
    /// Configuration.
    config: ChangeDetectionConfig,
    /// Indexed document metadata by path.
    metadata: RwLock<HashMap<String, IndexedDocumentMeta>>,
    /// Persistence path.
    persist_path: Option<PathBuf>,
}

impl ChangeDetector {
    /// Create a new change detector.
    pub fn new(config: ChangeDetectionConfig) -> Self {
        Self {
            config,
            metadata: RwLock::new(HashMap::new()),
            persist_path: None,
        }
    }

    /// Create with persistence.
    pub async fn with_persistence(config: ChangeDetectionConfig, path: PathBuf) -> crate::error::Result<Self> {
        let detector = Self {
            config,
            metadata: RwLock::new(HashMap::new()),
            persist_path: Some(path.clone()),
        };

        // Load existing state
        if path.exists() {
            detector.load().await?;
        }

        Ok(detector)
    }

    /// Load state from disk.
    async fn load(&self) -> crate::error::Result<()> {
        if let Some(ref path) = self.persist_path {
            if path.exists() {
                let data = tokio::fs::read_to_string(path).await?;
                let loaded: HashMap<String, IndexedDocumentMeta> = serde_json::from_str(&data)?;
                let mut metadata = self.metadata.write().await;
                *metadata = loaded;
            }
        }
        Ok(())
    }

    /// Save state to disk.
    pub async fn save(&self) -> crate::error::Result<()> {
        if let Some(ref path) = self.persist_path {
            let metadata = self.metadata.read().await;
            let data = serde_json::to_string_pretty(&*metadata)?;
            tokio::fs::write(path, data).await?;
        }
        Ok(())
    }

    /// Compute SHA-256 hash of content.
    pub fn compute_hash(content: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content);
        format!("{:x}", hasher.finalize())
    }

    /// Detect changes for a list of source items.
    pub async fn detect_changes(
        &self,
        items: &[SourceItem],
        get_content: impl Fn(&SourceItem) -> Option<Vec<u8>>,
    ) -> Vec<ChangeEvent> {
        let metadata = self.metadata.read().await;
        let mut events = Vec::new();

        // Track which paths we've seen
        let mut seen_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

        for item in items {
            seen_paths.insert(item.uri.clone());
            let change_type = self.detect_item_change(item, &metadata, &get_content);

            events.push(ChangeEvent {
                item: item.clone(),
                change_type,
                previous: metadata.get(&item.uri).cloned(),
            });
        }

        // Detect deleted items (in metadata but not in current scan)
        for (path, meta) in metadata.iter() {
            if !seen_paths.contains(path) {
                events.push(ChangeEvent {
                    item: SourceItem {
                        id: meta.doc_id.clone(),
                        uri: meta.path.clone(),
                        mime_type: String::new(),
                        size: meta.size,
                        modified: meta.mtime,
                        metadata: serde_json::json!({}),
                    },
                    change_type: ChangeType::Deleted,
                    previous: Some(meta.clone()),
                });
            }
        }

        events
    }

    /// Detect change type for a single item.
    fn detect_item_change(
        &self,
        item: &SourceItem,
        metadata: &HashMap<String, IndexedDocumentMeta>,
        get_content: &impl Fn(&SourceItem) -> Option<Vec<u8>>,
    ) -> ChangeType {
        let existing = match metadata.get(&item.uri) {
            Some(m) => m,
            None => return ChangeType::New,
        };

        match self.config.strategy {
            ChangeDetectionStrategy::Hash => {
                if let Some(content) = get_content(item) {
                    let hash = Self::compute_hash(&content);
                    if hash != existing.content_hash {
                        ChangeType::Modified
                    } else {
                        ChangeType::Unchanged
                    }
                } else {
                    // Can't compute hash, fall back to mtime
                    if item.modified > existing.mtime {
                        ChangeType::Modified
                    } else {
                        ChangeType::Unchanged
                    }
                }
            }
            ChangeDetectionStrategy::Mtime => {
                if item.modified > existing.mtime {
                    ChangeType::Modified
                } else {
                    ChangeType::Unchanged
                }
            }
            ChangeDetectionStrategy::Both => {
                // First check mtime (fast)
                if item.modified <= existing.mtime && item.size == existing.size {
                    if self.config.verify_hash {
                        // Verify with hash
                        if let Some(content) = get_content(item) {
                            let hash = Self::compute_hash(&content);
                            if hash != existing.content_hash {
                                return ChangeType::Modified;
                            }
                        }
                    }
                    return ChangeType::Unchanged;
                }

                // Mtime changed, check hash to confirm
                if let Some(content) = get_content(item) {
                    let hash = Self::compute_hash(&content);
                    if hash != existing.content_hash {
                        ChangeType::Modified
                    } else {
                        // Hash is same, just mtime changed (touch, copy, etc.)
                        ChangeType::Unchanged
                    }
                } else {
                    // Can't verify, assume modified
                    ChangeType::Modified
                }
            }
        }
    }

    /// Register a successfully indexed document.
    pub async fn register(
        &self,
        item: &SourceItem,
        source_id: &str,
        content_hash: String,
        chunk_count: usize,
    ) -> crate::error::Result<()> {
        let meta = IndexedDocumentMeta {
            doc_id: item.id.clone(),
            path: item.uri.clone(),
            source_id: source_id.to_string(),
            content_hash,
            mtime: item.modified,
            indexed_at: Utc::now(),
            chunk_count,
            size: item.size,
        };

        let mut metadata = self.metadata.write().await;
        metadata.insert(item.uri.clone(), meta);

        // Auto-save if persistence enabled
        drop(metadata);
        self.save().await?;

        Ok(())
    }

    /// Remove a document from tracking.
    pub async fn remove(&self, path: &str) -> crate::error::Result<Option<IndexedDocumentMeta>> {
        let mut metadata = self.metadata.write().await;
        let removed = metadata.remove(path);

        drop(metadata);
        self.save().await?;

        Ok(removed)
    }

    /// Remove all documents for a source.
    pub async fn remove_source(&self, source_id: &str) -> crate::error::Result<usize> {
        let mut metadata = self.metadata.write().await;
        let before = metadata.len();
        metadata.retain(|_, m| m.source_id != source_id);
        let removed = before - metadata.len();

        drop(metadata);
        self.save().await?;

        Ok(removed)
    }

    /// Clear all tracking data.
    pub async fn clear(&self) -> crate::error::Result<()> {
        let mut metadata = self.metadata.write().await;
        metadata.clear();

        drop(metadata);
        self.save().await?;

        Ok(())
    }

    /// Get metadata for a document.
    pub async fn get(&self, path: &str) -> Option<IndexedDocumentMeta> {
        let metadata = self.metadata.read().await;
        metadata.get(path).cloned()
    }

    /// Get all metadata for a source.
    pub async fn get_source_metadata(&self, source_id: &str) -> Vec<IndexedDocumentMeta> {
        let metadata = self.metadata.read().await;
        metadata
            .values()
            .filter(|m| m.source_id == source_id)
            .cloned()
            .collect()
    }

    /// Get statistics.
    pub async fn stats(&self) -> ChangeDetectorStats {
        let metadata = self.metadata.read().await;

        let mut total_size = 0u64;
        let mut total_chunks = 0usize;
        let mut sources: HashMap<String, usize> = HashMap::new();

        for meta in metadata.values() {
            total_size += meta.size;
            total_chunks += meta.chunk_count;
            *sources.entry(meta.source_id.clone()).or_default() += 1;
        }

        ChangeDetectorStats {
            document_count: metadata.len(),
            total_size,
            total_chunks,
            source_count: sources.len(),
        }
    }
}

/// Statistics about tracked documents.
#[derive(Debug, Clone, Default)]
pub struct ChangeDetectorStats {
    /// Number of documents tracked.
    pub document_count: usize,
    /// Total size of tracked documents.
    pub total_size: u64,
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Number of sources.
    pub source_count: usize,
}

/// Filter for processing only changed items.
pub struct ChangeFilter {
    /// Items that need processing (New or Modified).
    pub to_process: Vec<ChangeEvent>,
    /// Items that are unchanged.
    pub unchanged: Vec<ChangeEvent>,
    /// Items that were deleted.
    pub deleted: Vec<ChangeEvent>,
}

impl ChangeFilter {
    /// Create from change events.
    pub fn from_events(events: Vec<ChangeEvent>) -> Self {
        let mut to_process = Vec::new();
        let mut unchanged = Vec::new();
        let mut deleted = Vec::new();

        for event in events {
            match event.change_type {
                ChangeType::New | ChangeType::Modified => to_process.push(event),
                ChangeType::Unchanged => unchanged.push(event),
                ChangeType::Deleted => deleted.push(event),
            }
        }

        Self {
            to_process,
            unchanged,
            deleted,
        }
    }

    /// Check if there are any changes.
    pub fn has_changes(&self) -> bool {
        !self.to_process.is_empty() || !self.deleted.is_empty()
    }

    /// Get summary of changes.
    pub fn summary(&self) -> String {
        format!(
            "Changes: {} new/modified, {} unchanged, {} deleted",
            self.to_process.len(),
            self.unchanged.len(),
            self.deleted.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_item(uri: &str, size: u64, modified: DateTime<Utc>) -> SourceItem {
        SourceItem {
            id: format!("id_{}", uri),
            uri: uri.to_string(),
            mime_type: "text/plain".to_string(),
            size,
            modified,
            metadata: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn test_detect_new_item() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default());
        let item = create_test_item("test.txt", 100, Utc::now());

        let events = detector.detect_changes(&[item], |_| None).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].change_type, ChangeType::New);
    }

    #[tokio::test]
    async fn test_detect_unchanged() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default());
        let now = Utc::now();
        let item = create_test_item("test.txt", 100, now);

        // Register the item
        detector
            .register(&item, "source1", "hash123".to_string(), 5)
            .await
            .unwrap();

        // Check again - should be unchanged
        let events = detector.detect_changes(&[item.clone()], |_| Some(b"content".to_vec())).await;
        assert_eq!(events.len(), 1);
        // Mtime is same, size is same, so unchanged
        assert_eq!(events[0].change_type, ChangeType::Unchanged);
    }

    #[tokio::test]
    async fn test_detect_modified_by_mtime() {
        let mut config = ChangeDetectionConfig::default();
        config.strategy = ChangeDetectionStrategy::Mtime;
        let detector = ChangeDetector::new(config);

        let old_time = Utc::now() - chrono::Duration::hours(1);
        let item = create_test_item("test.txt", 100, old_time);

        // Register with old time
        detector
            .register(&item, "source1", "hash123".to_string(), 5)
            .await
            .unwrap();

        // Create updated item with new mtime
        let new_item = create_test_item("test.txt", 100, Utc::now());
        let events = detector.detect_changes(&[new_item], |_| None).await;

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].change_type, ChangeType::Modified);
    }

    #[tokio::test]
    async fn test_detect_deleted() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default());
        let item = create_test_item("test.txt", 100, Utc::now());

        // Register the item
        detector
            .register(&item, "source1", "hash123".to_string(), 5)
            .await
            .unwrap();

        // Check with empty list - should detect deletion
        let events = detector.detect_changes(&[], |_| None).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].change_type, ChangeType::Deleted);
    }

    #[tokio::test]
    async fn test_hash_change_detection() {
        let mut config = ChangeDetectionConfig::default();
        config.strategy = ChangeDetectionStrategy::Hash;
        let detector = ChangeDetector::new(config);

        let item = create_test_item("test.txt", 100, Utc::now());
        let original_content = b"original content";
        let hash = ChangeDetector::compute_hash(original_content);

        // Register with original hash
        detector
            .register(&item, "source1", hash, 5)
            .await
            .unwrap();

        // Check with different content
        let modified_content = b"modified content";
        let events = detector
            .detect_changes(&[item], |_| Some(modified_content.to_vec()))
            .await;

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].change_type, ChangeType::Modified);
    }

    #[tokio::test]
    async fn test_change_filter() {
        let events = vec![
            ChangeEvent {
                item: create_test_item("new.txt", 100, Utc::now()),
                change_type: ChangeType::New,
                previous: None,
            },
            ChangeEvent {
                item: create_test_item("unchanged.txt", 100, Utc::now()),
                change_type: ChangeType::Unchanged,
                previous: None,
            },
            ChangeEvent {
                item: create_test_item("deleted.txt", 100, Utc::now()),
                change_type: ChangeType::Deleted,
                previous: None,
            },
        ];

        let filter = ChangeFilter::from_events(events);

        assert_eq!(filter.to_process.len(), 1);
        assert_eq!(filter.unchanged.len(), 1);
        assert_eq!(filter.deleted.len(), 1);
        assert!(filter.has_changes());
    }

    #[tokio::test]
    async fn test_remove_source() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default());

        // Register items from two sources
        let item1 = create_test_item("a.txt", 100, Utc::now());
        let item2 = create_test_item("b.txt", 100, Utc::now());

        detector
            .register(&item1, "source1", "h1".to_string(), 1)
            .await
            .unwrap();
        detector
            .register(&item2, "source2", "h2".to_string(), 1)
            .await
            .unwrap();

        // Remove source1
        let removed = detector.remove_source("source1").await.unwrap();
        assert_eq!(removed, 1);

        // Verify source2 still exists
        assert!(detector.get("b.txt").await.is_some());
        assert!(detector.get("a.txt").await.is_none());
    }

    #[tokio::test]
    async fn test_stats() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default());

        let item1 = create_test_item("a.txt", 100, Utc::now());
        let item2 = create_test_item("b.txt", 200, Utc::now());

        detector
            .register(&item1, "source1", "h1".to_string(), 2)
            .await
            .unwrap();
        detector
            .register(&item2, "source1", "h2".to_string(), 3)
            .await
            .unwrap();

        let stats = detector.stats().await;
        assert_eq!(stats.document_count, 2);
        assert_eq!(stats.total_size, 300);
        assert_eq!(stats.total_chunks, 5);
        assert_eq!(stats.source_count, 1);
    }
}

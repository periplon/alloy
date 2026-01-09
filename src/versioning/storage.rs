//! Version storage backends.
//!
//! Provides storage for document version history with support for
//! full content storage and delta compression.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;

/// Type of change that created a version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChangeType {
    /// Initial document creation.
    Created,
    /// Content was modified.
    ContentModified,
    /// Only metadata was modified.
    MetadataModified,
    /// Document was deleted.
    Deleted,
    /// Document was restored from a previous version.
    Restored { from_version: String },
}

/// Storage type for version content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VersionStorageType {
    /// Full content is stored.
    Full { content: String },
    /// Delta from a base version.
    Delta {
        base_version: String,
        operations: Vec<DeltaOperation>,
    },
}

/// Delta operation for incremental storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaOperation {
    /// Insert text at position.
    Insert { position: usize, text: String },
    /// Delete characters at position.
    Delete { position: usize, length: usize },
    /// Replace text at position.
    Replace {
        position: usize,
        length: usize,
        text: String,
    },
}

/// A specific version of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentVersion {
    /// Unique version ID.
    pub version_id: String,
    /// Document ID this version belongs to.
    pub document_id: String,
    /// Sequential version number.
    pub version_number: u64,
    /// When this version was created.
    pub timestamp: DateTime<Utc>,
    /// User who made the change (if known).
    pub author: Option<String>,
    /// Type of change.
    pub change_type: ChangeType,
    /// SHA-256 hash of the content.
    pub content_hash: String,
    /// How the content is stored.
    pub storage: VersionStorageType,
    /// Size of content in bytes.
    pub size_bytes: usize,
}

impl DocumentVersion {
    /// Compute SHA-256 hash of content.
    pub fn compute_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Metadata about a version (lightweight for listing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// Version ID.
    pub version_id: String,
    /// Sequential version number.
    pub version_number: u64,
    /// When created.
    pub timestamp: DateTime<Utc>,
    /// Who made the change.
    pub author: Option<String>,
    /// Type of change.
    pub change_type: ChangeType,
    /// Content size in bytes.
    pub size_bytes: usize,
    /// Content hash.
    pub content_hash: String,
}

impl From<&DocumentVersion> for VersionMetadata {
    fn from(v: &DocumentVersion) -> Self {
        Self {
            version_id: v.version_id.clone(),
            version_number: v.version_number,
            timestamp: v.timestamp,
            author: v.author.clone(),
            change_type: v.change_type.clone(),
            size_bytes: v.size_bytes,
            content_hash: v.content_hash.clone(),
        }
    }
}

/// Version cleanup statistics.
#[derive(Debug, Clone, Default)]
pub struct CleanupStats {
    /// Number of versions removed.
    pub versions_removed: usize,
    /// Bytes freed.
    pub bytes_freed: u64,
    /// Documents affected.
    pub documents_affected: usize,
}

/// Trait for version storage backends.
#[async_trait]
pub trait VersionStorage: Send + Sync {
    /// Store a new version.
    async fn store_version(&self, version: DocumentVersion) -> crate::error::Result<()>;

    /// Get a version by ID.
    async fn get_version(&self, version_id: &str) -> crate::error::Result<Option<DocumentVersion>>;

    /// Get a version by document ID and version number.
    async fn get_version_by_number(
        &self,
        doc_id: &str,
        version_number: u64,
    ) -> crate::error::Result<Option<DocumentVersion>>;

    /// Get the latest version number for a document.
    async fn get_latest_version_number(&self, doc_id: &str) -> crate::error::Result<Option<u64>>;

    /// Get the latest version for a document.
    async fn get_latest_version(
        &self,
        doc_id: &str,
    ) -> crate::error::Result<Option<DocumentVersion>>;

    /// List all versions for a document.
    async fn list_versions(&self, doc_id: &str) -> crate::error::Result<Vec<VersionMetadata>>;

    /// Reconstruct full content for a version.
    async fn reconstruct_content(&self, version_id: &str) -> crate::error::Result<String>;

    /// Delete a specific version.
    async fn delete_version(&self, version_id: &str) -> crate::error::Result<()>;

    /// Delete all versions for a document.
    async fn delete_document_versions(&self, doc_id: &str) -> crate::error::Result<usize>;

    /// Get total storage size.
    async fn total_size(&self) -> crate::error::Result<u64>;
}

/// In-memory version storage (for testing and simple use cases).
#[allow(dead_code)]
pub struct InMemoryVersionStorage {
    /// Versions by version ID.
    versions: RwLock<HashMap<String, DocumentVersion>>,
    /// Version IDs by document ID, ordered by version number.
    doc_versions: RwLock<HashMap<String, Vec<String>>>,
}

impl InMemoryVersionStorage {
    /// Create a new in-memory storage.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            versions: RwLock::new(HashMap::new()),
            doc_versions: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryVersionStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VersionStorage for InMemoryVersionStorage {
    async fn store_version(&self, version: DocumentVersion) -> crate::error::Result<()> {
        let version_id = version.version_id.clone();
        let doc_id = version.document_id.clone();

        let mut versions = self.versions.write().await;
        let mut doc_versions = self.doc_versions.write().await;

        versions.insert(version_id.clone(), version);
        doc_versions
            .entry(doc_id)
            .or_insert_with(Vec::new)
            .push(version_id);

        Ok(())
    }

    async fn get_version(&self, version_id: &str) -> crate::error::Result<Option<DocumentVersion>> {
        let versions = self.versions.read().await;
        Ok(versions.get(version_id).cloned())
    }

    async fn get_version_by_number(
        &self,
        doc_id: &str,
        version_number: u64,
    ) -> crate::error::Result<Option<DocumentVersion>> {
        let versions = self.versions.read().await;
        let doc_versions = self.doc_versions.read().await;

        if let Some(version_ids) = doc_versions.get(doc_id) {
            for vid in version_ids {
                if let Some(v) = versions.get(vid) {
                    if v.version_number == version_number {
                        return Ok(Some(v.clone()));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn get_latest_version_number(&self, doc_id: &str) -> crate::error::Result<Option<u64>> {
        let versions = self.versions.read().await;
        let doc_versions = self.doc_versions.read().await;

        if let Some(version_ids) = doc_versions.get(doc_id) {
            let max = version_ids
                .iter()
                .filter_map(|vid| versions.get(vid).map(|v| v.version_number))
                .max();
            return Ok(max);
        }

        Ok(None)
    }

    async fn get_latest_version(
        &self,
        doc_id: &str,
    ) -> crate::error::Result<Option<DocumentVersion>> {
        if let Some(num) = self.get_latest_version_number(doc_id).await? {
            return self.get_version_by_number(doc_id, num).await;
        }
        Ok(None)
    }

    async fn list_versions(&self, doc_id: &str) -> crate::error::Result<Vec<VersionMetadata>> {
        let versions = self.versions.read().await;
        let doc_versions = self.doc_versions.read().await;

        let mut result = Vec::new();
        if let Some(version_ids) = doc_versions.get(doc_id) {
            for vid in version_ids {
                if let Some(v) = versions.get(vid) {
                    result.push(VersionMetadata::from(v));
                }
            }
        }

        // Sort by version number descending
        result.sort_by(|a, b| b.version_number.cmp(&a.version_number));
        Ok(result)
    }

    async fn reconstruct_content(&self, version_id: &str) -> crate::error::Result<String> {
        let versions = self.versions.read().await;

        let version = versions
            .get(version_id)
            .ok_or_else(|| crate::error::StorageError::NotFound(version_id.to_string()))?
            .clone();

        drop(versions); // Release lock before any operations

        match version.storage {
            VersionStorageType::Full { content } => Ok(content),
            VersionStorageType::Delta {
                base_version,
                operations,
            } => {
                // Recursively reconstruct base content
                let mut content = self.reconstruct_content(&base_version).await?;

                // Apply delta operations
                for op in operations {
                    match op {
                        DeltaOperation::Insert { position, text } => {
                            if position <= content.len() {
                                content.insert_str(position, &text);
                            }
                        }
                        DeltaOperation::Delete { position, length } => {
                            if position + length <= content.len() {
                                content.drain(position..position + length);
                            }
                        }
                        DeltaOperation::Replace {
                            position,
                            length,
                            text,
                        } => {
                            if position + length <= content.len() {
                                content.drain(position..position + length);
                                content.insert_str(position, &text);
                            }
                        }
                    }
                }

                Ok(content)
            }
        }
    }

    async fn delete_version(&self, version_id: &str) -> crate::error::Result<()> {
        let mut versions = self.versions.write().await;
        let mut doc_versions = self.doc_versions.write().await;

        if let Some(v) = versions.remove(version_id) {
            if let Some(ids) = doc_versions.get_mut(&v.document_id) {
                ids.retain(|id| id != version_id);
            }
        }

        Ok(())
    }

    async fn delete_document_versions(&self, doc_id: &str) -> crate::error::Result<usize> {
        let mut versions = self.versions.write().await;
        let mut doc_versions = self.doc_versions.write().await;

        let count = if let Some(ids) = doc_versions.remove(doc_id) {
            for id in &ids {
                versions.remove(id);
            }
            ids.len()
        } else {
            0
        };

        Ok(count)
    }

    async fn total_size(&self) -> crate::error::Result<u64> {
        let versions = self.versions.read().await;
        let size: u64 = versions.values().map(|v| v.size_bytes as u64).sum();
        Ok(size)
    }
}

/// File-based version storage.
pub struct FileVersionStorage {
    /// Base directory for version files.
    base_dir: PathBuf,
    /// In-memory index of versions.
    index: RwLock<HashMap<String, DocumentVersion>>,
    /// Document to version IDs mapping.
    doc_index: RwLock<HashMap<String, Vec<String>>>,
    /// Store full version every N versions.
    #[allow(dead_code)]
    delta_threshold: usize,
}

impl FileVersionStorage {
    /// Create a new file-based version storage.
    pub async fn new(base_dir: PathBuf, delta_threshold: usize) -> crate::error::Result<Self> {
        tokio::fs::create_dir_all(&base_dir).await?;

        let storage = Self {
            base_dir,
            index: RwLock::new(HashMap::new()),
            doc_index: RwLock::new(HashMap::new()),
            delta_threshold,
        };

        // Load existing index
        storage.load_index().await?;

        Ok(storage)
    }

    /// Load version index from disk.
    async fn load_index(&self) -> crate::error::Result<()> {
        let index_path = self.base_dir.join("index.json");
        if !index_path.exists() {
            return Ok(());
        }

        let data = tokio::fs::read_to_string(&index_path).await?;
        let saved: SavedIndex = serde_json::from_str(&data)?;

        let mut index = self.index.write().await;
        let mut doc_index = self.doc_index.write().await;

        *index = saved.versions;
        *doc_index = saved.doc_versions;

        Ok(())
    }

    /// Save version index to disk.
    async fn save_index(&self) -> crate::error::Result<()> {
        let index = self.index.read().await;
        let doc_index = self.doc_index.read().await;

        let saved = SavedIndex {
            versions: index.clone(),
            doc_versions: doc_index.clone(),
        };

        let data = serde_json::to_string_pretty(&saved)?;
        let index_path = self.base_dir.join("index.json");
        tokio::fs::write(&index_path, data).await?;

        Ok(())
    }

    /// Get content file path for a version.
    fn content_path(&self, version_id: &str) -> PathBuf {
        self.base_dir.join(format!("{}.content", version_id))
    }

    /// Save content to file.
    async fn save_content(&self, version_id: &str, content: &str) -> crate::error::Result<()> {
        let path = self.content_path(version_id);
        tokio::fs::write(&path, content).await?;
        Ok(())
    }

    /// Load content from file.
    async fn load_content(&self, version_id: &str) -> crate::error::Result<String> {
        let path = self.content_path(version_id);
        let content = tokio::fs::read_to_string(&path).await?;
        Ok(content)
    }
}

#[derive(Serialize, Deserialize)]
struct SavedIndex {
    versions: HashMap<String, DocumentVersion>,
    doc_versions: HashMap<String, Vec<String>>,
}

#[async_trait]
impl VersionStorage for FileVersionStorage {
    async fn store_version(&self, version: DocumentVersion) -> crate::error::Result<()> {
        let version_id = version.version_id.clone();
        let doc_id = version.document_id.clone();

        // Save content if it's a full version
        if let VersionStorageType::Full { ref content } = version.storage {
            self.save_content(&version_id, content).await?;
        }

        // Update index
        {
            let mut index = self.index.write().await;
            let mut doc_index = self.doc_index.write().await;

            index.insert(version_id.clone(), version);
            doc_index
                .entry(doc_id)
                .or_insert_with(Vec::new)
                .push(version_id);
        }

        // Save index
        self.save_index().await?;

        Ok(())
    }

    async fn get_version(&self, version_id: &str) -> crate::error::Result<Option<DocumentVersion>> {
        let index = self.index.read().await;
        Ok(index.get(version_id).cloned())
    }

    async fn get_version_by_number(
        &self,
        doc_id: &str,
        version_number: u64,
    ) -> crate::error::Result<Option<DocumentVersion>> {
        let index = self.index.read().await;
        let doc_index = self.doc_index.read().await;

        if let Some(version_ids) = doc_index.get(doc_id) {
            for vid in version_ids {
                if let Some(v) = index.get(vid) {
                    if v.version_number == version_number {
                        return Ok(Some(v.clone()));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn get_latest_version_number(&self, doc_id: &str) -> crate::error::Result<Option<u64>> {
        let index = self.index.read().await;
        let doc_index = self.doc_index.read().await;

        if let Some(version_ids) = doc_index.get(doc_id) {
            let max = version_ids
                .iter()
                .filter_map(|vid| index.get(vid).map(|v| v.version_number))
                .max();
            return Ok(max);
        }

        Ok(None)
    }

    async fn get_latest_version(
        &self,
        doc_id: &str,
    ) -> crate::error::Result<Option<DocumentVersion>> {
        if let Some(num) = self.get_latest_version_number(doc_id).await? {
            return self.get_version_by_number(doc_id, num).await;
        }
        Ok(None)
    }

    async fn list_versions(&self, doc_id: &str) -> crate::error::Result<Vec<VersionMetadata>> {
        let index = self.index.read().await;
        let doc_index = self.doc_index.read().await;

        let mut result = Vec::new();
        if let Some(version_ids) = doc_index.get(doc_id) {
            for vid in version_ids {
                if let Some(v) = index.get(vid) {
                    result.push(VersionMetadata::from(v));
                }
            }
        }

        result.sort_by(|a, b| b.version_number.cmp(&a.version_number));
        Ok(result)
    }

    async fn reconstruct_content(&self, version_id: &str) -> crate::error::Result<String> {
        let index = self.index.read().await;

        let version = index
            .get(version_id)
            .ok_or_else(|| crate::error::StorageError::NotFound(version_id.to_string()))?
            .clone();

        drop(index);

        match &version.storage {
            VersionStorageType::Full { .. } => {
                // Load from file
                self.load_content(version_id).await
            }
            VersionStorageType::Delta {
                base_version,
                operations,
            } => {
                let mut content = self.reconstruct_content(base_version).await?;

                for op in operations {
                    match op {
                        DeltaOperation::Insert { position, text } => {
                            if *position <= content.len() {
                                content.insert_str(*position, text);
                            }
                        }
                        DeltaOperation::Delete { position, length } => {
                            if *position + *length <= content.len() {
                                content.drain(*position..*position + *length);
                            }
                        }
                        DeltaOperation::Replace {
                            position,
                            length,
                            text,
                        } => {
                            if *position + *length <= content.len() {
                                content.drain(*position..*position + *length);
                                content.insert_str(*position, text);
                            }
                        }
                    }
                }

                Ok(content)
            }
        }
    }

    async fn delete_version(&self, version_id: &str) -> crate::error::Result<()> {
        {
            let mut index = self.index.write().await;
            let mut doc_index = self.doc_index.write().await;

            if let Some(v) = index.remove(version_id) {
                if let Some(ids) = doc_index.get_mut(&v.document_id) {
                    ids.retain(|id| id != version_id);
                }

                // Delete content file
                let path = self.content_path(version_id);
                let _ = tokio::fs::remove_file(&path).await;
            }
        }

        self.save_index().await?;
        Ok(())
    }

    async fn delete_document_versions(&self, doc_id: &str) -> crate::error::Result<usize> {
        let count;
        {
            let mut index = self.index.write().await;
            let mut doc_index = self.doc_index.write().await;

            count = if let Some(ids) = doc_index.remove(doc_id) {
                for id in &ids {
                    index.remove(id);
                    let path = self.content_path(id);
                    let _ = tokio::fs::remove_file(&path).await;
                }
                ids.len()
            } else {
                0
            };
        }

        self.save_index().await?;
        Ok(count)
    }

    async fn total_size(&self) -> crate::error::Result<u64> {
        let index = self.index.read().await;
        let size: u64 = index.values().map(|v| v.size_bytes as u64).sum();
        Ok(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_storage() {
        let storage = InMemoryVersionStorage::new();

        let version = DocumentVersion {
            version_id: "v1".to_string(),
            document_id: "doc1".to_string(),
            version_number: 1,
            timestamp: Utc::now(),
            author: Some("test".to_string()),
            change_type: ChangeType::Created,
            content_hash: "abc123".to_string(),
            storage: VersionStorageType::Full {
                content: "Hello, World!".to_string(),
            },
            size_bytes: 13,
        };

        storage.store_version(version).await.unwrap();

        let retrieved = storage.get_version("v1").await.unwrap();
        assert!(retrieved.is_some());

        let content = storage.reconstruct_content("v1").await.unwrap();
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_version_listing() {
        let storage = InMemoryVersionStorage::new();

        for i in 1..=3 {
            let version = DocumentVersion {
                version_id: format!("v{}", i),
                document_id: "doc1".to_string(),
                version_number: i,
                timestamp: Utc::now(),
                author: None,
                change_type: ChangeType::ContentModified,
                content_hash: format!("hash{}", i),
                storage: VersionStorageType::Full {
                    content: format!("Content v{}", i),
                },
                size_bytes: 10,
            };
            storage.store_version(version).await.unwrap();
        }

        let versions = storage.list_versions("doc1").await.unwrap();
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].version_number, 3); // Most recent first
    }

    #[tokio::test]
    async fn test_delta_reconstruction() {
        let storage = InMemoryVersionStorage::new();

        // Store base version
        let v1 = DocumentVersion {
            version_id: "v1".to_string(),
            document_id: "doc1".to_string(),
            version_number: 1,
            timestamp: Utc::now(),
            author: None,
            change_type: ChangeType::Created,
            content_hash: "hash1".to_string(),
            storage: VersionStorageType::Full {
                content: "Hello, World!".to_string(),
            },
            size_bytes: 13,
        };
        storage.store_version(v1).await.unwrap();

        // Store delta version
        let v2 = DocumentVersion {
            version_id: "v2".to_string(),
            document_id: "doc1".to_string(),
            version_number: 2,
            timestamp: Utc::now(),
            author: None,
            change_type: ChangeType::ContentModified,
            content_hash: "hash2".to_string(),
            storage: VersionStorageType::Delta {
                base_version: "v1".to_string(),
                operations: vec![DeltaOperation::Replace {
                    position: 7,
                    length: 5,
                    text: "Rust".to_string(),
                }],
            },
            size_bytes: 12,
        };
        storage.store_version(v2).await.unwrap();

        let content = storage.reconstruct_content("v2").await.unwrap();
        assert_eq!(content, "Hello, Rust!");
    }

    #[tokio::test]
    async fn test_file_storage() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = FileVersionStorage::new(temp_dir.path().to_path_buf(), 10)
            .await
            .unwrap();

        let version = DocumentVersion {
            version_id: "v1".to_string(),
            document_id: "doc1".to_string(),
            version_number: 1,
            timestamp: Utc::now(),
            author: Some("test".to_string()),
            change_type: ChangeType::Created,
            content_hash: "abc123".to_string(),
            storage: VersionStorageType::Full {
                content: "Test content".to_string(),
            },
            size_bytes: 12,
        };

        storage.store_version(version).await.unwrap();

        let content = storage.reconstruct_content("v1").await.unwrap();
        assert_eq!(content, "Test content");
    }
}

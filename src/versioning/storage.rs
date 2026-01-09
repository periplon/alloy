//! Version storage backends.
//!
//! Provides storage for document version history with support for
//! full content storage, delta compression, and optional zstd compression.

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

/// Chunk-level versioning for embeddings.
/// Tracks individual chunks within a document version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkVersion {
    /// Unique chunk identifier.
    pub chunk_id: String,
    /// SHA-256 hash of chunk content.
    pub content_hash: String,
    /// SHA-256 hash of chunk embedding.
    pub embedding_hash: String,
    /// Position of chunk in document (0-indexed).
    pub position: usize,
    /// Byte offset in the original document.
    pub byte_offset: usize,
    /// Byte length of the chunk.
    pub byte_length: usize,
}

impl ChunkVersion {
    /// Create a new chunk version.
    pub fn new(
        chunk_id: String,
        content: &str,
        embedding: &[f32],
        position: usize,
        byte_offset: usize,
        byte_length: usize,
    ) -> Self {
        Self {
            chunk_id,
            content_hash: Self::hash_content(content),
            embedding_hash: Self::hash_embedding(embedding),
            position,
            byte_offset,
            byte_length,
        }
    }

    /// Compute SHA-256 hash of content.
    pub fn hash_content(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Compute SHA-256 hash of embedding.
    pub fn hash_embedding(embedding: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for f in embedding {
            hasher.update(f.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

/// Compression method for version storage.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CompressionMethod {
    /// No compression.
    #[default]
    None,
    /// Gzip compression.
    Gzip,
    /// Zstd compression (recommended for best ratio/speed).
    Zstd,
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
    /// SHA-256 hash of the metadata (for detecting metadata-only changes).
    #[serde(default)]
    pub metadata_hash: Option<String>,
    /// How the content is stored.
    pub storage: VersionStorageType,
    /// Size of content in bytes.
    pub size_bytes: usize,
    /// Chunk-level version information (for tracking embedding changes).
    #[serde(default)]
    pub chunks: Option<Vec<ChunkVersion>>,
    /// Compression method used for stored content.
    #[serde(default)]
    pub compression: CompressionMethod,
    /// Optional tags for the version (e.g., "release", "milestone").
    #[serde(default)]
    pub tags: Vec<String>,
}

impl DocumentVersion {
    /// Compute SHA-256 hash of content.
    pub fn compute_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Compute SHA-256 hash of metadata.
    pub fn compute_metadata_hash(metadata: &serde_json::Value) -> String {
        let mut hasher = Sha256::new();
        // Serialize metadata in a canonical way
        let json = serde_json::to_string(metadata).unwrap_or_default();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get the number of chunks in this version.
    pub fn chunk_count(&self) -> usize {
        self.chunks.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// Check if this version has chunk information.
    pub fn has_chunks(&self) -> bool {
        self.chunks.as_ref().map(|c| !c.is_empty()).unwrap_or(false)
    }

    /// Add a tag to this version.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Check if version has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Compression utilities for version content.
pub mod compression {
    use super::CompressionMethod;
    use std::io::{Read, Write};

    /// Compress content using the specified method.
    pub fn compress(content: &[u8], method: CompressionMethod) -> crate::error::Result<Vec<u8>> {
        match method {
            CompressionMethod::None => Ok(content.to_vec()),
            CompressionMethod::Gzip => {
                use flate2::write::GzEncoder;
                use flate2::Compression;
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(content)?;
                Ok(encoder.finish()?)
            }
            CompressionMethod::Zstd => {
                // Use zstd compression level 3 (good balance of speed/ratio)
                let compressed = zstd::encode_all(content, 3)?;
                Ok(compressed)
            }
        }
    }

    /// Decompress content using the specified method.
    pub fn decompress(data: &[u8], method: CompressionMethod) -> crate::error::Result<Vec<u8>> {
        match method {
            CompressionMethod::None => Ok(data.to_vec()),
            CompressionMethod::Gzip => {
                use flate2::read::GzDecoder;
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
            CompressionMethod::Zstd => {
                let decompressed = zstd::decode_all(data)?;
                Ok(decompressed)
            }
        }
    }

    /// Get the compression ratio (original_size / compressed_size).
    pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if compressed_size == 0 {
            return 1.0;
        }
        original_size as f64 / compressed_size as f64
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
    /// Number of chunks in this version.
    #[serde(default)]
    pub chunk_count: usize,
    /// Tags associated with this version.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Compression method used.
    #[serde(default)]
    pub compression: CompressionMethod,
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
            chunk_count: v.chunk_count(),
            tags: v.tags.clone(),
            compression: v.compression,
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

    fn create_test_version(id: &str, doc_id: &str, num: u64, content: &str) -> DocumentVersion {
        DocumentVersion {
            version_id: id.to_string(),
            document_id: doc_id.to_string(),
            version_number: num,
            timestamp: Utc::now(),
            author: None,
            change_type: if num == 1 {
                ChangeType::Created
            } else {
                ChangeType::ContentModified
            },
            content_hash: DocumentVersion::compute_hash(content),
            metadata_hash: None,
            storage: VersionStorageType::Full {
                content: content.to_string(),
            },
            size_bytes: content.len(),
            chunks: None,
            compression: CompressionMethod::None,
            tags: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_in_memory_storage() {
        let storage = InMemoryVersionStorage::new();

        let version = create_test_version("v1", "doc1", 1, "Hello, World!");

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
            let version =
                create_test_version(&format!("v{}", i), "doc1", i, &format!("Content v{}", i));
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
        let v1 = create_test_version("v1", "doc1", 1, "Hello, World!");
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
            metadata_hash: None,
            storage: VersionStorageType::Delta {
                base_version: "v1".to_string(),
                operations: vec![DeltaOperation::Replace {
                    position: 7,
                    length: 5,
                    text: "Rust".to_string(),
                }],
            },
            size_bytes: 12,
            chunks: None,
            compression: CompressionMethod::None,
            tags: Vec::new(),
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

        let version = create_test_version("v1", "doc1", 1, "Test content");

        storage.store_version(version).await.unwrap();

        let content = storage.reconstruct_content("v1").await.unwrap();
        assert_eq!(content, "Test content");
    }

    #[tokio::test]
    async fn test_chunk_version() {
        let chunk = ChunkVersion::new(
            "chunk1".to_string(),
            "Test chunk content",
            &[0.1, 0.2, 0.3, 0.4],
            0,
            0,
            18,
        );

        assert_eq!(chunk.position, 0);
        assert!(!chunk.content_hash.is_empty());
        assert!(!chunk.embedding_hash.is_empty());
    }

    #[tokio::test]
    async fn test_compression() {
        let original = b"Hello, World! This is test content for compression testing.";

        // Test gzip compression
        let compressed = compression::compress(original, CompressionMethod::Gzip).unwrap();
        let decompressed = compression::decompress(&compressed, CompressionMethod::Gzip).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());

        // Test zstd compression
        let compressed = compression::compress(original, CompressionMethod::Zstd).unwrap();
        let decompressed = compression::decompress(&compressed, CompressionMethod::Zstd).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());

        // Test no compression
        let compressed = compression::compress(original, CompressionMethod::None).unwrap();
        assert_eq!(original.as_slice(), compressed.as_slice());
    }

    #[test]
    fn test_document_version_helpers() {
        let mut version = create_test_version("v1", "doc1", 1, "Test content");

        // Test tag management
        assert!(!version.has_tag("important"));
        version.add_tag("important");
        assert!(version.has_tag("important"));

        // Test duplicate tag prevention
        version.add_tag("important");
        assert_eq!(version.tags.len(), 1);

        // Test chunk count
        assert_eq!(version.chunk_count(), 0);
        assert!(!version.has_chunks());

        version.chunks = Some(vec![ChunkVersion {
            chunk_id: "c1".to_string(),
            content_hash: "hash".to_string(),
            embedding_hash: "ehash".to_string(),
            position: 0,
            byte_offset: 0,
            byte_length: 10,
        }]);
        assert_eq!(version.chunk_count(), 1);
        assert!(version.has_chunks());
    }
}

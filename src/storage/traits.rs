//! Storage trait definitions.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// An indexed document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedDocument {
    /// Document ID
    pub id: String,
    /// Source ID
    pub source_id: String,
    /// File path or URI
    pub path: String,
    /// MIME type
    pub mime_type: String,
    /// File size in bytes
    pub size: u64,
    /// Full text content
    pub content: String,
    /// Last modified time
    pub modified_at: DateTime<Utc>,
    /// Indexed time
    pub indexed_at: DateTime<Utc>,
    /// Metadata
    pub metadata: serde_json::Value,
}

/// A vector chunk for storage.
#[derive(Debug, Clone)]
pub struct VectorChunk {
    /// Chunk ID
    pub id: String,
    /// Document ID
    pub document_id: String,
    /// Chunk text
    pub text: String,
    /// Embedding vector
    pub vector: Vec<f32>,
    /// Start offset in document
    pub start_offset: usize,
    /// End offset in document
    pub end_offset: usize,
}

/// Search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Query text
    pub text: String,
    /// Query embedding
    pub embedding: Option<Vec<f32>>,
    /// Maximum results
    pub limit: usize,
    /// Vector weight (0.0 = full-text only, 1.0 = vector only)
    pub vector_weight: f32,
    /// Source filter
    pub source_id: Option<String>,
    /// File type filter
    pub file_types: Vec<String>,
}

/// Search result from storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSearchResult {
    /// Document ID
    pub document_id: String,
    /// Chunk ID
    pub chunk_id: String,
    /// Matched text
    pub text: String,
    /// Relevance score
    pub score: f32,
    /// Vector similarity score
    pub vector_score: Option<f32>,
    /// BM25 score
    pub text_score: Option<f32>,
}

/// Index statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Number of documents
    pub document_count: usize,
    /// Number of chunks
    pub chunk_count: usize,
    /// Storage size in bytes
    pub storage_bytes: u64,
}

/// Trait for storage backends.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store a document and its vector chunks.
    async fn store(
        &self,
        doc: IndexedDocument,
        vectors: Vec<VectorChunk>,
    ) -> crate::error::Result<()>;

    /// Search for documents.
    async fn search(&self, query: SearchQuery) -> crate::error::Result<Vec<StorageSearchResult>>;

    /// Remove a document by ID.
    async fn remove(&self, doc_id: &str) -> crate::error::Result<()>;

    /// Get a document by ID.
    async fn get(&self, doc_id: &str) -> crate::error::Result<Option<IndexedDocument>>;

    /// Get storage statistics.
    async fn stats(&self) -> crate::error::Result<StorageStats>;
}

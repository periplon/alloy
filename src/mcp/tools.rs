//! MCP tool implementations for Alloy.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Request to index a path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPathRequest {
    /// Path to index (local path or S3 URI)
    pub path: String,
    /// Optional glob pattern to filter files
    #[serde(default)]
    pub pattern: Option<String>,
    /// Whether to watch for changes
    #[serde(default)]
    pub watch: bool,
    /// Recursive indexing
    #[serde(default = "default_true")]
    pub recursive: bool,
}

fn default_true() -> bool {
    true
}

/// Response from indexing operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPathResponse {
    /// Unique source ID
    pub source_id: String,
    /// Number of documents indexed
    pub documents_indexed: usize,
    /// Number of chunks created
    pub chunks_created: usize,
    /// Whether watching is enabled
    pub watching: bool,
    /// Status message
    pub message: String,
}

/// Search request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Search query
    pub query: String,
    /// Maximum number of results
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Weight for vector search (0.0 to 1.0)
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,
    /// Optional source filter
    #[serde(default)]
    pub source_id: Option<String>,
    /// Optional file type filter
    #[serde(default)]
    pub file_types: Vec<String>,
}

fn default_limit() -> usize {
    10
}

fn default_vector_weight() -> f32 {
    0.5
}

/// Search result item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID
    pub document_id: String,
    /// Chunk ID
    pub chunk_id: String,
    /// Source ID
    pub source_id: String,
    /// File path or URI
    pub path: String,
    /// Matched content
    pub content: String,
    /// Relevance score (0.0 to 1.0)
    pub score: f32,
    /// Highlighted matches
    #[serde(default)]
    pub highlights: Vec<String>,
    /// Metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Search response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results
    pub results: Vec<SearchResult>,
    /// Total number of matches
    pub total_matches: usize,
    /// Query execution time in milliseconds
    pub took_ms: u64,
    /// Whether query expansion was applied
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_expanded: Option<bool>,
    /// Expanded query text (if expansion was applied)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expanded_query: Option<String>,
    /// Whether reranking was applied
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reranked: Option<bool>,
}

/// Get document request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetDocumentRequest {
    /// Document ID
    pub document_id: String,
    /// Whether to include full content
    #[serde(default = "default_true")]
    pub include_content: bool,
}

/// Document details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentDetails {
    /// Document ID
    pub document_id: String,
    /// Source ID
    pub source_id: String,
    /// File path or URI
    pub path: String,
    /// MIME type
    pub mime_type: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Number of chunks
    pub chunk_count: usize,
    /// Document content (if requested)
    #[serde(default)]
    pub content: Option<String>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
    /// Indexed timestamp
    pub indexed_at: DateTime<Utc>,
    /// Metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// List sources response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListSourcesResponse {
    /// List of sources
    pub sources: Vec<SourceInfo>,
}

/// Source information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Source ID
    pub source_id: String,
    /// Source type (local, s3)
    pub source_type: String,
    /// Base path or URI
    pub path: String,
    /// Number of documents
    pub document_count: usize,
    /// Whether watching is enabled
    pub watching: bool,
    /// Last scan timestamp
    pub last_scan: DateTime<Utc>,
    /// Status
    pub status: String,
}

/// Remove source request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveSourceRequest {
    /// Source ID to remove
    pub source_id: String,
}

/// Remove source response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveSourceResponse {
    /// Whether removal was successful
    pub success: bool,
    /// Number of documents removed
    pub documents_removed: usize,
    /// Status message
    pub message: String,
}

/// Index statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of sources
    pub source_count: usize,
    /// Total number of documents
    pub document_count: usize,
    /// Total number of chunks
    pub chunk_count: usize,
    /// Total storage size in bytes
    pub storage_bytes: u64,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Storage backend type
    pub storage_backend: String,
    /// Embedding provider
    pub embedding_provider: String,
    /// Uptime in seconds
    pub uptime_secs: u64,
}

/// Configuration update request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureRequest {
    /// Configuration updates (partial)
    pub updates: serde_json::Value,
}

/// Configuration response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureResponse {
    /// Whether update was successful
    pub success: bool,
    /// Current configuration
    pub config: serde_json::Value,
    /// Status message
    pub message: String,
}

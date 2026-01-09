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

/// Cluster documents response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterDocumentsResponse {
    /// List of clusters
    pub clusters: Vec<ClusterInfo>,
    /// Document IDs that are outliers (not assigned to any cluster)
    pub outliers: Vec<String>,
    /// Clustering quality metrics
    pub metrics: ClusterMetrics,
    /// Algorithm used for clustering
    pub algorithm: String,
    /// Total documents processed
    pub total_documents: usize,
}

/// Information about a single cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    /// Cluster identifier
    pub cluster_id: usize,
    /// Human-readable label for the cluster
    pub label: String,
    /// Keywords describing the cluster content
    pub keywords: Vec<String>,
    /// Number of documents in this cluster
    pub size: usize,
    /// Coherence score (0.0 to 1.0, higher is more coherent)
    pub coherence_score: f64,
    /// Representative document IDs (closest to cluster centroid)
    pub representative_docs: Vec<String>,
}

/// Clustering quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMetrics {
    /// Silhouette score (-1 to 1, higher is better)
    pub silhouette_score: f64,
    /// Number of clusters
    pub num_clusters: usize,
    /// Number of outliers (for DBSCAN)
    pub num_outliers: usize,
}

// ============================================================================
// Deduplication Types
// ============================================================================

/// Check duplicate request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckDuplicateRequest {
    /// Content to check for duplicates.
    pub content: String,
    /// Optional document ID for the check.
    #[serde(default)]
    pub document_id: Option<String>,
}

/// Check duplicate response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckDuplicateResponse {
    /// Whether the content is a duplicate.
    pub is_duplicate: bool,
    /// ID of the original document if this is a duplicate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duplicate_of: Option<String>,
    /// Similarity score (1.0 for exact match, 0.0-1.0 for fuzzy).
    pub similarity: f32,
    /// Strategy that detected the duplicate.
    pub strategy: String,
    /// Human-readable message.
    pub message: String,
}

/// Deduplication statistics response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationStatsResponse {
    /// Whether deduplication is enabled.
    pub enabled: bool,
    /// Deduplication strategy in use.
    pub strategy: String,
    /// Similarity threshold for fuzzy matching.
    pub threshold: f32,
    /// Action taken when duplicates are found.
    pub action: String,
    /// Number of hash functions for MinHash.
    pub minhash_num_hashes: usize,
    /// Shingle size for MinHash.
    pub shingle_size: usize,
}

/// Clear deduplication index response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearDeduplicationResponse {
    /// Whether the operation was successful.
    pub success: bool,
    /// Status message.
    pub message: String,
}

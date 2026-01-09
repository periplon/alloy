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

// ============================================================================
// Versioning Types
// ============================================================================

/// Get document history response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetDocumentHistoryResponse {
    /// Document ID.
    pub document_id: String,
    /// List of version metadata.
    pub versions: Vec<VersionInfo>,
    /// Total number of versions.
    pub total_versions: usize,
    /// Message.
    pub message: String,
}

/// Version information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Version ID.
    pub version_id: String,
    /// Sequential version number.
    pub version_number: u64,
    /// When this version was created.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// User who made the change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Type of change.
    pub change_type: String,
    /// Content size in bytes.
    pub size_bytes: usize,
    /// Content hash (SHA-256).
    pub content_hash: String,
}

/// Diff versions response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffVersionsResponse {
    /// Document ID.
    pub document_id: String,
    /// Version A info.
    pub version_a: VersionInfo,
    /// Version B info.
    pub version_b: VersionInfo,
    /// Unified diff output.
    pub unified_diff: String,
    /// Statistics about the diff.
    pub stats: DiffStatsInfo,
    /// Message.
    pub message: String,
}

/// Diff statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffStatsInfo {
    /// Lines added.
    pub lines_added: usize,
    /// Lines removed.
    pub lines_removed: usize,
    /// Lines unchanged.
    pub lines_unchanged: usize,
}

/// Restore version response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreVersionResponse {
    /// Document ID.
    pub document_id: String,
    /// Restored version info.
    pub restored_version: VersionInfo,
    /// The version that was restored from.
    pub restored_from: String,
    /// Success status.
    pub success: bool,
    /// Message.
    pub message: String,
}

/// Get version content response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetVersionContentResponse {
    /// Version ID.
    pub version_id: String,
    /// Document ID.
    pub document_id: String,
    /// Version number.
    pub version_number: u64,
    /// Full content of the version.
    pub content: String,
    /// Content size in bytes.
    pub size_bytes: usize,
    /// Message.
    pub message: String,
}

/// Versioning statistics response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningStatsResponse {
    /// Whether versioning is enabled.
    pub enabled: bool,
    /// Storage type.
    pub storage: String,
    /// Delta threshold.
    pub delta_threshold: usize,
    /// Total versions across all documents.
    pub total_versions: Option<usize>,
    /// Total storage used by versions.
    pub total_size_bytes: Option<u64>,
    /// Retention policy info.
    pub retention: VersioningRetentionInfo,
}

/// Versioning retention info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningRetentionInfo {
    /// Minimum versions to keep.
    pub min_versions: usize,
    /// Maximum versions to keep.
    pub max_versions: usize,
    /// Minimum age in days.
    pub min_age_days: u32,
    /// Maximum age in days.
    pub max_age_days: u32,
    /// Keep full versions.
    pub keep_full_versions: bool,
    /// Auto cleanup enabled.
    pub auto_cleanup: bool,
}

// ============================================================================
// Operations Types (Metrics, Backup, Cache)
// ============================================================================

/// Get metrics response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetMetricsResponse {
    /// Prometheus-format metrics text.
    pub prometheus: String,
    /// JSON-format metrics.
    pub json: serde_json::Value,
    /// Status message.
    pub message: String,
}

/// Create backup response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateBackupResponse {
    /// Backup ID.
    pub backup_id: String,
    /// Path to the backup file.
    pub path: String,
    /// Number of documents in the backup.
    pub document_count: usize,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Success status.
    pub success: bool,
    /// Status message.
    pub message: String,
}

/// List backups response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListBackupsResponse {
    /// List of backup metadata.
    pub backups: Vec<BackupInfo>,
    /// Status message.
    pub message: String,
}

/// Backup information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    /// Backup ID.
    pub backup_id: String,
    /// When the backup was created.
    pub created_at: DateTime<Utc>,
    /// Alloy version.
    pub version: String,
    /// Number of documents.
    pub document_count: usize,
    /// Number of chunks.
    pub chunk_count: usize,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Restore backup response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreBackupResponse {
    /// Backup ID that was restored.
    pub backup_id: String,
    /// Number of documents restored.
    pub documents_restored: usize,
    /// Number of chunks restored.
    pub chunks_restored: usize,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Success status.
    pub success: bool,
    /// Status message.
    pub message: String,
}

/// Export documents response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportDocumentsResponse {
    /// Path to the export file.
    pub path: String,
    /// Number of documents exported.
    pub document_count: usize,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Success status.
    pub success: bool,
    /// Status message.
    pub message: String,
}

/// Import documents response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportDocumentsResponse {
    /// Number of documents imported.
    pub documents_imported: usize,
    /// Number of chunks imported.
    pub chunks_imported: usize,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Success status.
    pub success: bool,
    /// Status message.
    pub message: String,
}

/// Get cache stats response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetCacheStatsResponse {
    /// Whether caching is enabled.
    pub enabled: bool,
    /// Number of embedding entries.
    pub embedding_entries: u64,
    /// Number of result entries.
    pub result_entries: u64,
    /// Approximate size of embedding cache in bytes.
    pub embedding_size_bytes: u64,
    /// Approximate size of result cache in bytes.
    pub result_size_bytes: u64,
    /// Cache configuration.
    pub config: CacheConfigInfo,
    /// Status message.
    pub message: String,
}

/// Cache configuration info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfigInfo {
    /// Maximum entries.
    pub max_entries: u64,
    /// TTL in seconds.
    pub ttl_secs: u64,
    /// Whether embedding caching is enabled.
    pub cache_embeddings: bool,
    /// Whether result caching is enabled.
    pub cache_results: bool,
}

/// Clear cache response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearCacheResponse {
    /// Success status.
    pub success: bool,
    /// Status message.
    pub message: String,
}

// ============================================================================
// Security Types (Authentication and ACL)
// ============================================================================

/// Get authentication status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetAuthStatusResponse {
    /// Whether authentication is enabled.
    pub enabled: bool,
    /// Authentication method.
    pub method: String,
    /// Whether the current request is authenticated.
    pub is_authenticated: bool,
    /// Current user ID (if authenticated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Current user roles.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Status message.
    pub message: String,
}

/// Set document ACL request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetDocumentAclRequest {
    /// Document ID.
    pub document_id: String,
    /// Owner user ID (optional, uses current user if not set).
    #[serde(default)]
    pub owner: Option<String>,
    /// ACL entries.
    pub entries: Vec<AclEntryParam>,
    /// Whether to inherit from source.
    #[serde(default)]
    pub inherit_from_source: Option<bool>,
}

/// ACL entry parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclEntryParam {
    /// Principal type: "user", "role", "group", "everyone", "authenticated".
    pub principal_type: String,
    /// Principal ID (for user, role, group).
    #[serde(default)]
    pub principal_id: Option<String>,
    /// Permissions: "read", "write", "delete", "admin".
    pub permissions: Vec<String>,
}

/// Set document ACL response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetDocumentAclResponse {
    /// Document ID.
    pub document_id: String,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Number of ACL entries set.
    pub entry_count: usize,
    /// Status message.
    pub message: String,
}

/// Get document ACL response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetDocumentAclResponse {
    /// Document ID.
    pub document_id: String,
    /// Owner user ID.
    pub owner: String,
    /// ACL entries.
    pub entries: Vec<AclEntryInfo>,
    /// Whether inheriting from source.
    pub inherit_from_source: bool,
    /// Whether the document is public.
    pub is_public: bool,
    /// When the ACL was created.
    pub created_at: DateTime<Utc>,
    /// When the ACL was last updated.
    pub updated_at: DateTime<Utc>,
    /// Status message.
    pub message: String,
}

/// ACL entry info for responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclEntryInfo {
    /// Principal type.
    pub principal_type: String,
    /// Principal ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub principal_id: Option<String>,
    /// Permissions.
    pub permissions: Vec<String>,
}

/// Check permission request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckPermissionRequest {
    /// Document ID.
    pub document_id: String,
    /// User ID to check (optional, uses current user if not set).
    #[serde(default)]
    pub user_id: Option<String>,
    /// Permission to check.
    pub permission: String,
}

/// Check permission response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckPermissionResponse {
    /// Document ID.
    pub document_id: String,
    /// User ID that was checked.
    pub user_id: String,
    /// Permission that was checked.
    pub permission: String,
    /// Whether permission is granted.
    pub allowed: bool,
    /// Reason for the decision.
    pub reason: String,
    /// All permissions the user has on this document.
    pub granted_permissions: Vec<String>,
}

/// Delete document ACL response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteDocumentAclResponse {
    /// Document ID.
    pub document_id: String,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Status message.
    pub message: String,
}

/// Get ACL stats response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetAclStatsResponse {
    /// Whether ACL is enabled.
    pub enabled: bool,
    /// ACL configuration.
    pub config: AclConfigInfo,
    /// Number of documents with ACLs.
    pub document_acl_count: usize,
    /// Number of sources with ACLs.
    pub source_acl_count: usize,
    /// Role definitions.
    pub roles: Vec<RoleInfo>,
    /// Status message.
    pub message: String,
}

/// ACL configuration info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclConfigInfo {
    /// Default public access.
    pub default_public: bool,
    /// Default authenticated read.
    pub default_authenticated_read: bool,
    /// Enforce on search.
    pub enforce_on_search: bool,
    /// Enforce on get.
    pub enforce_on_get: bool,
    /// Enforce on delete.
    pub enforce_on_delete: bool,
}

/// Role info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleInfo {
    /// Role name.
    pub name: String,
    /// Roles this role inherits from.
    pub inherits_from: Vec<String>,
    /// Direct permissions.
    pub permissions: Vec<String>,
}

/// Generate token response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTokenResponse {
    /// Generated JWT token.
    pub token: String,
    /// User ID.
    pub user_id: String,
    /// Roles.
    pub roles: Vec<String>,
    /// Expiry time (Unix timestamp).
    pub expires_at: i64,
    /// Status message.
    pub message: String,
}

//! Configuration settings for Alloy MCP server.

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub processing: ProcessingConfig,
    pub search: SearchConfig,
    pub indexing: IndexingConfig,
    pub operations: OperationsConfig,
    pub security: SecurityConfig,
    pub integration: IntegrationConfig,
}

impl Config {
    /// Load configuration from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(ConfigError::ReadFile)?;
        Self::parse(&content)
    }

    /// Parse configuration from a TOML string.
    pub fn parse(content: &str) -> Result<Self> {
        let config: Config = toml::from_str(content).map_err(ConfigError::Parse)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from default locations or use defaults.
    pub fn load() -> Result<Self> {
        // Try standard config locations
        let config_paths = [
            // Current directory
            PathBuf::from("config.toml"),
            PathBuf::from("alloy.toml"),
            // User config directory
            dirs::config_dir()
                .map(|p| p.join("alloy/config.toml"))
                .unwrap_or_default(),
            // Home directory
            dirs::home_dir()
                .map(|p| p.join(".alloy/config.toml"))
                .unwrap_or_default(),
        ];

        for path in &config_paths {
            if path.exists() {
                tracing::info!("Loading config from: {}", path.display());
                return Self::from_file(path);
            }
        }

        tracing::info!("No config file found, using defaults");
        Ok(Config::default())
    }

    /// Validate the configuration.
    fn validate(&self) -> Result<()> {
        // Validate embedding config
        if self.embedding.provider == EmbeddingProvider::Api {
            if self.embedding.api.base_url.is_empty() {
                return Err(ConfigError::MissingField("embedding.api.base_url".to_string()).into());
            }
            if self.embedding.api.model.is_empty() {
                return Err(ConfigError::MissingField("embedding.api.model".to_string()).into());
            }
        }

        // Validate storage config
        if self.storage.backend == StorageBackendType::Qdrant && self.storage.qdrant.url.is_empty()
        {
            return Err(ConfigError::MissingField("storage.qdrant.url".to_string()).into());
        }

        // Validate processing config
        if self.processing.chunk_size == 0 {
            return Err(ConfigError::Invalid("chunk_size must be > 0".to_string()).into());
        }

        Ok(())
    }

    /// Expand the data directory path.
    pub fn data_dir(&self) -> Result<PathBuf> {
        let expanded = shellexpand::tilde(&self.storage.data_dir);
        Ok(PathBuf::from(expanded.as_ref()))
    }
}

/// Server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Transport type: "stdio" or "http"
    pub transport: TransportType,
    /// HTTP port (only used when transport is "http")
    pub http_port: u16,
    /// Maximum concurrent indexing tasks
    pub max_concurrent_tasks: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            transport: TransportType::Stdio,
            http_port: 8080,
            max_concurrent_tasks: 4,
        }
    }
}

/// Transport type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransportType {
    Stdio,
    Http,
}

/// Embedding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Provider type: "local" or "api"
    pub provider: EmbeddingProvider,
    /// Model name for local embeddings
    pub model: String,
    /// API configuration
    pub api: ApiEmbeddingConfig,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::Local,
            model: "BAAI/bge-small-en-v1.5".to_string(),
            api: ApiEmbeddingConfig::default(),
        }
    }
}

/// Embedding provider enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    Local,
    Api,
}

/// API embedding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiEmbeddingConfig {
    /// Base URL for the embedding API
    pub base_url: String,
    /// Model name
    pub model: String,
    /// API key (loaded from environment if not set)
    pub api_key: Option<String>,
    /// Batch size for embedding requests
    pub batch_size: usize,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for ApiEmbeddingConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: None,
            batch_size: 100,
            timeout_secs: 30,
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Backend type: "embedded" or "qdrant"
    pub backend: StorageBackendType,
    /// Data directory for embedded storage
    pub data_dir: String,
    /// Qdrant configuration
    pub qdrant: QdrantConfig,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackendType::Embedded,
            data_dir: "~/.local/share/alloy".to_string(),
            qdrant: QdrantConfig::default(),
        }
    }
}

/// Storage backend type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackendType {
    Embedded,
    Qdrant,
}

/// Qdrant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QdrantConfig {
    /// Qdrant server URL
    pub url: String,
    /// Collection name
    pub collection: String,
    /// API key (optional)
    pub api_key: Option<String>,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            collection: "alloy".to_string(),
            api_key: None,
        }
    }
}

/// Processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessingConfig {
    /// Chunk size in tokens
    pub chunk_size: usize,
    /// Overlap between chunks in tokens
    pub chunk_overlap: usize,
    /// Image processing configuration
    pub image: ImageProcessingConfig,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 64,
            image: ImageProcessingConfig::default(),
        }
    }
}

/// Image processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageProcessingConfig {
    /// Enable OCR for images
    pub ocr: bool,
    /// Enable CLIP embeddings for images
    pub clip: bool,
    /// Enable Vision API descriptions
    pub vision_api: bool,
}

impl Default for ImageProcessingConfig {
    fn default() -> Self {
        Self {
            ocr: true,
            clip: true,
            vision_api: false,
        }
    }
}

/// Search configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SearchConfig {
    /// Reranking configuration
    pub reranking: RerankerConfig,
    /// Query expansion configuration
    pub expansion: QueryExpansionConfig,
    /// Clustering configuration
    pub clustering: ClusteringConfig,
    /// Caching configuration
    pub cache: CacheConfig,
}

/// Cross-encoder reranking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankerConfig {
    /// Enable reranking
    pub enabled: bool,
    /// Reranker type
    pub reranker_type: RerankerType,
    /// Model for local cross-encoder (used when reranker_type is LocalCrossEncoder)
    /// Options: bge-reranker-base, bge-reranker-v2-m3, jina-reranker-v1-turbo-en, jina-reranker-v2-base-multilingual
    pub model: String,
    /// Number of top candidates to rerank
    pub top_k: usize,
    /// Final number of results after reranking
    pub final_k: usize,
    /// Minimum score threshold for reranked results
    pub min_score: Option<f32>,
    /// API endpoint for cross-encoder API (used when reranker_type is CrossEncoder)
    pub api_url: Option<String>,
    /// API key for cross-encoder API (used when reranker_type is CrossEncoder)
    pub api_key: Option<String>,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reranker_type: RerankerType::ScoreBased,
            model: "bge-reranker-base".to_string(),
            top_k: 100,
            final_k: 10,
            min_score: None,
            api_url: None,
            api_key: None,
        }
    }
}

/// Reranker type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankerType {
    /// Score-based reranking using embedding similarity
    ScoreBased,
    /// Local cross-encoder model (fastembed-based, no API required)
    LocalCrossEncoder,
    /// Cross-encoder model (requires API)
    CrossEncoder,
    /// LLM-based reranking (requires API)
    Llm,
}

/// Query expansion configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QueryExpansionConfig {
    /// Enable query expansion
    pub enabled: bool,
    /// Expansion method
    pub method: QueryExpansionMethod,
    /// Maximum number of expansion terms
    pub max_expansions: usize,
    /// Similarity threshold for embedding-based expansion
    pub similarity_threshold: f32,
    /// Use pseudo-relevance feedback (expand based on top results)
    pub pseudo_relevance: bool,
    /// Number of top documents to use for pseudo-relevance feedback
    pub prf_top_k: usize,
    /// LLM API URL for LLM-based expansion (defaults to OpenAI)
    pub llm_api_url: Option<String>,
    /// LLM API key (loaded from environment if not set)
    pub llm_api_key: Option<String>,
    /// LLM model name for expansion
    pub llm_model: String,
}

impl Default for QueryExpansionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: QueryExpansionMethod::Embedding,
            max_expansions: 3,
            similarity_threshold: 0.7,
            pseudo_relevance: false,
            prf_top_k: 5,
            llm_api_url: None,
            llm_api_key: None,
            llm_model: "gpt-4o-mini".to_string(),
        }
    }
}

/// Query expansion method enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryExpansionMethod {
    /// Use embedding similarity to find related terms
    Embedding,
    /// Use string similarity and word stems
    Synonym,
    /// Combine embedding and synonym approaches
    Hybrid,
    /// Use language model to generate query variations
    Llm,
}

/// Semantic clustering configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusteringConfig {
    /// Enable clustering
    pub enabled: bool,
    /// Default clustering algorithm
    pub algorithm: ClusteringAlgorithm,
    /// Default number of clusters (0 = auto-detect)
    pub default_num_clusters: usize,
    /// Minimum cluster size for DBSCAN/HDBSCAN
    pub min_cluster_size: usize,
    /// Cache clustering results
    pub cache_results: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Generate labels for clusters
    pub generate_labels: bool,
    /// Maximum keywords per cluster label
    pub max_keywords: usize,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: ClusteringAlgorithm::KMeans,
            default_num_clusters: 5,
            min_cluster_size: 3,
            cache_results: true,
            cache_ttl_secs: 3600,
            generate_labels: true,
            max_keywords: 5,
        }
    }
}

/// Clustering algorithm enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusteringAlgorithm {
    /// K-Means clustering
    KMeans,
    /// DBSCAN density-based clustering
    Dbscan,
    /// Gaussian Mixture Model
    Gmm,
}

/// Cache configuration for search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    /// Maximum number of cached entries
    pub max_entries: u64,
    /// TTL for cached entries in seconds
    pub ttl_secs: u64,
    /// Cache embeddings
    pub cache_embeddings: bool,
    /// Cache search results
    pub cache_results: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            ttl_secs: 3600,
            cache_embeddings: true,
            cache_results: true,
        }
    }
}

// ============================================================================
// Data Management Configuration
// ============================================================================

/// Data management configuration (indexing section).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct IndexingConfig {
    /// Incremental update settings.
    pub incremental: IncrementalConfig,
    /// Deduplication settings.
    pub deduplication: DeduplicationConfig,
    /// Versioning settings.
    pub versioning: VersioningConfig,
}

/// Incremental indexing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IncrementalConfig {
    /// Enable incremental indexing.
    pub enabled: bool,
    /// Change detection strategy.
    pub change_detection: ChangeDetectionStrategy,
    /// Whether to verify content hash even if mtime unchanged.
    pub verify_hash: bool,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            change_detection: ChangeDetectionStrategy::Both,
            verify_hash: false,
        }
    }
}

/// Change detection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeDetectionStrategy {
    /// Use content hash (SHA-256).
    Hash,
    /// Use modification time only.
    Mtime,
    /// Use both hash and mtime (recommended).
    Both,
}

/// Deduplication configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeduplicationConfig {
    /// Enable deduplication.
    pub enabled: bool,
    /// Deduplication strategy.
    pub strategy: DeduplicationStrategy,
    /// Similarity threshold for fuzzy matching (0.0-1.0).
    pub threshold: f32,
    /// Action to take when duplicate is detected.
    pub action: DeduplicationAction,
    /// Number of hash functions for MinHash.
    pub minhash_num_hashes: usize,
    /// Shingle size for MinHash (number of consecutive tokens).
    pub shingle_size: usize,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: DeduplicationStrategy::Exact,
            threshold: 0.85,
            action: DeduplicationAction::Skip,
            minhash_num_hashes: 128,
            shingle_size: 3,
        }
    }
}

/// Deduplication strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeduplicationStrategy {
    /// Exact content hash matching.
    Exact,
    /// MinHash locality-sensitive hashing.
    MinHash,
    /// Embedding cosine similarity.
    Semantic,
}

/// Action to take when a duplicate is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeduplicationAction {
    /// Skip indexing duplicates.
    Skip,
    /// Index with reference to original.
    Flag,
    /// Update the existing document.
    Update,
}

/// Document versioning configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VersioningConfig {
    /// Enable versioning.
    pub enabled: bool,
    /// Storage type: "memory" or "file".
    pub storage: String,
    /// Store full version every N versions (0 = always full).
    pub delta_threshold: usize,
    /// Compression method for version content: "none", "gzip", or "zstd".
    pub compression: VersionCompressionType,
    /// Retention policy settings.
    pub retention: VersionRetentionConfig,
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            storage: "memory".to_string(),
            delta_threshold: 10,
            compression: VersionCompressionType::default(),
            retention: VersionRetentionConfig::default(),
        }
    }
}

/// Compression type for version storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VersionCompressionType {
    /// No compression (fastest, largest storage).
    #[default]
    None,
    /// Gzip compression (good compression, moderate speed).
    Gzip,
    /// Zstd compression (best ratio/speed balance, recommended).
    Zstd,
}

/// Version retention policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VersionRetentionConfig {
    /// Minimum number of versions to keep.
    pub min_versions: usize,
    /// Maximum number of versions to keep (0 = unlimited).
    pub max_versions: usize,
    /// Keep versions newer than this many days.
    pub min_age_days: u32,
    /// Delete versions older than this many days.
    pub max_age_days: u32,
    /// Always keep full versions (for delta reconstruction).
    pub keep_full_versions: bool,
    /// Enable automatic cleanup.
    pub auto_cleanup: bool,
    /// Cleanup interval in hours.
    pub cleanup_interval_hours: u32,
}

impl Default for VersionRetentionConfig {
    fn default() -> Self {
        Self {
            min_versions: 5,
            max_versions: 100,
            min_age_days: 7,
            max_age_days: 365,
            keep_full_versions: true,
            auto_cleanup: false,
            cleanup_interval_hours: 24,
        }
    }
}

// ============================================================================
// Operations Configuration
// ============================================================================

/// Operations configuration (metrics, health, backup).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct OperationsConfig {
    /// Metrics configuration.
    pub metrics: MetricsConfig,
    /// Health check configuration.
    pub health: HealthConfig,
    /// Backup configuration.
    pub backup: BackupConfig,
}

/// Metrics configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    /// Enable metrics collection.
    pub enabled: bool,
    /// Enable Prometheus endpoint at /metrics.
    pub prometheus_enabled: bool,
    /// Metrics update interval in seconds.
    pub update_interval_secs: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prometheus_enabled: true,
            update_interval_secs: 15,
        }
    }
}

/// Health check configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HealthConfig {
    /// Enable health endpoints.
    pub enabled: bool,
    /// Health check timeout in milliseconds.
    pub timeout_ms: u64,
    /// Include detailed health checks (storage, embedding).
    pub detailed_checks: bool,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_ms: 5000,
            detailed_checks: true,
        }
    }
}

/// Backup configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackupConfig {
    /// Enable backup functionality.
    pub enabled: bool,
    /// Backup directory (defaults to data_dir/backups).
    pub backup_dir: Option<String>,
    /// Maximum number of backups to keep (0 = unlimited).
    pub max_backups: usize,
    /// Include embeddings in backup (increases size significantly).
    pub include_embeddings: bool,
    /// Compress backups.
    pub compress: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backup_dir: None,
            max_backups: 10,
            include_embeddings: true,
            compress: false,
        }
    }
}

// ============================================================================
// Security Configuration
// ============================================================================

/// Security configuration (authentication and access control).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SecurityConfig {
    /// Authentication configuration.
    pub auth: AuthConfig,
    /// Access control configuration.
    pub acl: AclConfig,
}

/// Authentication configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    /// Enable authentication.
    pub enabled: bool,
    /// Authentication method.
    pub method: AuthMethod,
    /// API keys for API key authentication.
    /// Can be loaded from environment variable ALLOY_API_KEYS (comma-separated).
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// JWT configuration.
    pub jwt: JwtConfig,
    /// Basic auth credentials (username -> password hash).
    #[serde(default)]
    pub basic_auth: std::collections::HashMap<String, String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: AuthMethod::ApiKey,
            api_keys: Vec::new(),
            jwt: JwtConfig::default(),
            basic_auth: std::collections::HashMap::new(),
        }
    }
}

/// Authentication method enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// API key authentication (X-API-Key header or Bearer token).
    #[default]
    ApiKey,
    /// JWT (JSON Web Token) authentication.
    Jwt,
    /// Basic HTTP authentication.
    Basic,
}

/// JWT configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct JwtConfig {
    /// JWT secret (can be loaded from environment variable ALLOY_JWT_SECRET).
    pub secret: String,
    /// JWT issuer.
    pub issuer: String,
    /// JWT audience.
    pub audience: String,
    /// Token expiry in seconds (0 = no expiry check).
    pub expiry_secs: u64,
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret: String::new(),
            issuer: "alloy".to_string(),
            audience: "alloy-users".to_string(),
            expiry_secs: 3600,
        }
    }
}

/// Access control list (ACL) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AclConfig {
    /// Enable ACL enforcement.
    pub enabled: bool,
    /// Default public access for new documents.
    pub default_public: bool,
    /// Allow read access for any authenticated user by default.
    pub default_authenticated_read: bool,
    /// Enforce ACL on search operations.
    pub enforce_on_search: bool,
    /// Enforce ACL on document retrieval.
    pub enforce_on_get: bool,
    /// Enforce ACL on document deletion.
    pub enforce_on_delete: bool,
    /// Role definitions.
    #[serde(default)]
    pub roles: Vec<RoleDefinition>,
}

impl Default for AclConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_public: true,
            default_authenticated_read: true,
            enforce_on_search: true,
            enforce_on_get: true,
            enforce_on_delete: true,
            roles: vec![
                RoleDefinition {
                    name: "admin".to_string(),
                    inherits_from: Vec::new(),
                    permissions: vec![
                        "read".to_string(),
                        "write".to_string(),
                        "delete".to_string(),
                        "admin".to_string(),
                    ],
                },
                RoleDefinition {
                    name: "editor".to_string(),
                    inherits_from: vec!["viewer".to_string()],
                    permissions: vec!["write".to_string()],
                },
                RoleDefinition {
                    name: "viewer".to_string(),
                    inherits_from: Vec::new(),
                    permissions: vec!["read".to_string()],
                },
            ],
        }
    }
}

/// Role definition for ACL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleDefinition {
    /// Role name.
    pub name: String,
    /// Roles this role inherits from.
    #[serde(default)]
    pub inherits_from: Vec<String>,
    /// Direct permissions granted to this role.
    pub permissions: Vec<String>,
}

// ============================================================================
// Integration Configuration
// ============================================================================

/// Integration configuration (webhooks, REST API, Web UI).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct IntegrationConfig {
    /// Webhook configuration.
    pub webhooks: WebhooksConfig,
    /// REST API configuration.
    pub rest_api: RestApiConfig,
    /// Web UI configuration.
    pub web_ui: WebUiConfig,
}

/// Webhooks configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebhooksConfig {
    /// Enable webhooks.
    pub enabled: bool,
    /// Configured webhooks.
    #[serde(default)]
    pub endpoints: Vec<WebhookEndpointConfig>,
    /// Maximum retries for failed webhook deliveries.
    pub max_retries: usize,
    /// Timeout in seconds for webhook requests.
    pub timeout_secs: u64,
}

impl Default for WebhooksConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoints: Vec::new(),
            max_retries: 3,
            timeout_secs: 30,
        }
    }
}

/// Individual webhook endpoint configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEndpointConfig {
    /// Target URL for webhook payloads.
    pub url: String,
    /// Events to subscribe to.
    pub events: Vec<String>,
    /// Secret for HMAC signature.
    #[serde(default)]
    pub secret: Option<String>,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
}

/// REST API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RestApiConfig {
    /// Enable REST API.
    pub enabled: bool,
    /// API prefix.
    pub prefix: String,
    /// Enable CORS.
    pub enable_cors: bool,
    /// Allowed CORS origins.
    #[serde(default)]
    pub cors_origins: Vec<String>,
}

impl Default for RestApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prefix: "/api/v1".to_string(),
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// Web UI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebUiConfig {
    /// Enable web UI.
    pub enabled: bool,
    /// Path prefix for web UI.
    pub path_prefix: String,
}

impl Default for WebUiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path_prefix: "/ui".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.transport, TransportType::Stdio);
        assert_eq!(config.embedding.provider, EmbeddingProvider::Local);
        assert_eq!(config.storage.backend, StorageBackendType::Embedded);
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
            [server]
            transport = "http"
            http_port = 9090

            [embedding]
            provider = "local"
            model = "BAAI/bge-base-en-v1.5"

            [storage]
            backend = "embedded"
            data_dir = "/tmp/alloy"

            [processing]
            chunk_size = 256
            chunk_overlap = 32
        "#;

        let config = Config::parse(toml).unwrap();
        assert_eq!(config.server.transport, TransportType::Http);
        assert_eq!(config.server.http_port, 9090);
        assert_eq!(config.processing.chunk_size, 256);
    }

    #[test]
    fn test_validate_missing_api_url() {
        let toml = r#"
            [embedding]
            provider = "api"

            [embedding.api]
            base_url = ""
            model = "text-embedding-3-small"
        "#;

        let result = Config::parse(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_zero_chunk_size() {
        let toml = r#"
            [processing]
            chunk_size = 0
        "#;

        let result = Config::parse(toml);
        assert!(result.is_err());
    }
}

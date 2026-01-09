//! Configuration settings for Alloy MCP server.

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub processing: ProcessingConfig,
    pub search: SearchConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            embedding: EmbeddingConfig::default(),
            storage: StorageConfig::default(),
            processing: ProcessingConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(ConfigError::ReadFile)?;
        Self::from_str(&content)
    }

    /// Parse configuration from a TOML string.
    pub fn from_str(content: &str) -> Result<Self> {
        let config: Config = toml::from_str(content)
            .map_err(ConfigError::Parse)?;
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
        if self.storage.backend == StorageBackendType::Qdrant {
            if self.storage.qdrant.url.is_empty() {
                return Err(ConfigError::MissingField("storage.qdrant.url".to_string()).into());
            }
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            reranking: RerankerConfig::default(),
            expansion: QueryExpansionConfig::default(),
            clustering: ClusteringConfig::default(),
            cache: CacheConfig::default(),
        }
    }
}

/// Cross-encoder reranking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankerConfig {
    /// Enable reranking
    pub enabled: bool,
    /// Reranker type
    pub reranker_type: RerankerType,
    /// Number of top candidates to rerank
    pub top_k: usize,
    /// Final number of results after reranking
    pub final_k: usize,
    /// Minimum score threshold for reranked results
    pub min_score: Option<f32>,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reranker_type: RerankerType::ScoreBased,
            top_k: 100,
            final_k: 10,
            min_score: None,
        }
    }
}

/// Reranker type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankerType {
    /// Score-based reranking using embedding similarity
    ScoreBased,
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

        let config = Config::from_str(toml).unwrap();
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

        let result = Config::from_str(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_zero_chunk_size() {
        let toml = r#"
            [processing]
            chunk_size = 0
        "#;

        let result = Config::from_str(toml);
        assert!(result.is_err());
    }
}

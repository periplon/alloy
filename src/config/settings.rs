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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            embedding: EmbeddingConfig::default(),
            storage: StorageConfig::default(),
            processing: ProcessingConfig::default(),
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

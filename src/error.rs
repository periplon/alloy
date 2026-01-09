//! Error types for Alloy MCP server.

use thiserror::Error;

/// Main error type for Alloy operations.
#[derive(Error, Debug)]
pub enum AlloyError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Source error: {0}")]
    Source(#[from] SourceError),

    #[error("Processing error: {0}")]
    Processing(#[from] ProcessingError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Search error: {0}")]
    Search(#[from] SearchError),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Configuration-related errors.
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    ReadFile(#[source] std::io::Error),

    #[error("Failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),

    #[error("Invalid configuration: {0}")]
    Invalid(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Path expansion failed: {0}")]
    PathExpansion(String),
}

/// Source-related errors (local files, S3).
#[derive(Error, Debug)]
pub enum SourceError {
    #[error("Path not found: {0}")]
    PathNotFound(String),

    #[error("Access denied: {0}")]
    AccessDenied(String),

    #[error("Invalid URI: {0}")]
    InvalidUri(String),

    #[error("S3 error: {0}")]
    S3(String),

    #[error("Watch error: {0}")]
    Watch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Processing-related errors (text extraction, parsing).
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("Unsupported content type: {0}")]
    UnsupportedType(String),

    #[error("Extraction failed: {0}")]
    Extraction(String),

    #[error("PDF error: {0}")]
    Pdf(String),

    #[error("DOCX error: {0}")]
    Docx(String),

    #[error("OCR error: {0}")]
    Ocr(String),

    #[error("Chunking error: {0}")]
    Chunking(String),
}

/// Embedding-related errors.
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Batch too large: {0} (max {1})")]
    BatchTooLarge(usize, usize),
}

/// Storage-related errors.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Schema mismatch: {0}")]
    SchemaMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Search-related errors.
#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Fusion error: {0}")]
    Fusion(String),

    #[error("Timeout after {0}ms")]
    Timeout(u64),

    #[error("Reranking error: {0}")]
    Reranking(String),

    #[error("Query expansion error: {0}")]
    QueryExpansion(String),

    #[error("Clustering error: {0}")]
    Clustering(String),
}

/// Result type alias for Alloy operations.
pub type Result<T> = std::result::Result<T, AlloyError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AlloyError::Config(ConfigError::MissingField("embedding.model".to_string()));
        assert!(err.to_string().contains("embedding.model"));
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: AlloyError = io_err.into();
        assert!(matches!(err, AlloyError::Io(_)));
    }
}

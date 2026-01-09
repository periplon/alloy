//! Embedding module for vector embeddings.
//!
//! This module provides embedding providers for generating vector representations
//! of text and images. It supports both local (ONNX-based) and API-based embeddings.
//!
//! # Providers
//!
//! - [`LocalEmbeddingProvider`]: Uses fastembed-rs for local ONNX-based embeddings.
//!   Supports models like BGE, MiniLM, and Jina embeddings.
//!
//! - [`ApiEmbeddingProvider`]: OpenAI-compatible API provider. Works with OpenAI,
//!   Voyage AI, Cohere, and any OpenAI-compatible endpoint.
//!
//! # Batch Processing
//!
//! The [`BatchEmbeddingProcessor`] wraps any embedding provider to handle:
//! - Automatic batching of large inputs
//! - Rate limiting for API providers
//! - Retry logic with exponential backoff
//! - Progress reporting
//!
//! # Example
//!
//! ```rust,ignore
//! use alloy::embedding::{LocalEmbeddingProvider, BatchEmbeddingProcessor, BatchConfig};
//!
//! // Create a local provider
//! let provider = LocalEmbeddingProvider::new("BAAI/bge-small-en-v1.5", false)?;
//!
//! // Wrap with batch processor for large inputs
//! let processor = BatchEmbeddingProcessor::new(
//!     provider,
//!     BatchConfig::default().with_batch_size(32),
//! );
//!
//! // Embed texts
//! let texts = vec!["Hello, world!".to_string()];
//! let embeddings = processor.embed_all(&texts).await?;
//! ```

mod api;
mod batch;
mod local;
mod traits;

pub use api::ApiEmbeddingProvider;
pub use batch::{BatchConfig, BatchEmbeddingProcessor, ProgressBatchProcessor, ProgressCallback};
pub use local::LocalEmbeddingProvider;
pub use traits::EmbeddingProvider;

use crate::config::{EmbeddingConfig, EmbeddingProvider as EmbeddingProviderType};
use crate::error::Result;

/// Create an embedding provider from configuration.
pub fn create_provider(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingProvider>> {
    match config.provider {
        EmbeddingProviderType::Local => {
            let provider = LocalEmbeddingProvider::new(&config.model, false)?;
            Ok(Box::new(provider))
        }
        EmbeddingProviderType::Api => {
            let provider = ApiEmbeddingProvider::from_config(&config.api)?;
            Ok(Box::new(provider))
        }
    }
}

/// Create a batch processor from configuration.
pub fn create_batch_processor(
    config: &EmbeddingConfig,
    batch_config: BatchConfig,
) -> Result<Box<dyn EmbeddingProvider>> {
    match config.provider {
        EmbeddingProviderType::Local => {
            let provider = LocalEmbeddingProvider::new(&config.model, false)?;
            let processor = BatchEmbeddingProcessor::new(provider, batch_config);
            Ok(Box::new(processor))
        }
        EmbeddingProviderType::Api => {
            let provider = ApiEmbeddingProvider::from_config(&config.api)?;
            // For API providers, default to rate limiting
            let batch_config = if batch_config.requests_per_second == 0 {
                batch_config.with_rate_limit(10) // Conservative default for APIs
            } else {
                batch_config
            };
            let processor = BatchEmbeddingProcessor::new(provider, batch_config);
            Ok(Box::new(processor))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_provider_local() {
        // This test would require model download, so we just test config parsing
        let config = EmbeddingConfig::default();
        assert_eq!(config.provider, EmbeddingProviderType::Local);
    }

    #[test]
    fn test_create_provider_api_missing_key() {
        std::env::remove_var("OPENAI_API_KEY");

        let mut config = EmbeddingConfig::default();
        config.provider = EmbeddingProviderType::Api;
        config.api.api_key = None;

        let result = create_provider(&config);
        assert!(result.is_err());
    }
}

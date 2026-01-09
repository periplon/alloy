//! Embedding trait definitions.

use async_trait::async_trait;
use bytes::Bytes;

/// Trait for embedding providers.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for text.
    async fn embed(&self, texts: &[String]) -> crate::error::Result<Vec<Vec<f32>>>;

    /// Generate embeddings for images (if supported).
    async fn embed_images(&self, _images: &[Bytes]) -> crate::error::Result<Vec<Vec<f32>>> {
        Err(crate::error::EmbeddingError::Api("Image embedding not supported".to_string()).into())
    }

    /// Return the embedding dimension.
    fn dimension(&self) -> usize;

    /// Return the maximum batch size.
    fn max_batch_size(&self) -> usize {
        100
    }
}

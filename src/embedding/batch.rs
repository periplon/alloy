//! Batch processing with rate limiting for embedding providers.

use async_trait::async_trait;
use governor::{
    clock::DefaultClock, middleware::NoOpMiddleware, state::NotKeyed, Quota, RateLimiter,
};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{AlloyError, EmbeddingError, Result};

use super::EmbeddingProvider;

/// Check if an error is a rate limit error.
fn is_rate_limit_error(error: &AlloyError) -> bool {
    matches!(error, AlloyError::Embedding(EmbeddingError::RateLimited))
}

type DefaultRateLimiter = RateLimiter<NotKeyed, governor::state::InMemoryState, DefaultClock, NoOpMiddleware>;

/// Batch embedding processor with rate limiting and retry logic.
pub struct BatchEmbeddingProcessor<P: EmbeddingProvider> {
    provider: Arc<P>,
    rate_limiter: Option<Arc<DefaultRateLimiter>>,
    batch_size: usize,
    max_retries: usize,
    retry_delay: Duration,
}

/// Configuration for batch processing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of texts per batch
    pub batch_size: usize,
    /// Requests per second limit (0 = no limit)
    pub requests_per_second: u32,
    /// Maximum number of retries on failure
    pub max_retries: usize,
    /// Delay between retries
    pub retry_delay: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            requests_per_second: 0, // No limit by default
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
        }
    }
}

impl BatchConfig {
    /// Create a new batch config with rate limiting.
    pub fn with_rate_limit(mut self, requests_per_second: u32) -> Self {
        self.requests_per_second = requests_per_second;
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
}

impl<P: EmbeddingProvider + 'static> BatchEmbeddingProcessor<P> {
    /// Create a new batch processor with the given provider and configuration.
    pub fn new(provider: P, config: BatchConfig) -> Self {
        let rate_limiter = if config.requests_per_second > 0 {
            let quota = Quota::per_second(NonZeroU32::new(config.requests_per_second).unwrap());
            Some(Arc::new(RateLimiter::direct(quota)))
        } else {
            None
        };

        // Use the smaller of configured batch size and provider's max batch size
        let batch_size = config.batch_size.min(provider.max_batch_size());

        Self {
            provider: Arc::new(provider),
            rate_limiter,
            batch_size,
            max_retries: config.max_retries,
            retry_delay: config.retry_delay,
        }
    }

    /// Create a new batch processor with default configuration.
    pub fn with_defaults(provider: P) -> Self {
        Self::new(provider, BatchConfig::default())
    }

    /// Process a large batch of texts, splitting into smaller batches and handling rate limiting.
    pub async fn embed_all(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            let chunk_embeddings = self.embed_with_retry(chunk).await?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embed a batch with retry logic.
    async fn embed_with_retry(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            // Wait for rate limiter if configured
            if let Some(ref limiter) = self.rate_limiter {
                limiter.until_ready().await;
            }

            match self.provider.embed(texts).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => {
                    // Check if this is a rate limit error by checking the error type
                    let is_rate_limited = is_rate_limit_error(&e);

                    if attempt < self.max_retries {
                        let delay = if is_rate_limited {
                            // Exponential backoff for rate limiting
                            self.retry_delay * (2u32.pow(attempt as u32))
                        } else {
                            self.retry_delay
                        };

                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = self.max_retries,
                            delay_ms = delay.as_millis(),
                            "Embedding request failed, retrying: {}",
                            e
                        );

                        tokio::time::sleep(delay).await;
                    }

                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Get the underlying provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Get the effective batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[async_trait]
impl<P: EmbeddingProvider + 'static> EmbeddingProvider for BatchEmbeddingProcessor<P> {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_all(texts).await
    }

    async fn embed_images(&self, images: &[bytes::Bytes]) -> Result<Vec<Vec<f32>>> {
        // For images, we don't batch currently
        self.provider.embed_images(images).await
    }

    fn dimension(&self) -> usize {
        self.provider.dimension()
    }

    fn max_batch_size(&self) -> usize {
        // The batch processor can handle any size by chunking
        usize::MAX
    }
}

/// Progress callback for long-running batch operations.
pub type ProgressCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Extended batch processor with progress reporting.
pub struct ProgressBatchProcessor<P: EmbeddingProvider> {
    processor: BatchEmbeddingProcessor<P>,
    progress_callback: Option<ProgressCallback>,
}

impl<P: EmbeddingProvider + 'static> ProgressBatchProcessor<P> {
    /// Create a new progress batch processor.
    pub fn new(provider: P, config: BatchConfig) -> Self {
        Self {
            processor: BatchEmbeddingProcessor::new(provider, config),
            progress_callback: None,
        }
    }

    /// Set a progress callback.
    pub fn with_progress(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Process all texts with progress reporting.
    pub async fn embed_all(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let total = texts.len();
        let mut all_embeddings = Vec::with_capacity(total);
        let mut processed = 0;

        for chunk in texts.chunks(self.processor.batch_size) {
            let chunk_embeddings = self.processor.embed_with_retry(chunk).await?;
            processed += chunk.len();
            all_embeddings.extend(chunk_embeddings);

            if let Some(ref callback) = self.progress_callback {
                callback(processed, total);
            }
        }

        Ok(all_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock embedding provider for testing.
    struct MockProvider {
        dimension: usize,
        call_count: AtomicUsize,
        max_batch: usize,
    }

    impl MockProvider {
        fn new(dimension: usize) -> Self {
            Self {
                dimension,
                call_count: AtomicUsize::new(0),
                max_batch: 10,
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(texts.iter().map(|_| vec![0.0; self.dimension]).collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn max_batch_size(&self) -> usize {
            self.max_batch
        }
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let provider = MockProvider::new(384);
        let processor = BatchEmbeddingProcessor::new(
            provider,
            BatchConfig::default().with_batch_size(5),
        );

        // Process 12 texts, should result in 3 batches (5, 5, 2)
        let texts: Vec<String> = (0..12).map(|i| format!("text {}", i)).collect();
        let embeddings = processor.embed_all(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 12);
        assert_eq!(processor.provider().call_count(), 3);
    }

    #[tokio::test]
    async fn test_empty_input() {
        let provider = MockProvider::new(384);
        let processor = BatchEmbeddingProcessor::with_defaults(provider);

        let embeddings = processor.embed_all(&[]).await.unwrap();
        assert!(embeddings.is_empty());
    }

    #[tokio::test]
    async fn test_single_batch() {
        let provider = MockProvider::new(384);
        let processor = BatchEmbeddingProcessor::new(
            provider,
            BatchConfig::default().with_batch_size(100),
        );

        let texts: Vec<String> = (0..5).map(|i| format!("text {}", i)).collect();
        let embeddings = processor.embed_all(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 5);
        // Should respect provider's max_batch_size (10), not our larger batch_size (100)
        assert_eq!(processor.provider().call_count(), 1);
    }

    #[tokio::test]
    async fn test_provider_trait_impl() {
        let provider = MockProvider::new(384);
        let processor = BatchEmbeddingProcessor::with_defaults(provider);

        // Test through the EmbeddingProvider trait
        let texts: Vec<String> = (0..5).map(|i| format!("text {}", i)).collect();
        let embeddings: Result<Vec<Vec<f32>>> =
            EmbeddingProvider::embed(&processor, &texts).await;

        assert!(embeddings.is_ok());
        assert_eq!(embeddings.unwrap().len(), 5);
    }

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::default()
            .with_batch_size(50)
            .with_rate_limit(10)
            .with_max_retries(5);

        assert_eq!(config.batch_size, 50);
        assert_eq!(config.requests_per_second, 10);
        assert_eq!(config.max_retries, 5);
    }

    #[tokio::test]
    async fn test_progress_callback() {
        let provider = MockProvider::new(384);
        let progress = Arc::new(AtomicUsize::new(0));
        let progress_clone = progress.clone();

        let processor = ProgressBatchProcessor::new(
            provider,
            BatchConfig::default().with_batch_size(5),
        )
        .with_progress(Box::new(move |processed, _total| {
            progress_clone.store(processed, Ordering::SeqCst);
        }));

        let texts: Vec<String> = (0..12).map(|i| format!("text {}", i)).collect();
        let _ = processor.embed_all(&texts).await.unwrap();

        assert_eq!(progress.load(Ordering::SeqCst), 12);
    }
}

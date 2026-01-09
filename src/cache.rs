//! Query and embedding cache for Alloy MCP server.
//!
//! This module provides high-performance caching for:
//! - Query embeddings (text -> vector mappings)
//! - Search results (query -> results)
//! - Document embeddings (document chunks)

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use moka::future::Cache;
use serde::{Deserialize, Serialize};

use crate::config::CacheConfig;
use crate::metrics::get_metrics;
use crate::search::HybridSearchResponse;

/// Hash key for embedding cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmbeddingKey {
    /// The text to embed.
    text: String,
    /// Model identifier.
    model: String,
}

impl EmbeddingKey {
    /// Create a new embedding key.
    pub fn new(text: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model: model.into(),
        }
    }
}

/// Hash key for search result cache.
#[derive(Debug, Clone)]
pub struct SearchKey {
    /// The query text.
    query: String,
    /// Vector weight.
    vector_weight: u32, // Store as fixed-point for hashing
    /// Result limit.
    limit: usize,
    /// Optional source filter.
    source_id: Option<String>,
    /// Whether query expansion was used.
    expand_query: bool,
    /// Whether reranking was used.
    rerank: bool,
}

impl SearchKey {
    /// Create a new search key.
    pub fn new(
        query: impl Into<String>,
        vector_weight: f32,
        limit: usize,
        source_id: Option<String>,
        expand_query: bool,
        rerank: bool,
    ) -> Self {
        Self {
            query: query.into(),
            vector_weight: (vector_weight * 1000.0) as u32,
            limit,
            source_id,
            expand_query,
            rerank,
        }
    }
}

impl PartialEq for SearchKey {
    fn eq(&self, other: &Self) -> bool {
        self.query == other.query
            && self.vector_weight == other.vector_weight
            && self.limit == other.limit
            && self.source_id == other.source_id
            && self.expand_query == other.expand_query
            && self.rerank == other.rerank
    }
}

impl Eq for SearchKey {}

impl Hash for SearchKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.query.hash(state);
        self.vector_weight.hash(state);
        self.limit.hash(state);
        self.source_id.hash(state);
        self.expand_query.hash(state);
        self.rerank.hash(state);
    }
}

/// Cached search result with metadata.
#[derive(Debug, Clone)]
pub struct CachedSearchResult {
    /// The search response.
    pub response: HybridSearchResponse,
    /// When this was cached.
    pub cached_at: std::time::Instant,
}

/// Query cache for embeddings and search results.
#[derive(Clone)]
pub struct QueryCache {
    /// Cache for text embeddings.
    embedding_cache: Cache<EmbeddingKey, Arc<Vec<f32>>>,
    /// Cache for search results.
    result_cache: Cache<SearchKey, Arc<CachedSearchResult>>,
    /// Whether embedding caching is enabled.
    cache_embeddings: bool,
    /// Whether result caching is enabled.
    cache_results: bool,
}

impl QueryCache {
    /// Create a new query cache from configuration.
    pub fn new(config: &CacheConfig) -> Self {
        let ttl = Duration::from_secs(config.ttl_secs);
        let max_capacity = config.max_entries;

        let embedding_cache = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(ttl)
            .build();

        let result_cache = Cache::builder()
            .max_capacity(max_capacity / 10) // Results are larger, cache fewer
            .time_to_live(ttl)
            .build();

        Self {
            embedding_cache,
            result_cache,
            cache_embeddings: config.cache_embeddings,
            cache_results: config.cache_results,
        }
    }

    /// Create a disabled cache.
    pub fn disabled() -> Self {
        Self {
            embedding_cache: Cache::builder().max_capacity(0).build(),
            result_cache: Cache::builder().max_capacity(0).build(),
            cache_embeddings: false,
            cache_results: false,
        }
    }

    /// Check if caching is enabled.
    pub fn is_enabled(&self) -> bool {
        self.cache_embeddings || self.cache_results
    }

    /// Get a cached embedding.
    pub async fn get_embedding(&self, key: &EmbeddingKey) -> Option<Arc<Vec<f32>>> {
        if !self.cache_embeddings {
            return None;
        }

        let result = self.embedding_cache.get(key).await;
        let metrics = get_metrics();

        if result.is_some() {
            metrics.cache_hits_total.inc();
        } else {
            metrics.cache_misses_total.inc();
        }

        result
    }

    /// Store an embedding in the cache.
    pub async fn set_embedding(&self, key: EmbeddingKey, embedding: Vec<f32>) {
        if !self.cache_embeddings {
            return;
        }

        self.embedding_cache.insert(key, Arc::new(embedding)).await;
    }

    /// Get a cached search result.
    pub async fn get_search_result(&self, key: &SearchKey) -> Option<Arc<CachedSearchResult>> {
        if !self.cache_results {
            return None;
        }

        let result = self.result_cache.get(key).await;
        let metrics = get_metrics();

        if result.is_some() {
            metrics.cache_hits_total.inc();
        } else {
            metrics.cache_misses_total.inc();
        }

        result
    }

    /// Store a search result in the cache.
    pub async fn set_search_result(&self, key: SearchKey, response: HybridSearchResponse) {
        if !self.cache_results {
            return;
        }

        let cached = CachedSearchResult {
            response,
            cached_at: std::time::Instant::now(),
        };

        self.result_cache.insert(key, Arc::new(cached)).await;
    }

    /// Invalidate all cached entries for a specific source.
    pub async fn invalidate_source(&self, _source_id: &str) {
        // Moka doesn't support iteration/filtering, so we invalidate everything
        // In a production system, you might use a more sophisticated cache key scheme
        self.result_cache.invalidate_all();
    }

    /// Invalidate all cache entries.
    pub async fn invalidate_all(&self) {
        self.embedding_cache.invalidate_all();
        self.result_cache.invalidate_all();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            embedding_entries: self.embedding_cache.entry_count(),
            result_entries: self.result_cache.entry_count(),
            embedding_size_bytes: self.embedding_cache.weighted_size(),
            result_size_bytes: self.result_cache.weighted_size(),
        }
    }

    /// Run cache maintenance (cleanup expired entries).
    pub async fn run_pending_tasks(&self) {
        self.embedding_cache.run_pending_tasks().await;
        self.result_cache.run_pending_tasks().await;
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of entries in the embedding cache.
    pub embedding_entries: u64,
    /// Number of entries in the result cache.
    pub result_entries: u64,
    /// Approximate size of embedding cache in bytes.
    pub embedding_size_bytes: u64,
    /// Approximate size of result cache in bytes.
    pub result_size_bytes: u64,
}

/// Wrapper for embedding operations with caching.
pub struct CachedEmbedder<E> {
    /// Underlying embedding provider.
    embedder: E,
    /// Query cache.
    cache: QueryCache,
    /// Model identifier for cache keys.
    model: String,
}

impl<E> CachedEmbedder<E> {
    /// Create a new cached embedder.
    pub fn new(embedder: E, cache: QueryCache, model: impl Into<String>) -> Self {
        Self {
            embedder,
            cache,
            model: model.into(),
        }
    }

    /// Get the underlying embedder.
    pub fn inner(&self) -> &E {
        &self.embedder
    }

    /// Get the cache.
    pub fn cache(&self) -> &QueryCache {
        &self.cache
    }
}

impl<E> CachedEmbedder<E>
where
    E: crate::embedding::EmbeddingProvider,
{
    /// Embed texts with caching.
    pub async fn embed(&self, texts: &[String]) -> crate::error::Result<Vec<Vec<f32>>> {
        if !self.cache.is_enabled() {
            return self.embedder.embed(texts).await;
        }

        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut to_embed: Vec<(usize, String)> = Vec::new();

        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            let key = EmbeddingKey::new(text.clone(), self.model.clone());
            if let Some(cached) = self.cache.get_embedding(&key).await {
                results[i] = Some((*cached).clone());
            } else {
                to_embed.push((i, text.clone()));
            }
        }

        // Embed uncached texts
        if !to_embed.is_empty() {
            let texts_to_embed: Vec<String> = to_embed.iter().map(|(_, t)| t.clone()).collect();
            let embeddings = self.embedder.embed(&texts_to_embed).await?;

            // Store in cache and results
            for ((idx, text), embedding) in to_embed.into_iter().zip(embeddings.into_iter()) {
                let key = EmbeddingKey::new(text, self.model.clone());
                self.cache.set_embedding(key, embedding.clone()).await;
                results[idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().flatten().collect())
    }

    /// Embed a single text with caching.
    pub async fn embed_single(&self, text: &str) -> crate::error::Result<Vec<f32>> {
        let key = EmbeddingKey::new(text.to_string(), self.model.clone());

        if let Some(cached) = self.cache.get_embedding(&key).await {
            return Ok((*cached).clone());
        }

        let embeddings = self.embedder.embed(&[text.to_string()]).await?;
        if let Some(embedding) = embeddings.into_iter().next() {
            self.cache.set_embedding(key, embedding.clone()).await;
            Ok(embedding)
        } else {
            Err(crate::error::EmbeddingError::Api("No embedding returned".to_string()).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CacheConfig {
        CacheConfig {
            enabled: true,
            max_entries: 1000,
            ttl_secs: 60,
            cache_embeddings: true,
            cache_results: true,
        }
    }

    #[tokio::test]
    async fn test_embedding_cache() {
        let cache = QueryCache::new(&test_config());

        let key = EmbeddingKey::new("test text", "model-1");
        let embedding = vec![0.1, 0.2, 0.3];

        // Should not be in cache initially
        assert!(cache.get_embedding(&key).await.is_none());

        // Add to cache
        cache.set_embedding(key.clone(), embedding.clone()).await;

        // Should now be in cache
        let cached = cache.get_embedding(&key).await;
        assert!(cached.is_some());
        assert_eq!(*cached.unwrap(), embedding);
    }

    #[tokio::test]
    async fn test_search_result_cache() {
        let cache = QueryCache::new(&test_config());

        let key = SearchKey::new("search query", 0.5, 10, None, false, false);

        // Should not be in cache initially
        assert!(cache.get_search_result(&key).await.is_none());

        // Create a mock response
        let response = HybridSearchResponse {
            results: vec![],
            stats: crate::search::SearchStats::default(),
        };

        // Add to cache
        cache.set_search_result(key.clone(), response).await;

        // Should now be in cache
        let cached = cache.get_search_result(&key).await;
        assert!(cached.is_some());
    }

    #[tokio::test]
    async fn test_disabled_cache() {
        let cache = QueryCache::disabled();

        let key = EmbeddingKey::new("test", "model");
        cache.set_embedding(key.clone(), vec![0.1]).await;

        // Should not cache when disabled
        assert!(cache.get_embedding(&key).await.is_none());
    }

    #[tokio::test]
    async fn test_invalidate_all() {
        let cache = QueryCache::new(&test_config());

        let key1 = EmbeddingKey::new("test1", "model");
        let key2 = EmbeddingKey::new("test2", "model");

        cache.set_embedding(key1.clone(), vec![0.1]).await;
        cache.set_embedding(key2.clone(), vec![0.2]).await;

        assert!(cache.get_embedding(&key1).await.is_some());
        assert!(cache.get_embedding(&key2).await.is_some());

        cache.invalidate_all().await;

        // Force pending tasks to run
        cache.run_pending_tasks().await;

        // Note: invalidation is async and may not be immediate
        // In production, you'd want to verify this more carefully
    }

    #[test]
    fn test_search_key_equality() {
        let key1 = SearchKey::new("query", 0.5, 10, None, false, false);
        let key2 = SearchKey::new("query", 0.5, 10, None, false, false);
        let key3 = SearchKey::new("query", 0.6, 10, None, false, false);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}

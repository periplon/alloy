//! Clustering result caching module.
//!
//! Provides dedicated caching for clustering results with configurable TTL
//! and capacity limits.

use std::time::Duration;

use moka::future::Cache;
use serde::{Deserialize, Serialize};

use super::ClusteringResult;
use crate::config::ClusteringConfig;

/// Cache key for clustering results.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusteringCacheKey {
    /// Sorted list of document IDs included in clustering.
    pub document_ids_hash: String,
    /// Algorithm used.
    pub algorithm: String,
    /// Number of clusters requested.
    pub num_clusters: Option<usize>,
    /// Source ID filter (if any).
    pub source_id: Option<String>,
}

impl ClusteringCacheKey {
    /// Create a new cache key from document IDs and parameters.
    pub fn new(
        document_ids: &[String],
        algorithm: &str,
        num_clusters: Option<usize>,
        source_id: Option<&str>,
    ) -> Self {
        // Create a hash of sorted document IDs for efficient comparison
        let mut sorted_ids = document_ids.to_vec();
        sorted_ids.sort();
        let ids_hash = format!("{:x}", md5_hash(&sorted_ids.join(",")));

        Self {
            document_ids_hash: ids_hash,
            algorithm: algorithm.to_string(),
            num_clusters,
            source_id: source_id.map(String::from),
        }
    }

    /// Convert to string for cache storage.
    pub fn to_cache_string(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.document_ids_hash,
            self.algorithm,
            self.num_clusters.map(|n| n.to_string()).unwrap_or_default(),
            self.source_id.as_deref().unwrap_or("")
        )
    }
}

/// Simple MD5-like hash for cache keys (using a basic hash function).
fn md5_hash(input: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

/// Cached clustering result with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedClusteringResult {
    /// The clustering result.
    pub result: ClusteringResult,
    /// When the result was cached.
    pub cached_at: chrono::DateTime<chrono::Utc>,
    /// Number of times this result was accessed from cache.
    pub hit_count: u64,
}

/// Statistics for the clustering cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringCacheStats {
    /// Total number of cache entries.
    pub entry_count: u64,
    /// Total number of cache hits.
    pub total_hits: u64,
    /// Total number of cache misses.
    pub total_misses: u64,
    /// Cache hit rate (0.0 to 1.0).
    pub hit_rate: f64,
    /// Maximum capacity.
    pub max_capacity: u64,
    /// TTL in seconds.
    pub ttl_secs: u64,
}

/// Dedicated cache for clustering results.
pub struct ClusteringCache {
    cache: Cache<String, CachedClusteringResult>,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    ttl_secs: u64,
    max_capacity: u64,
}

impl ClusteringCache {
    /// Create a new clustering cache from configuration.
    pub fn new(config: &ClusteringConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(config.max_cached_results as u64)
            .time_to_live(Duration::from_secs(config.cache_ttl_secs))
            .build();

        Self {
            cache,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            ttl_secs: config.cache_ttl_secs,
            max_capacity: config.max_cached_results as u64,
        }
    }

    /// Create a cache with custom settings.
    pub fn with_settings(max_capacity: u64, ttl_secs: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(Duration::from_secs(ttl_secs))
            .build();

        Self {
            cache,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            ttl_secs,
            max_capacity,
        }
    }

    /// Get a cached clustering result.
    pub async fn get(&self, key: &ClusteringCacheKey) -> Option<ClusteringResult> {
        let cache_key = key.to_cache_string();

        if let Some(mut cached) = self.cache.get(&cache_key).await {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            cached.hit_count += 1;
            // Update the entry with new hit count
            self.cache.insert(cache_key, cached.clone()).await;
            Some(cached.result)
        } else {
            self.misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }

    /// Store a clustering result in the cache.
    pub async fn insert(&self, key: &ClusteringCacheKey, result: ClusteringResult) {
        let cache_key = key.to_cache_string();
        let cached = CachedClusteringResult {
            result,
            cached_at: chrono::Utc::now(),
            hit_count: 0,
        };
        self.cache.insert(cache_key, cached).await;
    }

    /// Check if a key exists in the cache.
    pub async fn contains(&self, key: &ClusteringCacheKey) -> bool {
        let cache_key = key.to_cache_string();
        self.cache.get(&cache_key).await.is_some()
    }

    /// Remove a specific entry from the cache.
    pub async fn remove(&self, key: &ClusteringCacheKey) {
        let cache_key = key.to_cache_string();
        self.cache.invalidate(&cache_key).await;
    }

    /// Clear all entries from the cache.
    pub async fn clear(&self) {
        self.cache.invalidate_all();
        self.hits.store(0, std::sync::atomic::Ordering::Relaxed);
        self.misses.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> ClusteringCacheStats {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        ClusteringCacheStats {
            entry_count: self.cache.entry_count(),
            total_hits: hits,
            total_misses: misses,
            hit_rate,
            max_capacity: self.max_capacity,
            ttl_secs: self.ttl_secs,
        }
    }

    /// Get the number of entries in the cache.
    pub fn entry_count(&self) -> u64 {
        self.cache.entry_count()
    }

    /// Invalidate entries matching a source ID.
    pub async fn invalidate_source(&self, _source_id: &str) {
        // Note: moka doesn't support prefix invalidation directly
        // For now, we just clear all cache entries when a source changes
        // A more sophisticated implementation could track keys per source
        self.cache.invalidate_all();
    }
}

impl Default for ClusteringCache {
    fn default() -> Self {
        Self::with_settings(100, 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_result() -> ClusteringResult {
        use super::super::{Cluster, ClusteringMetrics};

        ClusteringResult {
            clusters: vec![Cluster {
                cluster_id: 0,
                label: "Test Cluster".to_string(),
                keywords: vec!["test".to_string()],
                document_ids: vec!["doc1".to_string(), "doc2".to_string()],
                size: 2,
                centroid: None,
                coherence_score: 0.8,
                representative_docs: vec!["doc1".to_string()],
            }],
            outliers: vec![],
            metrics: ClusteringMetrics {
                silhouette_score: 0.5,
                inertia: 100.0,
                calinski_harabasz_index: 50.0,
                davies_bouldin_index: 0.5,
                num_clusters: 1,
                num_outliers: 0,
                cluster_size_distribution: vec![2],
            },
            algorithm_used: "kmeans".to_string(),
            total_documents: 2,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_cache_insert_get() {
        let cache = ClusteringCache::default();
        let key = ClusteringCacheKey::new(
            &["doc1".to_string(), "doc2".to_string()],
            "kmeans",
            Some(3),
            None,
        );
        let result = create_test_result();

        // Insert
        cache.insert(&key, result.clone()).await;

        // Get
        let cached = cache.get(&key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().clusters.len(), 1);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = ClusteringCache::default();
        let key = ClusteringCacheKey::new(&["doc1".to_string()], "kmeans", Some(3), None);

        let cached = cache.get(&key).await;
        assert!(cached.is_none());

        let stats = cache.stats();
        assert_eq!(stats.total_misses, 1);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = ClusteringCache::default();
        let key = ClusteringCacheKey::new(&["doc1".to_string()], "kmeans", Some(3), None);
        let result = create_test_result();

        cache.insert(&key, result).await;
        assert!(cache.contains(&key).await);

        cache.clear().await;
        assert!(!cache.contains(&key).await);
    }

    #[test]
    fn test_cache_key_consistency() {
        let key1 = ClusteringCacheKey::new(
            &["doc1".to_string(), "doc2".to_string()],
            "kmeans",
            Some(3),
            None,
        );
        let key2 = ClusteringCacheKey::new(
            &["doc2".to_string(), "doc1".to_string()], // Different order
            "kmeans",
            Some(3),
            None,
        );

        // Should produce same hash since document IDs are sorted
        assert_eq!(key1.to_cache_string(), key2.to_cache_string());
    }

    #[test]
    fn test_cache_key_different_params() {
        let key1 = ClusteringCacheKey::new(&["doc1".to_string()], "kmeans", Some(3), None);
        let key2 = ClusteringCacheKey::new(&["doc1".to_string()], "dbscan", Some(3), None);

        // Different algorithms should produce different keys
        assert_ne!(key1.to_cache_string(), key2.to_cache_string());
    }
}

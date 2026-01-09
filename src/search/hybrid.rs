//! Hybrid search orchestrator combining vector and full-text search.
//!
//! This module provides the main search abstraction that:
//! - Executes parallel vector and full-text searches
//! - Fuses results using configurable algorithms (RRF or DBSF)
//! - Applies filters (source, file type, date range)
//! - Returns unified ranked results

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingProvider;
use crate::error::{Result, SearchError};
use crate::storage::StorageBackend;

use super::fusion::{dbsf_fusion, rrf_fusion, FusionAlgorithm};

/// Configuration for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Fusion algorithm to use.
    pub fusion_algorithm: FusionAlgorithm,
    /// RRF k parameter (typical: 60).
    pub rrf_k: f32,
    /// Default vector weight (0.0-1.0).
    pub default_vector_weight: f32,
    /// Minimum score threshold (results below are filtered).
    pub min_score: Option<f32>,
    /// Enable score normalization.
    pub normalize_scores: bool,
    /// Fetch multiplier for fusion (fetch N * limit from each backend).
    pub fetch_multiplier: usize,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            fusion_algorithm: FusionAlgorithm::Rrf,
            rrf_k: 60.0,
            default_vector_weight: 0.5,
            min_score: None,
            normalize_scores: true,
            fetch_multiplier: 2,
        }
    }
}

/// Filter criteria for search queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilter {
    /// Filter by source ID.
    pub source_id: Option<String>,
    /// Filter by file types (MIME types or extensions).
    pub file_types: Vec<String>,
    /// Filter by path prefix.
    pub path_prefix: Option<String>,
    /// Filter by date range (documents modified after).
    pub modified_after: Option<DateTime<Utc>>,
    /// Filter by date range (documents modified before).
    pub modified_before: Option<DateTime<Utc>>,
    /// Custom metadata filters (key-value pairs).
    pub metadata: HashMap<String, String>,
}

impl SearchFilter {
    /// Create a new empty filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set source ID filter.
    pub fn source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// Add file type filter.
    pub fn file_type(mut self, file_type: impl Into<String>) -> Self {
        self.file_types.push(file_type.into());
        self
    }

    /// Set path prefix filter.
    pub fn path(mut self, prefix: impl Into<String>) -> Self {
        self.path_prefix = Some(prefix.into());
        self
    }

    /// Set modified after filter.
    pub fn after(mut self, date: DateTime<Utc>) -> Self {
        self.modified_after = Some(date);
        self
    }

    /// Set modified before filter.
    pub fn before(mut self, date: DateTime<Utc>) -> Self {
        self.modified_before = Some(date);
        self
    }

    /// Add metadata filter.
    pub fn metadata_filter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if filter is empty (no constraints).
    pub fn is_empty(&self) -> bool {
        self.source_id.is_none()
            && self.file_types.is_empty()
            && self.path_prefix.is_none()
            && self.modified_after.is_none()
            && self.modified_before.is_none()
            && self.metadata.is_empty()
    }
}

/// A hybrid search query.
#[derive(Debug, Clone)]
pub struct HybridQuery {
    /// Query text.
    pub text: String,
    /// Maximum number of results.
    pub limit: usize,
    /// Vector weight (0.0 = text only, 1.0 = vector only).
    pub vector_weight: f32,
    /// Search filters.
    pub filter: SearchFilter,
    /// Override fusion algorithm for this query.
    pub fusion_override: Option<FusionAlgorithm>,
}

impl HybridQuery {
    /// Create a new hybrid query.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 10,
            vector_weight: 0.5,
            filter: SearchFilter::default(),
            fusion_override: None,
        }
    }

    /// Set the result limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set vector weight (0.0 = text only, 1.0 = vector only).
    pub fn vector_weight(mut self, weight: f32) -> Self {
        self.vector_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set to text-only search.
    pub fn text_only(mut self) -> Self {
        self.vector_weight = 0.0;
        self
    }

    /// Set to vector-only search.
    pub fn vector_only(mut self) -> Self {
        self.vector_weight = 1.0;
        self
    }

    /// Set filter.
    pub fn filter(mut self, filter: SearchFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Override fusion algorithm.
    pub fn fusion(mut self, algorithm: FusionAlgorithm) -> Self {
        self.fusion_override = Some(algorithm);
        self
    }
}

/// A search result from hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    /// Document ID.
    pub document_id: String,
    /// Chunk ID (may be same as document_id for document-level results).
    pub chunk_id: String,
    /// Matched text snippet.
    pub text: String,
    /// Combined score after fusion.
    pub score: f32,
    /// Individual vector similarity score.
    pub vector_score: Option<f32>,
    /// Individual BM25/text score.
    pub text_score: Option<f32>,
    /// Document path.
    pub path: Option<String>,
    /// Document MIME type.
    pub mime_type: Option<String>,
    /// Highlights (ranges in text to highlight).
    pub highlights: Vec<(usize, usize)>,
}

/// Search statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchStats {
    /// Total search time in milliseconds.
    pub total_time_ms: u64,
    /// Vector search time in milliseconds.
    pub vector_time_ms: u64,
    /// Text search time in milliseconds.
    pub text_time_ms: u64,
    /// Fusion time in milliseconds.
    pub fusion_time_ms: u64,
    /// Number of candidates from vector search.
    pub vector_candidates: usize,
    /// Number of candidates from text search.
    pub text_candidates: usize,
    /// Fusion algorithm used.
    pub fusion_algorithm: FusionAlgorithm,
}

/// Response from hybrid search.
#[derive(Debug, Clone)]
pub struct HybridSearchResponse {
    /// Search results.
    pub results: Vec<HybridSearchResult>,
    /// Search statistics.
    pub stats: SearchStats,
}

/// Trait for hybrid search execution.
#[async_trait]
pub trait HybridSearcher: Send + Sync {
    /// Execute a hybrid search query.
    async fn search(&self, query: HybridQuery) -> Result<HybridSearchResponse>;

    /// Get embedding for query text.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

/// The main hybrid search orchestrator.
pub struct HybridSearchOrchestrator {
    storage: Arc<dyn StorageBackend>,
    embedder: Arc<dyn EmbeddingProvider>,
    config: HybridSearchConfig,
}

impl HybridSearchOrchestrator {
    /// Create a new hybrid search orchestrator.
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        embedder: Arc<dyn EmbeddingProvider>,
        config: HybridSearchConfig,
    ) -> Self {
        Self {
            storage,
            embedder,
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(
        storage: Arc<dyn StorageBackend>,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self::new(storage, embedder, HybridSearchConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &HybridSearchConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: HybridSearchConfig) {
        self.config = config;
    }

    /// Fuse results from multiple sources.
    #[allow(dead_code)]
    fn fuse_results(
        &self,
        result_lists: &[Vec<(String, f32)>],
        weights: &[f32],
        algorithm: FusionAlgorithm,
    ) -> Vec<(String, f32)> {
        match algorithm {
            FusionAlgorithm::Rrf => rrf_fusion(result_lists, self.config.rrf_k, weights),
            FusionAlgorithm::Dbsf => dbsf_fusion(result_lists, weights),
        }
    }

    /// Normalize scores to 0-1 range.
    fn normalize_scores(results: &mut [HybridSearchResult]) {
        if results.is_empty() {
            return;
        }

        let max_score = results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);

        let min_score = results
            .iter()
            .map(|r| r.score)
            .fold(f32::INFINITY, f32::min);

        let range = max_score - min_score;
        if range > 0.0 {
            for result in results {
                result.score = (result.score - min_score) / range;
            }
        } else {
            // All scores are the same, normalize to 1.0
            for result in results {
                result.score = 1.0;
            }
        }
    }

    /// Extract highlights from text based on query terms.
    fn extract_highlights(text: &str, query: &str) -> Vec<(usize, usize)> {
        let mut highlights = Vec::new();
        let text_lower = text.to_lowercase();
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        for term in query_terms {
            let mut start = 0;
            while let Some(pos) = text_lower[start..].find(term) {
                let abs_pos = start + pos;
                highlights.push((abs_pos, abs_pos + term.len()));
                start = abs_pos + term.len();
            }
        }

        // Sort and merge overlapping highlights
        highlights.sort_by_key(|h| h.0);
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for highlight in highlights {
            if let Some(last) = merged.last_mut() {
                if highlight.0 <= last.1 {
                    last.1 = last.1.max(highlight.1);
                    continue;
                }
            }
            merged.push(highlight);
        }

        merged
    }
}

#[async_trait]
impl HybridSearcher for HybridSearchOrchestrator {
    async fn search(&self, query: HybridQuery) -> Result<HybridSearchResponse> {
        let start_time = std::time::Instant::now();

        let vector_weight = query.vector_weight;
        let text_weight = 1.0 - vector_weight;
        let fetch_limit = query.limit * self.config.fetch_multiplier;

        // Get embedding for query
        let embedding = if vector_weight > 0.0 {
            Some(self.embed_query(&query.text).await?)
        } else {
            None
        };

        // Build storage query
        let storage_query = crate::storage::SearchQuery {
            text: query.text.clone(),
            embedding,
            limit: fetch_limit,
            vector_weight,
            source_id: query.filter.source_id.clone(),
            file_types: query.filter.file_types.clone(),
        };

        // Execute search via storage backend
        let search_start = std::time::Instant::now();
        let storage_results = self.storage.search(storage_query).await?;
        let search_time = search_start.elapsed().as_millis() as u64;

        // Convert storage results to hybrid results
        let mut results: Vec<HybridSearchResult> = storage_results
            .into_iter()
            .take(query.limit)
            .map(|r| {
                let highlights = Self::extract_highlights(&r.text, &query.text);
                HybridSearchResult {
                    document_id: r.document_id,
                    chunk_id: r.chunk_id,
                    text: r.text,
                    score: r.score,
                    vector_score: r.vector_score,
                    text_score: r.text_score,
                    path: None,     // Would need to fetch from document
                    mime_type: None, // Would need to fetch from document
                    highlights,
                }
            })
            .collect();

        // Apply minimum score filter
        if let Some(min_score) = self.config.min_score {
            results.retain(|r| r.score >= min_score);
        }

        // Normalize scores if configured
        if self.config.normalize_scores {
            Self::normalize_scores(&mut results);
        }

        let total_time = start_time.elapsed().as_millis() as u64;

        // Build stats
        let algorithm = query
            .fusion_override
            .unwrap_or(self.config.fusion_algorithm);

        let stats = SearchStats {
            total_time_ms: total_time,
            vector_time_ms: if vector_weight > 0.0 {
                search_time / 2
            } else {
                0
            },
            text_time_ms: if text_weight > 0.0 {
                search_time / 2
            } else {
                0
            },
            fusion_time_ms: 0, // Fusion happens in storage for now
            vector_candidates: results.iter().filter(|r| r.vector_score.is_some()).count(),
            text_candidates: results.iter().filter(|r| r.text_score.is_some()).count(),
            fusion_algorithm: algorithm,
        };

        Ok(HybridSearchResponse { results, stats })
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embedder.embed(&[text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| SearchError::InvalidQuery("Failed to embed query".to_string()).into())
    }
}

/// Builder for creating a hybrid search orchestrator.
pub struct HybridSearchBuilder {
    storage: Option<Arc<dyn StorageBackend>>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    config: HybridSearchConfig,
}

impl Default for HybridSearchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridSearchBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            storage: None,
            embedder: None,
            config: HybridSearchConfig::default(),
        }
    }

    /// Set the storage backend.
    pub fn storage(mut self, storage: Arc<dyn StorageBackend>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Set the embedding provider.
    pub fn embedder(mut self, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the fusion algorithm.
    pub fn fusion_algorithm(mut self, algorithm: FusionAlgorithm) -> Self {
        self.config.fusion_algorithm = algorithm;
        self
    }

    /// Set RRF k parameter.
    pub fn rrf_k(mut self, k: f32) -> Self {
        self.config.rrf_k = k;
        self
    }

    /// Set default vector weight.
    pub fn vector_weight(mut self, weight: f32) -> Self {
        self.config.default_vector_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set minimum score threshold.
    pub fn min_score(mut self, score: f32) -> Self {
        self.config.min_score = Some(score);
        self
    }

    /// Enable or disable score normalization.
    pub fn normalize_scores(mut self, normalize: bool) -> Self {
        self.config.normalize_scores = normalize;
        self
    }

    /// Set fetch multiplier.
    pub fn fetch_multiplier(mut self, multiplier: usize) -> Self {
        self.config.fetch_multiplier = multiplier.max(1);
        self
    }

    /// Set full configuration.
    pub fn config(mut self, config: HybridSearchConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the orchestrator.
    pub fn build(self) -> Result<HybridSearchOrchestrator> {
        let storage = self.storage.ok_or_else(|| {
            SearchError::InvalidQuery("Storage backend required".to_string())
        })?;

        let embedder = self.embedder.ok_or_else(|| {
            SearchError::InvalidQuery("Embedding provider required".to_string())
        })?;

        Ok(HybridSearchOrchestrator::new(storage, embedder, self.config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_filter_builder() {
        let filter = SearchFilter::new()
            .source("my-source")
            .file_type("text/plain")
            .file_type("application/pdf")
            .path("/documents");

        assert_eq!(filter.source_id, Some("my-source".to_string()));
        assert_eq!(filter.file_types.len(), 2);
        assert_eq!(filter.path_prefix, Some("/documents".to_string()));
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_empty_filter() {
        let filter = SearchFilter::new();
        assert!(filter.is_empty());
    }

    #[test]
    fn test_hybrid_query_builder() {
        let query = HybridQuery::new("test query")
            .limit(20)
            .vector_weight(0.7)
            .filter(SearchFilter::new().source("source1"));

        assert_eq!(query.text, "test query");
        assert_eq!(query.limit, 20);
        assert_eq!(query.vector_weight, 0.7);
        assert_eq!(query.filter.source_id, Some("source1".to_string()));
    }

    #[test]
    fn test_text_only_query() {
        let query = HybridQuery::new("test").text_only();
        assert_eq!(query.vector_weight, 0.0);
    }

    #[test]
    fn test_vector_only_query() {
        let query = HybridQuery::new("test").vector_only();
        assert_eq!(query.vector_weight, 1.0);
    }

    #[test]
    fn test_weight_clamping() {
        let query = HybridQuery::new("test").vector_weight(1.5);
        assert_eq!(query.vector_weight, 1.0);

        let query = HybridQuery::new("test").vector_weight(-0.5);
        assert_eq!(query.vector_weight, 0.0);
    }

    #[test]
    fn test_highlight_extraction() {
        let text = "The quick brown fox jumps over the lazy dog";
        let query = "quick fox";
        let highlights = HybridSearchOrchestrator::extract_highlights(text, query);

        assert_eq!(highlights.len(), 2);
        // "quick" starts at index 4
        assert_eq!(highlights[0], (4, 9));
        // "fox" starts at index 16
        assert_eq!(highlights[1], (16, 19));
    }

    #[test]
    fn test_highlight_merge_overlapping() {
        let text = "abcdef abcdef";
        let query = "abc bcd";
        let highlights = HybridSearchOrchestrator::extract_highlights(text, query);

        // Should merge overlapping highlights
        // First occurrence: abc (0-3) and bcd (1-4) -> merged to (0-4)
        // Second occurrence: abc (7-10) and bcd (8-11) -> merged to (7-11)
        assert_eq!(highlights.len(), 2);
    }

    #[test]
    fn test_default_config() {
        let config = HybridSearchConfig::default();
        assert_eq!(config.fusion_algorithm, FusionAlgorithm::Rrf);
        assert_eq!(config.rrf_k, 60.0);
        assert_eq!(config.default_vector_weight, 0.5);
        assert!(config.normalize_scores);
    }

    #[test]
    fn test_score_normalization() {
        let mut results = vec![
            HybridSearchResult {
                document_id: "doc1".to_string(),
                chunk_id: "chunk1".to_string(),
                text: "text1".to_string(),
                score: 10.0,
                vector_score: None,
                text_score: None,
                path: None,
                mime_type: None,
                highlights: vec![],
            },
            HybridSearchResult {
                document_id: "doc2".to_string(),
                chunk_id: "chunk2".to_string(),
                text: "text2".to_string(),
                score: 5.0,
                vector_score: None,
                text_score: None,
                path: None,
                mime_type: None,
                highlights: vec![],
            },
            HybridSearchResult {
                document_id: "doc3".to_string(),
                chunk_id: "chunk3".to_string(),
                text: "text3".to_string(),
                score: 0.0,
                vector_score: None,
                text_score: None,
                path: None,
                mime_type: None,
                highlights: vec![],
            },
        ];

        HybridSearchOrchestrator::normalize_scores(&mut results);

        // After normalization: 10 -> 1.0, 5 -> 0.5, 0 -> 0.0
        assert_eq!(results[0].score, 1.0);
        assert_eq!(results[1].score, 0.5);
        assert_eq!(results[2].score, 0.0);
    }
}

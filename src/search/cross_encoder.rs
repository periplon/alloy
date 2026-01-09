//! Cross-encoder reranking module for high-precision result reranking.
//!
//! This module provides cross-encoder based reranking using local ONNX models
//! via the fastembed library. Cross-encoders evaluate query-document pairs
//! jointly, enabling deeper semantic understanding compared to bi-encoders.
//!
//! # Supported Models
//!
//! - `BGERerankerBase`: BAAI/bge-reranker-base (1.04 GB) - MIT license
//! - `BGERerankerV2M3`: rozgo/bge-reranker-v2-m3 - Multilingual support
//! - `JINARerankerV1TurboEn`: jinaai/jina-reranker-v1-turbo-en - Fast English reranking
//! - `JINARerankerV2BaseMultilingual`: jinaai/jina-reranker-v2-base-multilingual
//!
//! # Example
//!
//! ```rust,ignore
//! use alloy::search::cross_encoder::{LocalCrossEncoder, CrossEncoderConfig};
//!
//! let config = CrossEncoderConfig::default();
//! let encoder = LocalCrossEncoder::new(config)?;
//!
//! let query = "What is machine learning?";
//! let documents = vec![
//!     "Machine learning is a subset of artificial intelligence.",
//!     "The weather today is sunny.",
//! ];
//!
//! let scores = encoder.score(query, &documents)?;
//! // scores[0] will be higher than scores[1] due to relevance
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::config::RerankerConfig;
use crate::error::{Result, SearchError};

use super::reranker::Reranker;
use super::HybridSearchResult;

/// Cross-encoder model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CrossEncoderModel {
    /// BAAI/bge-reranker-base - General purpose, high quality
    #[default]
    BgeRerankerBase,
    /// rozgo/bge-reranker-v2-m3 - Multilingual support
    BgeRerankerV2M3,
    /// jinaai/jina-reranker-v1-turbo-en - Fast English reranking
    JinaRerankerV1TurboEn,
    /// jinaai/jina-reranker-v2-base-multilingual - Multilingual
    JinaRerankerV2BaseMultilingual,
}

impl CrossEncoderModel {
    /// Convert to fastembed RerankerModel.
    pub fn to_fastembed_model(&self) -> RerankerModel {
        match self {
            CrossEncoderModel::BgeRerankerBase => RerankerModel::BGERerankerBase,
            CrossEncoderModel::BgeRerankerV2M3 => RerankerModel::BGERerankerV2M3,
            CrossEncoderModel::JinaRerankerV1TurboEn => RerankerModel::JINARerankerV1TurboEn,
            CrossEncoderModel::JinaRerankerV2BaseMultilingual => {
                RerankerModel::JINARerankerV2BaseMultiligual
            }
        }
    }

    /// Get human-readable model name.
    pub fn display_name(&self) -> &'static str {
        match self {
            CrossEncoderModel::BgeRerankerBase => "BAAI/bge-reranker-base",
            CrossEncoderModel::BgeRerankerV2M3 => "rozgo/bge-reranker-v2-m3",
            CrossEncoderModel::JinaRerankerV1TurboEn => "jinaai/jina-reranker-v1-turbo-en",
            CrossEncoderModel::JinaRerankerV2BaseMultilingual => {
                "jinaai/jina-reranker-v2-base-multilingual"
            }
        }
    }

    /// Parse model name from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bge-reranker-base" | "baai/bge-reranker-base" | "bge_reranker_base" => {
                Some(CrossEncoderModel::BgeRerankerBase)
            }
            "bge-reranker-v2-m3" | "rozgo/bge-reranker-v2-m3" | "bge_reranker_v2_m3" => {
                Some(CrossEncoderModel::BgeRerankerV2M3)
            }
            "jina-reranker-v1-turbo-en"
            | "jinaai/jina-reranker-v1-turbo-en"
            | "jina_reranker_v1_turbo_en" => Some(CrossEncoderModel::JinaRerankerV1TurboEn),
            "jina-reranker-v2-base-multilingual"
            | "jinaai/jina-reranker-v2-base-multilingual"
            | "jina_reranker_v2_base_multilingual" => {
                Some(CrossEncoderModel::JinaRerankerV2BaseMultilingual)
            }
            _ => None,
        }
    }
}

/// Configuration for the local cross-encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEncoderConfig {
    /// Model to use for reranking.
    pub model: CrossEncoderModel,
    /// Show download progress when loading model.
    pub show_download_progress: bool,
    /// Batch size for reranking (None = process all at once).
    pub batch_size: Option<usize>,
    /// Cache directory for model files (None = use default).
    pub cache_dir: Option<String>,
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            model: CrossEncoderModel::default(),
            show_download_progress: true,
            batch_size: Some(32),
            cache_dir: None,
        }
    }
}

/// Local cross-encoder for high-precision reranking using fastembed.
///
/// This reranker uses ONNX-based cross-encoder models to score query-document
/// pairs. Unlike bi-encoders that embed query and document separately,
/// cross-encoders process the pair jointly, achieving higher accuracy.
pub struct LocalCrossEncoder {
    /// The fastembed reranker model.
    model: Arc<RwLock<TextRerank>>,
    /// Configuration.
    config: CrossEncoderConfig,
}

impl LocalCrossEncoder {
    /// Create a new local cross-encoder.
    ///
    /// This will download the model if not cached locally.
    pub fn new(config: CrossEncoderConfig) -> Result<Self> {
        tracing::info!(
            "Initializing local cross-encoder with model: {}",
            config.model.display_name()
        );

        let mut options = RerankInitOptions::new(config.model.to_fastembed_model())
            .with_show_download_progress(config.show_download_progress);

        if let Some(ref cache_dir) = config.cache_dir {
            options = options.with_cache_dir(cache_dir.into());
        }

        let model = TextRerank::try_new(options).map_err(|e| {
            SearchError::Reranking(format!("Failed to initialize cross-encoder: {}", e))
        })?;

        tracing::info!("Cross-encoder initialized successfully");

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            config,
        })
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(CrossEncoderConfig::default())
    }

    /// Score query-document pairs.
    ///
    /// Returns a vector of relevance scores, one per document.
    /// Higher scores indicate more relevant documents.
    pub fn score<S: AsRef<str>>(&self, query: &str, documents: &[S]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let docs: Vec<&str> = documents.iter().map(|d| d.as_ref()).collect();

        let mut model = self.model.write();
        let results = model
            .rerank(query, docs, false, self.config.batch_size)
            .map_err(|e| SearchError::Reranking(format!("Reranking failed: {}", e)))?;

        // Results are sorted by score descending, but we need scores in original order
        // Create a map from index to score
        let mut scores = vec![0.0f32; documents.len()];
        for result in results {
            if result.index < scores.len() {
                scores[result.index] = result.score;
            }
        }

        Ok(scores)
    }

    /// Get the model name.
    pub fn model_name(&self) -> &'static str {
        self.config.model.display_name()
    }
}

// LocalCrossEncoder is Send + Sync because TextRerank is wrapped in Arc<RwLock>
unsafe impl Send for LocalCrossEncoder {}
unsafe impl Sync for LocalCrossEncoder {}

/// Reranker implementation using local cross-encoder.
pub struct LocalCrossEncoderReranker {
    encoder: LocalCrossEncoder,
}

impl LocalCrossEncoderReranker {
    /// Create a new local cross-encoder reranker.
    pub fn new(config: CrossEncoderConfig) -> Result<Self> {
        Ok(Self {
            encoder: LocalCrossEncoder::new(config)?,
        })
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(CrossEncoderConfig::default())
    }

    /// Create from model name string.
    pub fn from_model_name(model_name: &str) -> Result<Self> {
        let model = CrossEncoderModel::parse(model_name).ok_or_else(|| {
            SearchError::Reranking(format!(
                "Unknown cross-encoder model: {}. Supported: bge-reranker-base, \
                 bge-reranker-v2-m3, jina-reranker-v1-turbo-en, jina-reranker-v2-base-multilingual",
                model_name
            ))
        })?;

        let config = CrossEncoderConfig {
            model,
            ..Default::default()
        };

        Self::new(config)
    }
}

#[async_trait]
impl Reranker for LocalCrossEncoderReranker {
    async fn rerank(
        &self,
        query: &str,
        mut results: Vec<HybridSearchResult>,
        config: &RerankerConfig,
    ) -> Result<Vec<HybridSearchResult>> {
        if results.is_empty() {
            return Ok(results);
        }

        // Take top_k candidates for reranking
        let candidates: Vec<_> = results.drain(..results.len().min(config.top_k)).collect();

        // Extract texts for reranking
        let texts: Vec<&str> = candidates.iter().map(|r| r.text.as_str()).collect();

        // Get cross-encoder scores
        let scores = self.encoder.score(query, &texts)?;

        // Combine results with scores
        let mut scored: Vec<(HybridSearchResult, f32)> =
            candidates.into_iter().zip(scores).collect();

        // Sort by cross-encoder score (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply minimum score filter if configured
        if let Some(min_score) = config.min_score {
            scored.retain(|(_, score)| *score >= min_score);
        }

        // Take final_k results and update scores
        let reranked: Vec<HybridSearchResult> = scored
            .into_iter()
            .take(config.final_k)
            .map(|(mut result, score)| {
                result.score = score;
                result
            })
            .collect();

        Ok(reranked)
    }

    fn name(&self) -> &str {
        "local_cross_encoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_encoder_model_from_str() {
        assert_eq!(
            CrossEncoderModel::parse("bge-reranker-base"),
            Some(CrossEncoderModel::BgeRerankerBase)
        );
        assert_eq!(
            CrossEncoderModel::parse("BAAI/bge-reranker-base"),
            Some(CrossEncoderModel::BgeRerankerBase)
        );
        assert_eq!(
            CrossEncoderModel::parse("bge_reranker_base"),
            Some(CrossEncoderModel::BgeRerankerBase)
        );
        assert_eq!(
            CrossEncoderModel::parse("jina-reranker-v1-turbo-en"),
            Some(CrossEncoderModel::JinaRerankerV1TurboEn)
        );
        assert_eq!(CrossEncoderModel::parse("unknown-model"), None);
    }

    #[test]
    fn test_cross_encoder_config_default() {
        let config = CrossEncoderConfig::default();
        assert_eq!(config.model, CrossEncoderModel::BgeRerankerBase);
        assert!(config.show_download_progress);
        assert_eq!(config.batch_size, Some(32));
        assert!(config.cache_dir.is_none());
    }

    #[test]
    fn test_cross_encoder_model_display_name() {
        assert_eq!(
            CrossEncoderModel::BgeRerankerBase.display_name(),
            "BAAI/bge-reranker-base"
        );
        assert_eq!(
            CrossEncoderModel::JinaRerankerV1TurboEn.display_name(),
            "jinaai/jina-reranker-v1-turbo-en"
        );
    }

    // Integration test - requires model download
    #[test]
    #[ignore = "requires model download"]
    fn test_local_cross_encoder_reranking() {
        let config = CrossEncoderConfig {
            model: CrossEncoderModel::BgeRerankerBase,
            show_download_progress: false,
            batch_size: Some(2),
            cache_dir: None,
        };

        let encoder = LocalCrossEncoder::new(config).unwrap();

        let query = "What is machine learning?";
        let documents = vec![
            "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
            "The weather forecast predicts sunny skies tomorrow.",
            "Deep learning uses neural networks with multiple layers.",
        ];

        let scores = encoder.score(query, &documents).unwrap();

        assert_eq!(scores.len(), 3);
        // The first and third documents should have higher scores than the second
        assert!(
            scores[0] > scores[1],
            "ML document should score higher than weather"
        );
        assert!(
            scores[2] > scores[1],
            "Deep learning should score higher than weather"
        );
    }
}

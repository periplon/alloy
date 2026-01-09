//! Reranking module for improving search precision.
//!
//! This module provides reranking capabilities to improve search results
//! by re-scoring candidates using more sophisticated scoring methods.
//!
//! # Supported Reranking Methods
//!
//! - **Score-based**: Uses embedding cosine similarity for reranking
//! - **Cross-encoder**: Uses cross-encoder models for query-document scoring (API-based)
//! - **LLM-based**: Uses language models to evaluate relevance (API-based)

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::RerankerConfig;
use crate::embedding::EmbeddingProvider;
use crate::error::Result;

use super::HybridSearchResult;

/// Trait for reranking search results.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank search results based on query relevance.
    async fn rerank(
        &self,
        query: &str,
        results: Vec<HybridSearchResult>,
        config: &RerankerConfig,
    ) -> Result<Vec<HybridSearchResult>>;

    /// Get the name of this reranker.
    fn name(&self) -> &str;
}

/// Result of reranking with additional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankedResult {
    /// The original search result.
    pub result: HybridSearchResult,
    /// The reranking score.
    pub rerank_score: f32,
    /// Original rank before reranking.
    pub original_rank: usize,
    /// New rank after reranking.
    pub new_rank: usize,
}

/// Score-based reranker using embedding similarity.
///
/// This reranker computes the cosine similarity between the query embedding
/// and document chunk embeddings to rerank results.
pub struct ScoreBasedReranker {
    embedder: Arc<dyn EmbeddingProvider>,
}

impl ScoreBasedReranker {
    /// Create a new score-based reranker.
    pub fn new(embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self { embedder }
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl Reranker for ScoreBasedReranker {
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

        // Get query embedding
        let query_embeddings = self.embedder.embed(&[query.to_string()]).await?;
        let query_embedding = query_embeddings.into_iter().next().unwrap_or_default();

        if query_embedding.is_empty() {
            // If embedding failed, return original order
            return Ok(candidates);
        }

        // Get embeddings for all candidate texts
        let texts: Vec<String> = candidates.iter().map(|r| r.text.clone()).collect();
        let text_embeddings = self.embedder.embed(&texts).await?;

        // Compute reranking scores
        let mut scored: Vec<(HybridSearchResult, f32)> = candidates
            .into_iter()
            .zip(text_embeddings.iter())
            .map(|(result, embedding)| {
                let similarity = Self::cosine_similarity(&query_embedding, embedding);
                // Combine with original score using weighted average
                let combined_score = 0.4 * result.score + 0.6 * similarity;
                (result, combined_score)
            })
            .collect();

        // Sort by reranking score (descending)
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
        "score_based"
    }
}

/// Cross-encoder reranker for high-precision reranking.
///
/// This uses a cross-encoder model that takes (query, document) pairs
/// and produces a relevance score. More accurate but slower.
pub struct CrossEncoderReranker {
    /// API endpoint for cross-encoder
    api_url: String,
    /// API key
    api_key: Option<String>,
    /// HTTP client
    client: reqwest::Client,
}

impl CrossEncoderReranker {
    /// Create a new cross-encoder reranker.
    pub fn new(api_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            api_url: api_url.into(),
            api_key,
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct CrossEncoderRequest {
    query: String,
    documents: Vec<String>,
}

#[derive(Deserialize)]
struct CrossEncoderResponse {
    scores: Vec<f32>,
}

#[async_trait]
impl Reranker for CrossEncoderReranker {
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

        // Prepare documents for cross-encoder
        let documents: Vec<String> = candidates.iter().map(|r| r.text.clone()).collect();

        // Build request
        let request = CrossEncoderRequest {
            query: query.to_string(),
            documents,
        };

        // Send request to cross-encoder API
        let mut req_builder = self.client.post(&self.api_url).json(&request);

        if let Some(ref api_key) = self.api_key {
            req_builder = req_builder.bearer_auth(api_key);
        }

        let response = req_builder.send().await.map_err(|e| {
            crate::error::SearchError::Reranking(format!("Cross-encoder API error: {}", e))
        })?;

        if !response.status().is_success() {
            return Err(crate::error::SearchError::Reranking(format!(
                "Cross-encoder API returned status: {}",
                response.status()
            ))
            .into());
        }

        let scores: CrossEncoderResponse = response.json().await.map_err(|e| {
            crate::error::SearchError::Reranking(format!("Failed to parse response: {}", e))
        })?;

        // Combine results with scores
        let mut scored: Vec<(HybridSearchResult, f32)> = candidates
            .into_iter()
            .zip(scores.scores.into_iter())
            .collect();

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
        "cross_encoder"
    }
}

/// LLM-based reranker for semantic relevance scoring.
///
/// Uses a language model to evaluate the relevance of each document
/// to the query, providing natural language understanding.
pub struct LlmReranker {
    /// API endpoint for LLM
    api_url: String,
    /// API key
    api_key: Option<String>,
    /// Model name
    model: String,
    /// HTTP client
    client: reqwest::Client,
}

impl LlmReranker {
    /// Create a new LLM reranker.
    pub fn new(api_url: impl Into<String>, api_key: Option<String>, model: impl Into<String>) -> Self {
        Self {
            api_url: api_url.into(),
            api_key,
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct LlmRerankRequest {
    model: String,
    messages: Vec<LlmMessage>,
    temperature: f32,
}

#[derive(Serialize)]
struct LlmMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct LlmRerankResponse {
    choices: Vec<LlmChoice>,
}

#[derive(Deserialize)]
struct LlmChoice {
    message: LlmMessageResponse,
}

#[derive(Deserialize)]
struct LlmMessageResponse {
    content: String,
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        mut results: Vec<HybridSearchResult>,
        config: &RerankerConfig,
    ) -> Result<Vec<HybridSearchResult>> {
        if results.is_empty() {
            return Ok(results);
        }

        // Take top_k candidates for reranking (limit to avoid token limits)
        let max_candidates = config.top_k.min(20); // LLM context limit
        let candidates: Vec<_> = results.drain(..results.len().min(max_candidates)).collect();

        // Build prompt for relevance scoring
        let mut documents_text = String::new();
        for (i, result) in candidates.iter().enumerate() {
            documents_text.push_str(&format!(
                "[{}] {}\n\n",
                i + 1,
                result.text.chars().take(500).collect::<String>()
            ));
        }

        let prompt = format!(
            "Rate the relevance of each document to the query on a scale of 0.0 to 1.0.\n\
            Query: {}\n\n\
            Documents:\n{}\n\
            Respond with a JSON array of scores in order, e.g., [0.9, 0.7, 0.3, ...]",
            query, documents_text
        );

        let request = LlmRerankRequest {
            model: self.model.clone(),
            messages: vec![LlmMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            temperature: 0.0,
        };

        let mut req_builder = self.client.post(&self.api_url).json(&request);

        if let Some(ref api_key) = self.api_key {
            req_builder = req_builder.bearer_auth(api_key);
        }

        let response = req_builder.send().await.map_err(|e| {
            crate::error::SearchError::Reranking(format!("LLM API error: {}", e))
        })?;

        if !response.status().is_success() {
            // Fall back to original ranking on API failure
            tracing::warn!("LLM reranking failed, using original ranking");
            return Ok(candidates);
        }

        let llm_response: LlmRerankResponse = response.json().await.map_err(|e| {
            crate::error::SearchError::Reranking(format!("Failed to parse LLM response: {}", e))
        })?;

        // Parse scores from response
        let content = &llm_response.choices.first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let scores: Vec<f32> = serde_json::from_str(content).unwrap_or_else(|_| {
            // If parsing fails, try to extract numbers
            content
                .split(|c: char| !c.is_numeric() && c != '.')
                .filter_map(|s| s.parse::<f32>().ok())
                .collect()
        });

        // Combine results with scores
        let mut scored: Vec<(HybridSearchResult, f32)> = candidates
            .into_iter()
            .zip(scores.into_iter().chain(std::iter::repeat(0.5)))
            .collect();

        // Sort by LLM score (descending)
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
        "llm"
    }
}

/// Factory for creating rerankers based on configuration.
pub struct RerankerFactory;

impl RerankerFactory {
    /// Create a reranker based on configuration.
    pub fn create(
        config: &RerankerConfig,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Box<dyn Reranker> {
        match config.reranker_type {
            crate::config::RerankerType::ScoreBased => {
                Box::new(ScoreBasedReranker::new(embedder))
            }
            crate::config::RerankerType::CrossEncoder => {
                // Default cross-encoder API endpoint
                Box::new(CrossEncoderReranker::new(
                    "http://localhost:8080/rerank",
                    None,
                ))
            }
            crate::config::RerankerType::Llm => {
                // Default to OpenAI-compatible API
                Box::new(LlmReranker::new(
                    "https://api.openai.com/v1/chat/completions",
                    std::env::var("OPENAI_API_KEY").ok(),
                    "gpt-4o-mini",
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        // Same vector should have similarity 1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((ScoreBasedReranker::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors should have similarity 0.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((ScoreBasedReranker::cosine_similarity(&a, &b)).abs() < 0.001);

        // Opposite vectors should have similarity -1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((ScoreBasedReranker::cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let empty: Vec<f32> = vec![];
        let a = vec![1.0, 0.0];
        assert_eq!(ScoreBasedReranker::cosine_similarity(&empty, &a), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(ScoreBasedReranker::cosine_similarity(&a, &b), 0.0);
    }
}

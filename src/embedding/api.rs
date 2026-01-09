//! API-based embedding provider (OpenAI-compatible).

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::config::ApiEmbeddingConfig;
use crate::error::{EmbeddingError, Result};

use super::EmbeddingProvider;

/// OpenAI-compatible API embedding provider.
pub struct ApiEmbeddingProvider {
    client: Client,
    base_url: String,
    model: String,
    api_key: String,
    dimension: usize,
    max_batch_size: usize,
}

/// OpenAI embedding request format.
#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
}

/// OpenAI embedding response format.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

#[derive(Debug, Deserialize)]
struct Usage {
    #[allow(dead_code)]
    prompt_tokens: u32,
    #[allow(dead_code)]
    total_tokens: u32,
}

/// OpenAI error response format.
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: Option<String>,
    #[allow(dead_code)]
    code: Option<String>,
}

impl ApiEmbeddingProvider {
    /// Create a new API embedding provider from configuration.
    pub fn from_config(config: &ApiEmbeddingConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                EmbeddingError::Api(
                    "API key not provided and OPENAI_API_KEY env var not set".to_string(),
                )
            })?;

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| EmbeddingError::Api(format!("Failed to create HTTP client: {}", e)))?;

        let dimension = Self::get_model_dimension(&config.model);

        Ok(Self {
            client,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            model: config.model.clone(),
            api_key,
            dimension,
            max_batch_size: config.batch_size,
        })
    }

    /// Create a new API embedding provider with explicit parameters.
    pub fn new(base_url: &str, model: &str, api_key: &str, timeout_secs: u64) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| EmbeddingError::Api(format!("Failed to create HTTP client: {}", e)))?;

        let dimension = Self::get_model_dimension(model);

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
            dimension,
            max_batch_size: 100,
        })
    }

    fn get_model_dimension(model: &str) -> usize {
        match model {
            // OpenAI models
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            // Voyage AI models
            "voyage-large-2" => 1536,
            "voyage-code-2" => 1536,
            "voyage-2" => 1024,
            // Cohere models
            "embed-english-v3.0" => 1024,
            "embed-multilingual-v3.0" => 1024,
            "embed-english-light-v3.0" => 384,
            // Default for unknown models
            _ => 1536,
        }
    }

    /// Make an embedding request to the API.
    async fn request_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);

        let request = EmbeddingRequest {
            model: &self.model,
            input: texts,
            encoding_format: Some("float"),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EmbeddingError::Api("Request timed out".to_string())
                } else if e.is_connect() {
                    EmbeddingError::Api(format!("Connection failed: {}", e))
                } else {
                    EmbeddingError::Api(format!("Request failed: {}", e))
                }
            })?;

        let status = response.status();

        if status.is_success() {
            let result: EmbeddingResponse = response.json().await.map_err(|e| {
                EmbeddingError::Api(format!("Failed to parse response: {}", e))
            })?;

            // Sort by index to ensure correct order
            let mut embeddings: Vec<_> = result.data.into_iter().collect();
            embeddings.sort_by_key(|d| d.index);

            Ok(embeddings.into_iter().map(|d| d.embedding).collect())
        } else if status.as_u16() == 429 {
            Err(EmbeddingError::RateLimited.into())
        } else {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as OpenAI error format
            if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&error_text) {
                Err(EmbeddingError::Api(format!(
                    "API error ({}): {}",
                    status, error_response.error.message
                ))
                .into())
            } else {
                Err(EmbeddingError::Api(format!("API error ({}): {}", status, error_text)).into())
            }
        }
    }
}

#[async_trait]
impl EmbeddingProvider for ApiEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        if texts.len() > self.max_batch_size {
            return Err(EmbeddingError::BatchTooLarge(texts.len(), self.max_batch_size).into());
        }

        self.request_embeddings(texts).await
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dimension() {
        assert_eq!(ApiEmbeddingProvider::get_model_dimension("text-embedding-3-small"), 1536);
        assert_eq!(ApiEmbeddingProvider::get_model_dimension("text-embedding-3-large"), 3072);
        assert_eq!(ApiEmbeddingProvider::get_model_dimension("text-embedding-ada-002"), 1536);
        assert_eq!(ApiEmbeddingProvider::get_model_dimension("unknown-model"), 1536);
    }

    #[test]
    fn test_from_config_missing_api_key() {
        // Clear env var if set
        std::env::remove_var("OPENAI_API_KEY");

        let config = ApiEmbeddingConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: None,
            batch_size: 100,
            timeout_secs: 30,
        };

        let result = ApiEmbeddingProvider::from_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_config_with_api_key() {
        let config = ApiEmbeddingConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("test-key".to_string()),
            batch_size: 100,
            timeout_secs: 30,
        };

        let result = ApiEmbeddingProvider::from_config(&config);
        assert!(result.is_ok());

        let provider = result.unwrap();
        assert_eq!(provider.dimension(), 1536);
        assert_eq!(provider.max_batch_size(), 100);
    }

    #[test]
    fn test_base_url_normalization() {
        let config = ApiEmbeddingConfig {
            base_url: "https://api.openai.com/v1/".to_string(), // Note trailing slash
            model: "text-embedding-3-small".to_string(),
            api_key: Some("test-key".to_string()),
            batch_size: 100,
            timeout_secs: 30,
        };

        let provider = ApiEmbeddingProvider::from_config(&config).unwrap();
        assert!(!provider.base_url.ends_with('/'));
    }

    // Integration tests would require a real API key
    // Run with: OPENAI_API_KEY=xxx cargo test test_api_embedding_integration -- --ignored
    #[tokio::test]
    #[ignore = "requires API key"]
    async fn test_api_embedding_integration() {
        let config = ApiEmbeddingConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: None, // Will use env var
            batch_size: 100,
            timeout_secs: 30,
        };

        let provider = ApiEmbeddingProvider::from_config(&config).unwrap();
        let texts = vec!["Hello, world!".to_string()];
        let embeddings = provider.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 1536);
    }
}

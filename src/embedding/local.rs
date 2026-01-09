//! Local embedding provider using fastembed-rs.

use async_trait::async_trait;
use bytes::Bytes;
use fastembed::{EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Arc;
use tempfile::NamedTempFile;
use tokio::sync::Mutex;

use crate::error::{EmbeddingError, Result};

use super::EmbeddingProvider;

/// Local embedding provider using fastembed-rs with ONNX models.
pub struct LocalEmbeddingProvider {
    text_model: Arc<Mutex<TextEmbedding>>,
    image_model: Option<Arc<Mutex<ImageEmbedding>>>,
    dimension: usize,
    max_batch_size: usize,
}

impl LocalEmbeddingProvider {
    /// Create a new local embedding provider with the specified model.
    pub fn new(model_name: &str, enable_image: bool) -> Result<Self> {
        let text_model = Self::load_text_model(model_name)?;
        let dimension = Self::get_model_dimension(model_name);

        let image_model = if enable_image {
            Some(Arc::new(Mutex::new(Self::load_image_model()?)))
        } else {
            None
        };

        Ok(Self {
            text_model: Arc::new(Mutex::new(text_model)),
            image_model,
            dimension,
            max_batch_size: 32, // Conservative default for local models
        })
    }

    /// Create with default BGE-small model.
    pub fn default_model() -> Result<Self> {
        Self::new("BAAI/bge-small-en-v1.5", false)
    }

    fn load_text_model(model_name: &str) -> Result<TextEmbedding> {
        let model = Self::parse_model_name(model_name)?;

        let options = InitOptions::new(model).with_show_download_progress(true);

        TextEmbedding::try_new(options)
            .map_err(|e| EmbeddingError::ModelNotFound(format!("{}: {}", model_name, e)).into())
    }

    fn load_image_model() -> Result<ImageEmbedding> {
        let options = fastembed::ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
            .with_show_download_progress(true);

        ImageEmbedding::try_new(options)
            .map_err(|e| EmbeddingError::ModelNotFound(format!("CLIP ViT-B/32: {}", e)).into())
    }

    fn parse_model_name(model_name: &str) -> Result<EmbeddingModel> {
        // Map common model names to fastembed models
        match model_name {
            "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => Ok(EmbeddingModel::BGESmallENV15),
            "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => Ok(EmbeddingModel::BGEBaseENV15),
            "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => Ok(EmbeddingModel::BGELargeENV15),
            "sentence-transformers/all-MiniLM-L6-v2" | "all-MiniLM-L6-v2" => {
                Ok(EmbeddingModel::AllMiniLML6V2)
            }
            "sentence-transformers/all-MiniLM-L12-v2" | "all-MiniLM-L12-v2" => {
                Ok(EmbeddingModel::AllMiniLML12V2)
            }
            "nomic-ai/nomic-embed-text-v1.5" | "nomic-embed-text-v1.5" => {
                Ok(EmbeddingModel::NomicEmbedTextV15)
            }
            "nomic-ai/nomic-embed-text-v1" | "nomic-embed-text-v1" => {
                Ok(EmbeddingModel::NomicEmbedTextV1)
            }
            "jinaai/jina-embeddings-v2-base-code" | "jina-embeddings-v2-base-code" => {
                Ok(EmbeddingModel::JinaEmbeddingsV2BaseCode)
            }
            "thenlper/gte-base-en-v1.5" | "gte-base-en-v1.5" => Ok(EmbeddingModel::GTEBaseENV15),
            "thenlper/gte-large-en-v1.5" | "gte-large-en-v1.5" => Ok(EmbeddingModel::GTELargeENV15),
            "intfloat/multilingual-e5-small" | "multilingual-e5-small" => {
                Ok(EmbeddingModel::MultilingualE5Small)
            }
            "intfloat/multilingual-e5-base" | "multilingual-e5-base" => {
                Ok(EmbeddingModel::MultilingualE5Base)
            }
            "intfloat/multilingual-e5-large" | "multilingual-e5-large" => {
                Ok(EmbeddingModel::MultilingualE5Large)
            }
            _ => Err(EmbeddingError::ModelNotFound(format!(
                "Unknown model: {}. Supported: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5, \
                all-MiniLM-L6-v2, all-MiniLM-L12-v2, nomic-embed-text-v1.5, jina-embeddings-v2-base-code, \
                gte-base-en-v1.5, gte-large-en-v1.5, multilingual-e5-small/base/large",
                model_name
            ))
            .into()),
        }
    }

    fn get_model_dimension(model_name: &str) -> usize {
        match model_name {
            s if s.contains("bge-small") => 384,
            s if s.contains("bge-base") => 768,
            s if s.contains("bge-large") => 1024,
            s if s.contains("MiniLM-L6") => 384,
            s if s.contains("MiniLM-L12") => 384,
            s if s.contains("nomic") => 768,
            s if s.contains("jina") => 768,
            s if s.contains("gte-base") => 768,
            s if s.contains("gte-large") => 1024,
            s if s.contains("multilingual-e5-small") => 384,
            s if s.contains("multilingual-e5-base") => 768,
            s if s.contains("multilingual-e5-large") => 1024,
            _ => 384, // Default fallback
        }
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        if texts.len() > self.max_batch_size {
            return Err(EmbeddingError::BatchTooLarge(texts.len(), self.max_batch_size).into());
        }

        // Clone texts for the blocking operation
        let texts = texts.to_vec();
        let model = self.text_model.clone();

        // Run the embedding in a blocking task since fastembed is synchronous
        let embeddings = tokio::task::spawn_blocking(move || {
            let mut model = model.blocking_lock();
            model.embed(texts, None)
        })
        .await
        .map_err(|e| EmbeddingError::Api(format!("Task join error: {}", e)))?
        .map_err(|e| EmbeddingError::Api(format!("Embedding failed: {}", e)))?;

        Ok(embeddings)
    }

    async fn embed_images(&self, images: &[Bytes]) -> Result<Vec<Vec<f32>>> {
        let image_model = self.image_model.as_ref().ok_or_else(|| {
            EmbeddingError::Api("Image embedding not enabled for this provider".to_string())
        })?;

        if images.is_empty() {
            return Ok(vec![]);
        }

        // Write images to temporary files since fastembed expects file paths
        let mut temp_files = Vec::with_capacity(images.len());
        let mut temp_paths = Vec::with_capacity(images.len());

        for img_bytes in images {
            // Detect image format from magic bytes
            let extension = detect_image_format(img_bytes).unwrap_or("png");
            let temp_file = NamedTempFile::with_suffix(&format!(".{}", extension))
                .map_err(|e| EmbeddingError::Api(format!("Failed to create temp file: {}", e)))?;

            std::fs::write(temp_file.path(), img_bytes)
                .map_err(|e| EmbeddingError::Api(format!("Failed to write temp file: {}", e)))?;

            temp_paths.push(temp_file.path().to_path_buf());
            temp_files.push(temp_file); // Keep temp files alive
        }

        let model = image_model.clone();

        let embeddings = tokio::task::spawn_blocking(move || {
            let mut model = model.blocking_lock();
            let path_strs: Vec<String> = temp_paths
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            model.embed(path_strs, None)
        })
        .await
        .map_err(|e| EmbeddingError::Api(format!("Task join error: {}", e)))?
        .map_err(|e| EmbeddingError::Api(format!("Image embedding failed: {}", e)))?;

        // Temp files are automatically deleted when temp_files goes out of scope
        drop(temp_files);

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

/// Detect image format from magic bytes.
fn detect_image_format(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() < 4 {
        return None;
    }

    match &bytes[0..4] {
        [0x89, b'P', b'N', b'G'] => Some("png"),
        [0xFF, 0xD8, 0xFF, _] => Some("jpg"),
        [b'G', b'I', b'F', b'8'] => Some("gif"),
        [b'R', b'I', b'F', b'F'] if bytes.len() >= 12 && &bytes[8..12] == b"WEBP" => Some("webp"),
        [b'B', b'M', _, _] => Some("bmp"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require downloading models, so they're marked as ignored by default
    // Run with: cargo test --features test_local_embeddings -- --ignored

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn test_local_embedding_creation() {
        let provider = LocalEmbeddingProvider::default_model();
        assert!(provider.is_ok());
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn test_local_embedding_dimension() {
        let provider = LocalEmbeddingProvider::new("BAAI/bge-small-en-v1.5", false).unwrap();
        assert_eq!(provider.dimension(), 384);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn test_local_embed_text() {
        let provider = LocalEmbeddingProvider::default_model().unwrap();
        let texts = vec!["Hello, world!".to_string(), "This is a test.".to_string()];
        let embeddings = provider.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), provider.dimension());
        assert_eq!(embeddings[1].len(), provider.dimension());
    }

    #[test]
    fn test_model_dimension_mapping() {
        assert_eq!(
            LocalEmbeddingProvider::get_model_dimension("BAAI/bge-small-en-v1.5"),
            384
        );
        assert_eq!(
            LocalEmbeddingProvider::get_model_dimension("BAAI/bge-base-en-v1.5"),
            768
        );
        assert_eq!(
            LocalEmbeddingProvider::get_model_dimension("BAAI/bge-large-en-v1.5"),
            1024
        );
        assert_eq!(
            LocalEmbeddingProvider::get_model_dimension("all-MiniLM-L6-v2"),
            384
        );
    }

    #[test]
    fn test_parse_model_name() {
        assert!(LocalEmbeddingProvider::parse_model_name("BAAI/bge-small-en-v1.5").is_ok());
        assert!(LocalEmbeddingProvider::parse_model_name("bge-small-en-v1.5").is_ok());
        assert!(LocalEmbeddingProvider::parse_model_name("unknown-model").is_err());
    }

    #[test]
    fn test_detect_image_format() {
        assert_eq!(detect_image_format(&[0x89, b'P', b'N', b'G']), Some("png"));
        assert_eq!(detect_image_format(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("jpg"));
        assert_eq!(detect_image_format(&[b'G', b'I', b'F', b'8']), Some("gif"));
        assert_eq!(detect_image_format(&[b'B', b'M', 0, 0]), Some("bmp"));
        assert_eq!(detect_image_format(&[0, 0, 0, 0]), None);
    }
}

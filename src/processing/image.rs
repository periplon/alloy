//! Image processor for OCR, CLIP embeddings, and Vision API descriptions.

use async_trait::async_trait;
use bytes::Bytes;
use image::GenericImageView;
use tracing::{debug, warn};

use crate::config::ImageProcessingConfig;
use crate::error::{ProcessingError, Result};
use crate::processing::{ContentMetadata, ImageData, ProcessedContent, Processor};
use crate::sources::SourceItem;

/// Processor for image files.
pub struct ImageProcessor {
    config: ImageProcessingConfig,
    supported_types: Vec<&'static str>,
    vision_client: Option<VisionApiClient>,
}

impl ImageProcessor {
    /// Create a new image processor.
    pub fn new(config: ImageProcessingConfig) -> Self {
        let vision_client = if config.vision_api {
            VisionApiClient::from_env().ok()
        } else {
            None
        };

        Self {
            config,
            supported_types: vec![
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/gif",
                "image/webp",
                "image/bmp",
                "image/tiff",
                "image/*",
            ],
            vision_client,
        }
    }
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new(ImageProcessingConfig::default())
    }
}

#[async_trait]
impl Processor for ImageProcessor {
    async fn process(&self, content: Bytes, item: &SourceItem) -> Result<ProcessedContent> {
        // Load and analyze the image
        let img = image::load_from_memory(&content)
            .map_err(|e| ProcessingError::Extraction(format!("Failed to load image: {}", e)))?;

        let (width, height) = img.dimensions();

        debug!(
            uri = %item.uri,
            width,
            height,
            "Processing image"
        );

        let mut text_parts = Vec::new();
        let mut ocr_text = None;
        let mut description = None;

        // Perform OCR if enabled
        if self.config.ocr {
            match perform_ocr(&content).await {
                Ok(text) if !text.trim().is_empty() => {
                    text_parts.push(format!("OCR Text:\n{}", text));
                    ocr_text = Some(text);
                }
                Ok(_) => {
                    debug!(uri = %item.uri, "OCR returned empty text");
                }
                Err(e) => {
                    warn!(uri = %item.uri, error = %e, "OCR failed");
                }
            }
        }

        // Get Vision API description if enabled
        if self.config.vision_api {
            if let Some(ref client) = self.vision_client {
                match client.describe_image(&content, &item.mime_type).await {
                    Ok(desc) => {
                        text_parts.push(format!("Image Description:\n{}", desc));
                        description = Some(desc);
                    }
                    Err(e) => {
                        warn!(uri = %item.uri, error = %e, "Vision API failed");
                    }
                }
            }
        }

        // Combine all text
        let text = if text_parts.is_empty() {
            format!(
                "Image: {}x{} pixels, format: {}",
                width, height, item.mime_type
            )
        } else {
            text_parts.join("\n\n")
        };

        let metadata = ContentMetadata {
            extra: serde_json::json!({
                "width": width,
                "height": height,
                "format": &item.mime_type,
                "has_ocr": ocr_text.is_some(),
                "has_description": description.is_some(),
            }),
            ..ContentMetadata::from_text(&text)
        };

        // Create image data for potential CLIP embedding
        let image_data = ImageData {
            id: item.id.clone(),
            data: content,
            mime_type: item.mime_type.clone(),
            width,
            height,
            ocr_text,
            description,
            page: None,
        };

        Ok(ProcessedContent {
            text,
            chunks: Vec::new(),
            metadata,
            images: vec![image_data],
        })
    }

    fn supported_types(&self) -> &[&str] {
        &self.supported_types
    }

    fn name(&self) -> &str {
        "image"
    }
}

/// Perform OCR on an image.
/// Uses tesseract as a subprocess if available on the system.
async fn perform_ocr(content: &Bytes) -> Result<String> {
    // Try to run tesseract as a subprocess
    let temp_file = std::env::temp_dir().join(format!("alloy_ocr_{}.png", uuid::Uuid::new_v4()));

    // Save image to temp file
    if let Err(e) = std::fs::write(&temp_file, content) {
        return Err(ProcessingError::Ocr(format!("Failed to write temp file: {}", e)).into());
    }

    // Run tesseract
    let output = std::process::Command::new("tesseract")
        .arg(&temp_file)
        .arg("stdout")
        .arg("-l")
        .arg("eng")
        .output();

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_file);

    match output {
        Ok(out) if out.status.success() => Ok(String::from_utf8_lossy(&out.stdout).to_string()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            Err(ProcessingError::Ocr(format!("Tesseract failed: {}", stderr)).into())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Tesseract not installed - return empty string
            debug!("Tesseract not found, skipping OCR");
            Ok(String::new())
        }
        Err(e) => Err(ProcessingError::Ocr(format!("Failed to run tesseract: {}", e)).into()),
    }
}

/// Client for Vision API (OpenAI-compatible).
pub struct VisionApiClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl VisionApiClient {
    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .or_else(|_| std::env::var("VISION_API_KEY"))
            .map_err(|_| {
                ProcessingError::Extraction("OPENAI_API_KEY or VISION_API_KEY not set".to_string())
            })?;

        let base_url = std::env::var("VISION_API_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

        let model = std::env::var("VISION_API_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        Ok(Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            model,
        })
    }

    /// Create a client with explicit configuration.
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            model,
        }
    }

    /// Describe an image using the Vision API.
    pub async fn describe_image(&self, content: &Bytes, mime_type: &str) -> Result<String> {
        use base64::Engine;

        let base64_image = base64::engine::general_purpose::STANDARD.encode(content);
        let data_url = format!("data:{};base64,{}", mime_type, base64_image);

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail, focusing on text content, objects, and any relevant information for document indexing."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }],
            "max_tokens": 500
        });

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                ProcessingError::Extraction(format!("Vision API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ProcessingError::Extraction(format!(
                "Vision API returned {}: {}",
                status, text
            ))
            .into());
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ProcessingError::Extraction(format!("Failed to parse response: {}", e)))?;

        let description = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(description)
    }
}

/// CLIP embedding support for images.
/// This is a placeholder for fastembed CLIP integration.
#[derive(Default)]
pub struct ClipEmbedder {
    // Will be populated when fastembed is added
}

impl ClipEmbedder {
    /// Create a new CLIP embedder.
    pub fn new() -> Result<Self> {
        // Placeholder - will use fastembed for CLIP models
        Ok(Self {})
    }

    /// Generate CLIP embeddings for an image.
    pub async fn embed_image(&self, _content: &Bytes) -> Result<Vec<f32>> {
        // Placeholder - returns empty vector until fastembed integration
        // In production, this would use fastembed with a CLIP model
        Ok(Vec::new())
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        // CLIP ViT-B/32 dimension
        512
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::io::Cursor;

    fn make_item(mime_type: &str) -> SourceItem {
        SourceItem {
            id: "test".to_string(),
            uri: "/test/image.png".to_string(),
            mime_type: mime_type.to_string(),
            size: 100,
            modified: Utc::now(),
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    fn test_image_processor_supports() {
        let processor = ImageProcessor::default();
        assert!(processor.supports("image/png"));
        assert!(processor.supports("image/jpeg"));
        assert!(processor.supports("image/gif"));
        assert!(!processor.supports("text/plain"));
        assert!(!processor.supports("application/pdf"));
    }

    #[tokio::test]
    async fn test_process_png() {
        let processor = ImageProcessor::new(ImageProcessingConfig {
            ocr: false,
            clip: false,
            vision_api: false,
        });

        // Create a minimal valid PNG (1x1 white pixel)
        let png_data = create_minimal_png();
        let item = make_item("image/png");

        let result = processor.process(Bytes::from(png_data), &item).await;
        assert!(result.is_ok());

        let content = result.unwrap();
        assert!(!content.images.is_empty());
        assert_eq!(content.images[0].width, 1);
        assert_eq!(content.images[0].height, 1);
    }

    /// Create a minimal valid PNG image (1x1 white pixel).
    fn create_minimal_png() -> Vec<u8> {
        let mut img = image::RgbaImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));

        let mut bytes = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
        bytes
    }

    #[test]
    fn test_clip_embedder_dimension() {
        let embedder = ClipEmbedder::default();
        assert_eq!(embedder.dimension(), 512);
    }

    #[test]
    fn test_vision_client_new() {
        let client = VisionApiClient::new(
            "https://api.example.com".to_string(),
            "test-key".to_string(),
            "gpt-4o-mini".to_string(),
        );
        assert_eq!(client.base_url, "https://api.example.com");
        assert_eq!(client.model, "gpt-4o-mini");
    }
}

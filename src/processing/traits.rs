//! Processor trait definitions and common types.

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::sources::SourceItem;

/// Processed content from a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedContent {
    /// Full extracted text
    pub text: String,
    /// Text chunks for embedding
    pub chunks: Vec<TextChunk>,
    /// Extracted metadata
    pub metadata: ContentMetadata,
    /// Optional image data for CLIP embedding
    pub images: Vec<ImageData>,
}

impl ProcessedContent {
    /// Create a new ProcessedContent with just text.
    pub fn from_text(text: String) -> Self {
        Self {
            text,
            chunks: Vec::new(),
            metadata: ContentMetadata::default(),
            images: Vec::new(),
        }
    }

    /// Create a new ProcessedContent with text and metadata.
    pub fn with_metadata(text: String, metadata: ContentMetadata) -> Self {
        Self {
            text,
            chunks: Vec::new(),
            metadata,
            images: Vec::new(),
        }
    }
}

/// A chunk of text for embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// Chunk ID (unique within document)
    pub id: String,
    /// Chunk index (0-based)
    pub index: usize,
    /// Chunk text content
    pub text: String,
    /// Start offset in original text (character offset)
    pub start_offset: usize,
    /// End offset in original text (character offset)
    pub end_offset: usize,
    /// Optional section/heading context
    pub section: Option<String>,
}

impl TextChunk {
    /// Create a new text chunk.
    pub fn new(id: String, index: usize, text: String, start: usize, end: usize) -> Self {
        Self {
            id,
            index,
            text,
            start_offset: start,
            end_offset: end,
            section: None,
        }
    }

    /// Add section context to the chunk.
    pub fn with_section(mut self, section: String) -> Self {
        self.section = Some(section);
        self
    }
}

/// Metadata extracted from content.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentMetadata {
    /// Document title (if extractable)
    pub title: Option<String>,
    /// Document author
    pub author: Option<String>,
    /// Document language
    pub language: Option<String>,
    /// Page count (for PDFs)
    pub page_count: Option<usize>,
    /// Word count
    pub word_count: Option<usize>,
    /// Character count
    pub char_count: Option<usize>,
    /// Creation date
    pub created: Option<String>,
    /// Modification date
    pub modified: Option<String>,
    /// Additional key-value metadata
    pub extra: serde_json::Value,
}

impl ContentMetadata {
    /// Create metadata with word and character counts from text.
    pub fn from_text(text: &str) -> Self {
        Self {
            word_count: Some(text.split_whitespace().count()),
            char_count: Some(text.chars().count()),
            ..Default::default()
        }
    }
}

/// Image data for embedding or OCR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// Image identifier
    pub id: String,
    /// Raw image bytes
    #[serde(with = "bytes_serde")]
    pub data: Bytes,
    /// MIME type of the image
    pub mime_type: String,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// OCR-extracted text (if available)
    pub ocr_text: Option<String>,
    /// Vision API description (if available)
    pub description: Option<String>,
    /// Page number (for multi-page documents)
    pub page: Option<usize>,
}

/// Serde helper for Bytes
mod bytes_serde {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes)
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Bytes, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &s)
            .map(Bytes::from)
            .map_err(serde::de::Error::custom)
    }
}

/// Chunking configuration.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub chunk_overlap: usize,
    /// Minimum chunk size (chunks smaller than this will be merged)
    pub min_chunk_size: usize,
    /// Whether to respect sentence boundaries
    pub respect_sentences: bool,
    /// Whether to respect paragraph boundaries
    pub respect_paragraphs: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 64,
            min_chunk_size: 100,
            respect_sentences: true,
            respect_paragraphs: true,
        }
    }
}

impl ChunkConfig {
    /// Create a ChunkConfig from processing settings.
    pub fn from_settings(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            min_chunk_size: chunk_size / 5,
            ..Default::default()
        }
    }
}

/// Trait for document processors.
#[async_trait]
pub trait Processor: Send + Sync {
    /// Process content and extract text.
    async fn process(&self, content: Bytes, item: &SourceItem) -> Result<ProcessedContent>;

    /// Return the MIME types this processor supports.
    fn supported_types(&self) -> &[&str];

    /// Check if this processor supports the given MIME type.
    fn supports(&self, mime_type: &str) -> bool {
        self.supported_types()
            .iter()
            .any(|t| mime_matches(t, mime_type))
    }

    /// Get the processor name for logging.
    fn name(&self) -> &str;
}

/// A boxed processor for dynamic dispatch.
pub type BoxedProcessor = Box<dyn Processor>;

/// Check if a pattern matches a MIME type.
/// Supports wildcards like "text/*" or "image/*".
fn mime_matches(pattern: &str, mime_type: &str) -> bool {
    if pattern == mime_type {
        return true;
    }

    if let Some(prefix) = pattern.strip_suffix("/*") {
        if let Some(type_prefix) = mime_type.split('/').next() {
            return prefix == type_prefix;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mime_matches_exact() {
        assert!(mime_matches("text/plain", "text/plain"));
        assert!(!mime_matches("text/plain", "text/html"));
    }

    #[test]
    fn test_mime_matches_wildcard() {
        assert!(mime_matches("text/*", "text/plain"));
        assert!(mime_matches("text/*", "text/html"));
        assert!(mime_matches("image/*", "image/png"));
        assert!(!mime_matches("text/*", "image/png"));
    }

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 64);
    }

    #[test]
    fn test_content_metadata_from_text() {
        let text = "Hello world. This is a test.";
        let meta = ContentMetadata::from_text(text);
        assert_eq!(meta.word_count, Some(6));
        assert_eq!(meta.char_count, Some(28));
    }

    #[test]
    fn test_text_chunk_with_section() {
        let chunk = TextChunk::new("1".to_string(), 0, "text".to_string(), 0, 4)
            .with_section("Introduction".to_string());
        assert_eq!(chunk.section, Some("Introduction".to_string()));
    }
}

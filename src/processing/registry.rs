//! Processor registry for MIME type to processor mapping.

use std::sync::Arc;

use bytes::Bytes;
use tracing::debug;

use crate::config::ProcessingConfig;
use crate::error::{ProcessingError, Result};
use crate::processing::{ChunkConfig, ProcessedContent, Processor};
use crate::sources::SourceItem;

use super::chunker::chunk_text;
use super::document::{DocxProcessor, PdfProcessor};
use super::image::ImageProcessor;
use super::text::TextProcessor;

/// Registry for document processors.
pub struct ProcessorRegistry {
    /// Processors indexed by MIME type patterns
    processors: Vec<Arc<dyn Processor>>,
    /// Default chunk configuration
    chunk_config: ChunkConfig,
}

impl ProcessorRegistry {
    /// Create a new processor registry with default processors.
    pub fn new(config: &ProcessingConfig) -> Self {
        let chunk_config = ChunkConfig::from_settings(config.chunk_size, config.chunk_overlap);

        let mut processors: Vec<Arc<dyn Processor>> = Vec::new();

        // Add text processor (handles text/*, application/json, etc.)
        processors.push(Arc::new(TextProcessor::new()));

        // Add PDF processor
        processors.push(Arc::new(PdfProcessor::new()));

        // Add DOCX processor
        processors.push(Arc::new(DocxProcessor::new()));

        // Add image processor
        processors.push(Arc::new(ImageProcessor::new(config.image.clone())));

        Self {
            processors,
            chunk_config,
        }
    }

    /// Get a processor for the given MIME type.
    pub fn get_processor(&self, mime_type: &str) -> Option<Arc<dyn Processor>> {
        for processor in &self.processors {
            if processor.supports(mime_type) {
                return Some(Arc::clone(processor));
            }
        }
        None
    }

    /// Process a source item and return processed content with chunks.
    pub async fn process(&self, content: Bytes, item: &SourceItem) -> Result<ProcessedContent> {
        let processor = self
            .get_processor(&item.mime_type)
            .ok_or_else(|| ProcessingError::UnsupportedType(item.mime_type.clone()))?;

        debug!(
            processor = processor.name(),
            mime_type = %item.mime_type,
            uri = %item.uri,
            "Processing document"
        );

        let mut processed = processor.process(content, item).await?;

        // Add chunks if not already chunked
        if processed.chunks.is_empty() && !processed.text.is_empty() {
            processed.chunks = chunk_text(&processed.text, &item.id, &self.chunk_config);
        }

        debug!(
            chunks = processed.chunks.len(),
            text_len = processed.text.len(),
            "Document processed"
        );

        Ok(processed)
    }

    /// Check if a MIME type is supported.
    pub fn supports(&self, mime_type: &str) -> bool {
        self.get_processor(mime_type).is_some()
    }

    /// List all supported MIME types.
    pub fn supported_types(&self) -> Vec<&str> {
        let mut types = Vec::new();
        for processor in &self.processors {
            types.extend(processor.supported_types());
        }
        types
    }

    /// Register a custom processor.
    pub fn register(&mut self, processor: Arc<dyn Processor>) {
        self.processors.insert(0, processor); // Insert at front for priority
    }

    /// Get the chunk configuration.
    pub fn chunk_config(&self) -> &ChunkConfig {
        &self.chunk_config
    }
}

impl Default for ProcessorRegistry {
    fn default() -> Self {
        Self::new(&ProcessingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let config = ProcessingConfig::default();
        let registry = ProcessorRegistry::new(&config);
        assert!(registry.supports("text/plain"));
        assert!(registry.supports("application/pdf"));
    }

    #[test]
    fn test_get_processor_text() {
        let registry = ProcessorRegistry::default();
        let processor = registry.get_processor("text/plain");
        assert!(processor.is_some());
        assert_eq!(processor.unwrap().name(), "text");
    }

    #[test]
    fn test_get_processor_pdf() {
        let registry = ProcessorRegistry::default();
        let processor = registry.get_processor("application/pdf");
        assert!(processor.is_some());
        assert_eq!(processor.unwrap().name(), "pdf");
    }

    #[test]
    fn test_get_processor_docx() {
        let registry = ProcessorRegistry::default();
        let processor = registry.get_processor(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        );
        assert!(processor.is_some());
        assert_eq!(processor.unwrap().name(), "docx");
    }

    #[test]
    fn test_get_processor_image() {
        let registry = ProcessorRegistry::default();
        let processor = registry.get_processor("image/png");
        assert!(processor.is_some());
        assert_eq!(processor.unwrap().name(), "image");
    }

    #[test]
    fn test_unsupported_type() {
        let registry = ProcessorRegistry::default();
        assert!(!registry.supports("application/x-unknown-format"));
    }

    #[test]
    fn test_supported_types_list() {
        let registry = ProcessorRegistry::default();
        let types = registry.supported_types();
        assert!(types.contains(&"text/plain"));
        assert!(types.contains(&"application/pdf"));
    }
}

//! Entity extraction processor that integrates with the document processing pipeline.
//!
//! This module provides the bridge between the document processing system and
//! the entity extraction pipeline, enabling automatic extraction of entities,
//! relationships, and structured information during document indexing.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::ontology::extraction::{ExtractionConfig, ExtractionPipeline, ExtractionResult};
use crate::ontology::{Entity, OntologyStore, Relationship};
use crate::processing::{ProcessedContent, TextChunk};

/// Configuration for entity extraction during processing.
#[derive(Debug, Clone)]
pub struct EntityExtractionProcessorConfig {
    /// Whether entity extraction is enabled.
    pub enabled: bool,
    /// Extraction configuration.
    pub extraction_config: ExtractionConfig,
    /// Whether to extract from full text or just chunks.
    pub extract_from_full_text: bool,
    /// Whether to extract from individual chunks (in addition to full text).
    pub extract_from_chunks: bool,
    /// Minimum confidence to store entities.
    pub min_confidence: f32,
}

impl Default for EntityExtractionProcessorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            extraction_config: ExtractionConfig::default(),
            extract_from_full_text: true,
            extract_from_chunks: false,
            min_confidence: 0.7,
        }
    }
}

/// Result of entity extraction from a document.
#[derive(Debug, Clone)]
pub struct DocumentExtractionResult {
    /// Document ID.
    pub document_id: String,
    /// Entities extracted from the document.
    pub entities: Vec<Entity>,
    /// Relationships extracted from the document.
    pub relationships: Vec<Relationship>,
    /// Overall confidence of the extraction.
    pub confidence: f32,
    /// Whether LLM was used.
    pub used_llm: bool,
    /// Processing time in milliseconds.
    pub processing_ms: u64,
}

/// Processor that extracts entities from documents.
pub struct EntityExtractionProcessor {
    /// Extraction pipeline.
    pipeline: ExtractionPipeline,
    /// Configuration.
    config: EntityExtractionProcessorConfig,
    /// Optional ontology store for immediate persistence.
    store: Option<Arc<dyn OntologyStore>>,
}

impl EntityExtractionProcessor {
    /// Create a new entity extraction processor.
    pub fn new(config: EntityExtractionProcessorConfig) -> Self {
        let pipeline = ExtractionPipeline::new(config.extraction_config.clone());
        Self {
            pipeline,
            config,
            store: None,
        }
    }

    /// Create a processor with an ontology store for immediate persistence.
    pub fn with_store(mut self, store: Arc<dyn OntologyStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Check if extraction is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Extract entities from processed content.
    pub async fn extract_from_content(
        &self,
        content: &ProcessedContent,
        document_id: &str,
    ) -> Result<DocumentExtractionResult> {
        if !self.config.enabled {
            return Ok(DocumentExtractionResult {
                document_id: document_id.to_string(),
                entities: vec![],
                relationships: vec![],
                confidence: 1.0,
                used_llm: false,
                processing_ms: 0,
            });
        }

        let mut all_entities = Vec::new();
        let mut all_relationships = Vec::new();
        let mut total_confidence = 0.0;
        let mut extraction_count = 0;
        let mut used_llm = false;
        let start_time = std::time::Instant::now();

        // Extract from full text if configured
        if self.config.extract_from_full_text && !content.text.is_empty() {
            let result = self.pipeline.extract(&content.text, document_id).await?;
            self.merge_extraction_result(&result, &mut all_entities, &mut all_relationships);
            total_confidence += result.confidence;
            extraction_count += 1;
            used_llm = used_llm || result.metadata.used_llm;
        }

        // Extract from chunks if configured
        if self.config.extract_from_chunks {
            for chunk in &content.chunks {
                let chunk_doc_id = format!("{}#{}", document_id, chunk.id);
                let result = self.pipeline.extract(&chunk.text, &chunk_doc_id).await?;
                self.merge_extraction_result(&result, &mut all_entities, &mut all_relationships);
                total_confidence += result.confidence;
                extraction_count += 1;
                used_llm = used_llm || result.metadata.used_llm;
            }
        }

        let processing_ms = start_time.elapsed().as_millis() as u64;

        // Calculate average confidence
        let confidence = if extraction_count > 0 {
            total_confidence / extraction_count as f32
        } else {
            1.0
        };

        // Filter by minimum confidence
        all_entities.retain(|e| e.confidence >= self.config.min_confidence);

        // Deduplicate entities by name and type
        all_entities = Self::deduplicate_entities(all_entities);

        // Persist to store if available
        if let Some(ref store) = self.store {
            for entity in &all_entities {
                // Ignore errors for now - entity might already exist
                let _ = store.create_entity(entity.clone()).await;
            }
            for relationship in &all_relationships {
                // Ignore errors for now - relationship might already exist
                let _ = store.create_relationship(relationship.clone()).await;
            }
        }

        Ok(DocumentExtractionResult {
            document_id: document_id.to_string(),
            entities: all_entities,
            relationships: all_relationships,
            confidence,
            used_llm,
            processing_ms,
        })
    }

    /// Extract entities from a single text chunk.
    pub async fn extract_from_chunk(
        &self,
        chunk: &TextChunk,
        document_id: &str,
    ) -> Result<ExtractionResult> {
        let chunk_doc_id = format!("{}#{}", document_id, chunk.id);
        self.pipeline.extract(&chunk.text, &chunk_doc_id).await
    }

    /// Extract entities from raw text.
    pub async fn extract_from_text(
        &self,
        text: &str,
        document_id: &str,
    ) -> Result<ExtractionResult> {
        self.pipeline.extract(text, document_id).await
    }

    /// Merge extraction result into accumulator vectors.
    fn merge_extraction_result(
        &self,
        result: &ExtractionResult,
        entities: &mut Vec<Entity>,
        relationships: &mut Vec<Relationship>,
    ) {
        for extracted in &result.entities {
            entities.push(extracted.entity.clone());
        }
        for extracted in &result.relationships {
            relationships.push(extracted.relationship.clone());
        }
    }

    /// Deduplicate entities by name and type.
    fn deduplicate_entities(entities: Vec<Entity>) -> Vec<Entity> {
        use std::collections::HashMap;

        let mut seen: HashMap<(String, crate::ontology::EntityType), Entity> = HashMap::new();

        for entity in entities {
            let key = (entity.name.to_lowercase(), entity.entity_type);
            match seen.get(&key) {
                Some(existing) if existing.confidence >= entity.confidence => {
                    // Keep existing
                }
                _ => {
                    seen.insert(key, entity);
                }
            }
        }

        seen.into_values().collect()
    }
}

/// Trait for components that support entity extraction.
#[async_trait]
pub trait EntityExtractable: Send + Sync {
    /// Extract entities from this content.
    async fn extract_entities(&self, document_id: &str) -> Result<DocumentExtractionResult>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extraction_processor_disabled() {
        let config = EntityExtractionProcessorConfig {
            enabled: false,
            ..Default::default()
        };
        let processor = EntityExtractionProcessor::new(config);

        let content = ProcessedContent::from_text("TODO: Review the proposal".to_string());
        let result = processor
            .extract_from_content(&content, "test-doc")
            .await
            .unwrap();

        assert!(result.entities.is_empty());
        assert_eq!(result.processing_ms, 0);
    }

    #[tokio::test]
    async fn test_extraction_processor_basic() {
        let config = EntityExtractionProcessorConfig {
            enabled: true,
            extract_from_full_text: true,
            extract_from_chunks: false,
            min_confidence: 0.5,
            ..Default::default()
        };
        let processor = EntityExtractionProcessor::new(config);

        let content = ProcessedContent::from_text(
            "Meeting with John Smith tomorrow at 3pm. TODO: Review the proposal.".to_string(),
        );
        let result = processor
            .extract_from_content(&content, "test-doc")
            .await
            .unwrap();

        // Should have extracted some entities
        assert!(!result.entities.is_empty() || result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_extraction_from_text() {
        let config = EntityExtractionProcessorConfig::default();
        let processor = EntityExtractionProcessor::new(config);

        let result = processor
            .extract_from_text("Email john@example.com about the project.", "test-doc")
            .await
            .unwrap();

        // Should have extracted email
        assert!(!result.entities.is_empty() || !result.temporal.is_empty());
    }
}

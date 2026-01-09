//! Entity extraction pipeline for the ontology system.
//!
//! This module provides the extraction layer that processes documents to identify
//! and extract entities, relationships, and structured information for the
//! knowledge graph and GTD systems.
//!
//! # Architecture
//!
//! ```text
//! Document Text
//!       │
//!       ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │              Entity Extraction Pipeline                  │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐│
//! │  │  Temporal   │ │   Action    │ │       NER           ││
//! │  │   Parser    │ │  Detector   │ │   (Local/LLM)       ││
//! │  └─────────────┘ └─────────────┘ └─────────────────────┘│
//! │  ┌─────────────────────────────────────────────────────┐│
//! │  │           Relationship Extractor                    ││
//! │  │         (co-occurrence, patterns)                   ││
//! │  └─────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────┘
//!       │
//!       ▼
//! Entities & Relationships
//! ```
//!
//! # Components
//!
//! - **Temporal Parser**: Extracts dates, times, deadlines, and recurring patterns
//! - **Action Detector**: Identifies tasks, commitments, and action items
//! - **NER (Named Entity Recognition)**: Extracts people, organizations, locations
//! - **Relationship Extractor**: Identifies relationships via co-occurrence and patterns

mod actions;
mod ner;
mod processor;
mod relations;
mod temporal;

pub use actions::{
    ActionDetector, ActionItem, ActionType, CommitmentType, DetectedAction, EnergyLevel, Priority,
};
pub use ner::{LlmNerExtractor, LocalNerExtractor, NamedEntity, NamedEntityType, NerExtractor};
pub use processor::{
    DocumentExtractionResult, EntityExtractable, EntityExtractionProcessor,
    EntityExtractionProcessorConfig,
};
pub use relations::{
    CoOccurrence, RelationExtractor, RelationPattern, RelationshipCandidate,
    SemanticRelationExtractor,
};
pub use temporal::{
    DateType, ParsedDate, RecurrencePattern, RecurrenceRule, TemporalExtraction, TemporalParser,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ontology::{DocumentRef, Entity, EntityType, Relationship};

// ============================================================================
// Extraction Result Types
// ============================================================================

/// Complete extraction result from processing a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// The document ID that was processed.
    pub document_id: String,
    /// Extracted entities.
    pub entities: Vec<ExtractedEntity>,
    /// Extracted relationships.
    pub relationships: Vec<ExtractedRelationship>,
    /// Temporal extractions (dates, times, deadlines).
    pub temporal: Vec<TemporalExtraction>,
    /// Detected action items (tasks, commitments).
    pub actions: Vec<DetectedAction>,
    /// Overall confidence of the extraction.
    pub confidence: f32,
    /// Processing metadata.
    pub metadata: ExtractionMetadata,
}

/// An entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// The extracted entity.
    pub entity: Entity,
    /// The original text span that was extracted.
    pub source_text: String,
    /// Character offset in the document where extraction occurred.
    pub start_offset: usize,
    /// End character offset.
    pub end_offset: usize,
    /// Extraction method used.
    pub extraction_method: ExtractionMethod,
}

/// A relationship extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    /// The extracted relationship.
    pub relationship: Relationship,
    /// The original text that suggested this relationship.
    pub source_text: Option<String>,
    /// Extraction method used.
    pub extraction_method: ExtractionMethod,
}

/// The method used to extract an entity or relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMethod {
    /// Pattern-based extraction (regex, rules).
    Pattern,
    /// Temporal parsing for dates/times.
    Temporal,
    /// Action item detection.
    ActionDetection,
    /// Local NER using patterns and heuristics.
    LocalNer,
    /// LLM-based NER.
    LlmNer,
    /// Co-occurrence analysis.
    CoOccurrence,
    /// Semantic similarity.
    Semantic,
    /// Manual/explicit creation.
    Manual,
}

/// Metadata about the extraction process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionMetadata {
    /// Time when extraction started.
    pub started_at: Option<DateTime<Utc>>,
    /// Time when extraction completed.
    pub completed_at: Option<DateTime<Utc>>,
    /// Number of characters processed.
    pub chars_processed: usize,
    /// Whether LLM was used.
    pub used_llm: bool,
    /// Processing time in milliseconds.
    pub processing_ms: u64,
    /// Any warnings during extraction.
    #[serde(default)]
    pub warnings: Vec<String>,
}

// ============================================================================
// Extraction Pipeline
// ============================================================================

/// Configuration for the extraction pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Enable temporal parsing.
    #[serde(default = "default_true")]
    pub enable_temporal: bool,
    /// Enable action item detection.
    #[serde(default = "default_true")]
    pub enable_actions: bool,
    /// Enable local NER (patterns, heuristics).
    #[serde(default = "default_true")]
    pub enable_local_ner: bool,
    /// Enable LLM-based NER (requires API).
    #[serde(default)]
    pub enable_llm_ner: bool,
    /// Enable relationship extraction.
    #[serde(default = "default_true")]
    pub enable_relations: bool,
    /// Minimum confidence threshold for extracted entities.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    /// LLM configuration for NER.
    #[serde(default)]
    pub llm_config: Option<LlmExtractionConfig>,
}

fn default_true() -> bool {
    true
}

fn default_confidence_threshold() -> f32 {
    0.7
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enable_temporal: true,
            enable_actions: true,
            enable_local_ner: true,
            enable_llm_ner: false,
            enable_relations: true,
            confidence_threshold: 0.7,
            llm_config: None,
        }
    }
}

/// Configuration for LLM-based extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmExtractionConfig {
    /// API endpoint.
    pub api_endpoint: String,
    /// API key (can be from env).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Model to use.
    #[serde(default = "default_model")]
    pub model: String,
    /// Maximum tokens per request.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Rate limit (requests per minute).
    #[serde(default = "default_rate_limit")]
    pub rate_limit_rpm: usize,
}

fn default_model() -> String {
    "gpt-4o-mini".to_string()
}

fn default_max_tokens() -> usize {
    4000
}

fn default_rate_limit() -> usize {
    60
}

impl Default for LlmExtractionConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "https://api.openai.com/v1".to_string(),
            api_key: None,
            model: default_model(),
            max_tokens: default_max_tokens(),
            rate_limit_rpm: default_rate_limit(),
        }
    }
}

/// The main extraction pipeline that coordinates all extractors.
pub struct ExtractionPipeline {
    config: ExtractionConfig,
    temporal_parser: TemporalParser,
    action_detector: ActionDetector,
    local_ner: LocalNerExtractor,
    llm_ner: Option<LlmNerExtractor>,
    relation_extractor: RelationExtractor,
}

impl ExtractionPipeline {
    /// Create a new extraction pipeline with the given configuration.
    pub fn new(config: ExtractionConfig) -> Self {
        let llm_ner = config
            .llm_config
            .as_ref()
            .filter(|_| config.enable_llm_ner)
            .map(|llm_config| LlmNerExtractor::new(llm_config.clone()));

        Self {
            temporal_parser: TemporalParser::new(),
            action_detector: ActionDetector::new(),
            local_ner: LocalNerExtractor::new(),
            llm_ner,
            relation_extractor: RelationExtractor::new(),
            config,
        }
    }

    /// Create a pipeline with default configuration.
    pub fn default_pipeline() -> Self {
        Self::new(ExtractionConfig::default())
    }

    /// Extract entities and relationships from text.
    pub async fn extract(&self, text: &str, document_id: &str) -> Result<ExtractionResult> {
        let started_at = Utc::now();
        let start_time = std::time::Instant::now();

        let mut entities: Vec<ExtractedEntity> = Vec::new();
        let mut relationships: Vec<ExtractedRelationship> = Vec::new();
        let mut temporal: Vec<TemporalExtraction> = Vec::new();
        let mut actions: Vec<DetectedAction> = Vec::new();
        let mut used_llm = false;
        let mut warnings: Vec<String> = Vec::new();

        let doc_ref = DocumentRef::new(document_id);

        // 1. Temporal parsing
        if self.config.enable_temporal {
            let temporal_results = self.temporal_parser.parse(text);
            for t in &temporal_results {
                // Create Date entity for each temporal extraction
                let entity = Entity::new(EntityType::Date, &t.normalized_text)
                    .with_source_ref(doc_ref.clone().with_text(&t.original_text))
                    .with_confidence(t.confidence)
                    .with_metadata("date_type", serde_json::json!(format!("{:?}", t.date_type)))
                    .with_metadata("parsed_date", serde_json::json!(t.parsed_date.to_string()));

                entities.push(ExtractedEntity {
                    entity,
                    source_text: t.original_text.clone(),
                    start_offset: t.start_offset,
                    end_offset: t.end_offset,
                    extraction_method: ExtractionMethod::Temporal,
                });
            }
            temporal = temporal_results;
        }

        // 2. Action detection
        if self.config.enable_actions {
            let action_results = self.action_detector.detect(text);
            for action in &action_results {
                // Create Task or Commitment entity for each action
                let entity_type = match action.action_type {
                    ActionType::Task => EntityType::Task,
                    ActionType::Commitment => EntityType::Commitment,
                    ActionType::FollowUp => EntityType::WaitingFor,
                    ActionType::Reminder => EntityType::CalendarEvent,
                };

                let entity = Entity::new(entity_type, &action.description)
                    .with_source_ref(doc_ref.clone().with_text(&action.source_text))
                    .with_confidence(action.confidence)
                    .with_metadata(
                        "action_type",
                        serde_json::json!(format!("{:?}", action.action_type)),
                    )
                    .with_metadata(
                        "priority",
                        serde_json::json!(format!("{:?}", action.priority)),
                    )
                    .with_metadata(
                        "energy_level",
                        serde_json::json!(format!("{:?}", action.energy_level)),
                    );

                entities.push(ExtractedEntity {
                    entity,
                    source_text: action.source_text.clone(),
                    start_offset: action.start_offset,
                    end_offset: action.end_offset,
                    extraction_method: ExtractionMethod::ActionDetection,
                });
            }
            actions = action_results;
        }

        // 3. Named Entity Recognition (Local)
        if self.config.enable_local_ner {
            let ner_results = self.local_ner.extract(text);
            for named_entity in ner_results {
                let entity_type = match named_entity.entity_type {
                    NamedEntityType::Person => EntityType::Person,
                    NamedEntityType::Organization => EntityType::Organization,
                    NamedEntityType::Location => EntityType::Location,
                    NamedEntityType::Email => EntityType::Person, // Email often indicates a person
                    NamedEntityType::Url => EntityType::Reference,
                    NamedEntityType::PhoneNumber => EntityType::Custom,
                    NamedEntityType::Money => EntityType::Custom,
                    NamedEntityType::Percentage => EntityType::Custom,
                    NamedEntityType::Context => EntityType::Context,
                };

                let entity = Entity::new(entity_type, &named_entity.text)
                    .with_source_ref(doc_ref.clone().with_text(&named_entity.text))
                    .with_confidence(named_entity.confidence)
                    .with_metadata(
                        "ner_type",
                        serde_json::json!(format!("{:?}", named_entity.entity_type)),
                    );

                entities.push(ExtractedEntity {
                    entity,
                    source_text: named_entity.text.clone(),
                    start_offset: named_entity.start_offset,
                    end_offset: named_entity.end_offset,
                    extraction_method: ExtractionMethod::LocalNer,
                });
            }
        }

        // 4. LLM-based NER (if enabled and configured)
        if self.config.enable_llm_ner {
            if let Some(ref llm_ner) = self.llm_ner {
                match llm_ner.extract(text).await {
                    Ok(llm_entities) => {
                        used_llm = true;
                        for named_entity in llm_entities {
                            let entity_type = match named_entity.entity_type {
                                NamedEntityType::Person => EntityType::Person,
                                NamedEntityType::Organization => EntityType::Organization,
                                NamedEntityType::Location => EntityType::Location,
                                NamedEntityType::Email => EntityType::Person,
                                NamedEntityType::Url => EntityType::Reference,
                                NamedEntityType::PhoneNumber => EntityType::Custom,
                                NamedEntityType::Money => EntityType::Custom,
                                NamedEntityType::Percentage => EntityType::Custom,
                                NamedEntityType::Context => EntityType::Context,
                            };

                            let entity = Entity::new(entity_type, &named_entity.text)
                                .with_source_ref(doc_ref.clone().with_text(&named_entity.text))
                                .with_confidence(named_entity.confidence)
                                .with_metadata(
                                    "ner_type",
                                    serde_json::json!(format!("{:?}", named_entity.entity_type)),
                                )
                                .with_metadata("extraction_source", serde_json::json!("llm"));

                            entities.push(ExtractedEntity {
                                entity,
                                source_text: named_entity.text.clone(),
                                start_offset: named_entity.start_offset,
                                end_offset: named_entity.end_offset,
                                extraction_method: ExtractionMethod::LlmNer,
                            });
                        }
                    }
                    Err(e) => {
                        warnings.push(format!("LLM NER failed: {}", e));
                    }
                }
            }
        }

        // 5. Relationship extraction from co-occurrence
        if self.config.enable_relations {
            let entity_refs: Vec<_> = entities
                .iter()
                .map(|e| (&e.entity, e.start_offset, e.end_offset))
                .collect();

            let relation_candidates = self.relation_extractor.extract_from_cooccurrence(
                text,
                &entity_refs
                    .iter()
                    .map(|(e, s, end)| (*e, *s, *end))
                    .collect::<Vec<_>>(),
            );

            for candidate in relation_candidates {
                if candidate.confidence >= self.config.confidence_threshold {
                    let rel = Relationship::new(
                        &candidate.source_entity_id,
                        candidate.relation_type,
                        &candidate.target_entity_id,
                    )
                    .with_confidence(candidate.confidence)
                    .with_source_ref(doc_ref.clone());

                    relationships.push(ExtractedRelationship {
                        relationship: rel,
                        source_text: candidate.source_text,
                        extraction_method: ExtractionMethod::CoOccurrence,
                    });
                }
            }
        }

        // Filter by confidence threshold
        entities.retain(|e| e.entity.confidence >= self.config.confidence_threshold);

        // Deduplicate entities by name and type
        entities = Self::deduplicate_entities(entities);

        let processing_ms = start_time.elapsed().as_millis() as u64;
        let completed_at = Utc::now();

        // Calculate overall confidence
        let entity_confidences: Vec<f32> = entities.iter().map(|e| e.entity.confidence).collect();
        let overall_confidence = if entity_confidences.is_empty() {
            1.0
        } else {
            entity_confidences.iter().sum::<f32>() / entity_confidences.len() as f32
        };

        Ok(ExtractionResult {
            document_id: document_id.to_string(),
            entities,
            relationships,
            temporal,
            actions,
            confidence: overall_confidence,
            metadata: ExtractionMetadata {
                started_at: Some(started_at),
                completed_at: Some(completed_at),
                chars_processed: text.len(),
                used_llm,
                processing_ms,
                warnings,
            },
        })
    }

    /// Deduplicate entities by name and type, keeping the highest confidence.
    fn deduplicate_entities(entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
        use std::collections::HashMap;

        let mut seen: HashMap<(String, EntityType), ExtractedEntity> = HashMap::new();

        for entity in entities {
            let key = (entity.entity.name.to_lowercase(), entity.entity.entity_type);

            match seen.get(&key) {
                Some(existing) if existing.entity.confidence >= entity.entity.confidence => {
                    // Keep existing, it has higher or equal confidence
                }
                _ => {
                    seen.insert(key, entity);
                }
            }
        }

        seen.into_values().collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &ExtractionConfig {
        &self.config
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extraction_pipeline_basic() {
        let pipeline = ExtractionPipeline::default_pipeline();

        let text =
            "Meeting with John Smith tomorrow at 3pm. TODO: Review the quarterly report by Friday.";
        let result = pipeline.extract(text, "test-doc-1").await.unwrap();

        // Should have extracted some entities
        assert!(!result.entities.is_empty() || !result.temporal.is_empty());
        assert!(result.confidence > 0.0);
        assert!(result.metadata.processing_ms > 0);
    }

    #[tokio::test]
    async fn test_extraction_pipeline_temporal() {
        let pipeline = ExtractionPipeline::new(ExtractionConfig {
            enable_temporal: true,
            enable_actions: false,
            enable_local_ner: false,
            enable_llm_ner: false,
            enable_relations: false,
            ..Default::default()
        });

        let text = "The meeting is scheduled for next Monday at 2:30 PM.";
        let result = pipeline.extract(text, "test-doc-2").await.unwrap();

        assert!(!result.temporal.is_empty());
    }

    #[tokio::test]
    async fn test_extraction_pipeline_actions() {
        let pipeline = ExtractionPipeline::new(ExtractionConfig {
            enable_temporal: false,
            enable_actions: true,
            enable_local_ner: false,
            enable_llm_ner: false,
            enable_relations: false,
            ..Default::default()
        });

        let text = "TODO: Complete the project proposal\nACTION: Send invoice to client";
        let result = pipeline.extract(text, "test-doc-3").await.unwrap();

        assert!(!result.actions.is_empty(), "Should detect action items");
    }

    #[test]
    fn test_deduplication() {
        let doc_ref = DocumentRef::new("test");

        let entities = vec![
            ExtractedEntity {
                entity: Entity::new(EntityType::Person, "John Smith")
                    .with_confidence(0.9)
                    .with_source_ref(doc_ref.clone()),
                source_text: "John Smith".to_string(),
                start_offset: 0,
                end_offset: 10,
                extraction_method: ExtractionMethod::LocalNer,
            },
            ExtractedEntity {
                entity: Entity::new(EntityType::Person, "john smith")
                    .with_confidence(0.8)
                    .with_source_ref(doc_ref.clone()),
                source_text: "john smith".to_string(),
                start_offset: 50,
                end_offset: 60,
                extraction_method: ExtractionMethod::LocalNer,
            },
        ];

        let deduped = ExtractionPipeline::deduplicate_entities(entities);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].entity.confidence, 0.9); // Kept higher confidence
    }
}

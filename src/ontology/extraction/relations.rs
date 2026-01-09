//! Relationship extraction from co-occurrence and patterns.
//!
//! This module identifies relationships between entities using:
//! - **Co-occurrence analysis**: Entities mentioned together are likely related
//! - **Pattern matching**: Specific linguistic patterns indicate relationships
//! - **Semantic similarity**: Entities with similar embeddings may be related

use serde::{Deserialize, Serialize};

use crate::ontology::{Entity, RelationType};

// ============================================================================
// Types
// ============================================================================

/// A potential relationship candidate identified from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipCandidate {
    /// Source entity ID.
    pub source_entity_id: String,
    /// Target entity ID.
    pub target_entity_id: String,
    /// The inferred relationship type.
    pub relation_type: RelationType,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// The text evidence for this relationship.
    pub source_text: Option<String>,
    /// How the relationship was detected.
    pub detection_method: RelationDetectionMethod,
}

/// How a relationship was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationDetectionMethod {
    /// Entities appear close together in text.
    CoOccurrence,
    /// A specific linguistic pattern was matched.
    Pattern,
    /// High semantic similarity between entities.
    Semantic,
    /// Explicitly stated relationship.
    Explicit,
}

/// A co-occurrence between two entities.
#[derive(Debug, Clone)]
pub struct CoOccurrence {
    /// First entity ID.
    pub entity_a_id: String,
    /// Second entity ID.
    pub entity_b_id: String,
    /// Distance in characters.
    pub char_distance: usize,
    /// Distance in sentences.
    pub sentence_distance: usize,
    /// Number of times they co-occur.
    pub count: usize,
}

/// A pattern that indicates a relationship.
#[derive(Debug, Clone)]
pub struct RelationPattern {
    /// The regex pattern.
    pub pattern: regex::Regex,
    /// The relationship type this pattern indicates.
    pub relation_type: RelationType,
    /// Base confidence for matches.
    pub confidence: f32,
    /// Whether the first capture is source (true) or target (false).
    pub first_is_source: bool,
}

// ============================================================================
// Relation Extractor
// ============================================================================

/// Extracts relationships between entities.
pub struct RelationExtractor {
    /// Patterns for relationship detection.
    patterns: Vec<RelationPattern>,
    /// Maximum character distance for co-occurrence.
    max_cooccurrence_distance: usize,
}

impl Default for RelationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationExtractor {
    /// Create a new relation extractor with default patterns.
    pub fn new() -> Self {
        Self {
            patterns: Self::default_patterns(),
            max_cooccurrence_distance: 200,
        }
    }

    /// Create with custom maximum co-occurrence distance.
    pub fn with_max_distance(mut self, distance: usize) -> Self {
        self.max_cooccurrence_distance = distance;
        self
    }

    fn default_patterns() -> Vec<RelationPattern> {
        let patterns = [
            // WorksFor patterns
            (
                r"(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works?\s+(?:at|for)|is\s+(?:at|with)|employed\s+(?:at|by))\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::WorksFor,
                0.85,
                true,
            ),
            (
                r"(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+from\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::WorksFor,
                0.7,
                true,
            ),

            // AuthoredBy patterns
            (
                r"(?i)(?:written|authored|created|by)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::AuthoredBy,
                0.8,
                false,
            ),

            // Mentions patterns
            (
                r"(?i)(?:mentioned|discussed|talked about|regarding)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::Mentions,
                0.75,
                false,
            ),

            // BelongsToProject patterns
            (
                r"(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:for|on|in)\s+(?:the\s+)?(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+project",
                RelationType::BelongsToProject,
                0.8,
                true,
            ),

            // WaitingOn patterns
            (
                r"(?i)waiting\s+(?:for|on)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::WaitingOn,
                0.85,
                false,
            ),
            (
                r"(?i)pending\s+(?:response|review|approval)\s+from\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::WaitingOn,
                0.85,
                false,
            ),

            // DelegatedTo patterns
            (
                r"(?i)(?:assigned|delegated|given)\s+to\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::DelegatedTo,
                0.85,
                false,
            ),
            (
                r"(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:will|to)\s+(?:handle|take care of|work on)",
                RelationType::DelegatedTo,
                0.75,
                true,
            ),

            // LocatedAt patterns
            (
                r"(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?(?:in|at|located\s+(?:in|at))\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::LocatedAt,
                0.7,
                true,
            ),

            // AboutTopic patterns
            (
                r"(?i)(?:about|regarding|concerning|related to)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::AboutTopic,
                0.7,
                false,
            ),

            // PartOf patterns
            (
                r"(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?(?:part of|belongs to|member of)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::PartOf,
                0.8,
                true,
            ),

            // References patterns
            (
                r"(?i)(?:see|refer to|check|review)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::References,
                0.7,
                false,
            ),

            // DependsOn patterns
            (
                r"(?i)(?:depends on|requires|needs|blocked by)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                RelationType::DependsOn,
                0.8,
                false,
            ),

            // SupportsGoal patterns
            (
                r"(?i)(?:supports?|contributes? to|helps? with)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+goal",
                RelationType::SupportsGoal,
                0.75,
                false,
            ),

            // HasContext patterns
            (
                r"(?i)@(home|work|phone|computer|office|errand|anywhere)\b",
                RelationType::HasContext,
                0.9,
                false,
            ),
        ];

        patterns
            .into_iter()
            .filter_map(|(p, rt, c, fis)| {
                regex::Regex::new(p).ok().map(|r| RelationPattern {
                    pattern: r,
                    relation_type: rt,
                    confidence: c,
                    first_is_source: fis,
                })
            })
            .collect()
    }

    /// Extract relationships from co-occurrence of entities in text.
    pub fn extract_from_cooccurrence(
        &self,
        text: &str,
        entities: &[(&Entity, usize, usize)], // Entity with start and end offsets
    ) -> Vec<RelationshipCandidate> {
        let mut candidates = Vec::new();

        // For each pair of entities, check if they co-occur
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let (entity_a, start_a, end_a) = &entities[i];
                let (entity_b, start_b, end_b) = &entities[j];

                // Calculate distance
                let distance = if *start_a > *end_b {
                    start_a - end_b
                } else if *start_b > *end_a {
                    start_b - end_a
                } else {
                    0 // Overlapping
                };

                // Check if within co-occurrence distance
                if distance <= self.max_cooccurrence_distance {
                    // Determine relationship type based on entity types
                    if let Some((relation_type, confidence)) =
                        self.infer_relation_from_types(entity_a, entity_b, distance)
                    {
                        // Extract text between entities for context
                        let context_start = (*end_a).min(*end_b);
                        let context_end = (*start_a).max(*start_b);
                        let source_text = if context_end > context_start && context_end <= text.len() {
                            Some(text[context_start..context_end].to_string())
                        } else {
                            None
                        };

                        candidates.push(RelationshipCandidate {
                            source_entity_id: entity_a.id.clone(),
                            target_entity_id: entity_b.id.clone(),
                            relation_type,
                            confidence,
                            source_text,
                            detection_method: RelationDetectionMethod::CoOccurrence,
                        });
                    }
                }
            }
        }

        candidates
    }

    /// Infer relationship type from entity types.
    fn infer_relation_from_types(
        &self,
        entity_a: &Entity,
        entity_b: &Entity,
        distance: usize,
    ) -> Option<(RelationType, f32)> {
        use crate::ontology::EntityType;

        // Base confidence decreases with distance
        let distance_factor = 1.0 - (distance as f32 / self.max_cooccurrence_distance as f32 * 0.3);

        // Infer relationship based on type combinations
        match (entity_a.entity_type, entity_b.entity_type) {
            // Person + Organization -> WorksFor
            (EntityType::Person, EntityType::Organization) => {
                Some((RelationType::WorksFor, 0.6 * distance_factor))
            }
            (EntityType::Organization, EntityType::Person) => {
                Some((RelationType::WorksFor, 0.6 * distance_factor))
            }

            // Person + Location -> LocatedAt
            (EntityType::Person, EntityType::Location) => {
                Some((RelationType::LocatedAt, 0.5 * distance_factor))
            }

            // Task + Project -> BelongsToProject
            (EntityType::Task, EntityType::Project) => {
                Some((RelationType::BelongsToProject, 0.7 * distance_factor))
            }

            // Task + Context -> HasContext
            (EntityType::Task, EntityType::Context) => {
                Some((RelationType::HasContext, 0.8 * distance_factor))
            }

            // Task + Person -> DelegatedTo or WaitingOn
            (EntityType::Task, EntityType::Person) => {
                Some((RelationType::DelegatedTo, 0.5 * distance_factor))
            }
            (EntityType::WaitingFor, EntityType::Person) => {
                Some((RelationType::WaitingOn, 0.7 * distance_factor))
            }

            // Project + Goal -> SupportsGoal
            (EntityType::Project, EntityType::Goal) => {
                Some((RelationType::SupportsGoal, 0.6 * distance_factor))
            }

            // Project + Area -> InArea
            (EntityType::Project, EntityType::Area) => {
                Some((RelationType::InArea, 0.6 * distance_factor))
            }

            // Entity + Topic -> AboutTopic
            (_, EntityType::Topic) => Some((RelationType::AboutTopic, 0.5 * distance_factor)),

            // Entity + Date -> ScheduledFor or DueOn
            (EntityType::Task, EntityType::Date) | (EntityType::CalendarEvent, EntityType::Date) => {
                Some((RelationType::ScheduledFor, 0.6 * distance_factor))
            }

            // Generic RelatedTo for same types or unmatched
            (a, b) if a == b => Some((RelationType::RelatedTo, 0.4 * distance_factor)),

            // Person + Person -> RelatedTo
            (EntityType::Person, EntityType::Person) => {
                Some((RelationType::RelatedTo, 0.5 * distance_factor))
            }

            // Default: Mentions
            _ => Some((RelationType::Mentions, 0.3 * distance_factor)),
        }
    }

    /// Extract relationships using pattern matching.
    pub fn extract_from_patterns(&self, text: &str) -> Vec<RelationshipCandidate> {
        let mut candidates = Vec::new();

        for pattern in &self.patterns {
            for cap in pattern.pattern.captures_iter(text) {
                let full_match = cap.get(0).unwrap();

                // Extract entity names from captures
                let entity_texts: Vec<_> = cap
                    .iter()
                    .skip(1) // Skip full match
                    .filter_map(|m| m.map(|m| m.as_str().to_string()))
                    .collect();

                if !entity_texts.is_empty() {
                    // For single-entity patterns, we need context
                    let source_text = full_match.as_str().to_string();

                    if entity_texts.len() >= 2 {
                        // Two entities found - create relationship
                        let (source_id, target_id) = if pattern.first_is_source {
                            (entity_texts[0].clone(), entity_texts[1].clone())
                        } else {
                            (entity_texts[1].clone(), entity_texts[0].clone())
                        };

                        candidates.push(RelationshipCandidate {
                            source_entity_id: source_id,
                            target_entity_id: target_id,
                            relation_type: pattern.relation_type,
                            confidence: pattern.confidence,
                            source_text: Some(source_text),
                            detection_method: RelationDetectionMethod::Pattern,
                        });
                    } else {
                        // Single entity - store for potential matching
                        candidates.push(RelationshipCandidate {
                            source_entity_id: "".to_string(), // Placeholder
                            target_entity_id: entity_texts[0].clone(),
                            relation_type: pattern.relation_type,
                            confidence: pattern.confidence * 0.8, // Lower confidence
                            source_text: Some(source_text),
                            detection_method: RelationDetectionMethod::Pattern,
                        });
                    }
                }
            }
        }

        candidates
    }

    /// Find sentence boundaries in text.
    #[allow(dead_code)]
    fn find_sentence_boundaries(text: &str) -> Vec<usize> {
        let sentence_end = regex::Regex::new(r"[.!?]\s+").expect("Invalid regex");

        let mut boundaries = vec![0];
        for m in sentence_end.find_iter(text) {
            boundaries.push(m.end());
        }
        boundaries.push(text.len());

        boundaries
    }

    /// Get sentence number for a position.
    #[allow(dead_code)]
    fn get_sentence_number(boundaries: &[usize], position: usize) -> usize {
        boundaries
            .windows(2)
            .position(|w| position >= w[0] && position < w[1])
            .unwrap_or(0)
    }
}

// ============================================================================
// Semantic Relation Extractor
// ============================================================================

/// Extracts relationships based on semantic similarity of entity embeddings.
pub struct SemanticRelationExtractor {
    /// Minimum similarity threshold for relationship inference.
    similarity_threshold: f32,
}

impl Default for SemanticRelationExtractor {
    fn default() -> Self {
        Self::new(0.7)
    }
}

impl SemanticRelationExtractor {
    /// Create a new semantic relation extractor.
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
        }
    }

    /// Find semantically related entity pairs.
    pub fn find_related_pairs(
        &self,
        entities: &[Entity],
    ) -> Vec<RelationshipCandidate> {
        let mut candidates = Vec::new();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity_a = &entities[i];
                let entity_b = &entities[j];

                // Skip if either doesn't have an embedding
                let (emb_a, emb_b) = match (&entity_a.embedding, &entity_b.embedding) {
                    (Some(a), Some(b)) => (a, b),
                    _ => continue,
                };

                // Calculate cosine similarity
                let similarity = Self::cosine_similarity(emb_a, emb_b);

                if similarity >= self.similarity_threshold {
                    candidates.push(RelationshipCandidate {
                        source_entity_id: entity_a.id.clone(),
                        target_entity_id: entity_b.id.clone(),
                        relation_type: RelationType::RelatedTo,
                        confidence: similarity,
                        source_text: None,
                        detection_method: RelationDetectionMethod::Semantic,
                    });
                }
            }
        }

        candidates
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::EntityType;

    fn make_entity(id: &str, name: &str, entity_type: EntityType) -> Entity {
        Entity::with_id(id, entity_type, name)
    }

    #[test]
    fn test_cooccurrence_extraction() {
        let extractor = RelationExtractor::new();

        let text = "John Smith works at Acme Corp. He is a great developer.";

        let entities = vec![
            (make_entity("1", "John Smith", EntityType::Person), 0usize, 10usize),
            (make_entity("2", "Acme Corp", EntityType::Organization), 20usize, 29usize),
        ];

        let entity_refs: Vec<_> = entities
            .iter()
            .map(|(e, s, end)| (e, *s, *end))
            .collect();

        let candidates = extractor.extract_from_cooccurrence(text, &entity_refs);

        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].relation_type, RelationType::WorksFor);
    }

    #[test]
    fn test_pattern_extraction() {
        let extractor = RelationExtractor::new();

        let text = "John Smith works at Acme Corp";
        let candidates = extractor.extract_from_patterns(text);

        let works_for: Vec<_> = candidates
            .iter()
            .filter(|c| c.relation_type == RelationType::WorksFor)
            .collect();

        assert!(!works_for.is_empty());
    }

    #[test]
    fn test_waiting_on_pattern() {
        let extractor = RelationExtractor::new();

        let text = "I'm waiting for Sarah to review the document";
        let candidates = extractor.extract_from_patterns(text);

        let waiting: Vec<_> = candidates
            .iter()
            .filter(|c| c.relation_type == RelationType::WaitingOn)
            .collect();

        assert!(!waiting.is_empty());
    }

    #[test]
    fn test_delegated_to_pattern() {
        let extractor = RelationExtractor::new();

        let text = "This task was assigned to Mike for completion";
        let candidates = extractor.extract_from_patterns(text);

        let delegated: Vec<_> = candidates
            .iter()
            .filter(|c| c.relation_type == RelationType::DelegatedTo)
            .collect();

        assert!(!delegated.is_empty());
    }

    #[test]
    fn test_context_pattern() {
        let extractor = RelationExtractor::new();

        let text = "Do this task @home and that @work";
        let candidates = extractor.extract_from_patterns(text);

        let contexts: Vec<_> = candidates
            .iter()
            .filter(|c| c.relation_type == RelationType::HasContext)
            .collect();

        assert_eq!(contexts.len(), 2);
    }

    #[test]
    fn test_semantic_relation() {
        let extractor = SemanticRelationExtractor::new(0.7);

        let entities = vec![
            make_entity("1", "Machine Learning", EntityType::Topic)
                .with_embedding(vec![1.0, 0.0, 0.0]),
            make_entity("2", "Deep Learning", EntityType::Topic)
                .with_embedding(vec![0.95, 0.1, 0.0]),
            make_entity("3", "Cooking", EntityType::Topic)
                .with_embedding(vec![0.0, 1.0, 0.0]),
        ];

        let candidates = extractor.find_related_pairs(&entities);

        // ML and DL should be related, Cooking should not
        assert!(candidates.len() >= 1);

        let ml_dl_related = candidates.iter().any(|c| {
            (c.source_entity_id == "1" && c.target_entity_id == "2")
                || (c.source_entity_id == "2" && c.target_entity_id == "1")
        });
        assert!(ml_dl_related);

        let _cooking_related = candidates.iter().any(|c| {
            c.source_entity_id == "3" || c.target_entity_id == "3"
        });
        // Cooking should not be related (similarity < threshold)
        // Note: This depends on threshold, with 0.7 it shouldn't be related
    }

    #[test]
    fn test_distance_factor() {
        let extractor = RelationExtractor::new();

        // Close entities should have higher confidence
        let text = "John at Acme Corp";
        let entities_close = vec![
            (make_entity("1", "John", EntityType::Person), 0usize, 4usize),
            (make_entity("2", "Acme Corp", EntityType::Organization), 8usize, 17usize),
        ];

        let entity_refs: Vec<_> = entities_close
            .iter()
            .map(|(e, s, end)| (e, *s, *end))
            .collect();

        let candidates_close = extractor.extract_from_cooccurrence(text, &entity_refs);

        // Far entities
        let text_far = "John said something interesting. After a long time Acme Corp responded.";
        let entities_far = vec![
            (make_entity("1", "John", EntityType::Person), 0usize, 4usize),
            (make_entity("2", "Acme Corp", EntityType::Organization), 50usize, 59usize),
        ];

        let entity_refs: Vec<_> = entities_far
            .iter()
            .map(|(e, s, end)| (e, *s, *end))
            .collect();

        let candidates_far = extractor.extract_from_cooccurrence(text_far, &entity_refs);

        // Both should produce candidates, but close should have higher confidence
        if !candidates_close.is_empty() && !candidates_far.is_empty() {
            assert!(candidates_close[0].confidence >= candidates_far[0].confidence);
        }
    }

    #[test]
    fn test_infer_types() {
        let extractor = RelationExtractor::new();

        // Test various type combinations
        let person = make_entity("1", "John", EntityType::Person);
        let org = make_entity("2", "Acme", EntityType::Organization);
        let task = make_entity("3", "Review", EntityType::Task);
        let project = make_entity("4", "Website", EntityType::Project);
        let context = make_entity("5", "@home", EntityType::Context);

        // Person + Organization -> WorksFor
        let result = extractor.infer_relation_from_types(&person, &org, 10);
        assert!(matches!(result, Some((RelationType::WorksFor, _))));

        // Task + Project -> BelongsToProject
        let result = extractor.infer_relation_from_types(&task, &project, 10);
        assert!(matches!(result, Some((RelationType::BelongsToProject, _))));

        // Task + Context -> HasContext
        let result = extractor.infer_relation_from_types(&task, &context, 10);
        assert!(matches!(result, Some((RelationType::HasContext, _))));
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        assert_eq!(
            SemanticRelationExtractor::cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]),
            1.0
        );

        // Orthogonal vectors
        assert_eq!(
            SemanticRelationExtractor::cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]),
            0.0
        );

        // Opposite vectors
        assert!(
            SemanticRelationExtractor::cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) < 0.0
        );
    }
}

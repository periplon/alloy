//! Types for the Knowledge Graph query system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ontology::{Entity, EntityType, RelationType, Relationship};

// ============================================================================
// Query Types
// ============================================================================

/// Type of knowledge query to perform.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeQueryType {
    /// Semantic search: "What do I know about X?"
    #[default]
    SemanticSearch,
    /// Find specific entity by name or ID.
    EntityLookup,
    /// Query relationships: "Who works with person X?"
    RelationshipQuery,
    /// Summarize knowledge on a topic.
    TopicSummary,
    /// Get connected entities via graph traversal.
    ConnectedEntities,
    /// Find experts on a topic.
    ExpertFinding,
}

/// Parameters for knowledge graph queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeQueryParams {
    /// Natural language query or search term.
    pub query: String,
    /// Type of query to perform.
    #[serde(default)]
    pub query_type: KnowledgeQueryType,
    /// Filter by entity types.
    #[serde(default)]
    pub entity_types: Vec<EntityType>,
    /// Filter by relationship types (for relationship queries).
    #[serde(default)]
    pub relationship_types: Vec<RelationType>,
    /// Maximum depth for graph traversal.
    #[serde(default)]
    pub max_depth: Option<usize>,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum confidence threshold.
    #[serde(default)]
    pub min_confidence: Option<f32>,
    /// Starting entity ID (for traversal queries).
    #[serde(default)]
    pub from_entity_id: Option<String>,
    /// Include source document references.
    #[serde(default = "default_true")]
    pub include_sources: bool,
}

fn default_limit() -> usize {
    20
}

fn default_true() -> bool {
    true
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of a knowledge query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeQueryResult {
    /// Matched entities with relevance scores.
    pub entities: Vec<ScoredEntity>,
    /// Related relationships found.
    pub relationships: Vec<Relationship>,
    /// Source document summaries.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source_documents: Vec<DocumentSummary>,
    /// Generated summary (for topic queries).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Overall confidence of the results.
    pub confidence: f32,
    /// Query execution statistics.
    pub stats: QueryStats,
}

impl Default for KnowledgeQueryResult {
    fn default() -> Self {
        Self {
            entities: Vec::new(),
            relationships: Vec::new(),
            source_documents: Vec::new(),
            summary: None,
            confidence: 0.0,
            stats: QueryStats::default(),
        }
    }
}

/// An entity with a relevance/similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredEntity {
    /// The entity.
    pub entity: Entity,
    /// Relevance score (0.0 to 1.0).
    pub score: f32,
    /// Why this entity matched.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub match_reasons: Vec<String>,
}

impl ScoredEntity {
    pub fn new(entity: Entity, score: f32) -> Self {
        Self {
            entity,
            score,
            match_reasons: Vec::new(),
        }
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.match_reasons.push(reason.into());
        self
    }

    pub fn with_reasons(mut self, reasons: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.match_reasons
            .extend(reasons.into_iter().map(|r| r.into()));
        self
    }
}

/// Summary of a source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    /// Document ID.
    pub document_id: String,
    /// Document path or title.
    pub path: String,
    /// Snippet of relevant content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// Relevance score.
    pub relevance: f32,
}

/// Statistics about query execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Number of entities searched.
    pub entities_searched: usize,
    /// Number of relationships traversed.
    pub relationships_traversed: usize,
    /// Query execution time in milliseconds.
    pub execution_time_ms: u64,
    /// Whether embedding search was used.
    pub used_embedding_search: bool,
    /// Graph traversal depth reached.
    pub max_depth_reached: usize,
}

// ============================================================================
// Expert Finding Types
// ============================================================================

/// An expert on a topic with expertise score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expert {
    /// The person entity.
    pub person: Entity,
    /// Expertise score (0.0 to 1.0).
    pub expertise_score: f32,
    /// Topics they are expert in.
    pub topics: Vec<TopicExpertise>,
    /// Evidence for their expertise.
    pub evidence: Vec<ExpertiseEvidence>,
}

/// Expertise in a specific topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicExpertise {
    /// Topic name or entity ID.
    pub topic: String,
    /// Score for this topic (0.0 to 1.0).
    pub score: f32,
    /// Number of mentions/references.
    pub mention_count: usize,
}

/// Evidence supporting expertise claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertiseEvidence {
    /// Type of evidence.
    pub evidence_type: EvidenceType,
    /// Description of the evidence.
    pub description: String,
    /// Source document ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document: Option<String>,
    /// Weight of this evidence.
    pub weight: f32,
}

/// Types of expertise evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceType {
    /// Authored content on the topic.
    AuthoredContent,
    /// Mentioned in relation to topic.
    TopicMention,
    /// Part of project related to topic.
    ProjectInvolvement,
    /// Has relationship with topic experts.
    ExpertConnection,
    /// Semantic similarity to topic.
    SemanticSimilarity,
}

// ============================================================================
// Topic Summary Types
// ============================================================================

/// Summary of knowledge about a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicSummary {
    /// Topic name.
    pub topic: String,
    /// Generated summary text.
    pub summary: String,
    /// Key entities related to this topic.
    pub key_entities: Vec<ScoredEntity>,
    /// Sub-topics discovered.
    pub sub_topics: Vec<String>,
    /// Related topics.
    pub related_topics: Vec<RelatedTopic>,
    /// Knowledge gaps (areas with little information).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub knowledge_gaps: Vec<String>,
    /// Statistics about the topic.
    pub stats: TopicStats,
}

/// A related topic with connection strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedTopic {
    /// Topic name.
    pub topic: String,
    /// Connection strength (0.0 to 1.0).
    pub strength: f32,
    /// Type of relationship.
    pub relationship: String,
}

/// Statistics about a topic.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopicStats {
    /// Number of entities related to topic.
    pub entity_count: usize,
    /// Number of documents mentioning topic.
    pub document_count: usize,
    /// Number of relationships involving topic.
    pub relationship_count: usize,
    /// Average confidence of topic entities.
    pub average_confidence: f32,
    /// Most common entity types.
    pub entity_type_distribution: HashMap<String, usize>,
}

// ============================================================================
// Relationship Traversal Types
// ============================================================================

/// Result of relationship traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    /// Starting entity.
    pub start_entity: Entity,
    /// Discovered paths.
    pub paths: Vec<RelationshipPath>,
    /// All entities discovered during traversal.
    pub discovered_entities: Vec<ScoredEntity>,
    /// Statistics.
    pub stats: TraversalStats,
}

/// A path through relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipPath {
    /// Sequence of relationships in the path.
    pub relationships: Vec<Relationship>,
    /// Entities along the path (including start and end).
    pub entities: Vec<Entity>,
    /// Total path weight/strength.
    pub path_strength: f32,
}

/// Statistics about traversal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraversalStats {
    /// Number of nodes visited.
    pub nodes_visited: usize,
    /// Number of edges traversed.
    pub edges_traversed: usize,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
}

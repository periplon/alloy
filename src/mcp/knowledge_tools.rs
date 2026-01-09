//! Knowledge Graph MCP tool implementations for Alloy.
//!
//! This module provides MCP tools for knowledge graph functionality:
//! - `knowledge_query`: Unified query interface for knowledge graph
//! - Supports semantic search, entity lookup, relationship queries
//! - Expert finding, topic summarization, graph traversal

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::knowledge::{
    Expert, KnowledgeQueryResult, KnowledgeQueryType, RelatedTopic, ScoredEntity, TopicSummary,
    TraversalResult,
};
use crate::ontology::{EntityType, RelationType};

// ============================================================================
// Knowledge Query Tool Types
// ============================================================================

/// Type of knowledge query to perform.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeQueryAction {
    /// Semantic search: "What do I know about X?"
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

impl From<KnowledgeQueryAction> for KnowledgeQueryType {
    fn from(action: KnowledgeQueryAction) -> Self {
        match action {
            KnowledgeQueryAction::SemanticSearch => KnowledgeQueryType::SemanticSearch,
            KnowledgeQueryAction::EntityLookup => KnowledgeQueryType::EntityLookup,
            KnowledgeQueryAction::RelationshipQuery => KnowledgeQueryType::RelationshipQuery,
            KnowledgeQueryAction::TopicSummary => KnowledgeQueryType::TopicSummary,
            KnowledgeQueryAction::ConnectedEntities => KnowledgeQueryType::ConnectedEntities,
            KnowledgeQueryAction::ExpertFinding => KnowledgeQueryType::ExpertFinding,
        }
    }
}

/// Entity type filter for knowledge queries.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EntityTypeFilter {
    Person,
    Organization,
    Location,
    Topic,
    Concept,
    Project,
    Task,
    All,
}

impl EntityTypeFilter {
    pub fn to_entity_types(&self) -> Vec<EntityType> {
        match self {
            EntityTypeFilter::Person => vec![EntityType::Person],
            EntityTypeFilter::Organization => vec![EntityType::Organization],
            EntityTypeFilter::Location => vec![EntityType::Location],
            EntityTypeFilter::Topic => vec![EntityType::Topic],
            EntityTypeFilter::Concept => vec![EntityType::Concept],
            EntityTypeFilter::Project => vec![EntityType::Project],
            EntityTypeFilter::Task => vec![EntityType::Task],
            EntityTypeFilter::All => Vec::new(),
        }
    }
}

/// Relationship type filter for knowledge queries.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelationTypeFilter {
    WorksFor,
    AuthoredBy,
    Mentions,
    RelatedTo,
    AboutTopic,
    BelongsToProject,
    DependsOn,
    All,
}

impl RelationTypeFilter {
    pub fn to_relation_types(&self) -> Vec<RelationType> {
        match self {
            RelationTypeFilter::WorksFor => vec![RelationType::WorksFor],
            RelationTypeFilter::AuthoredBy => vec![RelationType::AuthoredBy],
            RelationTypeFilter::Mentions => vec![RelationType::Mentions],
            RelationTypeFilter::RelatedTo => vec![RelationType::RelatedTo],
            RelationTypeFilter::AboutTopic => vec![RelationType::AboutTopic],
            RelationTypeFilter::BelongsToProject => vec![RelationType::BelongsToProject],
            RelationTypeFilter::DependsOn => vec![RelationType::DependsOn],
            RelationTypeFilter::All => Vec::new(),
        }
    }
}

/// Parameters for the knowledge_query tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct KnowledgeQueryParams {
    /// Natural language query or search term.
    pub query: String,
    /// Type of query to perform.
    #[serde(default = "default_query_type")]
    pub query_type: KnowledgeQueryAction,
    /// Filter by entity type.
    #[serde(default)]
    pub entity_type: Option<EntityTypeFilter>,
    /// Filter by relationship type (for relationship queries).
    #[serde(default)]
    pub relationship_type: Option<RelationTypeFilter>,
    /// Maximum depth for graph traversal (default: 2).
    #[serde(default)]
    pub max_depth: Option<usize>,
    /// Maximum number of results (default: 20).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Minimum confidence threshold (0.0 to 1.0).
    #[serde(default)]
    pub min_confidence: Option<f32>,
    /// Starting entity ID (for traversal queries).
    #[serde(default)]
    pub from_entity_id: Option<String>,
    /// Include source document references.
    #[serde(default = "default_true")]
    pub include_sources: bool,
}

fn default_query_type() -> KnowledgeQueryAction {
    KnowledgeQueryAction::SemanticSearch
}

fn default_true() -> bool {
    true
}

#[allow(clippy::derivable_impls)] // include_sources defaults to true, not false
impl Default for KnowledgeQueryParams {
    fn default() -> Self {
        Self {
            query: String::new(),
            query_type: KnowledgeQueryAction::SemanticSearch,
            entity_type: None,
            relationship_type: None,
            max_depth: None,
            limit: None,
            min_confidence: None,
            from_entity_id: None,
            include_sources: true,
        }
    }
}

impl From<KnowledgeQueryParams> for crate::knowledge::KnowledgeQueryParams {
    fn from(params: KnowledgeQueryParams) -> Self {
        Self {
            query: params.query,
            query_type: params.query_type.into(),
            entity_types: params
                .entity_type
                .map(|t| t.to_entity_types())
                .unwrap_or_default(),
            relationship_types: params
                .relationship_type
                .map(|t| t.to_relation_types())
                .unwrap_or_default(),
            max_depth: params.max_depth,
            limit: params.limit.unwrap_or(20),
            min_confidence: params.min_confidence,
            from_entity_id: params.from_entity_id,
            include_sources: params.include_sources,
        }
    }
}

/// Entity info for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInfo {
    /// Entity ID.
    pub id: String,
    /// Entity name.
    pub name: String,
    /// Entity type.
    pub entity_type: String,
    /// Relevance score.
    pub score: f32,
    /// Match reasons.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub match_reasons: Vec<String>,
    /// Entity aliases.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

impl From<ScoredEntity> for EntityInfo {
    fn from(scored: ScoredEntity) -> Self {
        Self {
            id: scored.entity.id,
            name: scored.entity.name,
            entity_type: format!("{:?}", scored.entity.entity_type),
            score: scored.score,
            match_reasons: scored.match_reasons,
            aliases: scored.entity.aliases,
        }
    }
}

/// Relationship info for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipInfo {
    /// Relationship ID.
    pub id: String,
    /// Source entity ID.
    pub source_id: String,
    /// Relationship type.
    pub relationship_type: String,
    /// Target entity ID.
    pub target_id: String,
    /// Confidence score.
    pub confidence: f32,
}

/// Source document info for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDocInfo {
    /// Document ID.
    pub document_id: String,
    /// Document path.
    pub path: String,
    /// Content snippet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// Relevance score.
    pub relevance: f32,
}

/// Query statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatsInfo {
    /// Entities searched.
    pub entities_searched: usize,
    /// Relationships traversed.
    pub relationships_traversed: usize,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
    /// Whether embedding search was used.
    pub used_embedding_search: bool,
}

/// Response for knowledge query operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeQueryResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Matched entities.
    pub entities: Vec<EntityInfo>,
    /// Related relationships.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub relationships: Vec<RelationshipInfo>,
    /// Source documents.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source_documents: Vec<SourceDocInfo>,
    /// Generated summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Overall confidence.
    pub confidence: f32,
    /// Query statistics.
    pub stats: QueryStatsInfo,
    /// Status message.
    pub message: String,
}

impl KnowledgeQueryResponse {
    pub fn from_result(result: KnowledgeQueryResult) -> Self {
        Self {
            success: true,
            entities: result.entities.into_iter().map(EntityInfo::from).collect(),
            relationships: result
                .relationships
                .into_iter()
                .map(|r| RelationshipInfo {
                    id: r.id,
                    source_id: r.source_entity_id,
                    relationship_type: format!("{:?}", r.relationship_type),
                    target_id: r.target_entity_id,
                    confidence: r.confidence,
                })
                .collect(),
            source_documents: result
                .source_documents
                .into_iter()
                .map(|d| SourceDocInfo {
                    document_id: d.document_id,
                    path: d.path,
                    snippet: d.snippet,
                    relevance: d.relevance,
                })
                .collect(),
            summary: result.summary,
            confidence: result.confidence,
            stats: QueryStatsInfo {
                entities_searched: result.stats.entities_searched,
                relationships_traversed: result.stats.relationships_traversed,
                execution_time_ms: result.stats.execution_time_ms,
                used_embedding_search: result.stats.used_embedding_search,
            },
            message: "Query executed successfully".to_string(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            entities: Vec::new(),
            relationships: Vec::new(),
            source_documents: Vec::new(),
            summary: None,
            confidence: 0.0,
            stats: QueryStatsInfo {
                entities_searched: 0,
                relationships_traversed: 0,
                execution_time_ms: 0,
                used_embedding_search: false,
            },
            message: message.into(),
        }
    }
}

// ============================================================================
// Expert Finding Tool Types
// ============================================================================

/// Parameters for the expert_find tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ExpertFindParams {
    /// Topic to find experts for.
    #[serde(default)]
    pub topic: String,
    /// Maximum number of experts to return (default: 10).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Minimum expertise score (0.0 to 1.0).
    #[serde(default)]
    pub min_score: Option<f32>,
}

/// Expert info for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertInfo {
    /// Person ID.
    pub id: String,
    /// Person name.
    pub name: String,
    /// Expertise score.
    pub expertise_score: f32,
    /// Topics they are expert in.
    pub topics: Vec<TopicExpertiseInfo>,
    /// Evidence for expertise.
    pub evidence: Vec<String>,
}

/// Topic expertise info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicExpertiseInfo {
    pub topic: String,
    pub score: f32,
    pub mention_count: usize,
}

impl From<Expert> for ExpertInfo {
    fn from(expert: Expert) -> Self {
        Self {
            id: expert.person.id,
            name: expert.person.name,
            expertise_score: expert.expertise_score,
            topics: expert
                .topics
                .into_iter()
                .map(|t| TopicExpertiseInfo {
                    topic: t.topic,
                    score: t.score,
                    mention_count: t.mention_count,
                })
                .collect(),
            evidence: expert.evidence.into_iter().map(|e| e.description).collect(),
        }
    }
}

/// Response for expert finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertFindResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Found experts.
    pub experts: Vec<ExpertInfo>,
    /// Topic searched.
    pub topic: String,
    /// Status message.
    pub message: String,
}

impl ExpertFindResponse {
    pub fn success(topic: String, experts: Vec<Expert>) -> Self {
        let expert_count = experts.len();
        Self {
            success: true,
            experts: experts.into_iter().map(ExpertInfo::from).collect(),
            topic: topic.clone(),
            message: format!("Found {} experts on '{}'", expert_count, topic),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            experts: Vec::new(),
            topic: String::new(),
            message: message.into(),
        }
    }
}

// ============================================================================
// Topic Summary Tool Types
// ============================================================================

/// Parameters for the topic_summarize tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TopicSummarizeParams {
    /// Topic to summarize.
    pub topic: String,
    /// Maximum number of entities to include (default: 20).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Include related topics.
    #[serde(default = "default_true")]
    pub include_related: bool,
}

#[allow(clippy::derivable_impls)] // include_related defaults to true, not false
impl Default for TopicSummarizeParams {
    fn default() -> Self {
        Self {
            topic: String::new(),
            limit: None,
            include_related: true,
        }
    }
}

/// Related topic info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedTopicInfo {
    pub topic: String,
    pub strength: f32,
    pub relationship: String,
}

impl From<RelatedTopic> for RelatedTopicInfo {
    fn from(related: RelatedTopic) -> Self {
        Self {
            topic: related.topic,
            strength: related.strength,
            relationship: related.relationship,
        }
    }
}

/// Response for topic summarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicSummarizeResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Topic name.
    pub topic: String,
    /// Generated summary.
    pub summary: String,
    /// Key entities.
    pub key_entities: Vec<EntityInfo>,
    /// Sub-topics discovered.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sub_topics: Vec<String>,
    /// Related topics.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub related_topics: Vec<RelatedTopicInfo>,
    /// Statistics.
    pub stats: TopicStatsInfo,
    /// Status message.
    pub message: String,
}

/// Topic statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicStatsInfo {
    pub entity_count: usize,
    pub document_count: usize,
    pub relationship_count: usize,
    pub average_confidence: f32,
}

impl TopicSummarizeResponse {
    pub fn from_summary(summary: TopicSummary) -> Self {
        Self {
            success: true,
            topic: summary.topic.clone(),
            summary: summary.summary,
            key_entities: summary
                .key_entities
                .into_iter()
                .map(EntityInfo::from)
                .collect(),
            sub_topics: summary.sub_topics,
            related_topics: summary
                .related_topics
                .into_iter()
                .map(RelatedTopicInfo::from)
                .collect(),
            stats: TopicStatsInfo {
                entity_count: summary.stats.entity_count,
                document_count: summary.stats.document_count,
                relationship_count: summary.stats.relationship_count,
                average_confidence: summary.stats.average_confidence,
            },
            message: format!("Topic '{}' summarized successfully", summary.topic),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            topic: String::new(),
            summary: String::new(),
            key_entities: Vec::new(),
            sub_topics: Vec::new(),
            related_topics: Vec::new(),
            stats: TopicStatsInfo {
                entity_count: 0,
                document_count: 0,
                relationship_count: 0,
                average_confidence: 0.0,
            },
            message: message.into(),
        }
    }
}

// ============================================================================
// Graph Traversal Tool Types
// ============================================================================

/// Parameters for the graph_traverse tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct GraphTraverseParams {
    /// Starting entity ID.
    #[serde(default)]
    pub from_entity_id: String,
    /// Relationship type filter.
    #[serde(default)]
    pub relationship_type: Option<RelationTypeFilter>,
    /// Maximum traversal depth (default: 2).
    #[serde(default)]
    pub max_depth: Option<usize>,
    /// Maximum number of results (default: 50).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Path info for graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathInfo {
    /// Entities in the path.
    pub entities: Vec<String>,
    /// Relationship types in the path.
    pub relationships: Vec<String>,
    /// Path strength.
    pub strength: f32,
}

/// Response for graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraverseResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Starting entity.
    pub start_entity: EntityInfo,
    /// Discovered entities.
    pub discovered_entities: Vec<EntityInfo>,
    /// Paths found.
    pub paths: Vec<PathInfo>,
    /// Traversal statistics.
    pub stats: TraversalStatsInfo,
    /// Status message.
    pub message: String,
}

/// Traversal statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalStatsInfo {
    pub nodes_visited: usize,
    pub edges_traversed: usize,
    pub max_depth_reached: usize,
    pub execution_time_ms: u64,
}

impl GraphTraverseResponse {
    pub fn from_result(result: TraversalResult) -> Self {
        let node_count = result.discovered_entities.len();
        let path_count = result.paths.len();

        Self {
            success: true,
            start_entity: EntityInfo {
                id: result.start_entity.id.clone(),
                name: result.start_entity.name.clone(),
                entity_type: format!("{:?}", result.start_entity.entity_type),
                score: 1.0,
                match_reasons: vec!["Starting entity".to_string()],
                aliases: result.start_entity.aliases.clone(),
            },
            discovered_entities: result
                .discovered_entities
                .into_iter()
                .map(EntityInfo::from)
                .collect(),
            paths: result
                .paths
                .into_iter()
                .map(|p| PathInfo {
                    entities: p.entities.iter().map(|e| e.name.clone()).collect(),
                    relationships: p
                        .relationships
                        .iter()
                        .map(|r| format!("{:?}", r.relationship_type))
                        .collect(),
                    strength: p.path_strength,
                })
                .collect(),
            stats: TraversalStatsInfo {
                nodes_visited: result.stats.nodes_visited,
                edges_traversed: result.stats.edges_traversed,
                max_depth_reached: result.stats.max_depth_reached,
                execution_time_ms: result.stats.execution_time_ms,
            },
            message: format!(
                "Traversal complete: {} nodes visited, {} paths found",
                node_count, path_count
            ),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            start_entity: EntityInfo {
                id: String::new(),
                name: String::new(),
                entity_type: String::new(),
                score: 0.0,
                match_reasons: Vec::new(),
                aliases: Vec::new(),
            },
            discovered_entities: Vec::new(),
            paths: Vec::new(),
            stats: TraversalStatsInfo {
                nodes_visited: 0,
                edges_traversed: 0,
                max_depth_reached: 0,
                execution_time_ms: 0,
            },
            message: message.into(),
        }
    }
}

// ============================================================================
// Natural Language Query Tool Types
// ============================================================================

/// Parameters for the unified query tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct NaturalQueryParams {
    /// Natural language query (e.g., "What are my @phone tasks?", "Who knows about Kubernetes?").
    #[serde(default)]
    pub query: String,
    /// Maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Response for natural language queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalQueryResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Interpreted query type.
    pub interpreted_as: String,
    /// Query result.
    pub result: serde_json::Value,
    /// Natural language answer.
    pub answer: String,
    /// Status message.
    pub message: String,
}

impl NaturalQueryResponse {
    pub fn success(interpreted_as: String, result: serde_json::Value, answer: String) -> Self {
        Self {
            success: true,
            interpreted_as,
            result,
            answer: answer.clone(),
            message: "Query processed successfully".to_string(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            interpreted_as: String::new(),
            result: serde_json::Value::Null,
            answer: String::new(),
            message: message.into(),
        }
    }
}

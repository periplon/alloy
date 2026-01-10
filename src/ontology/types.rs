//! Core types for the Ontology system.
//!
//! This module defines the entity and relationship types that form the
//! foundation of the semantic knowledge graph and GTD system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Entity Types
// ============================================================================

/// A semantic entity in the ontology.
///
/// Entities are typed nodes in the knowledge graph that can represent
/// people, projects, tasks, concepts, and other meaningful items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier for the entity.
    pub id: String,
    /// The type of entity.
    pub entity_type: EntityType,
    /// Primary name of the entity.
    pub name: String,
    /// Alternative names or aliases.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Semantic embedding vector for similarity search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// When the entity was created.
    pub created_at: DateTime<Utc>,
    /// When the entity was last updated.
    pub updated_at: DateTime<Utc>,
    /// References to source documents where this entity was found.
    #[serde(default)]
    pub source_refs: Vec<DocumentRef>,
    /// Additional metadata as key-value pairs.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Confidence score for extracted entities (0.0-1.0).
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    1.0
}

impl Entity {
    /// Create a new entity with the given type and name.
    pub fn new(entity_type: EntityType, name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            entity_type,
            name: name.into(),
            aliases: Vec::new(),
            embedding: None,
            created_at: now,
            updated_at: now,
            source_refs: Vec::new(),
            metadata: HashMap::new(),
            confidence: 1.0,
        }
    }

    /// Create a new entity with a specific ID.
    pub fn with_id(
        id: impl Into<String>,
        entity_type: EntityType,
        name: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            entity_type,
            name: name.into(),
            aliases: Vec::new(),
            embedding: None,
            created_at: now,
            updated_at: now,
            source_refs: Vec::new(),
            metadata: HashMap::new(),
            confidence: 1.0,
        }
    }

    /// Add an alias to the entity.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Add multiple aliases to the entity.
    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases.extend(aliases.into_iter().map(|a| a.into()));
        self
    }

    /// Set the embedding vector.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add a source document reference.
    pub fn with_source_ref(mut self, doc_ref: DocumentRef) -> Self {
        self.source_refs.push(doc_ref);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set the confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Check if this entity matches a query string (name or aliases).
    pub fn matches_query(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();
        self.name.to_lowercase().contains(&query_lower)
            || self
                .aliases
                .iter()
                .any(|a| a.to_lowercase().contains(&query_lower))
    }
}

/// The type classification of an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    // GTD Core Types
    /// A project requiring multiple actions to complete.
    Project,
    /// A single next action.
    Task,
    /// An item delegated to someone else.
    WaitingFor,
    /// A deferred item for later consideration.
    SomedayMaybe,
    /// Reference or knowledge material.
    Reference,
    /// A time-specific commitment on the calendar.
    CalendarEvent,

    // GTD Organizational Types
    /// A context for actions (e.g., @home, @work, @phone).
    Context,
    /// An area of focus or responsibility.
    Area,
    /// A 1-2 year goal.
    Goal,
    /// A 3-5 year vision.
    Vision,
    /// Life purpose or guiding principles.
    Purpose,

    // Knowledge Graph Types
    /// A person.
    Person,
    /// A company, team, or organization.
    Organization,
    /// A physical location.
    Location,
    /// A subject area or topic.
    Topic,
    /// An abstract concept or idea.
    Concept,
    /// A temporal reference (specific date/time).
    Date,
    /// A promise or commitment made/received.
    Commitment,

    // Generic
    /// A custom or unclassified entity type.
    Custom,
}

impl EntityType {
    /// Check if this is a GTD-related type.
    pub fn is_gtd_type(&self) -> bool {
        matches!(
            self,
            EntityType::Project
                | EntityType::Task
                | EntityType::WaitingFor
                | EntityType::SomedayMaybe
                | EntityType::Reference
                | EntityType::CalendarEvent
                | EntityType::Context
                | EntityType::Area
                | EntityType::Goal
                | EntityType::Vision
                | EntityType::Purpose
        )
    }

    /// Check if this is a knowledge graph type.
    pub fn is_knowledge_type(&self) -> bool {
        matches!(
            self,
            EntityType::Person
                | EntityType::Organization
                | EntityType::Location
                | EntityType::Topic
                | EntityType::Concept
                | EntityType::Date
                | EntityType::Commitment
        )
    }

    /// Get a human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            EntityType::Project => "Project",
            EntityType::Task => "Task",
            EntityType::WaitingFor => "Waiting For",
            EntityType::SomedayMaybe => "Someday/Maybe",
            EntityType::Reference => "Reference",
            EntityType::CalendarEvent => "Calendar Event",
            EntityType::Context => "Context",
            EntityType::Area => "Area",
            EntityType::Goal => "Goal",
            EntityType::Vision => "Vision",
            EntityType::Purpose => "Purpose",
            EntityType::Person => "Person",
            EntityType::Organization => "Organization",
            EntityType::Location => "Location",
            EntityType::Topic => "Topic",
            EntityType::Concept => "Concept",
            EntityType::Date => "Date",
            EntityType::Commitment => "Commitment",
            EntityType::Custom => "Custom",
        }
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ============================================================================
// Relationship Types
// ============================================================================

/// A relationship between two entities.
///
/// Relationships are directed edges in the knowledge graph connecting
/// a source entity to a target entity with a typed relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Unique identifier for the relationship.
    pub id: String,
    /// ID of the source entity.
    pub source_entity_id: String,
    /// Type of relationship.
    pub relationship_type: RelationType,
    /// ID of the target entity.
    pub target_entity_id: String,
    /// Confidence score for the relationship (0.0-1.0).
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// When the relationship was created.
    pub created_at: DateTime<Utc>,
    /// References to source documents where this relationship was found.
    #[serde(default)]
    pub source_refs: Vec<DocumentRef>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Relationship {
    /// Create a new relationship between two entities.
    pub fn new(
        source_entity_id: impl Into<String>,
        relationship_type: RelationType,
        target_entity_id: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            source_entity_id: source_entity_id.into(),
            relationship_type,
            target_entity_id: target_entity_id.into(),
            confidence: 1.0,
            created_at: Utc::now(),
            source_refs: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new relationship with a specific ID.
    pub fn with_id(
        id: impl Into<String>,
        source_entity_id: impl Into<String>,
        relationship_type: RelationType,
        target_entity_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            source_entity_id: source_entity_id.into(),
            relationship_type,
            target_entity_id: target_entity_id.into(),
            confidence: 1.0,
            created_at: Utc::now(),
            source_refs: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add a source document reference.
    pub fn with_source_ref(mut self, doc_ref: DocumentRef) -> Self {
        self.source_refs.push(doc_ref);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// The type of relationship between entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    // GTD Relationships
    /// Task belongs to a project.
    BelongsToProject,
    /// Task/action has a specific context.
    HasContext,
    /// Project is in an area of focus.
    InArea,
    /// Project supports a goal.
    SupportsGoal,
    /// Task is waiting on someone.
    WaitingOn,
    /// Task was delegated to someone.
    DelegatedTo,
    /// Task is blocked by another task.
    BlockedBy,
    /// Task depends on another task.
    DependsOn,
    /// Task/event is scheduled for a time.
    ScheduledFor,
    /// Task has a due date.
    DueOn,

    // Knowledge Graph Relationships
    /// Document/entity mentions another entity.
    Mentions,
    /// Entity is semantically related to another.
    RelatedTo,
    /// Document was authored by a person.
    AuthoredBy,
    /// Entity is about a topic.
    AboutTopic,
    /// Entity is located at a location.
    LocatedAt,
    /// Person works for an organization.
    WorksFor,
    /// Person committed to something.
    CommittedTo,
    /// Task/project references a document.
    References,
    /// Entity is part of another entity.
    PartOf,
    /// Entity contains another entity.
    Contains,
    /// Entities are synonyms or aliases.
    SameAs,

    // Generic
    /// A custom relationship type.
    Custom,
}

impl RelationType {
    /// Get the inverse relationship type, if applicable.
    pub fn inverse(&self) -> Option<RelationType> {
        match self {
            RelationType::BelongsToProject => Some(RelationType::Contains),
            RelationType::Contains => Some(RelationType::PartOf),
            RelationType::PartOf => Some(RelationType::Contains),
            RelationType::BlockedBy => Some(RelationType::BlockedBy), // Symmetric concept
            RelationType::DependsOn => Some(RelationType::DependsOn),
            RelationType::WorksFor => None, // No direct inverse
            RelationType::AuthoredBy => None,
            RelationType::SameAs => Some(RelationType::SameAs), // Symmetric
            _ => None,
        }
    }

    /// Check if this is a GTD relationship.
    pub fn is_gtd_relation(&self) -> bool {
        matches!(
            self,
            RelationType::BelongsToProject
                | RelationType::HasContext
                | RelationType::InArea
                | RelationType::SupportsGoal
                | RelationType::WaitingOn
                | RelationType::DelegatedTo
                | RelationType::BlockedBy
                | RelationType::DependsOn
                | RelationType::ScheduledFor
                | RelationType::DueOn
        )
    }

    /// Get a human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            RelationType::BelongsToProject => "belongs to project",
            RelationType::HasContext => "has context",
            RelationType::InArea => "in area",
            RelationType::SupportsGoal => "supports goal",
            RelationType::WaitingOn => "waiting on",
            RelationType::DelegatedTo => "delegated to",
            RelationType::BlockedBy => "blocked by",
            RelationType::DependsOn => "depends on",
            RelationType::ScheduledFor => "scheduled for",
            RelationType::DueOn => "due on",
            RelationType::Mentions => "mentions",
            RelationType::RelatedTo => "related to",
            RelationType::AuthoredBy => "authored by",
            RelationType::AboutTopic => "about topic",
            RelationType::LocatedAt => "located at",
            RelationType::WorksFor => "works for",
            RelationType::CommittedTo => "committed to",
            RelationType::References => "references",
            RelationType::PartOf => "part of",
            RelationType::Contains => "contains",
            RelationType::SameAs => "same as",
            RelationType::Custom => "custom",
        }
    }
}

impl std::fmt::Display for RelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ============================================================================
// Deletion Strategy Types
// ============================================================================

/// Strategy for handling knowledge (entities/relationships) when removing a source.
///
/// This determines what happens to entities and relationships that reference
/// documents from a source being removed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeletionStrategy {
    /// Remove source references from entities/relationships.
    /// Delete only orphans (entities with no remaining source refs).
    /// This is the safest default that preserves knowledge with other sources.
    #[default]
    RemoveSourceRefs,
    /// Delete ALL entities/relationships that had ANY reference to the deleted source.
    /// Use this for complete cleanup when you want to remove all traces of a source.
    DeleteAffected,
    /// Only remove source references, never delete knowledge.
    /// Allows orphaned entities/relationships to remain in the store.
    /// Use this to preserve knowledge even if sources are removed.
    PreserveKnowledge,
}

impl DeletionStrategy {
    /// Get a human-readable description of the strategy.
    pub fn description(&self) -> &'static str {
        match self {
            DeletionStrategy::RemoveSourceRefs => {
                "Remove source references; delete only orphaned entities"
            }
            DeletionStrategy::DeleteAffected => {
                "Delete all entities/relationships referencing the source"
            }
            DeletionStrategy::PreserveKnowledge => "Remove references only; preserve all knowledge",
        }
    }
}

impl std::str::FromStr for DeletionStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "remove_source_refs" | "removesourcerefs" | "default" => {
                Ok(DeletionStrategy::RemoveSourceRefs)
            }
            "delete_affected" | "deleteaffected" | "delete" => {
                Ok(DeletionStrategy::DeleteAffected)
            }
            "preserve_knowledge" | "preserveknowledge" | "preserve" => {
                Ok(DeletionStrategy::PreserveKnowledge)
            }
            _ => Err(format!(
                "Invalid deletion strategy: '{}'. Valid options: remove_source_refs, delete_affected, preserve_knowledge",
                s
            )),
        }
    }
}

impl std::fmt::Display for DeletionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeletionStrategy::RemoveSourceRefs => write!(f, "remove_source_refs"),
            DeletionStrategy::DeleteAffected => write!(f, "delete_affected"),
            DeletionStrategy::PreserveKnowledge => write!(f, "preserve_knowledge"),
        }
    }
}

/// Result of unlinking documents from knowledge (entities/relationships).
///
/// This captures the impact of removing document references from the ontology store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnlinkResult {
    /// Number of entities that had source refs removed (but were not deleted).
    pub entities_updated: usize,
    /// Number of entities that were deleted (orphaned or affected).
    pub entities_deleted: usize,
    /// Number of relationships that had source refs removed (but were not deleted).
    pub relationships_updated: usize,
    /// Number of relationships that were deleted (orphaned or affected).
    pub relationships_deleted: usize,
    /// IDs of entities that became orphaned (only populated with PreserveKnowledge strategy).
    #[serde(default)]
    pub orphaned_entity_ids: Vec<String>,
    /// IDs of relationships that became orphaned (only populated with PreserveKnowledge strategy).
    #[serde(default)]
    pub orphaned_relationship_ids: Vec<String>,
}

impl UnlinkResult {
    /// Check if any changes were made.
    pub fn has_changes(&self) -> bool {
        self.entities_updated > 0
            || self.entities_deleted > 0
            || self.relationships_updated > 0
            || self.relationships_deleted > 0
    }

    /// Get total number of entities affected (updated + deleted).
    pub fn total_entities_affected(&self) -> usize {
        self.entities_updated + self.entities_deleted
    }

    /// Get total number of relationships affected (updated + deleted).
    pub fn total_relationships_affected(&self) -> usize {
        self.relationships_updated + self.relationships_deleted
    }

    /// Merge another UnlinkResult into this one.
    pub fn merge(&mut self, other: UnlinkResult) {
        self.entities_updated += other.entities_updated;
        self.entities_deleted += other.entities_deleted;
        self.relationships_updated += other.relationships_updated;
        self.relationships_deleted += other.relationships_deleted;
        self.orphaned_entity_ids.extend(other.orphaned_entity_ids);
        self.orphaned_relationship_ids
            .extend(other.orphaned_relationship_ids);
    }
}

/// Result of removing a source with cascading deletion.
///
/// This captures the full impact of source removal on documents and knowledge.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceRemovalResult {
    /// Number of documents removed from storage.
    pub documents_removed: usize,
    /// Result of unlinking documents from the knowledge graph.
    pub knowledge_result: UnlinkResult,
}

impl SourceRemovalResult {
    /// Check if any changes were made.
    pub fn has_changes(&self) -> bool {
        self.documents_removed > 0 || self.knowledge_result.has_changes()
    }

    /// Create a summary message describing the removal.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if self.documents_removed > 0 {
            parts.push(format!("{} documents removed", self.documents_removed));
        }

        let kr = &self.knowledge_result;
        if kr.entities_deleted > 0 {
            parts.push(format!("{} entities deleted", kr.entities_deleted));
        }
        if kr.entities_updated > 0 {
            parts.push(format!("{} entities updated", kr.entities_updated));
        }
        if kr.relationships_deleted > 0 {
            parts.push(format!(
                "{} relationships deleted",
                kr.relationships_deleted
            ));
        }
        if kr.relationships_updated > 0 {
            parts.push(format!(
                "{} relationships updated",
                kr.relationships_updated
            ));
        }
        if !kr.orphaned_entity_ids.is_empty() {
            parts.push(format!(
                "{} orphaned entities",
                kr.orphaned_entity_ids.len()
            ));
        }
        if !kr.orphaned_relationship_ids.is_empty() {
            parts.push(format!(
                "{} orphaned relationships",
                kr.orphaned_relationship_ids.len()
            ));
        }

        if parts.is_empty() {
            "No changes made".to_string()
        } else {
            parts.join(", ")
        }
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// A reference to a source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRef {
    /// Document ID in the Alloy index.
    pub document_id: String,
    /// Optional chunk ID within the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<String>,
    /// The text that was extracted from this location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extracted_text: Option<String>,
    /// Confidence of this extraction.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

impl DocumentRef {
    /// Create a new document reference.
    pub fn new(document_id: impl Into<String>) -> Self {
        Self {
            document_id: document_id.into(),
            chunk_id: None,
            extracted_text: None,
            confidence: 1.0,
        }
    }

    /// Set the chunk ID.
    pub fn with_chunk(mut self, chunk_id: impl Into<String>) -> Self {
        self.chunk_id = Some(chunk_id.into());
        self
    }

    /// Set the extracted text.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.extracted_text = Some(text.into());
        self
    }

    /// Set the confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

// ============================================================================
// Query and Filter Types
// ============================================================================

/// Filter criteria for querying entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityFilter {
    /// Filter by entity types.
    #[serde(default)]
    pub entity_types: Vec<EntityType>,
    /// Filter by name pattern (case-insensitive contains).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_pattern: Option<String>,
    /// Filter by minimum confidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_confidence: Option<f32>,
    /// Filter by source document ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document_id: Option<String>,
    /// Filter by creation date range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_after: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_before: Option<DateTime<Utc>>,
    /// Filter by metadata key existence.
    #[serde(default)]
    pub has_metadata_keys: Vec<String>,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    100
}

impl Default for EntityFilter {
    fn default() -> Self {
        Self {
            entity_types: Vec::new(),
            name_pattern: None,
            min_confidence: None,
            source_document_id: None,
            created_after: None,
            created_before: None,
            has_metadata_keys: Vec::new(),
            limit: 100,
            offset: 0,
        }
    }
}

impl EntityFilter {
    /// Create a new filter for specific entity types.
    pub fn by_types(types: impl IntoIterator<Item = EntityType>) -> Self {
        Self {
            entity_types: types.into_iter().collect(),
            ..Default::default()
        }
    }

    /// Create a filter for a name pattern.
    pub fn by_name(pattern: impl Into<String>) -> Self {
        Self {
            name_pattern: Some(pattern.into()),
            ..Default::default()
        }
    }

    /// Add a type filter.
    pub fn with_type(mut self, entity_type: EntityType) -> Self {
        self.entity_types.push(entity_type);
        self
    }

    /// Set the limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the offset.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Set minimum confidence.
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// Check if an entity matches this filter.
    pub fn matches(&self, entity: &Entity) -> bool {
        // Check entity types
        if !self.entity_types.is_empty() && !self.entity_types.contains(&entity.entity_type) {
            return false;
        }

        // Check name pattern
        if let Some(ref pattern) = self.name_pattern {
            if !entity.matches_query(pattern) {
                return false;
            }
        }

        // Check confidence
        if let Some(min_conf) = self.min_confidence {
            if entity.confidence < min_conf {
                return false;
            }
        }

        // Check source document
        if let Some(ref doc_id) = self.source_document_id {
            if !entity.source_refs.iter().any(|r| &r.document_id == doc_id) {
                return false;
            }
        }

        // Check creation dates
        if let Some(after) = self.created_after {
            if entity.created_at < after {
                return false;
            }
        }
        if let Some(before) = self.created_before {
            if entity.created_at > before {
                return false;
            }
        }

        // Check metadata keys
        for key in &self.has_metadata_keys {
            if !entity.metadata.contains_key(key) {
                return false;
            }
        }

        true
    }
}

/// Filter criteria for querying relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipFilter {
    /// Filter by relationship types.
    #[serde(default)]
    pub relationship_types: Vec<RelationType>,
    /// Filter by source entity ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_entity_id: Option<String>,
    /// Filter by target entity ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_entity_id: Option<String>,
    /// Filter by either source or target entity ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub involves_entity_id: Option<String>,
    /// Filter by minimum confidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_confidence: Option<f32>,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

impl Default for RelationshipFilter {
    fn default() -> Self {
        Self {
            relationship_types: Vec::new(),
            source_entity_id: None,
            target_entity_id: None,
            involves_entity_id: None,
            min_confidence: None,
            limit: 100,
            offset: 0,
        }
    }
}

impl RelationshipFilter {
    /// Create a filter for relationships from a source entity.
    pub fn from_entity(entity_id: impl Into<String>) -> Self {
        Self {
            source_entity_id: Some(entity_id.into()),
            ..Default::default()
        }
    }

    /// Create a filter for relationships to a target entity.
    pub fn to_entity(entity_id: impl Into<String>) -> Self {
        Self {
            target_entity_id: Some(entity_id.into()),
            ..Default::default()
        }
    }

    /// Create a filter for relationships involving an entity (either source or target).
    pub fn involving(entity_id: impl Into<String>) -> Self {
        Self {
            involves_entity_id: Some(entity_id.into()),
            ..Default::default()
        }
    }

    /// Add a relationship type filter.
    pub fn with_type(mut self, relation_type: RelationType) -> Self {
        self.relationship_types.push(relation_type);
        self
    }

    /// Set the limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Check if a relationship matches this filter.
    pub fn matches(&self, rel: &Relationship) -> bool {
        // Check relationship types
        if !self.relationship_types.is_empty()
            && !self.relationship_types.contains(&rel.relationship_type)
        {
            return false;
        }

        // Check source entity
        if let Some(ref src_id) = self.source_entity_id {
            if &rel.source_entity_id != src_id {
                return false;
            }
        }

        // Check target entity
        if let Some(ref tgt_id) = self.target_entity_id {
            if &rel.target_entity_id != tgt_id {
                return false;
            }
        }

        // Check involves entity
        if let Some(ref inv_id) = self.involves_entity_id {
            if &rel.source_entity_id != inv_id && &rel.target_entity_id != inv_id {
                return false;
            }
        }

        // Check confidence
        if let Some(min_conf) = self.min_confidence {
            if rel.confidence < min_conf {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Statistics Types
// ============================================================================

/// Statistics about the ontology store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OntologyStats {
    /// Total number of entities.
    pub entity_count: usize,
    /// Number of entities by type.
    pub entities_by_type: HashMap<String, usize>,
    /// Total number of relationships.
    pub relationship_count: usize,
    /// Number of relationships by type.
    pub relationships_by_type: HashMap<String, usize>,
    /// Number of entities with embeddings.
    pub entities_with_embeddings: usize,
    /// Average confidence across all entities.
    pub average_entity_confidence: f32,
    /// Average confidence across all relationships.
    pub average_relationship_confidence: f32,
}

// ============================================================================
// Update Types
// ============================================================================

/// Update operations for an entity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityUpdate {
    /// New name (if changing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// New entity type (if changing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_type: Option<EntityType>,
    /// Aliases to add.
    #[serde(default)]
    pub add_aliases: Vec<String>,
    /// Aliases to remove.
    #[serde(default)]
    pub remove_aliases: Vec<String>,
    /// New embedding vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Source refs to add.
    #[serde(default)]
    pub add_source_refs: Vec<DocumentRef>,
    /// Document IDs to remove from source_refs.
    /// All source refs matching these document IDs will be removed.
    #[serde(default)]
    pub remove_source_refs: Vec<String>,
    /// Metadata to set (overwrites existing keys).
    #[serde(default)]
    pub set_metadata: HashMap<String, serde_json::Value>,
    /// Metadata keys to remove.
    #[serde(default)]
    pub remove_metadata_keys: Vec<String>,
    /// New confidence score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl EntityUpdate {
    /// Create an update to change the name.
    pub fn name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Default::default()
        }
    }

    /// Add an alias.
    pub fn add_alias(mut self, alias: impl Into<String>) -> Self {
        self.add_aliases.push(alias.into());
        self
    }

    /// Set metadata.
    pub fn set_meta(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.set_metadata.insert(key.into(), value);
        self
    }

    /// Set a new embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Apply this update to an entity.
    pub fn apply_to(&self, entity: &mut Entity) {
        if let Some(ref name) = self.name {
            entity.name = name.clone();
        }
        if let Some(entity_type) = self.entity_type {
            entity.entity_type = entity_type;
        }
        for alias in &self.add_aliases {
            if !entity.aliases.contains(alias) {
                entity.aliases.push(alias.clone());
            }
        }
        for alias in &self.remove_aliases {
            entity.aliases.retain(|a| a != alias);
        }
        if let Some(ref embedding) = self.embedding {
            entity.embedding = Some(embedding.clone());
        }
        for source_ref in &self.add_source_refs {
            entity.source_refs.push(source_ref.clone());
        }
        // Remove source refs by document ID
        if !self.remove_source_refs.is_empty() {
            entity
                .source_refs
                .retain(|r| !self.remove_source_refs.contains(&r.document_id));
        }
        for (key, value) in &self.set_metadata {
            entity.metadata.insert(key.clone(), value.clone());
        }
        for key in &self.remove_metadata_keys {
            entity.metadata.remove(key);
        }
        if let Some(confidence) = self.confidence {
            entity.confidence = confidence.clamp(0.0, 1.0);
        }
        entity.updated_at = Utc::now();
    }

    /// Create an update that removes source refs by document ID.
    pub fn remove_source_ref(mut self, document_id: impl Into<String>) -> Self {
        self.remove_source_refs.push(document_id.into());
        self
    }

    /// Create an update that removes multiple source refs by document IDs.
    pub fn remove_source_refs_by_ids(mut self, document_ids: Vec<String>) -> Self {
        self.remove_source_refs.extend(document_ids);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new(EntityType::Person, "John Doe");
        assert_eq!(entity.name, "John Doe");
        assert_eq!(entity.entity_type, EntityType::Person);
        assert!(!entity.id.is_empty());
        assert_eq!(entity.confidence, 1.0);
    }

    #[test]
    fn test_entity_builder_pattern() {
        let entity = Entity::new(EntityType::Project, "Website Redesign")
            .with_alias("web redesign")
            .with_alias("site update")
            .with_confidence(0.95)
            .with_metadata("priority", serde_json::json!("high"));

        assert_eq!(entity.aliases.len(), 2);
        assert_eq!(entity.confidence, 0.95);
        assert!(entity.metadata.contains_key("priority"));
    }

    #[test]
    fn test_entity_matches_query() {
        let entity = Entity::new(EntityType::Person, "John Doe").with_alias("Johnny");

        assert!(entity.matches_query("john"));
        assert!(entity.matches_query("Doe"));
        assert!(entity.matches_query("johnny"));
        assert!(!entity.matches_query("jane"));
    }

    #[test]
    fn test_relationship_creation() {
        let rel = Relationship::new("entity1", RelationType::WorksFor, "entity2");
        assert_eq!(rel.source_entity_id, "entity1");
        assert_eq!(rel.target_entity_id, "entity2");
        assert_eq!(rel.relationship_type, RelationType::WorksFor);
        assert_eq!(rel.confidence, 1.0);
    }

    #[test]
    fn test_entity_type_classification() {
        assert!(EntityType::Project.is_gtd_type());
        assert!(EntityType::Task.is_gtd_type());
        assert!(!EntityType::Person.is_gtd_type());

        assert!(EntityType::Person.is_knowledge_type());
        assert!(EntityType::Organization.is_knowledge_type());
        assert!(!EntityType::Project.is_knowledge_type());
    }

    #[test]
    fn test_entity_filter() {
        let entity = Entity::new(EntityType::Person, "John Doe").with_confidence(0.8);

        let filter = EntityFilter::by_types([EntityType::Person]).with_min_confidence(0.7);
        assert!(filter.matches(&entity));
        assert_eq!(filter.limit, 100); // Default limit should be 100

        let filter = EntityFilter::by_types([EntityType::Organization]);
        assert!(!filter.matches(&entity));

        let filter = EntityFilter::by_name("john");
        assert!(filter.matches(&entity));
        assert_eq!(filter.limit, 100); // Default limit should be 100
    }

    #[test]
    fn test_relationship_filter() {
        let rel = Relationship::new("e1", RelationType::WorksFor, "e2");

        let filter = RelationshipFilter::from_entity("e1");
        assert!(filter.matches(&rel));

        let filter = RelationshipFilter::to_entity("e2");
        assert!(filter.matches(&rel));

        let filter = RelationshipFilter::involving("e1");
        assert!(filter.matches(&rel));

        let filter = RelationshipFilter::involving("e3");
        assert!(!filter.matches(&rel));
    }

    #[test]
    fn test_entity_update() {
        let mut entity = Entity::new(EntityType::Person, "John");

        let update = EntityUpdate::name("John Doe")
            .add_alias("Johnny")
            .set_meta("role", serde_json::json!("developer"));

        update.apply_to(&mut entity);

        assert_eq!(entity.name, "John Doe");
        assert!(entity.aliases.contains(&"Johnny".to_string()));
        assert!(entity.metadata.contains_key("role"));
    }

    #[test]
    fn test_document_ref() {
        let doc_ref = DocumentRef::new("doc123")
            .with_chunk("chunk456")
            .with_text("Some extracted text")
            .with_confidence(0.9);

        assert_eq!(doc_ref.document_id, "doc123");
        assert_eq!(doc_ref.chunk_id, Some("chunk456".to_string()));
        assert_eq!(doc_ref.confidence, 0.9);
    }

    #[test]
    fn test_deletion_strategy_default() {
        let strategy = DeletionStrategy::default();
        assert_eq!(strategy, DeletionStrategy::RemoveSourceRefs);
    }

    #[test]
    fn test_deletion_strategy_from_str() {
        assert_eq!(
            "remove_source_refs".parse::<DeletionStrategy>().unwrap(),
            DeletionStrategy::RemoveSourceRefs
        );
        assert_eq!(
            "delete_affected".parse::<DeletionStrategy>().unwrap(),
            DeletionStrategy::DeleteAffected
        );
        assert_eq!(
            "preserve_knowledge".parse::<DeletionStrategy>().unwrap(),
            DeletionStrategy::PreserveKnowledge
        );
        assert_eq!(
            "default".parse::<DeletionStrategy>().unwrap(),
            DeletionStrategy::RemoveSourceRefs
        );
        assert_eq!(
            "DELETE".parse::<DeletionStrategy>().unwrap(),
            DeletionStrategy::DeleteAffected
        );
        assert_eq!(
            "PRESERVE".parse::<DeletionStrategy>().unwrap(),
            DeletionStrategy::PreserveKnowledge
        );
        assert!("invalid".parse::<DeletionStrategy>().is_err());
    }

    #[test]
    fn test_deletion_strategy_display() {
        assert_eq!(
            format!("{}", DeletionStrategy::RemoveSourceRefs),
            "remove_source_refs"
        );
        assert_eq!(
            format!("{}", DeletionStrategy::DeleteAffected),
            "delete_affected"
        );
        assert_eq!(
            format!("{}", DeletionStrategy::PreserveKnowledge),
            "preserve_knowledge"
        );
    }

    #[test]
    fn test_unlink_result_has_changes() {
        let empty = UnlinkResult::default();
        assert!(!empty.has_changes());

        let with_updates = UnlinkResult {
            entities_updated: 1,
            ..Default::default()
        };
        assert!(with_updates.has_changes());

        let with_deletes = UnlinkResult {
            entities_deleted: 1,
            ..Default::default()
        };
        assert!(with_deletes.has_changes());
    }

    #[test]
    fn test_unlink_result_totals() {
        let result = UnlinkResult {
            entities_updated: 2,
            entities_deleted: 3,
            relationships_updated: 4,
            relationships_deleted: 5,
            ..Default::default()
        };

        assert_eq!(result.total_entities_affected(), 5);
        assert_eq!(result.total_relationships_affected(), 9);
    }

    #[test]
    fn test_unlink_result_merge() {
        let mut result1 = UnlinkResult {
            entities_updated: 1,
            entities_deleted: 2,
            relationships_updated: 3,
            relationships_deleted: 4,
            orphaned_entity_ids: vec!["e1".to_string()],
            orphaned_relationship_ids: vec!["r1".to_string()],
        };

        let result2 = UnlinkResult {
            entities_updated: 10,
            entities_deleted: 20,
            relationships_updated: 30,
            relationships_deleted: 40,
            orphaned_entity_ids: vec!["e2".to_string()],
            orphaned_relationship_ids: vec!["r2".to_string()],
        };

        result1.merge(result2);

        assert_eq!(result1.entities_updated, 11);
        assert_eq!(result1.entities_deleted, 22);
        assert_eq!(result1.relationships_updated, 33);
        assert_eq!(result1.relationships_deleted, 44);
        assert_eq!(result1.orphaned_entity_ids.len(), 2);
        assert_eq!(result1.orphaned_relationship_ids.len(), 2);
    }

    #[test]
    fn test_source_removal_result_has_changes() {
        let empty = SourceRemovalResult::default();
        assert!(!empty.has_changes());

        let with_docs = SourceRemovalResult {
            documents_removed: 1,
            ..Default::default()
        };
        assert!(with_docs.has_changes());

        let with_knowledge = SourceRemovalResult {
            documents_removed: 0,
            knowledge_result: UnlinkResult {
                entities_deleted: 1,
                ..Default::default()
            },
        };
        assert!(with_knowledge.has_changes());
    }

    #[test]
    fn test_source_removal_result_summary() {
        let empty = SourceRemovalResult::default();
        assert_eq!(empty.summary(), "No changes made");

        let full = SourceRemovalResult {
            documents_removed: 5,
            knowledge_result: UnlinkResult {
                entities_updated: 2,
                entities_deleted: 3,
                relationships_updated: 4,
                relationships_deleted: 1,
                orphaned_entity_ids: vec!["e1".to_string()],
                orphaned_relationship_ids: vec!["r1".to_string(), "r2".to_string()],
            },
        };

        let summary = full.summary();
        assert!(summary.contains("5 documents removed"));
        assert!(summary.contains("3 entities deleted"));
        assert!(summary.contains("2 entities updated"));
        assert!(summary.contains("1 relationships deleted"));
        assert!(summary.contains("4 relationships updated"));
        assert!(summary.contains("1 orphaned entities"));
        assert!(summary.contains("2 orphaned relationships"));
    }

    #[test]
    fn test_entity_update_remove_source_refs() {
        let mut entity = Entity::new(EntityType::Person, "John")
            .with_source_ref(DocumentRef::new("doc1"))
            .with_source_ref(DocumentRef::new("doc2"))
            .with_source_ref(DocumentRef::new("doc3"));

        assert_eq!(entity.source_refs.len(), 3);

        let update = EntityUpdate::default()
            .remove_source_refs_by_ids(vec!["doc1".to_string(), "doc3".to_string()]);

        update.apply_to(&mut entity);

        assert_eq!(entity.source_refs.len(), 1);
        assert_eq!(entity.source_refs[0].document_id, "doc2");
    }

    #[test]
    fn test_entity_update_remove_source_ref_single() {
        let mut entity = Entity::new(EntityType::Person, "John")
            .with_source_ref(DocumentRef::new("doc1"))
            .with_source_ref(DocumentRef::new("doc2"));

        let update = EntityUpdate::default().remove_source_ref("doc1");

        update.apply_to(&mut entity);

        assert_eq!(entity.source_refs.len(), 1);
        assert_eq!(entity.source_refs[0].document_id, "doc2");
    }

    #[test]
    fn test_deletion_strategy_serialization() {
        // Test JSON serialization uses snake_case
        let json = serde_json::to_string(&DeletionStrategy::RemoveSourceRefs).unwrap();
        assert_eq!(json, "\"remove_source_refs\"");

        let json = serde_json::to_string(&DeletionStrategy::DeleteAffected).unwrap();
        assert_eq!(json, "\"delete_affected\"");

        let json = serde_json::to_string(&DeletionStrategy::PreserveKnowledge).unwrap();
        assert_eq!(json, "\"preserve_knowledge\"");

        // Test deserialization
        let strategy: DeletionStrategy = serde_json::from_str("\"remove_source_refs\"").unwrap();
        assert_eq!(strategy, DeletionStrategy::RemoveSourceRefs);

        let strategy: DeletionStrategy = serde_json::from_str("\"delete_affected\"").unwrap();
        assert_eq!(strategy, DeletionStrategy::DeleteAffected);
    }
}

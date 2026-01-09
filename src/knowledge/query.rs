//! Knowledge Graph Query Engine.
//!
//! Provides semantic search, expert finding, topic summarization, and
//! relationship traversal capabilities over the ontology.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use crate::embedding::EmbeddingProvider;
use crate::error::Result;
use crate::ontology::{
    EmbeddedOntologyStore, Entity, EntityFilter, EntityType, OntologyStore, RelationType,
    Relationship,
};

use super::types::*;

// ============================================================================
// Knowledge Query Engine
// ============================================================================

/// Knowledge graph query engine.
///
/// Provides high-level query capabilities over the ontology store including
/// semantic search, expert finding, and topic summarization.
///
/// Uses dynamic dispatch for embedding provider to work with any provider type.
pub struct KnowledgeQueryEngine {
    /// Ontology store for entity and relationship access.
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    /// Embedding provider for semantic search.
    embedder: Arc<dyn EmbeddingProvider>,
}

impl KnowledgeQueryEngine {
    /// Create a new knowledge query engine.
    pub fn new(
        store: Arc<RwLock<EmbeddedOntologyStore>>,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self { store, embedder }
    }

    // ========================================================================
    // Main Query Entry Point
    // ========================================================================

    /// Execute a knowledge query.
    pub async fn query(&self, params: KnowledgeQueryParams) -> Result<KnowledgeQueryResult> {
        let start = Instant::now();

        let result = match params.query_type {
            KnowledgeQueryType::SemanticSearch => self.semantic_search(&params).await?,
            KnowledgeQueryType::EntityLookup => self.entity_lookup(&params).await?,
            KnowledgeQueryType::RelationshipQuery => self.relationship_query(&params).await?,
            KnowledgeQueryType::TopicSummary => self.topic_summary(&params).await?,
            KnowledgeQueryType::ConnectedEntities => self.connected_entities(&params).await?,
            KnowledgeQueryType::ExpertFinding => self.expert_finding(&params).await?,
        };

        // Add execution time to stats
        let mut result = result;
        result.stats.execution_time_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    // ========================================================================
    // Semantic Search
    // ========================================================================

    /// Perform semantic search over entities.
    async fn semantic_search(&self, params: &KnowledgeQueryParams) -> Result<KnowledgeQueryResult> {
        let mut stats = QueryStats {
            used_embedding_search: true,
            ..Default::default()
        };

        // Generate query embedding
        let embeddings = self
            .embedder
            .embed(std::slice::from_ref(&params.query))
            .await?;
        let query_embedding = embeddings.into_iter().next().unwrap_or_default();

        // Search by embedding
        let entity_types = if params.entity_types.is_empty() {
            None
        } else {
            Some(params.entity_types.as_slice())
        };

        let store = self.store.read().await;
        let scored_entities = store
            .search_entities_by_embedding(&query_embedding, entity_types, params.limit)
            .await?;

        stats.entities_searched = scored_entities.len();

        // Convert to ScoredEntity
        let entities: Vec<ScoredEntity> = scored_entities
            .into_iter()
            .filter(|(_, score)| params.min_confidence.is_none_or(|min| *score >= min))
            .map(|(entity, score)| {
                ScoredEntity::new(entity, score).with_reason("Semantic similarity to query")
            })
            .collect();

        // Get relationships for top entities
        let relationships = self
            .get_entity_relationships_inner(&store, &entities, params.limit)
            .await?;
        stats.relationships_traversed = relationships.len();

        // Get source documents if requested
        let source_documents = if params.include_sources {
            self.get_source_documents(&entities)
        } else {
            Vec::new()
        };

        // Calculate overall confidence
        let confidence = if entities.is_empty() {
            0.0
        } else {
            entities.iter().map(|e| e.score).sum::<f32>() / entities.len() as f32
        };

        Ok(KnowledgeQueryResult {
            entities,
            relationships,
            source_documents,
            summary: None,
            confidence,
            stats,
        })
    }

    // ========================================================================
    // Entity Lookup
    // ========================================================================

    /// Look up entities by name or ID.
    async fn entity_lookup(&self, params: &KnowledgeQueryParams) -> Result<KnowledgeQueryResult> {
        let mut stats = QueryStats::default();

        let store = self.store.read().await;

        // Try to find by ID first
        if let Ok(Some(entity)) = store.get_entity(&params.query).await {
            let scored = ScoredEntity::new(entity, 1.0).with_reason("Exact ID match");
            stats.entities_searched = 1;

            return Ok(KnowledgeQueryResult {
                entities: vec![scored],
                relationships: Vec::new(),
                source_documents: Vec::new(),
                summary: None,
                confidence: 1.0,
                stats,
            });
        }

        // Search by name
        let entities = store
            .find_entities_by_name(&params.query, params.limit)
            .await?;

        stats.entities_searched = entities.len();

        let scored_entities: Vec<ScoredEntity> = entities
            .into_iter()
            .map(|entity| {
                let score = calculate_name_match_score(&entity.name, &params.query);
                ScoredEntity::new(entity, score).with_reason("Name match")
            })
            .collect();

        let confidence = if scored_entities.is_empty() {
            0.0
        } else {
            scored_entities.iter().map(|e| e.score).sum::<f32>() / scored_entities.len() as f32
        };

        Ok(KnowledgeQueryResult {
            entities: scored_entities,
            relationships: Vec::new(),
            source_documents: Vec::new(),
            summary: None,
            confidence,
            stats,
        })
    }

    // ========================================================================
    // Relationship Query
    // ========================================================================

    /// Query relationships (e.g., "Who works with X?").
    async fn relationship_query(
        &self,
        params: &KnowledgeQueryParams,
    ) -> Result<KnowledgeQueryResult> {
        let mut stats = QueryStats::default();

        let store = self.store.read().await;

        // Find the source entity
        let source_entity = if let Some(ref entity_id) = params.from_entity_id {
            store.get_entity(entity_id).await?
        } else {
            // Search for entity by name
            let entities = store.find_entities_by_name(&params.query, 1).await?;
            entities.into_iter().next()
        };

        let Some(source) = source_entity else {
            return Ok(KnowledgeQueryResult::default());
        };

        // Get relationships from/to this entity
        let relationships = store.get_relationships_involving(&source.id).await?;
        stats.relationships_traversed = relationships.len();

        // Filter by relationship types if specified
        let filtered_relationships: Vec<Relationship> = if params.relationship_types.is_empty() {
            relationships
        } else {
            relationships
                .into_iter()
                .filter(|r| params.relationship_types.contains(&r.relationship_type))
                .collect()
        };

        // Get connected entities
        let connected_entity_ids: HashSet<String> = filtered_relationships
            .iter()
            .flat_map(|r| vec![r.source_entity_id.clone(), r.target_entity_id.clone()])
            .filter(|id| id != &source.id)
            .collect();

        let mut entities = Vec::new();
        for id in connected_entity_ids.iter().take(params.limit) {
            if let Ok(Some(entity)) = store.get_entity(id).await {
                stats.entities_searched += 1;
                let score = 0.8; // Base score for connected entities
                entities.push(
                    ScoredEntity::new(entity, score).with_reason("Connected via relationship"),
                );
            }
        }

        let source_scored = ScoredEntity::new(source, 1.0).with_reason("Query subject");
        let mut all_entities = vec![source_scored];
        all_entities.extend(entities);

        Ok(KnowledgeQueryResult {
            entities: all_entities,
            relationships: filtered_relationships,
            source_documents: Vec::new(),
            summary: None,
            confidence: 0.9,
            stats,
        })
    }

    // ========================================================================
    // Topic Summary
    // ========================================================================

    /// Generate a topic summary.
    pub async fn topic_summary(
        &self,
        params: &KnowledgeQueryParams,
    ) -> Result<KnowledgeQueryResult> {
        let mut stats = QueryStats::default();

        let store = self.store.read().await;

        // Find topic entity or create semantic search
        let topic_entities: Vec<Entity> = if let Some(ref entity_id) = params.from_entity_id {
            if let Ok(Some(entity)) = store.get_entity(entity_id).await {
                vec![entity]
            } else {
                Vec::new()
            }
        } else {
            // Search for topic entities
            let filter = EntityFilter::by_name(&params.query)
                .with_type(EntityType::Topic)
                .with_limit(params.limit);
            store.list_entities(filter).await?
        };

        stats.entities_searched = topic_entities.len();

        // If no topic entities found, do semantic search
        let entities: Vec<ScoredEntity> = if topic_entities.is_empty() {
            // Fall back to semantic search
            let embeddings = self
                .embedder
                .embed(std::slice::from_ref(&params.query))
                .await?;
            let query_embedding = embeddings.into_iter().next().unwrap_or_default();
            stats.used_embedding_search = true;

            let scored = store
                .search_entities_by_embedding(&query_embedding, None, params.limit)
                .await?;

            stats.entities_searched += scored.len();

            scored
                .into_iter()
                .map(|(entity, score)| {
                    ScoredEntity::new(entity, score).with_reason("Semantic match to topic")
                })
                .collect()
        } else {
            topic_entities
                .into_iter()
                .map(|e| ScoredEntity::new(e, 1.0).with_reason("Topic entity"))
                .collect()
        };

        // Get relationships for topic entities
        let relationships = self
            .get_entity_relationships_inner(&store, &entities, params.limit)
            .await?;
        stats.relationships_traversed = relationships.len();

        // Generate topic summary
        let summary = self.generate_topic_summary(&params.query, &entities, &relationships);

        let source_documents = if params.include_sources {
            self.get_source_documents(&entities)
        } else {
            Vec::new()
        };

        let confidence = if entities.is_empty() {
            0.0
        } else {
            entities.iter().map(|e| e.score).sum::<f32>() / entities.len() as f32
        };

        Ok(KnowledgeQueryResult {
            entities,
            relationships,
            source_documents,
            summary: Some(summary),
            confidence,
            stats,
        })
    }

    /// Generate a summary for a topic based on entities and relationships.
    fn generate_topic_summary(
        &self,
        topic: &str,
        entities: &[ScoredEntity],
        relationships: &[Relationship],
    ) -> String {
        let mut summary = format!("## Topic: {}\n\n", topic);

        // Group entities by type
        let mut by_type: HashMap<EntityType, Vec<&ScoredEntity>> = HashMap::new();
        for entity in entities {
            by_type
                .entry(entity.entity.entity_type)
                .or_default()
                .push(entity);
        }

        // Add key entities section
        summary.push_str("### Key Entities\n\n");
        for (entity_type, type_entities) in by_type.iter() {
            summary.push_str(&format!(
                "**{}**: {}\n",
                entity_type.display_name(),
                type_entities
                    .iter()
                    .take(5)
                    .map(|e| e.entity.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // Add relationships section
        if !relationships.is_empty() {
            summary.push_str("\n### Key Relationships\n\n");

            // Group by relationship type
            let mut by_rel_type: HashMap<RelationType, Vec<&Relationship>> = HashMap::new();
            for rel in relationships {
                by_rel_type
                    .entry(rel.relationship_type)
                    .or_default()
                    .push(rel);
            }

            for (rel_type, rels) in by_rel_type.iter() {
                summary.push_str(&format!(
                    "- {} ({} instances)\n",
                    rel_type.display_name(),
                    rels.len()
                ));
            }
        }

        // Add statistics
        summary.push_str(&format!(
            "\n### Statistics\n\n- Total entities: {}\n- Total relationships: {}\n",
            entities.len(),
            relationships.len()
        ));

        summary
    }

    // ========================================================================
    // Connected Entities (Graph Traversal)
    // ========================================================================

    /// Get connected entities via graph traversal.
    async fn connected_entities(
        &self,
        params: &KnowledgeQueryParams,
    ) -> Result<KnowledgeQueryResult> {
        let mut stats = QueryStats::default();

        let store = self.store.read().await;

        // Get the starting entity
        let start_entity = if let Some(ref entity_id) = params.from_entity_id {
            store.get_entity(entity_id).await?
        } else {
            let entities = store.find_entities_by_name(&params.query, 1).await?;
            entities.into_iter().next()
        };

        let Some(start) = start_entity else {
            return Ok(KnowledgeQueryResult::default());
        };

        let max_depth = params.max_depth.unwrap_or(2);
        stats.max_depth_reached = max_depth;

        // Get relationship type filter
        let rel_types = if params.relationship_types.is_empty() {
            None
        } else {
            Some(params.relationship_types.as_slice())
        };

        // Traverse graph
        let connected = store
            .get_connected_entities(&start.id, rel_types, max_depth)
            .await?;

        stats.entities_searched = connected.len();
        stats.relationships_traversed = connected.iter().map(|(_, rels)| rels.len()).sum();

        // Convert to scored entities (score based on distance)
        let mut entities: Vec<ScoredEntity> = connected
            .into_iter()
            .take(params.limit)
            .map(|(entity, path)| {
                let depth = path.len();
                let score = 1.0 / (depth as f32 + 1.0); // Closer = higher score
                ScoredEntity::new(entity, score)
                    .with_reason(format!("Connected at depth {}", depth))
            })
            .collect();

        // Add start entity at top
        let start_scored = ScoredEntity::new(start, 1.0).with_reason("Starting entity");
        entities.insert(0, start_scored);

        Ok(KnowledgeQueryResult {
            entities,
            relationships: Vec::new(), // Could include paths
            source_documents: Vec::new(),
            summary: None,
            confidence: 0.9,
            stats,
        })
    }

    // ========================================================================
    // Expert Finding
    // ========================================================================

    /// Find experts on a topic.
    pub async fn expert_finding(
        &self,
        params: &KnowledgeQueryParams,
    ) -> Result<KnowledgeQueryResult> {
        let mut stats = QueryStats {
            used_embedding_search: true,
            ..Default::default()
        };

        // Generate query embedding for topic
        let embeddings = self
            .embedder
            .embed(std::slice::from_ref(&params.query))
            .await?;
        let query_embedding = embeddings.into_iter().next().unwrap_or_default();

        let store = self.store.read().await;

        // Find entities semantically related to the topic
        let topic_related = store
            .search_entities_by_embedding(&query_embedding, None, params.limit * 5)
            .await?;

        stats.entities_searched = topic_related.len();

        // Find all Person entities
        let people = store
            .find_entities_by_type(EntityType::Person, 1000)
            .await?;

        // Score each person based on their connections to topic-related entities
        let mut expert_scores: HashMap<String, (Entity, f32, Vec<String>)> = HashMap::new();

        for person in people {
            let mut score = 0.0f32;
            let mut evidence: Vec<String> = Vec::new();

            // Get relationships from this person
            let rels = store.get_relationships_from(&person.id).await?;
            stats.relationships_traversed += rels.len();

            for rel in &rels {
                // Check if target is related to topic
                if let Some((related_entity, topic_score)) = topic_related
                    .iter()
                    .find(|(e, _)| e.id == rel.target_entity_id)
                {
                    // Add score based on relationship type and topic relevance
                    let rel_weight = match rel.relationship_type {
                        RelationType::AuthoredBy => 0.5, // Note: inverse direction
                        RelationType::AboutTopic => 0.4,
                        RelationType::WorksFor => 0.2,
                        RelationType::Mentions => 0.3,
                        _ => 0.1,
                    };

                    let contribution = rel_weight * topic_score;
                    score += contribution;
                    evidence.push(format!(
                        "{} {} (score: {:.2})",
                        rel.relationship_type.display_name(),
                        related_entity.name,
                        contribution
                    ));
                }
            }

            // Check if person was authored by (reverse relationship)
            let rels_to = store.get_relationships_to(&person.id).await?;
            for rel in &rels_to {
                if rel.relationship_type == RelationType::AuthoredBy {
                    // This person authored something related to the topic
                    if let Some((related_entity, topic_score)) = topic_related
                        .iter()
                        .find(|(e, _)| e.id == rel.source_entity_id)
                    {
                        score += 0.5 * topic_score;
                        evidence.push(format!(
                            "Authored {} (score: {:.2})",
                            related_entity.name,
                            0.5 * topic_score
                        ));
                    }
                }
            }

            // Check person's own embedding similarity to topic
            if let Some(ref person_embedding) = person.embedding {
                let similarity = cosine_similarity(&query_embedding, person_embedding);
                if similarity > 0.3 {
                    score += similarity * 0.3;
                    evidence.push(format!("Direct semantic similarity: {:.2}", similarity));
                }
            }

            if score > 0.0 {
                expert_scores.insert(person.id.clone(), (person, score, evidence));
            }
        }

        // Sort by score and take top experts
        let mut experts: Vec<(Entity, f32, Vec<String>)> = expert_scores.into_values().collect();
        experts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        experts.truncate(params.limit);

        // Convert to ScoredEntity
        let entities: Vec<ScoredEntity> = experts
            .into_iter()
            .map(|(entity, score, evidence)| {
                let normalized_score = (score / 5.0).min(1.0); // Normalize score
                ScoredEntity::new(entity, normalized_score).with_reasons(evidence)
            })
            .collect();

        let confidence = if entities.is_empty() {
            0.0
        } else {
            entities.iter().map(|e| e.score).sum::<f32>() / entities.len() as f32
        };

        // Generate summary
        let summary = format!(
            "Found {} potential experts on '{}'. Top expert: {}",
            entities.len(),
            params.query,
            entities
                .first()
                .map(|e| e.entity.name.as_str())
                .unwrap_or("none")
        );

        Ok(KnowledgeQueryResult {
            entities,
            relationships: Vec::new(),
            source_documents: Vec::new(),
            summary: Some(summary),
            confidence,
            stats,
        })
    }

    /// Find experts on a topic (public wrapper with structured result).
    pub async fn find_experts(&self, topic: &str, limit: usize) -> Result<Vec<Expert>> {
        let params = KnowledgeQueryParams {
            query: topic.to_string(),
            query_type: KnowledgeQueryType::ExpertFinding,
            limit,
            ..Default::default()
        };

        let result = self.query(params).await?;

        // Convert scored entities to Expert structs
        let experts = result
            .entities
            .into_iter()
            .map(|scored| {
                let evidence: Vec<ExpertiseEvidence> = scored
                    .match_reasons
                    .iter()
                    .map(|reason| ExpertiseEvidence {
                        evidence_type: EvidenceType::TopicMention,
                        description: reason.clone(),
                        source_document: None,
                        weight: 0.5,
                    })
                    .collect();

                Expert {
                    person: scored.entity,
                    expertise_score: scored.score,
                    topics: vec![TopicExpertise {
                        topic: topic.to_string(),
                        score: scored.score,
                        mention_count: evidence.len(),
                    }],
                    evidence,
                }
            })
            .collect();

        Ok(experts)
    }

    // ========================================================================
    // Topic Summarization (Public API)
    // ========================================================================

    /// Generate a comprehensive topic summary.
    pub async fn summarize_topic(&self, topic: &str, limit: usize) -> Result<TopicSummary> {
        let params = KnowledgeQueryParams {
            query: topic.to_string(),
            query_type: KnowledgeQueryType::TopicSummary,
            limit,
            include_sources: true,
            ..Default::default()
        };

        let result = self.query(params).await?;

        // Calculate entity type distribution
        let mut entity_type_distribution: HashMap<String, usize> = HashMap::new();
        for entity in &result.entities {
            *entity_type_distribution
                .entry(format!("{:?}", entity.entity.entity_type))
                .or_default() += 1;
        }

        // Find related topics from relationships
        let related_topics: Vec<RelatedTopic> = result
            .relationships
            .iter()
            .filter(|r| r.relationship_type == RelationType::AboutTopic)
            .filter_map(|r| {
                result
                    .entities
                    .iter()
                    .find(|e| e.entity.id == r.target_entity_id)
                    .map(|e| RelatedTopic {
                        topic: e.entity.name.clone(),
                        strength: r.confidence,
                        relationship: r.relationship_type.display_name().to_string(),
                    })
            })
            .collect();

        // Extract sub-topics from Topic entities
        let sub_topics: Vec<String> = result
            .entities
            .iter()
            .filter(|e| e.entity.entity_type == EntityType::Topic)
            .filter(|e| e.entity.name.to_lowercase() != topic.to_lowercase())
            .map(|e| e.entity.name.clone())
            .collect();

        Ok(TopicSummary {
            topic: topic.to_string(),
            summary: result.summary.unwrap_or_default(),
            key_entities: result.entities,
            sub_topics,
            related_topics,
            knowledge_gaps: Vec::new(), // Could be computed by finding low-confidence areas
            stats: TopicStats {
                entity_count: result.stats.entities_searched,
                document_count: result.source_documents.len(),
                relationship_count: result.stats.relationships_traversed,
                average_confidence: result.confidence,
                entity_type_distribution,
            },
        })
    }

    // ========================================================================
    // Relationship Traversal (Public API)
    // ========================================================================

    /// Traverse relationships from an entity.
    pub async fn traverse_relationships(
        &self,
        from_entity_id: &str,
        relation_types: Option<&[RelationType]>,
        max_depth: usize,
    ) -> Result<TraversalResult> {
        let start = Instant::now();

        let store = self.store.read().await;

        let start_entity = store.get_entity(from_entity_id).await?.ok_or_else(|| {
            crate::error::AlloyError::Storage(crate::error::StorageError::NotFound(format!(
                "Entity not found: {}",
                from_entity_id
            )))
        })?;

        let connected = store
            .get_connected_entities(from_entity_id, relation_types, max_depth)
            .await?;

        let mut paths = Vec::new();
        let mut discovered: Vec<ScoredEntity> = Vec::new();
        let mut nodes_visited = 1; // Start entity
        let mut edges_traversed = 0;
        let mut max_depth_reached = 0;

        for (entity, path_rels) in connected {
            let depth = path_rels.len();
            if depth > max_depth_reached {
                max_depth_reached = depth;
            }

            nodes_visited += 1;
            edges_traversed += path_rels.len();

            let score = 1.0 / (depth as f32 + 1.0);
            discovered.push(ScoredEntity::new(entity.clone(), score));

            // Build path
            let mut path_entities = vec![start_entity.clone()];
            for rel in &path_rels {
                if let Ok(Some(e)) = store.get_entity(&rel.target_entity_id).await {
                    path_entities.push(e);
                }
            }
            path_entities.push(entity);

            let path_strength = if path_rels.is_empty() {
                1.0
            } else {
                path_rels
                    .iter()
                    .map(|r| r.confidence)
                    .product::<f32>()
                    .powf(1.0 / path_rels.len() as f32)
            };

            paths.push(RelationshipPath {
                relationships: path_rels,
                entities: path_entities,
                path_strength,
            });
        }

        Ok(TraversalResult {
            start_entity,
            paths,
            discovered_entities: discovered,
            stats: TraversalStats {
                nodes_visited,
                edges_traversed,
                max_depth_reached,
                execution_time_ms: start.elapsed().as_millis() as u64,
            },
        })
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Get relationships for a set of entities (requires store lock held).
    async fn get_entity_relationships_inner(
        &self,
        store: &EmbeddedOntologyStore,
        entities: &[ScoredEntity],
        limit: usize,
    ) -> Result<Vec<Relationship>> {
        let mut all_rels = Vec::new();

        for entity in entities.iter().take(10) {
            // Limit API calls
            let rels = store.get_relationships_involving(&entity.entity.id).await?;
            all_rels.extend(rels);
        }

        // Deduplicate and limit
        let mut seen_ids: HashSet<String> = HashSet::new();
        let unique_rels: Vec<Relationship> = all_rels
            .into_iter()
            .filter(|r| seen_ids.insert(r.id.clone()))
            .take(limit)
            .collect();

        Ok(unique_rels)
    }

    /// Get source documents for entities.
    fn get_source_documents(&self, entities: &[ScoredEntity]) -> Vec<DocumentSummary> {
        let mut docs = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        for entity in entities {
            for source_ref in &entity.entity.source_refs {
                if seen_ids.insert(source_ref.document_id.clone()) {
                    docs.push(DocumentSummary {
                        document_id: source_ref.document_id.clone(),
                        path: source_ref.document_id.clone(), // Could be improved with actual path
                        snippet: source_ref.extracted_text.clone(),
                        relevance: entity.score,
                    });
                }
            }
        }

        docs
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate name match score using string similarity.
fn calculate_name_match_score(name: &str, query: &str) -> f32 {
    let name_lower = name.to_lowercase();
    let query_lower = query.to_lowercase();

    if name_lower == query_lower {
        return 1.0;
    }

    if name_lower.contains(&query_lower) {
        return 0.9;
    }

    // Use Jaro-Winkler similarity
    strsim::jaro_winkler(&name_lower, &query_lower) as f32
}

/// Calculate cosine similarity between two vectors.
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_match_score() {
        assert_eq!(calculate_name_match_score("John Doe", "John Doe"), 1.0);
        assert!(calculate_name_match_score("John Doe", "john doe") > 0.9);
        assert!(calculate_name_match_score("John Doe", "John") > 0.8);
        assert!(calculate_name_match_score("John Doe", "Jane") < 0.8);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![0.707, 0.707, 0.0];
        assert!((cosine_similarity(&a, &d) - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_scored_entity() {
        let entity = Entity::new(EntityType::Person, "John Doe");
        let scored = ScoredEntity::new(entity.clone(), 0.8)
            .with_reason("Test reason")
            .with_reasons(vec!["Reason 1", "Reason 2"]);

        assert_eq!(scored.score, 0.8);
        assert_eq!(scored.match_reasons.len(), 3);
    }
}

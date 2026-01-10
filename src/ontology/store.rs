//! Ontology storage trait and implementations.
//!
//! This module provides the storage abstraction for entities and relationships.

use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as AsyncMutex, RwLock};

use crate::error::{AlloyError, Result, StorageError};
use crate::ontology::{
    DocumentRef, Entity, EntityFilter, EntityType, EntityUpdate, OntologyStats, RelationType,
    Relationship, RelationshipFilter,
};

// ============================================================================
// OntologyStore Trait
// ============================================================================

/// Trait for ontology storage backends.
///
/// This trait defines the interface for storing and querying entities
/// and relationships in the semantic knowledge graph.
#[async_trait]
pub trait OntologyStore: Send + Sync {
    // ========================================================================
    // Entity Operations
    // ========================================================================

    /// Create a new entity.
    async fn create_entity(&self, entity: Entity) -> Result<Entity>;

    /// Get an entity by ID.
    async fn get_entity(&self, id: &str) -> Result<Option<Entity>>;

    /// Update an entity.
    async fn update_entity(&self, id: &str, update: EntityUpdate) -> Result<Entity>;

    /// Delete an entity by ID.
    ///
    /// This also removes all relationships involving this entity.
    async fn delete_entity(&self, id: &str) -> Result<bool>;

    /// List entities matching a filter.
    async fn list_entities(&self, filter: EntityFilter) -> Result<Vec<Entity>>;

    /// Find entities by name (case-insensitive, partial match).
    async fn find_entities_by_name(&self, name: &str, limit: usize) -> Result<Vec<Entity>>;

    /// Find entities by type.
    async fn find_entities_by_type(
        &self,
        entity_type: EntityType,
        limit: usize,
    ) -> Result<Vec<Entity>>;

    /// Find entities by source document.
    async fn find_entities_by_document(&self, document_id: &str) -> Result<Vec<Entity>>;

    /// Search entities by semantic similarity.
    ///
    /// Returns entities sorted by similarity to the query embedding.
    async fn search_entities_by_embedding(
        &self,
        embedding: &[f32],
        entity_types: Option<&[EntityType]>,
        limit: usize,
    ) -> Result<Vec<(Entity, f32)>>;

    // ========================================================================
    // Relationship Operations
    // ========================================================================

    /// Create a new relationship.
    async fn create_relationship(&self, relationship: Relationship) -> Result<Relationship>;

    /// Get a relationship by ID.
    async fn get_relationship(&self, id: &str) -> Result<Option<Relationship>>;

    /// Delete a relationship by ID.
    async fn delete_relationship(&self, id: &str) -> Result<bool>;

    /// List relationships matching a filter.
    async fn list_relationships(&self, filter: RelationshipFilter) -> Result<Vec<Relationship>>;

    /// Get all relationships from a source entity.
    async fn get_relationships_from(&self, entity_id: &str) -> Result<Vec<Relationship>>;

    /// Get all relationships to a target entity.
    async fn get_relationships_to(&self, entity_id: &str) -> Result<Vec<Relationship>>;

    /// Get all relationships involving an entity (either as source or target).
    async fn get_relationships_involving(&self, entity_id: &str) -> Result<Vec<Relationship>>;

    /// Find relationships of a specific type.
    async fn find_relationships_by_type(
        &self,
        relation_type: RelationType,
        limit: usize,
    ) -> Result<Vec<Relationship>>;

    // ========================================================================
    // Graph Traversal
    // ========================================================================

    /// Get connected entities (entities related to the given entity).
    async fn get_connected_entities(
        &self,
        entity_id: &str,
        relation_types: Option<&[RelationType]>,
        max_depth: usize,
    ) -> Result<Vec<(Entity, Vec<Relationship>)>>;

    /// Find paths between two entities.
    async fn find_paths(
        &self,
        from_entity_id: &str,
        to_entity_id: &str,
        max_depth: usize,
    ) -> Result<Vec<Vec<Relationship>>>;

    // ========================================================================
    // Bulk Operations
    // ========================================================================

    /// Create multiple entities in a batch.
    async fn create_entities_batch(&self, entities: Vec<Entity>) -> Result<Vec<Entity>>;

    /// Create multiple relationships in a batch.
    async fn create_relationships_batch(
        &self,
        relationships: Vec<Relationship>,
    ) -> Result<Vec<Relationship>>;

    /// Delete all entities and relationships from a source document.
    async fn delete_by_document(&self, document_id: &str) -> Result<usize>;

    // ========================================================================
    // Deduplication and Merging
    // ========================================================================

    /// Find potential duplicate entities.
    async fn find_duplicates(
        &self,
        entity_type: EntityType,
        similarity_threshold: f32,
    ) -> Result<Vec<(Entity, Entity, f32)>>;

    /// Merge two entities, keeping the first and absorbing the second.
    ///
    /// All relationships pointing to/from the second entity are redirected
    /// to the first entity. Aliases and source refs are combined.
    async fn merge_entities(&self, keep_id: &str, merge_id: &str) -> Result<Entity>;

    // ========================================================================
    // Statistics and Maintenance
    // ========================================================================

    /// Get statistics about the ontology store.
    async fn stats(&self) -> Result<OntologyStats>;

    /// Clear all data from the store.
    async fn clear(&self) -> Result<()>;
}

// ============================================================================
// Internal Data Structure
// ============================================================================

/// Internal data storage structure.
#[derive(Debug, Default)]
struct OntologyData {
    /// Entities indexed by ID.
    entities: HashMap<String, Entity>,
    /// Relationships indexed by ID.
    relationships: HashMap<String, Relationship>,
    /// Index: source_entity_id -> relationship IDs.
    rel_by_source: HashMap<String, Vec<String>>,
    /// Index: target_entity_id -> relationship IDs.
    rel_by_target: HashMap<String, Vec<String>>,
    /// Index: document_id -> entity IDs.
    entities_by_doc: HashMap<String, Vec<String>>,
    /// Index: document_id -> relationship IDs.
    relationships_by_doc: HashMap<String, Vec<String>>,
    /// Index: entity_type -> entity IDs.
    entities_by_type: HashMap<EntityType, Vec<String>>,
}

impl OntologyData {
    /// Add entity to type index.
    fn index_entity_type(&mut self, entity_id: &str, entity_type: EntityType) {
        self.entities_by_type
            .entry(entity_type)
            .or_default()
            .push(entity_id.to_string());
    }

    /// Remove entity from type index.
    fn unindex_entity_type(&mut self, entity_id: &str, entity_type: EntityType) {
        if let Some(ids) = self.entities_by_type.get_mut(&entity_type) {
            ids.retain(|id| id != entity_id);
        }
    }

    /// Add entity to document index.
    fn index_entity_doc(&mut self, entity_id: &str, doc_refs: &[DocumentRef]) {
        for doc_ref in doc_refs {
            self.entities_by_doc
                .entry(doc_ref.document_id.clone())
                .or_default()
                .push(entity_id.to_string());
        }
    }

    /// Remove entity from document index.
    fn unindex_entity_doc(&mut self, entity_id: &str) {
        for ids in self.entities_by_doc.values_mut() {
            ids.retain(|id| id != entity_id);
        }
    }

    /// Add relationship to document index.
    fn index_relationship_doc(&mut self, rel_id: &str, doc_refs: &[DocumentRef]) {
        for doc_ref in doc_refs {
            self.relationships_by_doc
                .entry(doc_ref.document_id.clone())
                .or_default()
                .push(rel_id.to_string());
        }
    }

    /// Remove relationship from document index.
    fn unindex_relationship_doc(&mut self, rel_id: &str) {
        for ids in self.relationships_by_doc.values_mut() {
            ids.retain(|id| id != rel_id);
        }
    }

    /// Add relationship to indices.
    fn index_relationship(&mut self, rel: &Relationship) {
        self.rel_by_source
            .entry(rel.source_entity_id.clone())
            .or_default()
            .push(rel.id.clone());

        self.rel_by_target
            .entry(rel.target_entity_id.clone())
            .or_default()
            .push(rel.id.clone());

        // Index by document
        self.index_relationship_doc(&rel.id, &rel.source_refs);
    }

    /// Remove relationship from indices.
    fn unindex_relationship(&mut self, rel: &Relationship) {
        if let Some(ids) = self.rel_by_source.get_mut(&rel.source_entity_id) {
            ids.retain(|id| id != &rel.id);
        }
        if let Some(ids) = self.rel_by_target.get_mut(&rel.target_entity_id) {
            ids.retain(|id| id != &rel.id);
        }

        // Remove from document index
        self.unindex_relationship_doc(&rel.id);
    }
}

// ============================================================================
// Embedded Implementation
// ============================================================================

/// In-memory ontology store with optional persistence.
///
/// This implementation stores entities and relationships in memory using
/// HashMaps for fast access, with optional JSON file persistence.
pub struct EmbeddedOntologyStore {
    /// All data protected by a single RwLock for consistent access.
    data: RwLock<OntologyData>,
    /// Optional persistence file path.
    persistence_path: Option<std::path::PathBuf>,
    /// Mutex for persistence operations.
    persist_lock: AsyncMutex<()>,
}

impl EmbeddedOntologyStore {
    /// Create a new in-memory ontology store without persistence.
    pub fn new() -> Self {
        Self {
            data: RwLock::new(OntologyData::default()),
            persistence_path: None,
            persist_lock: AsyncMutex::new(()),
        }
    }

    /// Create a new ontology store with file persistence.
    pub async fn with_persistence(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir).map_err(StorageError::Io)?;

        let persistence_path = data_dir.join("ontology.json");
        let store = Self {
            data: RwLock::new(OntologyData::default()),
            persistence_path: Some(persistence_path.clone()),
            persist_lock: AsyncMutex::new(()),
        };

        // Load existing data if present
        if persistence_path.exists() {
            store.load_from_file(&persistence_path).await?;
        }

        Ok(store)
    }

    /// Load data from a JSON file.
    async fn load_from_file(&self, path: &Path) -> Result<()> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(AlloyError::Io)?;

        let persisted: PersistenceData =
            serde_json::from_str(&content).map_err(AlloyError::Serialization)?;

        let mut data = self.data.write().await;

        for entity in persisted.entities {
            // Update type index
            data.index_entity_type(&entity.id, entity.entity_type);

            // Update document index
            data.index_entity_doc(&entity.id, &entity.source_refs);

            data.entities.insert(entity.id.clone(), entity);
        }

        for rel in persisted.relationships {
            // Update indices
            data.index_relationship(&rel);
            data.relationships.insert(rel.id.clone(), rel);
        }

        tracing::info!(
            "Loaded {} entities and {} relationships from {}",
            data.entities.len(),
            data.relationships.len(),
            path.display()
        );

        Ok(())
    }

    /// Persist data to file if persistence is enabled.
    async fn persist(&self) -> Result<()> {
        let Some(ref path) = self.persistence_path else {
            return Ok(());
        };

        let _lock = self.persist_lock.lock().await;

        let data = self.data.read().await;
        let entities: Vec<Entity> = data.entities.values().cloned().collect();
        let relationships: Vec<Relationship> = data.relationships.values().cloned().collect();
        drop(data);

        let persisted = PersistenceData {
            version: 1,
            entities,
            relationships,
        };

        let content =
            serde_json::to_string_pretty(&persisted).map_err(AlloyError::Serialization)?;

        // Write to temp file first, then rename for atomicity
        let temp_path = path.with_extension("json.tmp");
        tokio::fs::write(&temp_path, content)
            .await
            .map_err(AlloyError::Io)?;
        tokio::fs::rename(&temp_path, path)
            .await
            .map_err(AlloyError::Io)?;

        Ok(())
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

impl Default for EmbeddedOntologyStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OntologyStore for EmbeddedOntologyStore {
    // ========================================================================
    // Entity Operations
    // ========================================================================

    async fn create_entity(&self, entity: Entity) -> Result<Entity> {
        let mut data = self.data.write().await;

        let id = entity.id.clone();
        let entity_type = entity.entity_type;
        let doc_refs = entity.source_refs.clone();

        data.entities.insert(id.clone(), entity.clone());
        data.index_entity_type(&id, entity_type);
        data.index_entity_doc(&id, &doc_refs);

        drop(data);
        self.persist().await?;
        Ok(entity)
    }

    async fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        let data = self.data.read().await;
        Ok(data.entities.get(id).cloned())
    }

    async fn update_entity(&self, id: &str, update: EntityUpdate) -> Result<Entity> {
        let mut data = self.data.write().await;

        let entity = data
            .entities
            .get_mut(id)
            .ok_or_else(|| StorageError::NotFound(format!("Entity not found: {}", id)))?;

        let old_type = entity.entity_type;

        update.apply_to(entity);
        let updated = entity.clone();

        // Update type index if type changed
        if let Some(new_type) = update.entity_type {
            if new_type != old_type {
                data.unindex_entity_type(id, old_type);
                data.index_entity_type(id, new_type);
            }
        }

        // Update doc index if source refs added
        if !update.add_source_refs.is_empty() {
            data.index_entity_doc(id, &update.add_source_refs);
        }

        drop(data);
        self.persist().await?;
        Ok(updated)
    }

    async fn delete_entity(&self, id: &str) -> Result<bool> {
        let mut data = self.data.write().await;

        let entity = match data.entities.remove(id) {
            Some(e) => e,
            None => return Ok(false),
        };

        // Remove from indices
        data.unindex_entity_type(id, entity.entity_type);
        data.unindex_entity_doc(id);

        // Collect relationship IDs to delete
        let source_rels = data.rel_by_source.get(id).cloned().unwrap_or_default();
        let target_rels = data.rel_by_target.get(id).cloned().unwrap_or_default();
        let rel_ids: Vec<String> = source_rels.into_iter().chain(target_rels).collect();

        // Delete relationships
        for rel_id in rel_ids {
            if let Some(rel) = data.relationships.remove(&rel_id) {
                data.unindex_relationship(&rel);
            }
        }

        drop(data);
        self.persist().await?;
        Ok(true)
    }

    async fn list_entities(&self, filter: EntityFilter) -> Result<Vec<Entity>> {
        let data = self.data.read().await;

        let mut results: Vec<Entity> = data
            .entities
            .values()
            .filter(|e| filter.matches(e))
            .cloned()
            .collect();

        // Sort by updated_at (most recent first)
        results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        // Apply pagination
        let offset = filter.offset;
        let limit = filter.limit;
        Ok(results.into_iter().skip(offset).take(limit).collect())
    }

    async fn find_entities_by_name(&self, name: &str, limit: usize) -> Result<Vec<Entity>> {
        let name_lower = name.to_lowercase();
        let data = self.data.read().await;

        let mut results: Vec<Entity> = data
            .entities
            .values()
            .filter(|e| e.matches_query(&name_lower))
            .cloned()
            .collect();

        results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(results.into_iter().take(limit).collect())
    }

    async fn find_entities_by_type(
        &self,
        entity_type: EntityType,
        limit: usize,
    ) -> Result<Vec<Entity>> {
        let data = self.data.read().await;

        let ids = data
            .entities_by_type
            .get(&entity_type)
            .cloned()
            .unwrap_or_default();

        let mut results: Vec<Entity> = ids
            .iter()
            .filter_map(|id| data.entities.get(id).cloned())
            .collect();

        results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(results.into_iter().take(limit).collect())
    }

    async fn find_entities_by_document(&self, document_id: &str) -> Result<Vec<Entity>> {
        let data = self.data.read().await;

        let ids = data
            .entities_by_doc
            .get(document_id)
            .cloned()
            .unwrap_or_default();

        Ok(ids
            .iter()
            .filter_map(|id| data.entities.get(id).cloned())
            .collect())
    }

    async fn search_entities_by_embedding(
        &self,
        embedding: &[f32],
        entity_types: Option<&[EntityType]>,
        limit: usize,
    ) -> Result<Vec<(Entity, f32)>> {
        let data = self.data.read().await;

        let mut scored: Vec<(Entity, f32)> = data
            .entities
            .values()
            .filter(|e| {
                // Filter by type if specified
                if let Some(types) = entity_types {
                    if !types.contains(&e.entity_type) {
                        return false;
                    }
                }
                // Only consider entities with embeddings
                e.embedding.is_some()
            })
            .map(|e| {
                let sim = e
                    .embedding
                    .as_ref()
                    .map(|emb| Self::cosine_similarity(emb, embedding))
                    .unwrap_or(0.0);
                (e.clone(), sim)
            })
            .collect();

        // Sort by similarity (highest first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored.into_iter().take(limit).collect())
    }

    // ========================================================================
    // Relationship Operations
    // ========================================================================

    async fn create_relationship(&self, relationship: Relationship) -> Result<Relationship> {
        let mut data = self.data.write().await;

        // Verify both entities exist
        if !data.entities.contains_key(&relationship.source_entity_id) {
            return Err(StorageError::NotFound(format!(
                "Source entity not found: {}",
                relationship.source_entity_id
            ))
            .into());
        }
        if !data.entities.contains_key(&relationship.target_entity_id) {
            return Err(StorageError::NotFound(format!(
                "Target entity not found: {}",
                relationship.target_entity_id
            ))
            .into());
        }

        data.index_relationship(&relationship);
        data.relationships
            .insert(relationship.id.clone(), relationship.clone());

        drop(data);
        self.persist().await?;
        Ok(relationship)
    }

    async fn get_relationship(&self, id: &str) -> Result<Option<Relationship>> {
        let data = self.data.read().await;
        Ok(data.relationships.get(id).cloned())
    }

    async fn delete_relationship(&self, id: &str) -> Result<bool> {
        let mut data = self.data.write().await;

        let rel = match data.relationships.remove(id) {
            Some(r) => r,
            None => return Ok(false),
        };

        data.unindex_relationship(&rel);

        drop(data);
        self.persist().await?;
        Ok(true)
    }

    async fn list_relationships(&self, filter: RelationshipFilter) -> Result<Vec<Relationship>> {
        let data = self.data.read().await;

        let mut results: Vec<Relationship> = data
            .relationships
            .values()
            .filter(|r| filter.matches(r))
            .cloned()
            .collect();

        // Sort by created_at (most recent first)
        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        // Apply pagination
        let offset = filter.offset;
        let limit = filter.limit;
        Ok(results.into_iter().skip(offset).take(limit).collect())
    }

    async fn get_relationships_from(&self, entity_id: &str) -> Result<Vec<Relationship>> {
        let data = self.data.read().await;

        let rel_ids = data
            .rel_by_source
            .get(entity_id)
            .cloned()
            .unwrap_or_default();

        Ok(rel_ids
            .iter()
            .filter_map(|id| data.relationships.get(id).cloned())
            .collect())
    }

    async fn get_relationships_to(&self, entity_id: &str) -> Result<Vec<Relationship>> {
        let data = self.data.read().await;

        let rel_ids = data
            .rel_by_target
            .get(entity_id)
            .cloned()
            .unwrap_or_default();

        Ok(rel_ids
            .iter()
            .filter_map(|id| data.relationships.get(id).cloned())
            .collect())
    }

    async fn get_relationships_involving(&self, entity_id: &str) -> Result<Vec<Relationship>> {
        let data = self.data.read().await;

        let source_ids = data
            .rel_by_source
            .get(entity_id)
            .cloned()
            .unwrap_or_default();
        let target_ids = data
            .rel_by_target
            .get(entity_id)
            .cloned()
            .unwrap_or_default();

        let mut all_ids: Vec<String> = source_ids;
        for id in target_ids {
            if !all_ids.contains(&id) {
                all_ids.push(id);
            }
        }

        Ok(all_ids
            .iter()
            .filter_map(|id| data.relationships.get(id).cloned())
            .collect())
    }

    async fn find_relationships_by_type(
        &self,
        relation_type: RelationType,
        limit: usize,
    ) -> Result<Vec<Relationship>> {
        let data = self.data.read().await;

        let mut results: Vec<Relationship> = data
            .relationships
            .values()
            .filter(|r| r.relationship_type == relation_type)
            .cloned()
            .collect();

        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(results.into_iter().take(limit).collect())
    }

    // ========================================================================
    // Graph Traversal
    // ========================================================================

    async fn get_connected_entities(
        &self,
        entity_id: &str,
        relation_types: Option<&[RelationType]>,
        max_depth: usize,
    ) -> Result<Vec<(Entity, Vec<Relationship>)>> {
        use std::collections::{HashSet, VecDeque};

        let mut visited: HashSet<String> = HashSet::new();
        let mut results: Vec<(Entity, Vec<Relationship>)> = Vec::new();
        let mut queue: VecDeque<(String, usize, Vec<Relationship>)> = VecDeque::new();

        visited.insert(entity_id.to_string());
        queue.push_back((entity_id.to_string(), 0, Vec::new()));

        while let Some((current_id, depth, path)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Get all relationships from current entity
            let rels = self.get_relationships_from(&current_id).await?;

            for rel in rels {
                // Filter by relation types if specified
                if let Some(types) = relation_types {
                    if !types.contains(&rel.relationship_type) {
                        continue;
                    }
                }

                let target_id = &rel.target_entity_id;
                if visited.contains(target_id) {
                    continue;
                }

                visited.insert(target_id.clone());

                if let Some(entity) = self.get_entity(target_id).await? {
                    let mut new_path = path.clone();
                    new_path.push(rel.clone());
                    results.push((entity, new_path.clone()));
                    queue.push_back((target_id.clone(), depth + 1, new_path));
                }
            }
        }

        Ok(results)
    }

    async fn find_paths(
        &self,
        from_entity_id: &str,
        to_entity_id: &str,
        max_depth: usize,
    ) -> Result<Vec<Vec<Relationship>>> {
        use std::collections::{HashSet, VecDeque};

        let mut paths: Vec<Vec<Relationship>> = Vec::new();
        let mut queue: VecDeque<(String, Vec<Relationship>, HashSet<String>)> = VecDeque::new();

        let mut initial_visited = HashSet::new();
        initial_visited.insert(from_entity_id.to_string());
        queue.push_back((from_entity_id.to_string(), Vec::new(), initial_visited));

        while let Some((current_id, path, visited)) = queue.pop_front() {
            if path.len() >= max_depth {
                continue;
            }

            let rels = self.get_relationships_from(&current_id).await?;

            for rel in rels {
                let target_id = rel.target_entity_id.clone();

                if target_id == to_entity_id {
                    // Found a path
                    let mut complete_path = path.clone();
                    complete_path.push(rel);
                    paths.push(complete_path);
                } else if !visited.contains(&target_id) {
                    // Continue search
                    let mut new_path = path.clone();
                    new_path.push(rel);
                    let mut new_visited = visited.clone();
                    new_visited.insert(target_id.clone());
                    queue.push_back((target_id, new_path, new_visited));
                }
            }
        }

        Ok(paths)
    }

    // ========================================================================
    // Bulk Operations
    // ========================================================================

    async fn create_entities_batch(&self, entities: Vec<Entity>) -> Result<Vec<Entity>> {
        let mut data = self.data.write().await;
        let mut created = Vec::with_capacity(entities.len());

        for entity in entities {
            let id = entity.id.clone();
            let entity_type = entity.entity_type;
            let doc_refs = entity.source_refs.clone();

            data.entities.insert(id.clone(), entity.clone());
            data.index_entity_type(&id, entity_type);
            data.index_entity_doc(&id, &doc_refs);

            created.push(entity);
        }

        drop(data);
        self.persist().await?;
        Ok(created)
    }

    async fn create_relationships_batch(
        &self,
        relationships: Vec<Relationship>,
    ) -> Result<Vec<Relationship>> {
        let mut data = self.data.write().await;

        // Validate all relationships first
        for rel in &relationships {
            if !data.entities.contains_key(&rel.source_entity_id) {
                return Err(StorageError::NotFound(format!(
                    "Source entity not found: {}",
                    rel.source_entity_id
                ))
                .into());
            }
            if !data.entities.contains_key(&rel.target_entity_id) {
                return Err(StorageError::NotFound(format!(
                    "Target entity not found: {}",
                    rel.target_entity_id
                ))
                .into());
            }
        }

        let mut created = Vec::with_capacity(relationships.len());

        for rel in relationships {
            data.index_relationship(&rel);
            data.relationships.insert(rel.id.clone(), rel.clone());
            created.push(rel);
        }

        drop(data);
        self.persist().await?;
        Ok(created)
    }

    async fn delete_by_document(&self, document_id: &str) -> Result<usize> {
        let entity_ids: Vec<String> = {
            let data = self.data.read().await;
            data.entities_by_doc
                .get(document_id)
                .cloned()
                .unwrap_or_default()
        };

        let mut deleted = 0;
        for id in entity_ids {
            if self.delete_entity(&id).await? {
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    // ========================================================================
    // Deduplication and Merging
    // ========================================================================

    async fn find_duplicates(
        &self,
        entity_type: EntityType,
        similarity_threshold: f32,
    ) -> Result<Vec<(Entity, Entity, f32)>> {
        let entities = self.find_entities_by_type(entity_type, 10000).await?;
        let mut duplicates = Vec::new();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let e1 = &entities[i];
                let e2 = &entities[j];

                // Check embedding similarity if both have embeddings
                if let (Some(emb1), Some(emb2)) = (&e1.embedding, &e2.embedding) {
                    let sim = Self::cosine_similarity(emb1, emb2);
                    if sim >= similarity_threshold {
                        duplicates.push((e1.clone(), e2.clone(), sim));
                    }
                }

                // Check name similarity
                let name_sim =
                    strsim::jaro_winkler(&e1.name.to_lowercase(), &e2.name.to_lowercase()) as f32;
                if name_sim >= similarity_threshold {
                    // Avoid duplicates if already added via embedding
                    let already_added = duplicates
                        .iter()
                        .any(|(a, b, _)| a.id == e1.id && b.id == e2.id);
                    if !already_added {
                        duplicates.push((e1.clone(), e2.clone(), name_sim));
                    }
                }
            }
        }

        // Sort by similarity (highest first)
        duplicates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        Ok(duplicates)
    }

    async fn merge_entities(&self, keep_id: &str, merge_id: &str) -> Result<Entity> {
        // Get both entities
        let keep_entity = self
            .get_entity(keep_id)
            .await?
            .ok_or_else(|| StorageError::NotFound(format!("Entity not found: {}", keep_id)))?;

        let merge_entity = self
            .get_entity(merge_id)
            .await?
            .ok_or_else(|| StorageError::NotFound(format!("Entity not found: {}", merge_id)))?;

        // Build update with merged data
        let mut update = EntityUpdate::default();

        // Add aliases from merge entity
        update.add_aliases.push(merge_entity.name.clone());
        for alias in &merge_entity.aliases {
            if !keep_entity.aliases.contains(alias) && alias != &keep_entity.name {
                update.add_aliases.push(alias.clone());
            }
        }

        // Add source refs from merge entity
        update.add_source_refs = merge_entity.source_refs.clone();

        // Merge metadata (merge entity values don't overwrite)
        for (key, value) in merge_entity.metadata {
            if !keep_entity.metadata.contains_key(&key) {
                update.set_metadata.insert(key, value);
            }
        }

        // Apply update to keep entity
        let updated = self.update_entity(keep_id, update).await?;

        // Redirect all relationships from merge entity to keep entity
        let merge_rels = self.get_relationships_involving(merge_id).await?;
        for rel in merge_rels {
            // Create new relationship with redirected IDs
            let new_source = if rel.source_entity_id == merge_id {
                keep_id.to_string()
            } else {
                rel.source_entity_id.clone()
            };
            let new_target = if rel.target_entity_id == merge_id {
                keep_id.to_string()
            } else {
                rel.target_entity_id.clone()
            };

            // Skip self-referential relationships
            if new_source != new_target {
                let new_rel = Relationship::new(new_source, rel.relationship_type, new_target)
                    .with_confidence(rel.confidence);
                let _ = self.create_relationship(new_rel).await; // Ignore if exists
            }

            // Delete old relationship
            self.delete_relationship(&rel.id).await?;
        }

        // Delete merge entity
        self.delete_entity(merge_id).await?;

        Ok(updated)
    }

    // ========================================================================
    // Statistics and Maintenance
    // ========================================================================

    async fn stats(&self) -> Result<OntologyStats> {
        let data = self.data.read().await;

        let entity_count = data.entities.len();
        let relationship_count = data.relationships.len();

        // Count by type
        let mut entities_by_type: HashMap<String, usize> = HashMap::new();
        let mut total_entity_confidence: f32 = 0.0;
        let mut entities_with_embeddings = 0;

        for entity in data.entities.values() {
            *entities_by_type
                .entry(format!("{:?}", entity.entity_type))
                .or_default() += 1;
            total_entity_confidence += entity.confidence;
            if entity.embedding.is_some() {
                entities_with_embeddings += 1;
            }
        }

        let mut relationships_by_type: HashMap<String, usize> = HashMap::new();
        let mut total_relationship_confidence: f32 = 0.0;

        for rel in data.relationships.values() {
            *relationships_by_type
                .entry(format!("{:?}", rel.relationship_type))
                .or_default() += 1;
            total_relationship_confidence += rel.confidence;
        }

        let average_entity_confidence = if entity_count > 0 {
            total_entity_confidence / entity_count as f32
        } else {
            0.0
        };

        let average_relationship_confidence = if relationship_count > 0 {
            total_relationship_confidence / relationship_count as f32
        } else {
            0.0
        };

        Ok(OntologyStats {
            entity_count,
            entities_by_type,
            relationship_count,
            relationships_by_type,
            entities_with_embeddings,
            average_entity_confidence,
            average_relationship_confidence,
        })
    }

    async fn clear(&self) -> Result<()> {
        let mut data = self.data.write().await;
        *data = OntologyData::default();
        drop(data);

        self.persist().await?;
        Ok(())
    }
}

// ============================================================================
// Persistence Data Structure
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct PersistenceData {
    version: u32,
    entities: Vec<Entity>,
    relationships: Vec<Relationship>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_store() -> EmbeddedOntologyStore {
        EmbeddedOntologyStore::new()
    }

    #[tokio::test]
    async fn test_create_and_get_entity() {
        let store = create_test_store().await;

        let entity = Entity::new(EntityType::Person, "John Doe");
        let created = store.create_entity(entity.clone()).await.unwrap();

        assert_eq!(created.name, "John Doe");
        assert_eq!(created.entity_type, EntityType::Person);

        let retrieved = store.get_entity(&created.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "John Doe");
    }

    #[tokio::test]
    async fn test_update_entity() {
        let store = create_test_store().await;

        let entity = Entity::new(EntityType::Person, "John");
        let created = store.create_entity(entity).await.unwrap();

        let update = EntityUpdate::name("John Doe").add_alias("Johnny");
        let updated = store.update_entity(&created.id, update).await.unwrap();

        assert_eq!(updated.name, "John Doe");
        assert!(updated.aliases.contains(&"Johnny".to_string()));
    }

    #[tokio::test]
    async fn test_delete_entity() {
        let store = create_test_store().await;

        let entity = Entity::new(EntityType::Person, "John Doe");
        let created = store.create_entity(entity).await.unwrap();

        let deleted = store.delete_entity(&created.id).await.unwrap();
        assert!(deleted);

        let retrieved = store.get_entity(&created.id).await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_list_entities_with_filter() {
        let store = create_test_store().await;

        store
            .create_entity(Entity::new(EntityType::Person, "John"))
            .await
            .unwrap();
        store
            .create_entity(Entity::new(EntityType::Person, "Jane"))
            .await
            .unwrap();
        store
            .create_entity(Entity::new(EntityType::Organization, "Acme"))
            .await
            .unwrap();

        let filter = EntityFilter::by_types([EntityType::Person]);
        let results = store.list_entities(filter).await.unwrap();
        assert_eq!(results.len(), 2);

        let filter = EntityFilter::by_name("john");
        let results = store.list_entities(filter).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_create_and_get_relationship() {
        let store = create_test_store().await;

        let person = store
            .create_entity(Entity::new(EntityType::Person, "John"))
            .await
            .unwrap();
        let org = store
            .create_entity(Entity::new(EntityType::Organization, "Acme"))
            .await
            .unwrap();

        let rel = Relationship::new(&person.id, RelationType::WorksFor, &org.id);
        let created = store.create_relationship(rel).await.unwrap();

        assert_eq!(created.source_entity_id, person.id);
        assert_eq!(created.target_entity_id, org.id);
        assert_eq!(created.relationship_type, RelationType::WorksFor);

        let retrieved = store.get_relationship(&created.id).await.unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_get_relationships_from() {
        let store = create_test_store().await;

        let person = store
            .create_entity(Entity::new(EntityType::Person, "John"))
            .await
            .unwrap();
        let org1 = store
            .create_entity(Entity::new(EntityType::Organization, "Acme"))
            .await
            .unwrap();
        let org2 = store
            .create_entity(Entity::new(EntityType::Organization, "Beta"))
            .await
            .unwrap();

        store
            .create_relationship(Relationship::new(
                &person.id,
                RelationType::WorksFor,
                &org1.id,
            ))
            .await
            .unwrap();
        store
            .create_relationship(Relationship::new(
                &person.id,
                RelationType::WorksFor,
                &org2.id,
            ))
            .await
            .unwrap();

        let rels = store.get_relationships_from(&person.id).await.unwrap();
        assert_eq!(rels.len(), 2);
    }

    #[tokio::test]
    async fn test_delete_entity_cascades_relationships() {
        let store = create_test_store().await;

        let person = store
            .create_entity(Entity::new(EntityType::Person, "John"))
            .await
            .unwrap();
        let org = store
            .create_entity(Entity::new(EntityType::Organization, "Acme"))
            .await
            .unwrap();

        let rel = store
            .create_relationship(Relationship::new(
                &person.id,
                RelationType::WorksFor,
                &org.id,
            ))
            .await
            .unwrap();

        // Delete person should cascade to relationships
        store.delete_entity(&person.id).await.unwrap();

        let retrieved_rel = store.get_relationship(&rel.id).await.unwrap();
        assert!(retrieved_rel.is_none());
    }

    #[tokio::test]
    async fn test_get_connected_entities() {
        let store = create_test_store().await;

        let john = store
            .create_entity(Entity::new(EntityType::Person, "John"))
            .await
            .unwrap();
        let acme = store
            .create_entity(Entity::new(EntityType::Organization, "Acme"))
            .await
            .unwrap();
        let project = store
            .create_entity(Entity::new(EntityType::Project, "Website"))
            .await
            .unwrap();

        store
            .create_relationship(Relationship::new(
                &john.id,
                RelationType::WorksFor,
                &acme.id,
            ))
            .await
            .unwrap();
        store
            .create_relationship(Relationship::new(
                &john.id,
                RelationType::BelongsToProject,
                &project.id,
            ))
            .await
            .unwrap();

        let connected = store
            .get_connected_entities(&john.id, None, 2)
            .await
            .unwrap();
        assert_eq!(connected.len(), 2);
    }

    #[tokio::test]
    async fn test_find_paths() {
        let store = create_test_store().await;

        let a = store
            .create_entity(Entity::new(EntityType::Person, "A"))
            .await
            .unwrap();
        let b = store
            .create_entity(Entity::new(EntityType::Person, "B"))
            .await
            .unwrap();
        let c = store
            .create_entity(Entity::new(EntityType::Person, "C"))
            .await
            .unwrap();

        store
            .create_relationship(Relationship::new(&a.id, RelationType::RelatedTo, &b.id))
            .await
            .unwrap();
        store
            .create_relationship(Relationship::new(&b.id, RelationType::RelatedTo, &c.id))
            .await
            .unwrap();

        let paths = store.find_paths(&a.id, &c.id, 3).await.unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 2);
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let store = create_test_store().await;

        let entities = vec![
            Entity::new(EntityType::Person, "John"),
            Entity::new(EntityType::Person, "Jane"),
            Entity::new(EntityType::Person, "Bob"),
        ];

        let created = store.create_entities_batch(entities).await.unwrap();
        assert_eq!(created.len(), 3);

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.entity_count, 3);
    }

    #[tokio::test]
    async fn test_merge_entities() {
        let store = create_test_store().await;

        let e1 = store
            .create_entity(
                Entity::new(EntityType::Person, "John Doe")
                    .with_alias("John")
                    .with_metadata("email", serde_json::json!("john@example.com")),
            )
            .await
            .unwrap();

        let e2 = store
            .create_entity(
                Entity::new(EntityType::Person, "Johnny Doe")
                    .with_alias("Johnny")
                    .with_metadata("phone", serde_json::json!("555-1234")),
            )
            .await
            .unwrap();

        let org = store
            .create_entity(Entity::new(EntityType::Organization, "Acme"))
            .await
            .unwrap();

        // Create relationship with e2
        store
            .create_relationship(Relationship::new(&e2.id, RelationType::WorksFor, &org.id))
            .await
            .unwrap();

        // Merge e2 into e1
        let merged = store.merge_entities(&e1.id, &e2.id).await.unwrap();

        // Check merged entity has combined aliases
        assert!(merged.aliases.contains(&"John".to_string()));
        assert!(merged.aliases.contains(&"Johnny".to_string()));
        assert!(merged.aliases.contains(&"Johnny Doe".to_string()));

        // Check merged entity has combined metadata
        assert!(merged.metadata.contains_key("email"));
        assert!(merged.metadata.contains_key("phone"));

        // Check e2 no longer exists
        assert!(store.get_entity(&e2.id).await.unwrap().is_none());

        // Check relationship was redirected
        let rels = store.get_relationships_from(&e1.id).await.unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].target_entity_id, org.id);
    }

    #[tokio::test]
    async fn test_stats() {
        let store = create_test_store().await;

        store
            .create_entity(Entity::new(EntityType::Person, "John").with_confidence(0.9))
            .await
            .unwrap();
        store
            .create_entity(Entity::new(EntityType::Person, "Jane").with_confidence(0.8))
            .await
            .unwrap();
        store
            .create_entity(Entity::new(EntityType::Organization, "Acme").with_confidence(1.0))
            .await
            .unwrap();

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.entity_count, 3);
        assert_eq!(stats.entities_by_type.get("Person"), Some(&2));
        assert_eq!(stats.entities_by_type.get("Organization"), Some(&1));
        assert!((stats.average_entity_confidence - 0.9).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_clear() {
        let store = create_test_store().await;

        store
            .create_entity(Entity::new(EntityType::Person, "John"))
            .await
            .unwrap();
        store
            .create_entity(Entity::new(EntityType::Person, "Jane"))
            .await
            .unwrap();

        store.clear().await.unwrap();

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.relationship_count, 0);
    }

    #[tokio::test]
    async fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();

        // Create store and add data
        {
            let store = EmbeddedOntologyStore::with_persistence(temp_dir.path())
                .await
                .unwrap();
            store
                .create_entity(Entity::with_id("test-id", EntityType::Person, "John Doe"))
                .await
                .unwrap();
        }

        // Create new store from same path and verify data persisted
        {
            let store = EmbeddedOntologyStore::with_persistence(temp_dir.path())
                .await
                .unwrap();
            let entity = store.get_entity("test-id").await.unwrap();
            assert!(entity.is_some());
            assert_eq!(entity.unwrap().name, "John Doe");
        }
    }

    #[tokio::test]
    async fn test_search_by_embedding() {
        let store = create_test_store().await;

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.9, 0.1, 0.0];
        let vec3 = vec![0.0, 1.0, 0.0];

        store
            .create_entity(
                Entity::new(EntityType::Topic, "Machine Learning").with_embedding(vec1.clone()),
            )
            .await
            .unwrap();
        store
            .create_entity(Entity::new(EntityType::Topic, "Deep Learning").with_embedding(vec2))
            .await
            .unwrap();
        store
            .create_entity(Entity::new(EntityType::Topic, "Cooking").with_embedding(vec3))
            .await
            .unwrap();

        let results = store
            .search_entities_by_embedding(&vec1, None, 10)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.name, "Machine Learning"); // Exact match first
        assert_eq!(results[1].0.name, "Deep Learning"); // Similar second
        assert!(results[0].1 > results[1].1); // Higher similarity
    }
}

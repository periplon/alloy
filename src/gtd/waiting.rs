//! Waiting For tracking for GTD.
//!
//! This module manages items that are delegated to others.

use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{WaitingFilter, WaitingFor, WaitingStatus};
use crate::ontology::{
    EmbeddedOntologyStore, Entity, EntityFilter, EntityType, EntityUpdate, OntologyStore,
    RelationType, Relationship,
};

// ============================================================================
// Waiting Manager
// ============================================================================

/// Manages waiting-for items using the ontology store.
pub struct WaitingManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
}

impl WaitingManager {
    /// Create a new waiting manager with the given ontology store.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self { store }
    }

    /// Create a new waiting-for item.
    pub async fn create(&self, item: WaitingFor) -> Result<WaitingFor> {
        let store = self.store.read().await;

        // Create entity for the waiting item
        let mut entity = Entity::new(EntityType::WaitingFor, &item.description)
            .with_metadata("status", serde_json::json!(item.status))
            .with_metadata("delegated_to", serde_json::json!(item.delegated_to))
            .with_metadata("waiting_data", serde_json::to_value(&item)?);

        // Use the item ID
        entity.id = item.id.clone();

        store.create_entity(entity).await?;

        // Create relationship to person
        self.link_to_person(&store, &item.id, &item.delegated_to)
            .await?;

        // Create relationship to project if specified
        if let Some(ref project_id) = item.project_id {
            let rel = Relationship::new(&item.id, RelationType::BelongsToProject, project_id);
            let _ = store.create_relationship(rel).await;
        }

        Ok(item)
    }

    /// Get a waiting-for item by ID.
    pub async fn get(&self, id: &str) -> Result<Option<WaitingFor>> {
        let store = self.store.read().await;
        let entity = store.get_entity(id).await?;

        match entity {
            Some(e) if e.entity_type == EntityType::WaitingFor => Self::entity_to_waiting(&e),
            _ => Ok(None),
        }
    }

    /// Update a waiting-for item.
    pub async fn update(&self, id: &str, mut item: WaitingFor) -> Result<WaitingFor> {
        let store = self.store.read().await;

        item.updated_at = Utc::now();

        let update = EntityUpdate::name(&item.description)
            .set_meta("status", serde_json::json!(item.status))
            .set_meta("delegated_to", serde_json::json!(item.delegated_to))
            .set_meta("waiting_data", serde_json::to_value(&item)?);

        store.update_entity(id, update).await?;

        Ok(item)
    }

    /// Delete a waiting-for item.
    pub async fn delete(&self, id: &str) -> Result<bool> {
        let store = self.store.read().await;
        store.delete_entity(id).await
    }

    /// List waiting-for items matching the filter.
    pub async fn list(&self, filter: WaitingFilter) -> Result<Vec<WaitingFor>> {
        let store = self.store.read().await;

        let entity_filter = EntityFilter::by_types([EntityType::WaitingFor])
            .with_limit(filter.limit * 2)
            .with_offset(filter.offset);

        let entities = store.list_entities(entity_filter).await?;

        let mut items = Vec::new();
        for entity in entities {
            if let Ok(Some(mut item)) = Self::entity_to_waiting(&entity) {
                // Update overdue status
                if item.is_overdue() && item.status == WaitingStatus::Pending {
                    item.status = WaitingStatus::Overdue;
                }

                // Apply filters
                if self.item_matches_filter(&item, &filter) {
                    items.push(item);
                    if items.len() >= filter.limit {
                        break;
                    }
                }
            }
        }

        Ok(items)
    }

    /// Record a follow-up on a waiting item.
    pub async fn record_follow_up(&self, id: &str) -> Result<Option<WaitingFor>> {
        let item = match self.get(id).await? {
            Some(mut i) => {
                i.record_follow_up();
                i
            }
            None => return Ok(None),
        };

        self.update(id, item.clone()).await?;
        Ok(Some(item))
    }

    /// Resolve a waiting-for item.
    pub async fn resolve(&self, id: &str, resolution: &str) -> Result<Option<WaitingFor>> {
        let item = match self.get(id).await? {
            Some(mut i) => {
                i.resolve(resolution);
                i
            }
            None => return Ok(None),
        };

        self.update(id, item.clone()).await?;
        Ok(Some(item))
    }

    /// Get all overdue waiting items.
    pub async fn get_overdue(&self) -> Result<Vec<WaitingFor>> {
        self.list(WaitingFilter {
            overdue_only: true,
            ..Default::default()
        })
        .await
    }

    /// Get waiting items for a specific person.
    pub async fn get_by_person(&self, person: &str) -> Result<Vec<WaitingFor>> {
        self.list(WaitingFilter {
            delegated_to: Some(person.to_string()),
            status: Some(WaitingStatus::Pending),
            ..Default::default()
        })
        .await
    }

    /// Get waiting items for a specific project.
    pub async fn get_by_project(&self, project_id: &str) -> Result<Vec<WaitingFor>> {
        self.list(WaitingFilter {
            project_id: Some(project_id.to_string()),
            ..Default::default()
        })
        .await
    }

    /// Get all pending waiting items that need follow-up.
    pub async fn get_needing_follow_up(&self, days_since_last: i64) -> Result<Vec<WaitingFor>> {
        let items = self
            .list(WaitingFilter {
                status: Some(WaitingStatus::Pending),
                ..Default::default()
            })
            .await?;

        let threshold = Utc::now() - chrono::Duration::days(days_since_last);

        Ok(items
            .into_iter()
            .filter(|item| {
                let last_contact = item
                    .follow_up_dates
                    .last()
                    .copied()
                    .unwrap_or(item.delegated_date);
                last_contact < threshold
            })
            .collect())
    }

    // Helper methods

    fn item_matches_filter(&self, item: &WaitingFor, filter: &WaitingFilter) -> bool {
        // Filter by delegated to
        if let Some(ref person) = filter.delegated_to {
            if &item.delegated_to != person {
                return false;
            }
        }

        // Filter by project
        if let Some(ref project_id) = filter.project_id {
            if item.project_id.as_ref() != Some(project_id) {
                return false;
            }
        }

        // Filter by status
        if let Some(status) = filter.status {
            if item.status != status {
                return false;
            }
        }

        // Filter overdue only
        if filter.overdue_only && !item.is_overdue() && item.status != WaitingStatus::Overdue {
            return false;
        }

        true
    }

    async fn link_to_person(
        &self,
        store: &EmbeddedOntologyStore,
        waiting_id: &str,
        person_name: &str,
    ) -> Result<()> {
        // Find or create the person entity
        let person_entities = store.find_entities_by_name(person_name, 1).await?;
        let person_id = if let Some(person) = person_entities.first() {
            person.id.clone()
        } else {
            let person_entity = Entity::new(EntityType::Person, person_name);
            let created = store.create_entity(person_entity).await?;
            created.id
        };

        // Create relationship
        let relationship = Relationship::new(waiting_id, RelationType::WaitingOn, &person_id);
        store.create_relationship(relationship).await?;
        Ok(())
    }

    fn entity_to_waiting(entity: &Entity) -> Result<Option<WaitingFor>> {
        if entity.entity_type != EntityType::WaitingFor {
            return Ok(None);
        }

        // Try to get item from metadata
        if let Some(waiting_data) = entity.metadata.get("waiting_data") {
            let item: WaitingFor = serde_json::from_value(waiting_data.clone())?;
            return Ok(Some(item));
        }

        // Fallback: construct from entity fields
        let status = entity
            .metadata
            .get("status")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(WaitingStatus::Pending);

        let delegated_to = entity
            .metadata
            .get("delegated_to")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let item = WaitingFor {
            id: entity.id.clone(),
            description: entity.name.clone(),
            delegated_to,
            project_id: None,
            delegated_date: entity.created_at,
            expected_by: None,
            follow_up_dates: Vec::new(),
            status,
            resolution: None,
            source_document_id: None,
            created_at: entity.created_at,
            updated_at: entity.updated_at,
            metadata: entity.metadata.clone(),
        };

        Ok(Some(item))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    async fn create_test_manager() -> WaitingManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        WaitingManager::new(store)
    }

    #[tokio::test]
    async fn test_create_and_get_waiting() {
        let manager = create_test_manager().await;

        let item = WaitingFor::new("Response on proposal", "John");

        let created = manager.create(item.clone()).await.unwrap();
        assert_eq!(created.description, "Response on proposal");

        let retrieved = manager.get(&created.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().delegated_to, "John");
    }

    #[tokio::test]
    async fn test_list_waiting() {
        let manager = create_test_manager().await;

        manager
            .create(WaitingFor::new("Item 1", "John"))
            .await
            .unwrap();
        manager
            .create(WaitingFor::new("Item 2", "Sarah"))
            .await
            .unwrap();

        let items = manager.list(WaitingFilter::default()).await.unwrap();
        assert_eq!(items.len(), 2);
    }

    #[tokio::test]
    async fn test_filter_by_person() {
        let manager = create_test_manager().await;

        manager
            .create(WaitingFor::new("John's item", "John"))
            .await
            .unwrap();
        manager
            .create(WaitingFor::new("Sarah's item", "Sarah"))
            .await
            .unwrap();

        let johns = manager.get_by_person("John").await.unwrap();
        assert_eq!(johns.len(), 1);
        assert_eq!(johns[0].description, "John's item");
    }

    #[tokio::test]
    async fn test_record_follow_up() {
        let manager = create_test_manager().await;

        let item = manager
            .create(WaitingFor::new("Test", "Person"))
            .await
            .unwrap();
        assert!(item.follow_up_dates.is_empty());

        let updated = manager.record_follow_up(&item.id).await.unwrap().unwrap();
        assert_eq!(updated.follow_up_dates.len(), 1);
    }

    #[tokio::test]
    async fn test_resolve_waiting() {
        let manager = create_test_manager().await;

        let item = manager
            .create(WaitingFor::new("Test", "Person"))
            .await
            .unwrap();

        let resolved = manager
            .resolve(&item.id, "Got the response!")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.status, WaitingStatus::Resolved);
        assert_eq!(resolved.resolution, Some("Got the response!".to_string()));
    }

    #[tokio::test]
    async fn test_overdue_detection() {
        let manager = create_test_manager().await;

        // Create an overdue item
        let mut item = WaitingFor::new("Overdue item", "John");
        item.expected_by = Some(Utc::now() - Duration::days(1));
        manager.create(item).await.unwrap();

        // Create a non-overdue item
        let mut item2 = WaitingFor::new("Future item", "Sarah");
        item2.expected_by = Some(Utc::now() + Duration::days(7));
        manager.create(item2).await.unwrap();

        let overdue = manager.get_overdue().await.unwrap();
        assert_eq!(overdue.len(), 1);
        assert_eq!(overdue[0].description, "Overdue item");
    }
}

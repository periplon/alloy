//! Someday/Maybe management for GTD.
//!
//! This module manages deferred items for future consideration.

use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{SomedayFilter, SomedayItem, Task, TaskStatus};
use crate::ontology::{
    EmbeddedOntologyStore, Entity, EntityFilter, EntityType, EntityUpdate, OntologyStore,
};

// ============================================================================
// Someday Manager
// ============================================================================

/// Manages someday/maybe items using the ontology store.
pub struct SomedayManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
}

impl SomedayManager {
    /// Create a new someday manager with the given ontology store.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self { store }
    }

    /// Create a new someday/maybe item.
    pub async fn create(&self, item: SomedayItem) -> Result<SomedayItem> {
        let store = self.store.read().await;

        // Create entity for the someday item
        let mut entity = Entity::new(EntityType::SomedayMaybe, &item.description)
            .with_metadata("someday_data", serde_json::to_value(&item)?);

        if let Some(ref category) = item.category {
            entity = entity.with_metadata("category", serde_json::json!(category));
        }

        if let Some(ref trigger) = item.trigger {
            entity = entity.with_metadata("trigger", serde_json::json!(trigger));
        }

        // Use the item ID
        entity.id = item.id.clone();

        store.create_entity(entity).await?;

        Ok(item)
    }

    /// Get a someday/maybe item by ID.
    pub async fn get(&self, id: &str) -> Result<Option<SomedayItem>> {
        let store = self.store.read().await;
        let entity = store.get_entity(id).await?;

        match entity {
            Some(e) if e.entity_type == EntityType::SomedayMaybe => Self::entity_to_someday(&e),
            _ => Ok(None),
        }
    }

    /// Update a someday/maybe item.
    pub async fn update(&self, id: &str, mut item: SomedayItem) -> Result<SomedayItem> {
        let store = self.store.read().await;

        item.updated_at = Utc::now();

        let mut update = EntityUpdate::name(&item.description)
            .set_meta("someday_data", serde_json::to_value(&item)?);

        if let Some(ref category) = item.category {
            update = update.set_meta("category", serde_json::json!(category));
        }

        if let Some(ref trigger) = item.trigger {
            update = update.set_meta("trigger", serde_json::json!(trigger));
        }

        store.update_entity(id, update).await?;

        Ok(item)
    }

    /// Delete a someday/maybe item.
    pub async fn delete(&self, id: &str) -> Result<bool> {
        let store = self.store.read().await;
        store.delete_entity(id).await
    }

    /// List someday/maybe items matching the filter.
    pub async fn list(&self, filter: SomedayFilter) -> Result<Vec<SomedayItem>> {
        let store = self.store.read().await;

        let entity_filter = EntityFilter::by_types([EntityType::SomedayMaybe])
            .with_limit(filter.limit * 2)
            .with_offset(filter.offset);

        let entities = store.list_entities(entity_filter).await?;

        let mut items = Vec::new();
        for entity in entities {
            if let Ok(Some(item)) = Self::entity_to_someday(&entity) {
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

    /// Activate a someday item by converting it to a task.
    pub async fn activate(&self, id: &str) -> Result<Option<Task>> {
        let item = match self.get(id).await? {
            Some(i) => i,
            None => return Ok(None),
        };

        let store = self.store.read().await;

        // Create a new task from the someday item
        let task = Task::new(&item.description).with_status(TaskStatus::Next);

        // Create task entity
        let mut entity = Entity::new(EntityType::Task, &task.description)
            .with_metadata("status", serde_json::json!(task.status))
            .with_metadata("task_data", serde_json::to_value(&task)?)
            .with_metadata("activated_from_someday", serde_json::json!(id));

        entity.id = task.id.clone();
        store.create_entity(entity).await?;

        // Delete the someday item
        store.delete_entity(id).await?;

        Ok(Some(task))
    }

    /// Get all items due for review.
    pub async fn get_due_for_review(&self) -> Result<Vec<SomedayItem>> {
        self.list(SomedayFilter {
            due_for_review: true,
            ..Default::default()
        })
        .await
    }

    /// Get items by category.
    pub async fn get_by_category(&self, category: &str) -> Result<Vec<SomedayItem>> {
        self.list(SomedayFilter {
            category: Some(category.to_string()),
            ..Default::default()
        })
        .await
    }

    /// Get all unique categories.
    pub async fn get_categories(&self) -> Result<Vec<String>> {
        let items = self.list(SomedayFilter::default()).await?;

        let mut categories: Vec<String> = items.into_iter().filter_map(|i| i.category).collect();

        categories.sort();
        categories.dedup();

        Ok(categories)
    }

    // Helper methods

    fn item_matches_filter(&self, item: &SomedayItem, filter: &SomedayFilter) -> bool {
        // Filter by category
        if let Some(ref category) = filter.category {
            if item.category.as_ref() != Some(category) {
                return false;
            }
        }

        // Filter due for review
        if filter.due_for_review && !item.is_due_for_review() {
            return false;
        }

        true
    }

    fn entity_to_someday(entity: &Entity) -> Result<Option<SomedayItem>> {
        if entity.entity_type != EntityType::SomedayMaybe {
            return Ok(None);
        }

        // Try to get item from metadata
        if let Some(someday_data) = entity.metadata.get("someday_data") {
            let item: SomedayItem = serde_json::from_value(someday_data.clone())?;
            return Ok(Some(item));
        }

        // Fallback: construct from entity fields
        let category = entity
            .metadata
            .get("category")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let trigger = entity
            .metadata
            .get("trigger")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let item = SomedayItem {
            id: entity.id.clone(),
            description: entity.name.clone(),
            category,
            trigger,
            review_date: None,
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

    async fn create_test_manager() -> SomedayManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        SomedayManager::new(store)
    }

    #[tokio::test]
    async fn test_create_and_get_someday() {
        let manager = create_test_manager().await;

        let item = SomedayItem::new("Learn Rust").with_category("Learning");

        let created = manager.create(item.clone()).await.unwrap();
        assert_eq!(created.description, "Learn Rust");

        let retrieved = manager.get(&created.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().category, Some("Learning".to_string()));
    }

    #[tokio::test]
    async fn test_list_someday() {
        let manager = create_test_manager().await;

        manager
            .create(SomedayItem::new("Item 1").with_category("Travel"))
            .await
            .unwrap();
        manager
            .create(SomedayItem::new("Item 2").with_category("Learning"))
            .await
            .unwrap();

        let items = manager.list(SomedayFilter::default()).await.unwrap();
        assert_eq!(items.len(), 2);
    }

    #[tokio::test]
    async fn test_filter_by_category() {
        let manager = create_test_manager().await;

        manager
            .create(SomedayItem::new("Trip to Japan").with_category("Travel"))
            .await
            .unwrap();
        manager
            .create(SomedayItem::new("Learn Piano").with_category("Learning"))
            .await
            .unwrap();

        let travel = manager.get_by_category("Travel").await.unwrap();
        assert_eq!(travel.len(), 1);
        assert_eq!(travel[0].description, "Trip to Japan");
    }

    #[tokio::test]
    async fn test_get_categories() {
        let manager = create_test_manager().await;

        manager
            .create(SomedayItem::new("Item 1").with_category("Travel"))
            .await
            .unwrap();
        manager
            .create(SomedayItem::new("Item 2").with_category("Learning"))
            .await
            .unwrap();
        manager
            .create(SomedayItem::new("Item 3").with_category("Travel"))
            .await
            .unwrap();

        let categories = manager.get_categories().await.unwrap();
        assert_eq!(categories.len(), 2);
        assert!(categories.contains(&"Travel".to_string()));
        assert!(categories.contains(&"Learning".to_string()));
    }

    #[tokio::test]
    async fn test_activate_someday() {
        let manager = create_test_manager().await;

        let item = manager
            .create(SomedayItem::new("Do something"))
            .await
            .unwrap();

        let task = manager.activate(&item.id).await.unwrap();
        assert!(task.is_some());
        assert_eq!(task.unwrap().description, "Do something");

        // Original item should be deleted
        let retrieved = manager.get(&item.id).await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_due_for_review() {
        let manager = create_test_manager().await;

        // Item due for review
        let mut due_item = SomedayItem::new("Review me");
        due_item.review_date = Some(Utc::now() - Duration::days(1));
        manager.create(due_item).await.unwrap();

        // Item not due yet
        let mut future_item = SomedayItem::new("Not yet");
        future_item.review_date = Some(Utc::now() + Duration::days(30));
        manager.create(future_item).await.unwrap();

        let due = manager.get_due_for_review().await.unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].description, "Review me");
    }
}

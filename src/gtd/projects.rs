//! Project management for GTD.
//!
//! This module provides project CRUD operations and health scoring.

use chrono::{Duration, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{Project, ProjectFilter, ProjectHealth, ProjectStatus, Task};
use crate::ontology::{
    EmbeddedOntologyStore, Entity, EntityFilter, EntityType, EntityUpdate, OntologyStore,
    RelationType, Relationship,
};

// ============================================================================
// Project Manager
// ============================================================================

/// Manages GTD projects using the ontology store.
pub struct ProjectManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
}

impl ProjectManager {
    /// Create a new project manager with the given ontology store.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self { store }
    }

    /// Create a new project.
    pub async fn create(&self, project: Project) -> Result<Project> {
        let store = self.store.read().await;

        // Create entity for the project
        let mut entity = Entity::new(EntityType::Project, &project.name)
            .with_metadata("status", serde_json::json!(project.status))
            .with_metadata("project_data", serde_json::to_value(&project)?);

        if let Some(ref outcome) = project.outcome {
            entity = entity.with_metadata("outcome", serde_json::json!(outcome));
        }

        // Use the project ID
        entity.id = project.id.clone();

        store.create_entity(entity).await?;

        // Create relationships
        if let Some(ref area) = project.area {
            self.link_to_area(&store, &project.id, area).await?;
        }

        if let Some(ref goal) = project.supporting_goal {
            self.link_to_goal(&store, &project.id, goal).await?;
        }

        Ok(project)
    }

    /// Get a project by ID.
    pub async fn get(&self, id: &str) -> Result<Option<Project>> {
        let store = self.store.read().await;
        let entity = store.get_entity(id).await?;

        match entity {
            Some(e) if e.entity_type == EntityType::Project => Self::entity_to_project(&e),
            _ => Ok(None),
        }
    }

    /// Update a project.
    pub async fn update(&self, id: &str, project: Project) -> Result<Project> {
        let store = self.store.read().await;

        let update = EntityUpdate::name(&project.name)
            .set_meta("status", serde_json::json!(project.status))
            .set_meta("project_data", serde_json::to_value(&project)?);

        store.update_entity(id, update).await?;

        Ok(project)
    }

    /// Delete a project.
    pub async fn delete(&self, id: &str) -> Result<bool> {
        let store = self.store.read().await;
        store.delete_entity(id).await
    }

    /// List projects matching the filter.
    pub async fn list(&self, filter: ProjectFilter) -> Result<Vec<Project>> {
        let store = self.store.read().await;

        let entity_filter = EntityFilter::by_types([EntityType::Project])
            .with_limit(filter.limit)
            .with_offset(filter.offset);

        let entities = store.list_entities(entity_filter).await?;

        let mut projects = Vec::new();
        for entity in entities {
            if let Ok(Some(project)) = Self::entity_to_project(&entity) {
                // Apply additional filters
                if let Some(status) = filter.status {
                    if project.status != status {
                        continue;
                    }
                }

                if let Some(ref area) = filter.area {
                    if project.area.as_ref() != Some(area) {
                        continue;
                    }
                }

                if let Some(has_next) = filter.has_next_action {
                    let has_action = project.next_action_id.is_some();
                    if has_action != has_next {
                        continue;
                    }
                }

                if let Some(stalled_days) = filter.stalled_days {
                    if !project.is_stalled(stalled_days as i64) {
                        continue;
                    }
                }

                projects.push(project);
            }
        }

        Ok(projects)
    }

    /// Get the health of a project.
    pub async fn get_health(&self, id: &str) -> Result<Option<ProjectHealth>> {
        let project = match self.get(id).await? {
            Some(p) => p,
            None => return Ok(None),
        };

        let store = self.store.read().await;

        // Check if has next action
        let has_next_action = project.next_action_id.is_some();

        // Check for recent activity
        let recent_activity = project.updated_at > (Utc::now() - Duration::days(7));

        // Check for clear outcome
        let clear_outcome =
            project.outcome.is_some() && !project.outcome.as_ref().unwrap().is_empty();

        // Check for reasonable scope (max 20 tasks)
        let tasks = self.get_tasks_for_project(&store, id).await?;
        let reasonable_scope = tasks.len() <= 20;

        // Check for blockers
        let no_blockers = !tasks.iter().any(|t| !t.blocked_by.is_empty());

        // Check if linked to goal
        let linked_to_goal = project.supporting_goal.is_some();

        Ok(Some(ProjectHealth::calculate(
            has_next_action,
            recent_activity,
            clear_outcome,
            reasonable_scope,
            no_blockers,
            linked_to_goal,
        )))
    }

    /// Get all stalled projects.
    pub async fn get_stalled(&self, days: u32) -> Result<Vec<Project>> {
        self.list(ProjectFilter {
            status: Some(ProjectStatus::Active),
            stalled_days: Some(days),
            ..Default::default()
        })
        .await
    }

    /// Get projects without a next action.
    pub async fn get_without_next_action(&self) -> Result<Vec<Project>> {
        self.list(ProjectFilter {
            status: Some(ProjectStatus::Active),
            has_next_action: Some(false),
            ..Default::default()
        })
        .await
    }

    /// Archive a project.
    pub async fn archive(&self, id: &str) -> Result<Option<Project>> {
        let project = match self.get(id).await? {
            Some(mut p) => {
                p.status = ProjectStatus::Archived;
                p.updated_at = Utc::now();
                p
            }
            None => return Ok(None),
        };

        self.update(id, project.clone()).await?;
        Ok(Some(project))
    }

    /// Complete a project.
    pub async fn complete(&self, id: &str) -> Result<Option<Project>> {
        let project = match self.get(id).await? {
            Some(mut p) => {
                p.status = ProjectStatus::Completed;
                p.completed_at = Some(Utc::now());
                p.updated_at = Utc::now();
                p
            }
            None => return Ok(None),
        };

        self.update(id, project.clone()).await?;
        Ok(Some(project))
    }

    // Helper methods

    async fn link_to_area(
        &self,
        store: &EmbeddedOntologyStore,
        project_id: &str,
        area: &str,
    ) -> Result<()> {
        // Find or create the area entity
        let area_entities = store.find_entities_by_name(area, 1).await?;
        let area_id = if let Some(area_entity) = area_entities.first() {
            area_entity.id.clone()
        } else {
            let area_entity = Entity::new(EntityType::Area, area);
            let created = store.create_entity(area_entity).await?;
            created.id
        };

        // Create relationship
        let relationship = Relationship::new(project_id, RelationType::InArea, &area_id);
        store.create_relationship(relationship).await?;
        Ok(())
    }

    async fn link_to_goal(
        &self,
        store: &EmbeddedOntologyStore,
        project_id: &str,
        goal: &str,
    ) -> Result<()> {
        // Find or create the goal entity
        let goal_entities = store.find_entities_by_name(goal, 1).await?;
        let goal_id = if let Some(goal_entity) = goal_entities.first() {
            goal_entity.id.clone()
        } else {
            let goal_entity = Entity::new(EntityType::Goal, goal);
            let created = store.create_entity(goal_entity).await?;
            created.id
        };

        // Create relationship
        let relationship = Relationship::new(project_id, RelationType::SupportsGoal, &goal_id);
        store.create_relationship(relationship).await?;
        Ok(())
    }

    async fn get_tasks_for_project(
        &self,
        store: &EmbeddedOntologyStore,
        project_id: &str,
    ) -> Result<Vec<Task>> {
        // Get all relationships where tasks belong to this project
        let relationships = store.get_relationships_to(project_id).await?;

        let mut tasks = Vec::new();
        for rel in relationships {
            if rel.relationship_type == RelationType::BelongsToProject {
                if let Some(entity) = store.get_entity(&rel.source_entity_id).await? {
                    if entity.entity_type == EntityType::Task {
                        if let Some(task_data) = entity.metadata.get("task_data") {
                            if let Ok(task) = serde_json::from_value::<Task>(task_data.clone()) {
                                tasks.push(task);
                            }
                        }
                    }
                }
            }
        }

        Ok(tasks)
    }

    fn entity_to_project(entity: &Entity) -> Result<Option<Project>> {
        if entity.entity_type != EntityType::Project {
            return Ok(None);
        }

        // Try to get project from metadata
        if let Some(project_data) = entity.metadata.get("project_data") {
            let project: Project = serde_json::from_value(project_data.clone())?;
            return Ok(Some(project));
        }

        // Fallback: construct from entity fields
        let status = entity
            .metadata
            .get("status")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(ProjectStatus::Active);

        let outcome = entity
            .metadata
            .get("outcome")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let project = Project {
            id: entity.id.clone(),
            name: entity.name.clone(),
            outcome,
            status,
            area: None,
            supporting_goal: None,
            next_action_id: None,
            reference_doc_ids: Vec::new(),
            health_score: None,
            created_at: entity.created_at,
            updated_at: entity.updated_at,
            completed_at: None,
            metadata: entity.metadata.clone(),
        };

        Ok(Some(project))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_manager() -> ProjectManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        ProjectManager::new(store)
    }

    #[tokio::test]
    async fn test_create_and_get_project() {
        let manager = create_test_manager().await;

        let project = Project::new("Test Project").with_outcome("Complete the test");

        let created = manager.create(project.clone()).await.unwrap();
        assert_eq!(created.name, "Test Project");

        let retrieved = manager.get(&created.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test Project");
    }

    #[tokio::test]
    async fn test_list_projects() {
        let manager = create_test_manager().await;

        manager.create(Project::new("Project 1")).await.unwrap();
        manager.create(Project::new("Project 2")).await.unwrap();

        let projects = manager.list(ProjectFilter::default()).await.unwrap();
        assert_eq!(projects.len(), 2);
    }

    #[tokio::test]
    async fn test_complete_project() {
        let manager = create_test_manager().await;

        let project = manager.create(Project::new("Test")).await.unwrap();
        let completed = manager.complete(&project.id).await.unwrap();

        assert!(completed.is_some());
        assert_eq!(completed.unwrap().status, ProjectStatus::Completed);
    }

    #[tokio::test]
    async fn test_project_health() {
        let manager = create_test_manager().await;

        let project = Project::new("Test").with_outcome("Clear outcome");
        let created = manager.create(project).await.unwrap();

        let health = manager.get_health(&created.id).await.unwrap();
        assert!(health.is_some());

        let h = health.unwrap();
        assert!(h.clear_outcome);
        assert!(!h.has_next_action); // No next action set
    }
}

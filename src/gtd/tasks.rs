//! Task management for GTD.
//!
//! This module provides task CRUD operations and smart recommendations.

use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{
    EnergyLevel, Priority, RecommendParams, Task, TaskFilter, TaskRecommendation, TaskStatus,
};
use crate::ontology::{
    EmbeddedOntologyStore, Entity, EntityFilter, EntityType, EntityUpdate, OntologyStore,
    RelationType, Relationship,
};

// ============================================================================
// Task Manager
// ============================================================================

/// Manages GTD tasks using the ontology store.
pub struct TaskManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    /// Quick task threshold in minutes (2-minute rule).
    quick_task_threshold: u32,
}

impl TaskManager {
    /// Create a new task manager with the given ontology store.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self {
            store,
            quick_task_threshold: 2,
        }
    }

    /// Set the quick task threshold.
    pub fn with_quick_threshold(mut self, minutes: u32) -> Self {
        self.quick_task_threshold = minutes;
        self
    }

    /// Create a new task.
    pub async fn create(&self, task: Task) -> Result<Task> {
        let store = self.store.read().await;

        // Create entity for the task
        let mut entity = Entity::new(EntityType::Task, &task.description)
            .with_metadata("status", serde_json::json!(task.status))
            .with_metadata("task_data", serde_json::to_value(&task)?);

        // Use the task ID
        entity.id = task.id.clone();

        store.create_entity(entity).await?;

        // Create relationships
        if let Some(ref project_id) = task.project_id {
            let rel = Relationship::new(&task.id, RelationType::BelongsToProject, project_id);
            // Ignore errors if project doesn't exist
            let _ = store.create_relationship(rel).await;
        }

        // Create context relationships
        for context in &task.contexts {
            self.link_to_context(&store, &task.id, context).await?;
        }

        Ok(task)
    }

    /// Get a task by ID.
    pub async fn get(&self, id: &str) -> Result<Option<Task>> {
        let store = self.store.read().await;
        let entity = store.get_entity(id).await?;

        match entity {
            Some(e) if e.entity_type == EntityType::Task => Self::entity_to_task(&e),
            _ => Ok(None),
        }
    }

    /// Update a task.
    pub async fn update(&self, id: &str, mut task: Task) -> Result<Task> {
        let store = self.store.read().await;

        task.updated_at = Utc::now();

        let update = EntityUpdate::name(&task.description)
            .set_meta("status", serde_json::json!(task.status))
            .set_meta("task_data", serde_json::to_value(&task)?);

        store.update_entity(id, update).await?;

        Ok(task)
    }

    /// Delete a task.
    pub async fn delete(&self, id: &str) -> Result<bool> {
        let store = self.store.read().await;
        store.delete_entity(id).await
    }

    /// List tasks matching the filter.
    pub async fn list(&self, filter: TaskFilter) -> Result<Vec<Task>> {
        let store = self.store.read().await;

        let entity_filter = EntityFilter::by_types([EntityType::Task])
            .with_limit(filter.limit * 2) // Get more to account for filtering
            .with_offset(filter.offset);

        let entities = store.list_entities(entity_filter).await?;

        let mut tasks = Vec::new();
        for entity in entities {
            if let Ok(Some(task)) = Self::entity_to_task(&entity) {
                if self.task_matches_filter(&task, &filter) {
                    tasks.push(task);
                    if tasks.len() >= filter.limit {
                        break;
                    }
                }
            }
        }

        Ok(tasks)
    }

    /// Complete a task.
    pub async fn complete(&self, id: &str) -> Result<Option<Task>> {
        let task = match self.get(id).await? {
            Some(mut t) => {
                t.complete();
                t
            }
            None => return Ok(None),
        };

        self.update(id, task.clone()).await?;
        Ok(Some(task))
    }

    /// Get recommended tasks based on context and constraints.
    pub async fn recommend(&self, params: RecommendParams) -> Result<Vec<TaskRecommendation>> {
        let store = self.store.read().await;

        // Get all next actions
        let entity_filter = EntityFilter::by_types([EntityType::Task]).with_limit(500);
        let entities = store.list_entities(entity_filter).await?;

        let mut scored_tasks: Vec<(Task, f32, Vec<String>)> = Vec::new();

        for entity in entities {
            if let Ok(Some(task)) = Self::entity_to_task(&entity) {
                // Only consider active tasks
                if task.status != TaskStatus::Next && task.status != TaskStatus::Scheduled {
                    continue;
                }

                let (score, reasons) = self.score_task(&task, &params);
                if score > 0.0 {
                    scored_tasks.push((task, score, reasons));
                }
            }
        }

        // Sort by score (highest first)
        scored_tasks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        let recommendations: Vec<TaskRecommendation> = scored_tasks
            .into_iter()
            .take(params.limit)
            .map(|(task, score, reasons)| TaskRecommendation {
                task,
                score,
                reasons,
            })
            .collect();

        Ok(recommendations)
    }

    /// Get quick wins (2-minute tasks).
    pub async fn get_quick_wins(&self) -> Result<Vec<Task>> {
        let tasks = self.list(TaskFilter {
            status: Some(TaskStatus::Next),
            ..Default::default()
        }).await?;

        Ok(tasks
            .into_iter()
            .filter(|t| t.is_quick_task(self.quick_task_threshold))
            .collect())
    }

    /// Get overdue tasks.
    pub async fn get_overdue(&self) -> Result<Vec<Task>> {
        let tasks = self.list(TaskFilter {
            status: Some(TaskStatus::Next),
            ..Default::default()
        }).await?;

        Ok(tasks.into_iter().filter(|t| t.is_overdue()).collect())
    }

    /// Get tasks for a specific context.
    pub async fn get_by_context(&self, context: &str) -> Result<Vec<Task>> {
        self.list(TaskFilter {
            contexts: vec![context.to_string()],
            status: Some(TaskStatus::Next),
            ..Default::default()
        }).await
    }

    /// Get tasks for a specific project.
    pub async fn get_by_project(&self, project_id: &str) -> Result<Vec<Task>> {
        self.list(TaskFilter {
            project_id: Some(project_id.to_string()),
            ..Default::default()
        }).await
    }

    // Helper methods

    fn task_matches_filter(&self, task: &Task, filter: &TaskFilter) -> bool {
        // Filter by contexts
        if !filter.contexts.is_empty() {
            let has_context = filter.contexts.iter().any(|c| task.contexts.contains(c));
            if !has_context {
                return false;
            }
        }

        // Filter by project
        if let Some(ref project_id) = filter.project_id {
            if task.project_id.as_ref() != Some(project_id) {
                return false;
            }
        }

        // Filter by status
        if let Some(status) = filter.status {
            if task.status != status {
                return false;
            }
        }

        // Filter by energy level
        if let Some(energy) = filter.energy_level {
            if task.energy_level != energy {
                return false;
            }
        }

        // Filter by available time
        if let Some(time_available) = filter.time_available {
            if let Some(estimated) = task.estimated_minutes {
                if estimated > time_available {
                    return false;
                }
            }
        }

        // Filter by due date
        if let Some(due_before) = filter.due_before {
            if let Some(due) = task.due_date {
                if due > due_before {
                    return false;
                }
            } else {
                // No due date means it doesn't match "due before X"
                return false;
            }
        }

        // Filter by priority
        if let Some(priority) = filter.priority {
            if task.priority != priority {
                return false;
            }
        }

        true
    }

    fn score_task(&self, task: &Task, params: &RecommendParams) -> (f32, Vec<String>) {
        let mut score = 0.0f32;
        let mut reasons = Vec::new();

        // Context match (+30 points)
        if let Some(ref context) = params.current_context {
            if task.contexts.contains(context) || task.contexts.is_empty() {
                score += 30.0;
                if task.contexts.contains(context) {
                    reasons.push(format!("Matches your {} context", context));
                }
            } else {
                // Wrong context, heavily penalize
                score -= 50.0;
            }
        } else {
            // No context specified, give partial credit
            score += 15.0;
        }

        // Energy match (+20 points)
        if let Some(ref energy) = params.energy_level {
            if task.energy_level == *energy {
                score += 20.0;
                reasons.push(format!("Matches your {} energy level", format!("{:?}", energy).to_lowercase()));
            } else if *energy == EnergyLevel::Low && task.energy_level == EnergyLevel::Medium {
                // Medium tasks are ok when low energy
                score += 5.0;
            } else if task.energy_level == EnergyLevel::Low {
                // Low energy tasks are always ok
                score += 10.0;
            }
        } else {
            score += 10.0;
        }

        // Time fit (+20 points)
        if let Some(time_available) = params.time_available {
            if let Some(estimated) = task.estimated_minutes {
                if estimated <= time_available {
                    score += 20.0;
                    if estimated <= self.quick_task_threshold {
                        reasons.push(format!("Quick win ({} min)", estimated));
                    } else {
                        reasons.push(format!("Fits in {} min window", time_available));
                    }
                } else {
                    // Too long, penalize
                    score -= 30.0;
                }
            } else {
                // Unknown duration, partial credit
                score += 5.0;
            }
        } else {
            score += 10.0;
        }

        // Priority boost (+15 points)
        match task.priority {
            Priority::Critical => {
                score += 15.0;
                reasons.push("Critical priority".to_string());
            }
            Priority::High => {
                score += 10.0;
                reasons.push("High priority".to_string());
            }
            Priority::Normal => {
                score += 5.0;
            }
            Priority::Low => {
                // No penalty, just no bonus
            }
        }

        // Due date urgency (+20 points max)
        if let Some(due) = task.due_date {
            let now = Utc::now();
            if due < now {
                score += 20.0;
                reasons.push("Overdue!".to_string());
            } else {
                let hours_until = (due - now).num_hours();
                if hours_until <= 24 {
                    score += 15.0;
                    reasons.push("Due within 24 hours".to_string());
                } else if hours_until <= 72 {
                    score += 10.0;
                    reasons.push("Due within 3 days".to_string());
                } else if hours_until <= 168 {
                    score += 5.0;
                    reasons.push("Due this week".to_string());
                }
            }
        }

        // Quick wins get a bonus
        if task.is_quick_task(self.quick_task_threshold) {
            score += 10.0;
            if !reasons.iter().any(|r| r.contains("Quick")) {
                reasons.push(format!("Quick win (≤{} min)", self.quick_task_threshold));
            }
        }

        // Blocked tasks get penalized
        if !task.blocked_by.is_empty() {
            score -= 40.0;
        }

        (score.max(0.0), reasons)
    }

    async fn link_to_context(
        &self,
        store: &EmbeddedOntologyStore,
        task_id: &str,
        context: &str,
    ) -> Result<()> {
        // Find or create the context entity
        let context_entities = store.find_entities_by_name(context, 1).await?;
        let context_id = if let Some(ctx) = context_entities.first() {
            ctx.id.clone()
        } else {
            let ctx_entity = Entity::new(EntityType::Context, context);
            let created = store.create_entity(ctx_entity).await?;
            created.id
        };

        // Create relationship
        let relationship = Relationship::new(task_id, RelationType::HasContext, &context_id);
        store.create_relationship(relationship).await?;
        Ok(())
    }

    fn entity_to_task(entity: &Entity) -> Result<Option<Task>> {
        if entity.entity_type != EntityType::Task {
            return Ok(None);
        }

        // Try to get task from metadata
        if let Some(task_data) = entity.metadata.get("task_data") {
            let task: Task = serde_json::from_value(task_data.clone())?;
            return Ok(Some(task));
        }

        // Fallback: construct from entity fields
        let status = entity
            .metadata
            .get("status")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(TaskStatus::Next);

        let task = Task {
            id: entity.id.clone(),
            description: entity.name.clone(),
            project_id: None,
            contexts: Vec::new(),
            status,
            energy_level: EnergyLevel::Medium,
            estimated_minutes: None,
            due_date: None,
            scheduled_date: None,
            priority: Priority::Normal,
            waiting_on_person: None,
            blocked_by: Vec::new(),
            reference_doc_ids: Vec::new(),
            source_document_id: None,
            created_at: entity.created_at,
            updated_at: entity.updated_at,
            completed_at: None,
            metadata: entity.metadata.clone(),
        };

        Ok(Some(task))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_manager() -> TaskManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        TaskManager::new(store)
    }

    #[tokio::test]
    async fn test_create_and_get_task() {
        let manager = create_test_manager().await;

        let task = Task::new("Call John")
            .with_context("@phone")
            .with_duration(5);

        let created = manager.create(task.clone()).await.unwrap();
        assert_eq!(created.description, "Call John");

        let retrieved = manager.get(&created.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().description, "Call John");
    }

    #[tokio::test]
    async fn test_list_tasks() {
        let manager = create_test_manager().await;

        manager.create(Task::new("Task 1")).await.unwrap();
        manager.create(Task::new("Task 2")).await.unwrap();

        let tasks = manager.list(TaskFilter::default()).await.unwrap();
        assert_eq!(tasks.len(), 2);
    }

    #[tokio::test]
    async fn test_complete_task() {
        let manager = create_test_manager().await;

        let task = manager.create(Task::new("Test")).await.unwrap();
        let completed = manager.complete(&task.id).await.unwrap();

        assert!(completed.is_some());
        assert_eq!(completed.unwrap().status, TaskStatus::Done);
    }

    #[tokio::test]
    async fn test_filter_by_context() {
        let manager = create_test_manager().await;

        manager
            .create(Task::new("Phone task").with_context("@phone"))
            .await
            .unwrap();
        manager
            .create(Task::new("Computer task").with_context("@computer"))
            .await
            .unwrap();

        let phone_tasks = manager.get_by_context("@phone").await.unwrap();
        assert_eq!(phone_tasks.len(), 1);
        assert_eq!(phone_tasks[0].description, "Phone task");
    }

    #[tokio::test]
    async fn test_recommendations() {
        let manager = create_test_manager().await;

        manager
            .create(
                Task::new("Quick call")
                    .with_context("@phone")
                    .with_duration(2)
                    .with_energy(EnergyLevel::Low),
            )
            .await
            .unwrap();

        manager
            .create(
                Task::new("Complex analysis")
                    .with_context("@computer")
                    .with_duration(120)
                    .with_energy(EnergyLevel::High),
            )
            .await
            .unwrap();

        let recs = manager
            .recommend(RecommendParams {
                current_context: Some("@phone".to_string()),
                energy_level: Some(EnergyLevel::Low),
                time_available: Some(15),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(!recs.is_empty());
        assert_eq!(recs[0].task.description, "Quick call");
        assert!(!recs[0].reasons.is_empty());
    }

    #[tokio::test]
    async fn test_quick_wins() {
        let manager = create_test_manager().await;

        manager
            .create(Task::new("Quick").with_duration(2))
            .await
            .unwrap();
        manager
            .create(Task::new("Long").with_duration(60))
            .await
            .unwrap();

        let quick = manager.get_quick_wins().await.unwrap();
        assert_eq!(quick.len(), 1);
        assert_eq!(quick[0].description, "Quick");
    }

}

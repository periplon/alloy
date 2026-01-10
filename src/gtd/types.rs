//! GTD-specific types for task and project management.
//!
//! This module defines the core types for the Getting Things Done methodology,
//! including projects, tasks, waiting-for items, and someday/maybe items.

use chrono::{DateTime, Duration, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use crate::ontology::extraction::{EnergyLevel, Priority};

// ============================================================================
// Project Types
// ============================================================================

/// A GTD project requiring multiple actions to complete.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Project {
    /// Unique identifier for the project.
    pub id: String,
    /// Project name.
    pub name: String,
    /// Desired outcome statement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// Current status of the project.
    pub status: ProjectStatus,
    /// Area of focus this project belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub area: Option<String>,
    /// Goal this project supports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supporting_goal: Option<String>,
    /// The single next physical action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_action_id: Option<String>,
    /// Reference documents linked to this project.
    #[serde(default)]
    pub reference_doc_ids: Vec<String>,
    /// Computed health score (0-100).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_score: Option<f32>,
    /// When the project was created.
    pub created_at: DateTime<Utc>,
    /// When the project was last modified.
    pub updated_at: DateTime<Utc>,
    /// When the project was completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Project {
    /// Create a new project with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            outcome: None,
            status: ProjectStatus::Active,
            area: None,
            supporting_goal: None,
            next_action_id: None,
            reference_doc_ids: Vec::new(),
            health_score: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a project with a specific ID.
    pub fn with_id(id: impl Into<String>, name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            name: name.into(),
            outcome: None,
            status: ProjectStatus::Active,
            area: None,
            supporting_goal: None,
            next_action_id: None,
            reference_doc_ids: Vec::new(),
            health_score: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the outcome statement.
    pub fn with_outcome(mut self, outcome: impl Into<String>) -> Self {
        self.outcome = Some(outcome.into());
        self
    }

    /// Set the area of focus.
    pub fn with_area(mut self, area: impl Into<String>) -> Self {
        self.area = Some(area.into());
        self
    }

    /// Set the supporting goal.
    pub fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.supporting_goal = Some(goal.into());
        self
    }

    /// Set the status.
    pub fn with_status(mut self, status: ProjectStatus) -> Self {
        self.status = status;
        self
    }

    /// Check if the project is stalled (no activity in N days).
    pub fn is_stalled(&self, days: i64) -> bool {
        let threshold = Utc::now() - Duration::days(days);
        self.updated_at < threshold && self.status == ProjectStatus::Active
    }
}

/// Status of a project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectStatus {
    /// Actively being worked on.
    #[default]
    Active,
    /// Temporarily paused.
    OnHold,
    /// Successfully completed.
    Completed,
    /// No longer relevant, archived.
    Archived,
}

/// Health factors for a project.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectHealth {
    /// Overall score (0-100).
    pub score: f32,
    /// Has a defined next action.
    pub has_next_action: bool,
    /// Had activity in the last 7 days.
    pub recent_activity: bool,
    /// Has a clear outcome statement.
    pub clear_outcome: bool,
    /// Has reasonable scope (not too many tasks).
    pub reasonable_scope: bool,
    /// No blocking tasks.
    pub no_blockers: bool,
    /// Linked to a goal.
    pub linked_to_goal: bool,
    /// Recommendations for improving health.
    #[serde(default)]
    pub recommendations: Vec<String>,
}

impl ProjectHealth {
    /// Calculate health from factors.
    pub fn calculate(
        has_next_action: bool,
        recent_activity: bool,
        clear_outcome: bool,
        reasonable_scope: bool,
        no_blockers: bool,
        linked_to_goal: bool,
    ) -> Self {
        let mut score = 0.0;
        let mut recommendations = Vec::new();

        if has_next_action {
            score += 30.0;
        } else {
            recommendations.push("Define a clear next action for this project".to_string());
        }

        if recent_activity {
            score += 20.0;
        } else {
            recommendations.push("Project has been inactive - review and take action".to_string());
        }

        if clear_outcome {
            score += 15.0;
        } else {
            recommendations.push("Add a clear outcome statement".to_string());
        }

        if reasonable_scope {
            score += 15.0;
        } else {
            recommendations
                .push("Consider breaking this project into smaller projects".to_string());
        }

        if no_blockers {
            score += 10.0;
        } else {
            recommendations.push("Resolve blocking tasks".to_string());
        }

        if linked_to_goal {
            score += 10.0;
        } else {
            recommendations.push("Link this project to a goal for better alignment".to_string());
        }

        Self {
            score,
            has_next_action,
            recent_activity,
            clear_outcome,
            reasonable_scope,
            no_blockers,
            linked_to_goal,
            recommendations,
        }
    }
}

// ============================================================================
// Task Types
// ============================================================================

/// A GTD task - a single next physical action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Task {
    /// Unique identifier for the task.
    pub id: String,
    /// Task description (physical action).
    pub description: String,
    /// Project this task belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Contexts where this task can be done.
    #[serde(default)]
    pub contexts: Vec<String>,
    /// Current status of the task.
    pub status: TaskStatus,
    /// Energy level required.
    pub energy_level: EnergyLevel,
    /// Estimated duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_minutes: Option<u32>,
    /// Due date (hard deadline).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub due_date: Option<DateTime<Utc>>,
    /// Scheduled date (soft target).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduled_date: Option<DateTime<Utc>>,
    /// Priority level.
    pub priority: Priority,
    /// Person this task is waiting on (for waiting-for tasks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waiting_on_person: Option<String>,
    /// Tasks that block this task.
    #[serde(default)]
    pub blocked_by: Vec<String>,
    /// Reference documents linked to this task.
    #[serde(default)]
    pub reference_doc_ids: Vec<String>,
    /// Source document where this task was extracted from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document_id: Option<String>,
    /// When the task was created.
    pub created_at: DateTime<Utc>,
    /// When the task was last modified.
    pub updated_at: DateTime<Utc>,
    /// When the task was completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Task {
    /// Create a new task with the given description.
    pub fn new(description: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            project_id: None,
            contexts: Vec::new(),
            status: TaskStatus::Next,
            energy_level: EnergyLevel::Medium,
            estimated_minutes: None,
            due_date: None,
            scheduled_date: None,
            priority: Priority::Normal,
            waiting_on_person: None,
            blocked_by: Vec::new(),
            reference_doc_ids: Vec::new(),
            source_document_id: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a task with a specific ID.
    pub fn with_id(id: impl Into<String>, description: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            description: description.into(),
            project_id: None,
            contexts: Vec::new(),
            status: TaskStatus::Next,
            energy_level: EnergyLevel::Medium,
            estimated_minutes: None,
            due_date: None,
            scheduled_date: None,
            priority: Priority::Normal,
            waiting_on_person: None,
            blocked_by: Vec::new(),
            reference_doc_ids: Vec::new(),
            source_document_id: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the project.
    pub fn with_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Add a context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.contexts.push(context.into());
        self
    }

    /// Set contexts.
    pub fn with_contexts(mut self, contexts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.contexts = contexts.into_iter().map(|c| c.into()).collect();
        self
    }

    /// Set the status.
    pub fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = status;
        self
    }

    /// Set the energy level.
    pub fn with_energy(mut self, energy: EnergyLevel) -> Self {
        self.energy_level = energy;
        self
    }

    /// Set the estimated duration in minutes.
    pub fn with_duration(mut self, minutes: u32) -> Self {
        self.estimated_minutes = Some(minutes);
        self
    }

    /// Set the due date.
    pub fn with_due_date(mut self, due_date: DateTime<Utc>) -> Self {
        self.due_date = Some(due_date);
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Check if this is a quick task (2-minute rule).
    pub fn is_quick_task(&self, threshold_minutes: u32) -> bool {
        self.estimated_minutes
            .map(|m| m <= threshold_minutes)
            .unwrap_or(false)
    }

    /// Check if the task is overdue.
    pub fn is_overdue(&self) -> bool {
        if let Some(due) = self.due_date {
            return due < Utc::now() && self.status != TaskStatus::Done;
        }
        false
    }

    /// Mark the task as complete.
    pub fn complete(&mut self) {
        self.status = TaskStatus::Done;
        self.completed_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }
}

/// Status of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Ready to be worked on next.
    #[default]
    Next,
    /// Scheduled for a specific time.
    Scheduled,
    /// Waiting for someone else.
    Waiting,
    /// Deferred to someday/maybe.
    Someday,
    /// Completed.
    Done,
}

/// A task recommendation with scoring.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskRecommendation {
    /// The recommended task.
    pub task: Task,
    /// Recommendation score (higher is better).
    pub score: f32,
    /// Reasons why this task was recommended.
    pub reasons: Vec<String>,
}

// ============================================================================
// Waiting For Types
// ============================================================================

/// An item waiting for someone else.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WaitingFor {
    /// Unique identifier.
    pub id: String,
    /// Description of what we're waiting for.
    pub description: String,
    /// Who we're waiting on.
    pub delegated_to: String,
    /// Related project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// When this was delegated.
    pub delegated_date: DateTime<Utc>,
    /// When we expect a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_by: Option<DateTime<Utc>>,
    /// When we followed up.
    #[serde(default)]
    pub follow_up_dates: Vec<DateTime<Utc>>,
    /// Current status.
    pub status: WaitingStatus,
    /// Resolution notes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    /// Source document where this was extracted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document_id: Option<String>,
    /// When created.
    pub created_at: DateTime<Utc>,
    /// When last updated.
    pub updated_at: DateTime<Utc>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl WaitingFor {
    /// Create a new waiting-for item.
    pub fn new(description: impl Into<String>, delegated_to: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            delegated_to: delegated_to.into(),
            project_id: None,
            delegated_date: now,
            expected_by: None,
            follow_up_dates: Vec::new(),
            status: WaitingStatus::Pending,
            resolution: None,
            source_document_id: None,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Set the project.
    pub fn with_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Set the expected date.
    pub fn with_expected_by(mut self, date: DateTime<Utc>) -> Self {
        self.expected_by = Some(date);
        self
    }

    /// Check if this is overdue.
    pub fn is_overdue(&self) -> bool {
        if let Some(expected) = self.expected_by {
            return expected < Utc::now() && self.status == WaitingStatus::Pending;
        }
        false
    }

    /// Record a follow-up.
    pub fn record_follow_up(&mut self) {
        self.follow_up_dates.push(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Resolve the waiting item.
    pub fn resolve(&mut self, resolution: impl Into<String>) {
        self.status = WaitingStatus::Resolved;
        self.resolution = Some(resolution.into());
        self.updated_at = Utc::now();
    }
}

/// Status of a waiting-for item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WaitingStatus {
    /// Still waiting.
    #[default]
    Pending,
    /// Past expected date.
    Overdue,
    /// Received response/completed.
    Resolved,
}

// ============================================================================
// Someday/Maybe Types
// ============================================================================

/// A deferred item for future consideration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SomedayItem {
    /// Unique identifier.
    pub id: String,
    /// Description of the item.
    pub description: String,
    /// Category for organization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// What would trigger making this active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger: Option<String>,
    /// When to review/reconsider this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_date: Option<DateTime<Utc>>,
    /// Source document where this was extracted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document_id: Option<String>,
    /// When created.
    pub created_at: DateTime<Utc>,
    /// When last updated.
    pub updated_at: DateTime<Utc>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SomedayItem {
    /// Create a new someday/maybe item.
    pub fn new(description: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            category: None,
            trigger: None,
            review_date: None,
            source_document_id: None,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Set the trigger condition.
    pub fn with_trigger(mut self, trigger: impl Into<String>) -> Self {
        self.trigger = Some(trigger.into());
        self
    }

    /// Set the review date.
    pub fn with_review_date(mut self, date: DateTime<Utc>) -> Self {
        self.review_date = Some(date);
        self
    }

    /// Check if this item is due for review.
    pub fn is_due_for_review(&self) -> bool {
        if let Some(review) = self.review_date {
            return review <= Utc::now();
        }
        false
    }
}

// ============================================================================
// Filter Types
// ============================================================================

/// Filter criteria for projects.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectFilter {
    /// Filter by status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ProjectStatus>,
    /// Filter by area.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub area: Option<String>,
    /// Filter to projects with/without next actions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_next_action: Option<bool>,
    /// Filter to projects stalled for N days.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stalled_days: Option<u32>,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

impl Default for ProjectFilter {
    fn default() -> Self {
        Self {
            status: None,
            area: None,
            has_next_action: None,
            stalled_days: None,
            limit: 100,
            offset: 0,
        }
    }
}

/// Filter criteria for tasks.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskFilter {
    /// Filter by contexts.
    #[serde(default)]
    pub contexts: Vec<String>,
    /// Filter by project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Filter by status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<TaskStatus>,
    /// Filter by energy level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_level: Option<EnergyLevel>,
    /// Filter by available time (in minutes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_available: Option<u32>,
    /// Filter by due before date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub due_before: Option<DateTime<Utc>>,
    /// Filter by priority.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<Priority>,
    /// Filter by description (case-insensitive substring match).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description_contains: Option<String>,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

impl Default for TaskFilter {
    fn default() -> Self {
        Self {
            contexts: Vec::new(),
            project_id: None,
            status: None,
            energy_level: None,
            time_available: None,
            due_before: None,
            priority: None,
            description_contains: None,
            limit: 100,
            offset: 0,
        }
    }
}

/// Filter criteria for waiting-for items.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WaitingFilter {
    /// Filter by delegated to person.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delegated_to: Option<String>,
    /// Filter by project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Filter by status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<WaitingStatus>,
    /// Filter to only overdue items.
    #[serde(default)]
    pub overdue_only: bool,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

impl Default for WaitingFilter {
    fn default() -> Self {
        Self {
            delegated_to: None,
            project_id: None,
            status: None,
            overdue_only: false,
            limit: 100,
            offset: 0,
        }
    }
}

/// Filter criteria for someday/maybe items.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SomedayFilter {
    /// Filter by category.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Filter to items due for review.
    #[serde(default)]
    pub due_for_review: bool,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

impl Default for SomedayFilter {
    fn default() -> Self {
        Self {
            category: None,
            due_for_review: false,
            limit: 100,
            offset: 0,
        }
    }
}

fn default_limit() -> usize {
    100
}

// ============================================================================
// Recommendation Types
// ============================================================================

/// Parameters for task recommendations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RecommendParams {
    /// Current context (e.g., "@home", "@work").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_context: Option<String>,
    /// Current energy level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_level: Option<EnergyLevel>,
    /// Available time in minutes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_available: Option<u32>,
    /// Focus on a specific area.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus_area: Option<String>,
    /// Maximum number of recommendations.
    #[serde(default = "default_recommend_limit")]
    pub limit: usize,
}

impl Default for RecommendParams {
    fn default() -> Self {
        Self {
            current_context: None,
            energy_level: None,
            time_available: None,
            focus_area: None,
            limit: 5,
        }
    }
}

fn default_recommend_limit() -> usize {
    5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_creation() {
        let project = Project::new("Website Redesign")
            .with_outcome("Launch new website with improved UX")
            .with_area("Work");

        assert_eq!(project.name, "Website Redesign");
        assert!(project.outcome.is_some());
        assert_eq!(project.status, ProjectStatus::Active);
    }

    #[test]
    fn test_project_stalled() {
        let mut project = Project::new("Old Project");
        // Simulate old updated_at
        project.updated_at = Utc::now() - Duration::days(10);

        assert!(project.is_stalled(7));
        assert!(!project.is_stalled(14));
    }

    #[test]
    fn test_project_health() {
        let health = ProjectHealth::calculate(
            true, // has_next_action
            true, // recent_activity
            true, // clear_outcome
            true, // reasonable_scope
            true, // no_blockers
            true, // linked_to_goal
        );
        assert_eq!(health.score, 100.0);
        assert!(health.recommendations.is_empty());

        let unhealthy = ProjectHealth::calculate(false, false, false, false, false, false);
        assert_eq!(unhealthy.score, 0.0);
        assert!(!unhealthy.recommendations.is_empty());
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new("Call John about project")
            .with_context("@phone")
            .with_energy(EnergyLevel::Low)
            .with_duration(5);

        assert_eq!(task.description, "Call John about project");
        assert!(task.contexts.contains(&"@phone".to_string()));
        assert!(task.is_quick_task(10));
        assert!(!task.is_quick_task(2));
    }

    #[test]
    fn test_task_completion() {
        let mut task = Task::new("Test task");
        assert_eq!(task.status, TaskStatus::Next);
        assert!(task.completed_at.is_none());

        task.complete();
        assert_eq!(task.status, TaskStatus::Done);
        assert!(task.completed_at.is_some());
    }

    #[test]
    fn test_waiting_for() {
        let mut waiting = WaitingFor::new("Response on proposal", "John");
        assert_eq!(waiting.status, WaitingStatus::Pending);

        waiting.record_follow_up();
        assert_eq!(waiting.follow_up_dates.len(), 1);

        waiting.resolve("Got approval!");
        assert_eq!(waiting.status, WaitingStatus::Resolved);
    }

    #[test]
    fn test_someday_item() {
        let item = SomedayItem::new("Learn Rust")
            .with_category("Learning")
            .with_trigger("After current project ends");

        assert_eq!(item.description, "Learn Rust");
        assert_eq!(item.category, Some("Learning".to_string()));
    }
}

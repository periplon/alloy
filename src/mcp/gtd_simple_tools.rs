//! Simplified GTD MCP tools with one tool per action.
//!
//! These tools are designed to be easier for LLMs to use by having
//! separate tools for each action instead of a single tool with an action enum.

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::gtd::{EnergyLevel, Priority, ProjectStatus, TaskStatus, WaitingStatus};

// ============================================================================
// Task Tools - Simplified
// ============================================================================

/// Parameters for creating a new task.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskCreateParams {
    /// Task description (required). What needs to be done.
    pub description: String,
    /// Project ID to associate this task with.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Contexts where this task can be done (e.g., "@phone", "@computer", "@home").
    #[serde(default)]
    pub contexts: Option<Vec<String>>,
    /// Energy level required: "low", "medium", or "high".
    #[serde(default)]
    pub energy_level: Option<EnergyLevel>,
    /// Estimated duration in minutes.
    #[serde(default)]
    pub estimated_minutes: Option<u32>,
    /// Due date in ISO 8601 format (e.g., "2024-12-31T23:59:59Z").
    #[serde(default)]
    pub due_date: Option<DateTime<Utc>>,
    /// Priority: "low", "normal", "high", or "critical".
    #[serde(default)]
    pub priority: Option<Priority>,
}

/// Parameters for listing tasks.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskListParams {
    /// Filter by project ID.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Filter by contexts (e.g., ["@phone", "@computer"]).
    #[serde(default)]
    pub contexts: Option<Vec<String>>,
    /// Filter by status: "next", "scheduled", "waiting", "someday", or "done".
    #[serde(default)]
    pub status: Option<TaskStatus>,
    /// Filter by energy level: "low", "medium", or "high".
    #[serde(default)]
    pub energy_level: Option<EnergyLevel>,
    /// Filter by priority: "low", "normal", "high", or "critical".
    #[serde(default)]
    pub priority: Option<Priority>,
    /// Filter by description containing this text.
    #[serde(default)]
    pub description_contains: Option<String>,
    /// Maximum number of results (default: 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for getting a single task.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskGetParams {
    /// The task ID to retrieve (required).
    pub task_id: String,
}

/// Parameters for completing a task.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskCompleteParams {
    /// The task ID to mark as complete (required).
    pub task_id: String,
}

/// Parameters for deleting a task.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskDeleteParams {
    /// The task ID to delete (required).
    pub task_id: String,
}

/// Parameters for updating a task.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskUpdateParams {
    /// The task ID to update (required).
    pub task_id: String,
    /// New description (optional).
    #[serde(default)]
    pub description: Option<String>,
    /// New project ID (optional).
    #[serde(default)]
    pub project_id: Option<String>,
    /// New contexts (optional).
    #[serde(default)]
    pub contexts: Option<Vec<String>>,
    /// New status: "next", "scheduled", "waiting", "someday", or "done" (optional).
    #[serde(default)]
    pub status: Option<TaskStatus>,
    /// New energy level: "low", "medium", or "high" (optional).
    #[serde(default)]
    pub energy_level: Option<EnergyLevel>,
    /// New estimated duration in minutes (optional).
    #[serde(default)]
    pub estimated_minutes: Option<u32>,
    /// New due date in ISO 8601 format (optional).
    #[serde(default)]
    pub due_date: Option<DateTime<Utc>>,
    /// New priority: "low", "normal", "high", or "critical" (optional).
    #[serde(default)]
    pub priority: Option<Priority>,
}

/// Parameters for getting task recommendations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskRecommendParams {
    /// Current context (e.g., "@home", "@work", "@phone").
    #[serde(default)]
    pub current_context: Option<String>,
    /// Available time in minutes.
    #[serde(default)]
    pub time_available: Option<u32>,
    /// Current energy level: "low", "medium", or "high".
    #[serde(default)]
    pub energy_level: Option<EnergyLevel>,
    /// Maximum number of recommendations (default: 10).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for getting quick win tasks (2-minute tasks).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskQuickWinsParams {
    /// Maximum number of quick wins to return (default: 10).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for getting overdue tasks.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskOverdueParams {
    /// Maximum number of overdue tasks to return (default: 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

// ============================================================================
// Project Tools - Simplified
// ============================================================================

/// Parameters for creating a new project.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectCreateParams {
    /// Project name (required).
    pub name: String,
    /// Desired outcome - what does "done" look like?
    #[serde(default)]
    pub outcome: Option<String>,
    /// Area of focus (e.g., "Work", "Personal", "Health").
    #[serde(default)]
    pub area: Option<String>,
    /// Supporting goal this project contributes to.
    #[serde(default)]
    pub goal: Option<String>,
}

/// Parameters for listing projects.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectListParams {
    /// Filter by status: "active", "on_hold", "completed", or "archived".
    #[serde(default)]
    pub status: Option<ProjectStatus>,
    /// Filter by area.
    #[serde(default)]
    pub area: Option<String>,
    /// Filter to projects with (true) or without (false) next actions.
    #[serde(default)]
    pub has_next_action: Option<bool>,
    /// Filter to projects stalled for this many days.
    #[serde(default)]
    pub stalled_days: Option<u32>,
    /// Maximum number of results (default: 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for getting a single project.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectGetParams {
    /// The project ID to retrieve (required).
    pub project_id: String,
}

/// Parameters for updating a project.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectUpdateParams {
    /// The project ID to update (required).
    pub project_id: String,
    /// New project name (optional).
    #[serde(default)]
    pub name: Option<String>,
    /// New outcome statement (optional).
    #[serde(default)]
    pub outcome: Option<String>,
    /// New area (optional).
    #[serde(default)]
    pub area: Option<String>,
    /// New goal (optional).
    #[serde(default)]
    pub goal: Option<String>,
}

/// Parameters for archiving a project.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectArchiveParams {
    /// The project ID to archive (required).
    pub project_id: String,
}

/// Parameters for completing a project.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectCompleteParams {
    /// The project ID to mark as complete (required).
    pub project_id: String,
}

/// Parameters for getting project health.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectHealthParams {
    /// The project ID to check health for (required).
    pub project_id: String,
}

// ============================================================================
// Waiting-For Tools - Simplified
// ============================================================================

/// Parameters for adding a waiting-for item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WaitingAddParams {
    /// What you are waiting for (required).
    pub description: String,
    /// Person or entity you are waiting on (required).
    pub delegated_to: String,
    /// Related project ID (optional).
    #[serde(default)]
    pub project_id: Option<String>,
    /// Expected response date in ISO 8601 format (optional).
    #[serde(default)]
    pub expected_by: Option<DateTime<Utc>>,
}

/// Parameters for listing waiting-for items.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WaitingListParams {
    /// Filter by status: "pending", "overdue", or "resolved".
    #[serde(default)]
    pub status: Option<WaitingStatus>,
    /// Maximum number of results (default: 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for getting a waiting-for item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WaitingGetParams {
    /// The waiting-for item ID (required).
    pub item_id: String,
}

/// Parameters for resolving a waiting-for item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WaitingResolveParams {
    /// The waiting-for item ID (required).
    pub item_id: String,
    /// How it was resolved (optional).
    #[serde(default)]
    pub resolution: Option<String>,
}

/// Parameters for recording a follow-up on a waiting-for item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WaitingFollowUpParams {
    /// The waiting-for item ID (required).
    pub item_id: String,
}

// ============================================================================
// Someday/Maybe Tools - Simplified
// ============================================================================

/// Parameters for adding a someday/maybe item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SomedayAddParams {
    /// Description of the item (required).
    pub description: String,
    /// Category for organization (e.g., "Travel", "Learning", "Projects").
    #[serde(default)]
    pub category: Option<String>,
    /// Trigger condition - what would make this active?
    #[serde(default)]
    pub trigger: Option<String>,
    /// Review date in ISO 8601 format.
    #[serde(default)]
    pub review_date: Option<DateTime<Utc>>,
}

/// Parameters for listing someday/maybe items.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SomedayListParams {
    /// Filter by category.
    #[serde(default)]
    pub category: Option<String>,
    /// Maximum number of results (default: 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for getting a someday/maybe item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SomedayGetParams {
    /// The item ID (required).
    pub item_id: String,
}

/// Parameters for activating a someday/maybe item (convert to task).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SomedayActivateParams {
    /// The item ID to activate (required).
    pub item_id: String,
}

/// Parameters for archiving a someday/maybe item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SomedayArchiveParams {
    /// The item ID to archive (required).
    pub item_id: String,
}

// ============================================================================
// Calendar Tools - Simplified
// ============================================================================

/// Parameters for creating a calendar event.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarCreateParams {
    /// Event title (required).
    pub title: String,
    /// Start time in ISO 8601 format (required, e.g., "2024-12-31T14:00:00Z").
    pub start: DateTime<Utc>,
    /// Event description.
    #[serde(default)]
    pub description: Option<String>,
    /// Event type: "event", "meeting", "deadline", "reminder", "blocked_time", etc.
    #[serde(default)]
    pub event_type: Option<String>,
    /// End time in ISO 8601 format.
    #[serde(default)]
    pub end: Option<DateTime<Utc>>,
    /// Duration in minutes (alternative to end time).
    #[serde(default)]
    pub duration_minutes: Option<i64>,
    /// Is this an all-day event?
    #[serde(default)]
    pub all_day: Option<bool>,
    /// Location.
    #[serde(default)]
    pub location: Option<String>,
    /// Participants.
    #[serde(default)]
    pub participants: Option<Vec<String>>,
    /// Related project ID.
    #[serde(default)]
    pub project_id: Option<String>,
}

/// Parameters for listing calendar events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarListParams {
    /// Query type: "upcoming" (default), "today", "this_week", "next_week", "date_range".
    #[serde(default)]
    pub query_type: Option<String>,
    /// Start date for date_range queries (ISO 8601 format).
    #[serde(default)]
    pub start_date: Option<DateTime<Utc>>,
    /// End date for date_range queries (ISO 8601 format).
    #[serde(default)]
    pub end_date: Option<DateTime<Utc>>,
    /// Number of days for upcoming queries (default: 7).
    #[serde(default)]
    pub days: Option<i64>,
    /// Maximum number of results (default: 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for getting a calendar event.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarGetParams {
    /// The event ID (required).
    pub event_id: String,
}

/// Parameters for deleting a calendar event.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarDeleteParams {
    /// The event ID (required).
    pub event_id: String,
}

/// Parameters for finding free time slots.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarFindFreeTimeParams {
    /// Start of search range (ISO 8601 format, default: now).
    #[serde(default)]
    pub start_date: Option<DateTime<Utc>>,
    /// End of search range (ISO 8601 format, default: 7 days from now).
    #[serde(default)]
    pub end_date: Option<DateTime<Utc>>,
    /// Minimum slot duration in minutes (default: 30).
    #[serde(default)]
    pub min_duration_minutes: Option<u32>,
    /// Working hours start (HH:MM format, default: "09:00").
    #[serde(default)]
    pub working_hours_start: Option<String>,
    /// Working hours end (HH:MM format, default: "17:00").
    #[serde(default)]
    pub working_hours_end: Option<String>,
    /// Exclude weekends (default: true).
    #[serde(default)]
    pub exclude_weekends: Option<bool>,
    /// Maximum number of slots to return (default: 10).
    #[serde(default)]
    pub limit: Option<usize>,
}

// ============================================================================
// Commitment Tools - Simplified
// ============================================================================

/// Parameters for creating a commitment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentCreateParams {
    /// Description of the commitment (required).
    pub description: String,
    /// Type of commitment: "made" (I committed to someone) or "received" (someone committed to me). Default: "made".
    #[serde(default)]
    pub commitment_type: Option<String>,
    /// Person involved (who you made the commitment to, or who made it to you).
    #[serde(default)]
    pub person: Option<String>,
    /// Due date in ISO 8601 format.
    #[serde(default)]
    pub due_date: Option<DateTime<Utc>>,
    /// Related document ID (if extracted from a document).
    #[serde(default)]
    pub document_id: Option<String>,
}

/// Parameters for listing commitments.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentListParams {
    /// Filter by type: "made" or "received".
    #[serde(default)]
    pub commitment_type: Option<String>,
    /// Filter by status: "pending", "fulfilled", "cancelled", or "overdue".
    #[serde(default)]
    pub status: Option<String>,
    /// Filter by person involved.
    #[serde(default)]
    pub person: Option<String>,
    /// Maximum number of results (default: 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for getting a commitment by ID.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentGetParams {
    /// The commitment ID (required).
    pub commitment_id: String,
}

/// Parameters for fulfilling a commitment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentFulfillParams {
    /// The commitment ID to mark as fulfilled (required).
    pub commitment_id: String,
}

/// Parameters for cancelling a commitment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentCancelParams {
    /// The commitment ID to cancel (required).
    pub commitment_id: String,
}

/// Parameters for extracting commitments from text.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentExtractParams {
    /// Text to extract commitments from (required).
    pub text: String,
    /// Source document ID (optional).
    #[serde(default)]
    pub document_id: Option<String>,
}

/// Parameters for getting commitments made to a person.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentMadeToParams {
    /// The person to get commitments made to (required).
    pub person: String,
}

/// Parameters for getting commitments received from a person.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentReceivedFromParams {
    /// The person to get commitments received from (required).
    pub person: String,
}

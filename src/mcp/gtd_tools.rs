//! GTD MCP tool implementations for Alloy.
//!
//! This module provides MCP tools for GTD (Getting Things Done) functionality:
//! - `gtd_projects`: Project management (list, get, create, update, archive)
//! - `gtd_tasks`: Task management with recommendations
//! - `gtd_waiting`: Waiting-for tracking
//! - `gtd_someday`: Someday/maybe management

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::gtd::{
    EnergyLevel, Priority, Project, ProjectHealth, ProjectStatus, SomedayItem, Task,
    TaskRecommendation, TaskStatus, WaitingFor, WaitingStatus,
};

// ============================================================================
// Project Tool Types
// ============================================================================

/// Action to perform on projects.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectAction {
    /// List projects matching filters.
    List,
    /// Get a specific project by ID.
    Get,
    /// Create a new project.
    Create,
    /// Update an existing project.
    Update,
    /// Archive a project.
    Archive,
    /// Complete a project.
    Complete,
    /// Get project health score.
    Health,
}

/// Parameters for the gtd_projects tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GtdProjectsParams {
    /// The action to perform.
    pub action: ProjectAction,
    /// Project ID (required for get, update, archive, complete, health).
    #[serde(default)]
    pub project_id: Option<String>,
    /// Project name (required for create).
    #[serde(default)]
    pub name: Option<String>,
    /// Project outcome statement.
    #[serde(default)]
    pub outcome: Option<String>,
    /// Area of focus.
    #[serde(default)]
    pub area: Option<String>,
    /// Supporting goal.
    #[serde(default)]
    pub goal: Option<String>,
    /// Filter by status (for list action).
    #[serde(default)]
    pub status: Option<ProjectStatus>,
    /// Filter to projects with/without next actions.
    #[serde(default)]
    pub has_next_action: Option<bool>,
    /// Filter to stalled projects (inactive for N days).
    #[serde(default)]
    pub stalled_days: Option<u32>,
    /// Maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Response for project operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdProjectsResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// The project (for single operations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<Project>,
    /// List of projects (for list operations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projects: Option<Vec<Project>>,
    /// Project health (for health action).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health: Option<ProjectHealth>,
    /// Status message.
    pub message: String,
}

// ============================================================================
// Task Tool Types
// ============================================================================

/// Action to perform on tasks.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskAction {
    /// List tasks matching filters.
    List,
    /// Get a specific task by ID.
    Get,
    /// Create a new task.
    Create,
    /// Update an existing task.
    Update,
    /// Complete a task.
    Complete,
    /// Delete a task.
    Delete,
    /// Get task recommendations.
    Recommend,
    /// Get quick wins (2-minute tasks).
    QuickWins,
    /// Get overdue tasks.
    Overdue,
}

/// Parameters for the gtd_tasks tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GtdTasksParams {
    /// The action to perform.
    pub action: TaskAction,
    /// Task ID (required for get, update, complete, delete).
    #[serde(default)]
    pub task_id: Option<String>,
    /// Task description (required for create).
    #[serde(default)]
    pub description: Option<String>,
    /// Project ID to associate with.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Contexts (e.g., "@phone", "@computer").
    #[serde(default)]
    pub contexts: Option<Vec<String>>,
    /// Task status.
    #[serde(default)]
    pub status: Option<TaskStatus>,
    /// Energy level required.
    #[serde(default)]
    pub energy_level: Option<EnergyLevel>,
    /// Estimated duration in minutes.
    #[serde(default)]
    pub estimated_minutes: Option<u32>,
    /// Due date (ISO 8601 format).
    #[serde(default)]
    pub due_date: Option<DateTime<Utc>>,
    /// Priority level.
    #[serde(default)]
    pub priority: Option<Priority>,
    // Recommendation parameters
    /// Current context for recommendations.
    #[serde(default)]
    pub current_context: Option<String>,
    /// Available time in minutes for recommendations.
    #[serde(default)]
    pub time_available: Option<u32>,
    /// Maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Response for task operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdTasksResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// The task (for single operations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<Task>,
    /// List of tasks (for list/quick_wins/overdue).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tasks: Option<Vec<Task>>,
    /// Task recommendations (for recommend action).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommendations: Option<Vec<TaskRecommendation>>,
    /// Status message.
    pub message: String,
}

// ============================================================================
// Waiting Tool Types
// ============================================================================

/// Action to perform on waiting-for items.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WaitingAction {
    /// List waiting-for items.
    List,
    /// Get a specific item.
    Get,
    /// Add a new waiting-for item.
    Add,
    /// Record a follow-up.
    FollowUp,
    /// Resolve a waiting-for item.
    Resolve,
    /// Get overdue items.
    Overdue,
}

/// Parameters for the gtd_waiting tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GtdWaitingParams {
    /// The action to perform.
    pub action: WaitingAction,
    /// Item ID (required for get, follow_up, resolve).
    #[serde(default)]
    pub item_id: Option<String>,
    /// Description (required for add).
    #[serde(default)]
    pub description: Option<String>,
    /// Person waiting on (required for add).
    #[serde(default)]
    pub delegated_to: Option<String>,
    /// Related project ID.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Expected response date.
    #[serde(default)]
    pub expected_by: Option<DateTime<Utc>>,
    /// Resolution text (for resolve action).
    #[serde(default)]
    pub resolution: Option<String>,
    /// Filter by status.
    #[serde(default)]
    pub status: Option<WaitingStatus>,
    /// Maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Response for waiting-for operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdWaitingResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// The item (for single operations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<WaitingFor>,
    /// List of items (for list operations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<WaitingFor>>,
    /// Status message.
    pub message: String,
}

// ============================================================================
// Someday Tool Types
// ============================================================================

/// Action to perform on someday/maybe items.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SomedayAction {
    /// List someday/maybe items.
    List,
    /// Get a specific item.
    Get,
    /// Add a new item.
    Add,
    /// Update an existing item.
    Update,
    /// Activate (convert to task).
    Activate,
    /// Archive/delete an item.
    Archive,
    /// Get categories.
    Categories,
    /// Get items due for review.
    DueForReview,
}

/// Parameters for the gtd_someday tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GtdSomedayParams {
    /// The action to perform.
    pub action: SomedayAction,
    /// Item ID (required for get, update, activate, archive).
    #[serde(default)]
    pub item_id: Option<String>,
    /// Description (required for add).
    #[serde(default)]
    pub description: Option<String>,
    /// Category for organization.
    #[serde(default)]
    pub category: Option<String>,
    /// Trigger condition (what would make this active).
    #[serde(default)]
    pub trigger: Option<String>,
    /// Review date.
    #[serde(default)]
    pub review_date: Option<DateTime<Utc>>,
    /// Maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Response for someday/maybe operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdSomedayResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// The item (for single operations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<SomedayItem>,
    /// List of items (for list operations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<SomedayItem>>,
    /// Activated task (for activate action).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<Task>,
    /// Categories (for categories action).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categories: Option<Vec<String>>,
    /// Status message.
    pub message: String,
}

// ============================================================================
// Helper Functions for Building Responses
// ============================================================================

impl GtdProjectsResponse {
    pub fn success_single(project: Project, message: impl Into<String>) -> Self {
        Self {
            success: true,
            project: Some(project),
            projects: None,
            health: None,
            message: message.into(),
        }
    }

    pub fn success_list(projects: Vec<Project>, message: impl Into<String>) -> Self {
        Self {
            success: true,
            project: None,
            projects: Some(projects),
            health: None,
            message: message.into(),
        }
    }

    pub fn success_health(health: ProjectHealth, message: impl Into<String>) -> Self {
        Self {
            success: true,
            project: None,
            projects: None,
            health: Some(health),
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            project: None,
            projects: None,
            health: None,
            message: message.into(),
        }
    }
}

impl GtdTasksResponse {
    pub fn success_single(task: Task, message: impl Into<String>) -> Self {
        Self {
            success: true,
            task: Some(task),
            tasks: None,
            recommendations: None,
            message: message.into(),
        }
    }

    pub fn success_list(tasks: Vec<Task>, message: impl Into<String>) -> Self {
        Self {
            success: true,
            task: None,
            tasks: Some(tasks),
            recommendations: None,
            message: message.into(),
        }
    }

    pub fn success_recommendations(
        recs: Vec<TaskRecommendation>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            success: true,
            task: None,
            tasks: None,
            recommendations: Some(recs),
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            task: None,
            tasks: None,
            recommendations: None,
            message: message.into(),
        }
    }
}

impl GtdWaitingResponse {
    pub fn success_single(item: WaitingFor, message: impl Into<String>) -> Self {
        Self {
            success: true,
            item: Some(item),
            items: None,
            message: message.into(),
        }
    }

    pub fn success_list(items: Vec<WaitingFor>, message: impl Into<String>) -> Self {
        Self {
            success: true,
            item: None,
            items: Some(items),
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            item: None,
            items: None,
            message: message.into(),
        }
    }
}

impl GtdSomedayResponse {
    pub fn success_single(item: SomedayItem, message: impl Into<String>) -> Self {
        Self {
            success: true,
            item: Some(item),
            items: None,
            task: None,
            categories: None,
            message: message.into(),
        }
    }

    pub fn success_list(items: Vec<SomedayItem>, message: impl Into<String>) -> Self {
        Self {
            success: true,
            item: None,
            items: Some(items),
            task: None,
            categories: None,
            message: message.into(),
        }
    }

    pub fn success_activated(task: Task, message: impl Into<String>) -> Self {
        Self {
            success: true,
            item: None,
            items: None,
            task: Some(task),
            categories: None,
            message: message.into(),
        }
    }

    pub fn success_categories(categories: Vec<String>, message: impl Into<String>) -> Self {
        Self {
            success: true,
            item: None,
            items: None,
            task: None,
            categories: Some(categories),
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            item: None,
            items: None,
            task: None,
            categories: None,
            message: message.into(),
        }
    }
}

// ============================================================================
// Inbox Tool Types
// ============================================================================

/// Action to perform on inbox.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InboxAction {
    /// Process inbox items with auto-classification.
    Process,
    /// Classify a single text item.
    Classify,
    /// Classify an item manually.
    ClassifyAs,
    /// Get inbox to zero (auto-classify high confidence).
    InboxZero,
}

/// Parameters for the gtd_inbox tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GtdInboxParams {
    /// The action to perform.
    pub action: InboxAction,
    /// Text content to classify (for classify action).
    #[serde(default)]
    pub content: Option<String>,
    /// Document ID to process items from.
    #[serde(default)]
    pub document_id: Option<String>,
    /// Source ID to process items from.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Inbox item ID (for classify_as).
    #[serde(default)]
    pub item_id: Option<String>,
    /// Target GTD type (for classify_as).
    #[serde(default)]
    pub target_type: Option<String>,
    /// Project ID to associate with.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Contexts to apply.
    #[serde(default)]
    pub contexts: Option<Vec<String>>,
    /// Enable auto-classification.
    #[serde(default = "default_true")]
    pub auto_classify: bool,
    /// Confidence threshold for auto-classification.
    #[serde(default = "default_threshold")]
    pub auto_threshold: f32,
    /// Maximum number of items to process.
    #[serde(default = "default_batch")]
    pub batch_size: usize,
}

fn default_true() -> bool {
    true
}

fn default_threshold() -> f32 {
    0.85
}

fn default_batch() -> usize {
    50
}

impl Default for GtdInboxParams {
    fn default() -> Self {
        Self {
            action: InboxAction::Process,
            content: None,
            document_id: None,
            source_id: None,
            item_id: None,
            target_type: None,
            project_id: None,
            contexts: None,
            auto_classify: true,
            auto_threshold: 0.85,
            batch_size: 50,
        }
    }
}

/// Response for inbox operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdInboxResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Classified inbox item (for classify action).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<crate::gtd::InboxItem>,
    /// All processed items.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<crate::gtd::InboxItem>>,
    /// Auto-classified results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_classified: Option<Vec<crate::gtd::ClassificationResult>>,
    /// Items needing manual review.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub needs_review: Option<Vec<crate::gtd::InboxItem>>,
    /// Quick wins identified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quick_wins: Option<Vec<crate::gtd::InboxItem>>,
    /// Processing summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<crate::gtd::ProcessingSummary>,
    /// Status message.
    pub message: String,
}

impl GtdInboxResponse {
    pub fn success_item(item: crate::gtd::InboxItem, message: impl Into<String>) -> Self {
        Self {
            success: true,
            item: Some(item),
            items: None,
            auto_classified: None,
            needs_review: None,
            quick_wins: None,
            summary: None,
            message: message.into(),
        }
    }

    pub fn success_process(
        response: crate::gtd::ProcessInboxResponse,
        message: impl Into<String>,
    ) -> Self {
        Self {
            success: true,
            item: None,
            items: Some(response.items),
            auto_classified: Some(response.auto_classified),
            needs_review: Some(response.needs_review),
            quick_wins: Some(response.quick_wins),
            summary: Some(response.summary),
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            item: None,
            items: None,
            auto_classified: None,
            needs_review: None,
            quick_wins: None,
            summary: None,
            message: message.into(),
        }
    }
}

// ============================================================================
// Recommend Tool Types
// ============================================================================

/// Parameters for the gtd_recommend tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GtdRecommendParams {
    /// Current context (e.g., "@home", "@work", "@phone").
    #[serde(default)]
    pub current_context: Option<String>,
    /// Current energy level.
    #[serde(default)]
    pub energy_level: Option<EnergyLevel>,
    /// Available time in minutes.
    #[serde(default)]
    pub time_available: Option<u32>,
    /// Focus on a specific area.
    #[serde(default)]
    pub focus_area: Option<String>,
    /// Focus on a specific project.
    #[serde(default)]
    pub focus_project: Option<String>,
    /// Get quick wins only.
    #[serde(default)]
    pub quick_wins_only: bool,
    /// Maximum number of recommendations.
    #[serde(default = "default_recommend_limit")]
    pub limit: usize,
}

fn default_recommend_limit() -> usize {
    10
}

impl Default for GtdRecommendParams {
    fn default() -> Self {
        Self {
            current_context: None,
            energy_level: None,
            time_available: None,
            focus_area: None,
            focus_project: None,
            quick_wins_only: false,
            limit: 10,
        }
    }
}

/// Response for recommendation operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdRecommendResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Recommended tasks (from the advanced recommendation engine).
    pub recommendations: Vec<crate::gtd::recommend::TaskRecommendation>,
    /// Context used for recommendations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<crate::gtd::RecommendContext>,
    /// Reasoning explanation.
    pub reasoning: String,
    /// Status message.
    pub message: String,
}

impl GtdRecommendResponse {
    pub fn success(response: crate::gtd::RecommendResponse, message: impl Into<String>) -> Self {
        Self {
            success: true,
            recommendations: response.recommendations,
            context: Some(response.context),
            reasoning: response.reasoning,
            message: message.into(),
        }
    }

    pub fn success_list(
        recommendations: Vec<crate::gtd::recommend::TaskRecommendation>,
        reasoning: String,
        message: impl Into<String>,
    ) -> Self {
        Self {
            success: true,
            recommendations,
            context: None,
            reasoning,
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            recommendations: Vec::new(),
            context: None,
            reasoning: String::new(),
            message: message.into(),
        }
    }
}

// ============================================================================
// Weekly Review Tool Types
// ============================================================================

/// Parameters for the gtd_weekly_review tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GtdWeeklyReviewParams {
    /// Review type.
    #[serde(default = "default_review_type")]
    pub review_type: ReviewType,
    /// Week ending date (ISO 8601 format).
    #[serde(default)]
    pub week_ending: Option<DateTime<Utc>>,
    /// Sections to include.
    #[serde(default)]
    pub sections: Option<Vec<String>>,
    /// Number of days to look back.
    #[serde(default = "default_lookback")]
    pub lookback_days: u32,
    /// Number of days to look ahead.
    #[serde(default = "default_lookahead")]
    pub lookahead_days: u32,
    /// Days of inactivity to consider project stalled.
    #[serde(default = "default_stalled")]
    pub stalled_days: u32,
    /// Include detailed recommendations.
    #[serde(default = "default_true")]
    pub include_recommendations: bool,
}

/// Type of review to generate.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReviewType {
    /// Full weekly review.
    Weekly,
    /// Quick daily review.
    Daily,
    /// Custom sections.
    Custom,
}

fn default_review_type() -> ReviewType {
    ReviewType::Weekly
}

fn default_lookback() -> u32 {
    7
}

fn default_lookahead() -> u32 {
    7
}

fn default_stalled() -> u32 {
    7
}

impl Default for GtdWeeklyReviewParams {
    fn default() -> Self {
        Self {
            review_type: ReviewType::Weekly,
            week_ending: None,
            sections: None,
            lookback_days: 7,
            lookahead_days: 7,
            stalled_days: 7,
            include_recommendations: true,
        }
    }
}

/// Response for weekly review operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdWeeklyReviewResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// The weekly review report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<crate::gtd::WeeklyReviewReport>,
    /// Status message.
    pub message: String,
}

impl GtdWeeklyReviewResponse {
    pub fn success(report: crate::gtd::WeeklyReviewReport, message: impl Into<String>) -> Self {
        Self {
            success: true,
            report: Some(report),
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            report: None,
            message: message.into(),
        }
    }
}

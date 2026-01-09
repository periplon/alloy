//! Weekly Review functionality for GTD.
//!
//! This module provides comprehensive weekly review reports and assistance
//! following David Allen's GTD methodology.

use chrono::{DateTime, Datelike, Duration, Utc, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{
    Project, ProjectFilter, ProjectHealth, ProjectStatus, SomedayFilter, SomedayItem, Task,
    TaskFilter, TaskStatus, WaitingFilter, WaitingFor, WaitingStatus,
};
use crate::gtd::{ProjectManager, SomedayManager, TaskManager, WaitingManager};
use crate::ontology::{EmbeddedOntologyStore, EntityType, OntologyStore};
use schemars::JsonSchema;

// ============================================================================
// Weekly Review Types
// ============================================================================

/// Sections to include in the weekly review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReviewSection {
    /// Status of unprocessed inbox items.
    InboxStatus,
    /// Tasks completed this week.
    CompletedTasks,
    /// Project health check.
    ProjectsReview,
    /// Projects needing attention.
    StalledProjects,
    /// Waiting-for items needing follow-up.
    WaitingFor,
    /// Upcoming calendar preview.
    UpcomingCalendar,
    /// Someday/maybe review.
    SomedayMaybe,
    /// Area of focus review.
    AreasCheck,
    /// Overdue items.
    OverdueItems,
    /// Quick wins available.
    QuickWins,
    /// All sections.
    All,
}

/// Parameters for generating a weekly review.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WeeklyReviewParams {
    /// Week ending date (defaults to current week).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub week_ending: Option<DateTime<Utc>>,
    /// Sections to include.
    #[serde(default = "default_sections")]
    pub sections: Vec<ReviewSection>,
    /// Number of days to look back for completed tasks.
    #[serde(default = "default_lookback_days")]
    pub lookback_days: u32,
    /// Number of days to look ahead for upcoming items.
    #[serde(default = "default_lookahead_days")]
    pub lookahead_days: u32,
    /// Days without activity to consider stalled.
    #[serde(default = "default_stalled_days")]
    pub stalled_days: u32,
    /// Include detailed recommendations.
    #[serde(default = "default_true")]
    pub include_recommendations: bool,
}

fn default_sections() -> Vec<ReviewSection> {
    vec![ReviewSection::All]
}

fn default_lookback_days() -> u32 {
    7
}

fn default_lookahead_days() -> u32 {
    7
}

fn default_stalled_days() -> u32 {
    7
}

fn default_true() -> bool {
    true
}

impl Default for WeeklyReviewParams {
    fn default() -> Self {
        Self {
            week_ending: None,
            sections: vec![ReviewSection::All],
            lookback_days: 7,
            lookahead_days: 7,
            stalled_days: 7,
            include_recommendations: true,
        }
    }
}

/// The complete weekly review report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyReviewReport {
    /// The period covered by this review.
    pub period: ReviewPeriod,
    /// Inbox status.
    pub inbox: InboxStatus,
    /// Completed tasks summary.
    pub completed_tasks: CompletedTasksSummary,
    /// Active projects with health.
    pub active_projects: ProjectsSummary,
    /// Stalled projects needing attention.
    pub stalled_projects: Vec<ProjectWithHealth>,
    /// Waiting-for items summary.
    pub waiting_for: WaitingForSummary,
    /// Upcoming events and deadlines.
    pub upcoming: UpcomingSummary,
    /// Someday/maybe items due for review.
    pub someday_maybe: SomedayMaybeSummary,
    /// Areas of focus status.
    pub areas: AreasSummary,
    /// Quick wins available.
    pub quick_wins: Vec<Task>,
    /// Overall GTD system health.
    pub system_health: SystemHealth,
    /// Recommendations for actions.
    pub recommendations: Vec<Recommendation>,
    /// When this report was generated.
    pub generated_at: DateTime<Utc>,
}

/// The period covered by the review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewPeriod {
    /// Start of the review period.
    pub start: DateTime<Utc>,
    /// End of the review period.
    pub end: DateTime<Utc>,
    /// Week number.
    pub week_number: u32,
    /// Year.
    pub year: i32,
}

/// Inbox status summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxStatus {
    /// Total items in inbox.
    pub total_count: usize,
    /// Items needing processing.
    pub unprocessed_count: usize,
    /// Is inbox at zero?
    pub at_zero: bool,
    /// Oldest unprocessed item age in days.
    pub oldest_item_days: Option<u32>,
}

/// Summary of completed tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedTasksSummary {
    /// Total tasks completed in period.
    pub total_completed: usize,
    /// Completed by day.
    pub by_day: HashMap<String, usize>,
    /// Completed by project.
    pub by_project: HashMap<String, usize>,
    /// Completed by context.
    pub by_context: HashMap<String, usize>,
    /// Notable completions.
    pub notable_completions: Vec<Task>,
}

/// Summary of projects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectsSummary {
    /// Total active projects.
    pub active_count: usize,
    /// Projects with next actions defined.
    pub with_next_action: usize,
    /// Projects without next actions (need attention).
    pub without_next_action: usize,
    /// Average project health score.
    pub average_health: f32,
    /// Projects by status.
    pub by_status: HashMap<String, usize>,
    /// Projects by area.
    pub by_area: HashMap<String, usize>,
    /// Top projects by health.
    pub healthiest: Vec<ProjectWithHealth>,
    /// Bottom projects by health.
    pub least_healthy: Vec<ProjectWithHealth>,
}

/// A project with its health score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectWithHealth {
    /// The project.
    pub project: Project,
    /// Health information.
    pub health: ProjectHealth,
}

/// Summary of waiting-for items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaitingForSummary {
    /// Total pending items.
    pub pending_count: usize,
    /// Overdue items.
    pub overdue_count: usize,
    /// Items by person.
    pub by_person: HashMap<String, usize>,
    /// Items needing follow-up.
    pub needing_follow_up: Vec<WaitingFor>,
    /// Overdue items list.
    pub overdue_items: Vec<WaitingFor>,
}

/// Summary of upcoming items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpcomingSummary {
    /// Tasks due this week.
    pub tasks_due: Vec<Task>,
    /// Upcoming deadlines.
    pub deadlines: Vec<UpcomingDeadline>,
    /// Days with heavy workload.
    pub busy_days: Vec<String>,
}

/// An upcoming deadline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpcomingDeadline {
    /// Description.
    pub description: String,
    /// Due date.
    pub due_date: DateTime<Utc>,
    /// Days until due.
    pub days_until: i64,
    /// Type (task, project, etc.).
    pub item_type: String,
    /// Related project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
}

/// Summary of someday/maybe items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomedayMaybeSummary {
    /// Total items.
    pub total_count: usize,
    /// Items due for review.
    pub due_for_review: Vec<SomedayItem>,
    /// Items by category.
    pub by_category: HashMap<String, usize>,
    /// Suggestions to activate.
    pub suggested_activations: Vec<SomedayItem>,
}

/// Summary of areas of focus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AreasSummary {
    /// All areas.
    pub areas: Vec<AreaStatus>,
    /// Areas needing attention (no recent activity).
    pub neglected: Vec<String>,
    /// Attention distribution.
    pub attention_distribution: HashMap<String, f32>,
}

/// Status of a single area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AreaStatus {
    /// Area name.
    pub name: String,
    /// Active projects in this area.
    pub active_projects: usize,
    /// Pending tasks in this area.
    pub pending_tasks: usize,
    /// Last activity date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_activity: Option<DateTime<Utc>>,
    /// Relative attention score.
    pub attention_score: f32,
}

/// Overall GTD system health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Overall score (0-100).
    pub score: f32,
    /// Health rating.
    pub rating: HealthRating,
    /// Contributing factors.
    pub factors: SystemHealthFactors,
    /// Summary message.
    pub summary: String,
}

/// Health rating categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthRating {
    /// Excellent (90-100).
    Excellent,
    /// Good (70-89).
    Good,
    /// Needs Attention (50-69).
    NeedsAttention,
    /// Critical (0-49).
    Critical,
}

/// Factors contributing to system health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthFactors {
    /// Inbox at zero or low.
    pub inbox_clear: bool,
    /// All projects have next actions.
    pub projects_healthy: bool,
    /// No overdue waiting items.
    pub waiting_current: bool,
    /// No overdue tasks.
    pub tasks_current: bool,
    /// Someday items reviewed.
    pub someday_reviewed: bool,
    /// Areas balanced.
    pub areas_balanced: bool,
}

/// A recommendation for action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority (1-5, 1 is highest).
    pub priority: u8,
    /// Category.
    pub category: String,
    /// Action to take.
    pub action: String,
    /// Why this is recommended.
    pub reason: String,
    /// Related item ID if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_item_id: Option<String>,
}

// ============================================================================
// Review Manager
// ============================================================================

/// Manages GTD weekly review functionality.
pub struct ReviewManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    task_manager: TaskManager,
    project_manager: ProjectManager,
    waiting_manager: WaitingManager,
    someday_manager: SomedayManager,
}

impl ReviewManager {
    /// Create a new review manager.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self {
            task_manager: TaskManager::new(store.clone()),
            project_manager: ProjectManager::new(store.clone()),
            waiting_manager: WaitingManager::new(store.clone()),
            someday_manager: SomedayManager::new(store.clone()),
            store,
        }
    }

    /// Generate a weekly review report.
    pub async fn generate_weekly_review(
        &self,
        params: WeeklyReviewParams,
    ) -> Result<WeeklyReviewReport> {
        let now = Utc::now();
        let week_ending = params.week_ending.unwrap_or(now);

        let period = self.calculate_period(week_ending);
        let include_all = params.sections.contains(&ReviewSection::All);

        // Generate each section
        let inbox = if include_all || params.sections.contains(&ReviewSection::InboxStatus) {
            self.get_inbox_status().await?
        } else {
            InboxStatus {
                total_count: 0,
                unprocessed_count: 0,
                at_zero: true,
                oldest_item_days: None,
            }
        };

        let completed_tasks =
            if include_all || params.sections.contains(&ReviewSection::CompletedTasks) {
                self.get_completed_tasks(period.start, period.end).await?
            } else {
                CompletedTasksSummary::default()
            };

        let active_projects =
            if include_all || params.sections.contains(&ReviewSection::ProjectsReview) {
                self.get_projects_summary().await?
            } else {
                ProjectsSummary::default()
            };

        let stalled_projects =
            if include_all || params.sections.contains(&ReviewSection::StalledProjects) {
                self.get_stalled_projects(params.stalled_days).await?
            } else {
                Vec::new()
            };

        let waiting_for = if include_all || params.sections.contains(&ReviewSection::WaitingFor) {
            self.get_waiting_for_summary().await?
        } else {
            WaitingForSummary::default()
        };

        let upcoming = if include_all || params.sections.contains(&ReviewSection::UpcomingCalendar)
        {
            self.get_upcoming_summary(params.lookahead_days).await?
        } else {
            UpcomingSummary::default()
        };

        let someday_maybe = if include_all || params.sections.contains(&ReviewSection::SomedayMaybe)
        {
            self.get_someday_maybe_summary().await?
        } else {
            SomedayMaybeSummary::default()
        };

        let areas = if include_all || params.sections.contains(&ReviewSection::AreasCheck) {
            self.get_areas_summary().await?
        } else {
            AreasSummary::default()
        };

        let quick_wins = if include_all || params.sections.contains(&ReviewSection::QuickWins) {
            self.task_manager.get_quick_wins().await?
        } else {
            Vec::new()
        };

        // Calculate system health
        let system_health = self
            .calculate_system_health(&inbox, &active_projects, &waiting_for, &someday_maybe)
            .await;

        // Generate recommendations
        let recommendations = if params.include_recommendations {
            self.generate_recommendations(
                &inbox,
                &active_projects,
                &stalled_projects,
                &waiting_for,
                &upcoming,
                &someday_maybe,
                &quick_wins,
            )
            .await
        } else {
            Vec::new()
        };

        Ok(WeeklyReviewReport {
            period,
            inbox,
            completed_tasks,
            active_projects,
            stalled_projects,
            waiting_for,
            upcoming,
            someday_maybe,
            areas,
            quick_wins,
            system_health,
            recommendations,
            generated_at: Utc::now(),
        })
    }

    /// Get a quick daily review (subset of weekly).
    pub async fn generate_daily_review(&self) -> Result<WeeklyReviewReport> {
        self.generate_weekly_review(WeeklyReviewParams {
            sections: vec![
                ReviewSection::InboxStatus,
                ReviewSection::OverdueItems,
                ReviewSection::UpcomingCalendar,
                ReviewSection::QuickWins,
            ],
            lookback_days: 1,
            lookahead_days: 1,
            include_recommendations: true,
            ..Default::default()
        })
        .await
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    fn calculate_period(&self, week_ending: DateTime<Utc>) -> ReviewPeriod {
        // Find the end of the week (Sunday)
        let days_to_sunday = match week_ending.weekday() {
            Weekday::Sun => 0,
            Weekday::Mon => 6,
            Weekday::Tue => 5,
            Weekday::Wed => 4,
            Weekday::Thu => 3,
            Weekday::Fri => 2,
            Weekday::Sat => 1,
        };

        let end = week_ending + Duration::days(days_to_sunday as i64);
        let start = end - Duration::days(6);
        let week_number = start.iso_week().week();
        let year = start.year();

        ReviewPeriod {
            start,
            end,
            week_number,
            year,
        }
    }

    async fn get_inbox_status(&self) -> Result<InboxStatus> {
        let store = self.store.read().await;

        // Get entities that might be inbox items (tasks/commitments without gtd_processed)
        let tasks = store.find_entities_by_type(EntityType::Task, 500).await?;
        let commitments = store
            .find_entities_by_type(EntityType::Commitment, 500)
            .await?;

        let all_items: Vec<_> = tasks.into_iter().chain(commitments).collect();

        let unprocessed: Vec<_> = all_items
            .iter()
            .filter(|e| !e.metadata.contains_key("gtd_processed"))
            .collect();

        let oldest_item_days = unprocessed
            .iter()
            .map(|e| (Utc::now() - e.created_at).num_days() as u32)
            .max();

        Ok(InboxStatus {
            total_count: all_items.len(),
            unprocessed_count: unprocessed.len(),
            at_zero: unprocessed.is_empty(),
            oldest_item_days,
        })
    }

    async fn get_completed_tasks(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<CompletedTasksSummary> {
        let tasks = self
            .task_manager
            .list(TaskFilter {
                status: Some(TaskStatus::Done),
                ..Default::default()
            })
            .await?;

        let completed_in_period: Vec<Task> = tasks
            .into_iter()
            .filter(|t| {
                t.completed_at
                    .map(|c| c >= start && c <= end)
                    .unwrap_or(false)
            })
            .collect();

        let mut by_day: HashMap<String, usize> = HashMap::new();
        let mut by_project: HashMap<String, usize> = HashMap::new();
        let mut by_context: HashMap<String, usize> = HashMap::new();

        for task in &completed_in_period {
            if let Some(completed) = task.completed_at {
                let day = completed.format("%Y-%m-%d").to_string();
                *by_day.entry(day).or_default() += 1;
            }

            if let Some(ref proj) = task.project_id {
                *by_project.entry(proj.clone()).or_default() += 1;
            }

            for ctx in &task.contexts {
                *by_context.entry(ctx.clone()).or_default() += 1;
            }
        }

        // Notable completions (high priority or had due date)
        let notable: Vec<Task> = completed_in_period
            .iter()
            .filter(|t| {
                t.priority == crate::gtd::types::Priority::High
                    || t.priority == crate::gtd::types::Priority::Critical
                    || t.due_date.is_some()
            })
            .take(5)
            .cloned()
            .collect();

        Ok(CompletedTasksSummary {
            total_completed: completed_in_period.len(),
            by_day,
            by_project,
            by_context,
            notable_completions: notable,
        })
    }

    async fn get_projects_summary(&self) -> Result<ProjectsSummary> {
        let projects = self.project_manager.list(ProjectFilter::default()).await?;

        let active: Vec<&Project> = projects
            .iter()
            .filter(|p| p.status == ProjectStatus::Active)
            .collect();

        let with_next_action = active.iter().filter(|p| p.next_action_id.is_some()).count();
        let without_next_action = active.len() - with_next_action;

        let mut by_status: HashMap<String, usize> = HashMap::new();
        let mut by_area: HashMap<String, usize> = HashMap::new();

        for project in &projects {
            *by_status
                .entry(format!("{:?}", project.status))
                .or_default() += 1;
            if let Some(ref area) = project.area {
                *by_area.entry(area.clone()).or_default() += 1;
            }
        }

        // Get health for active projects
        let mut projects_with_health: Vec<ProjectWithHealth> = Vec::new();
        for project in &active {
            if let Some(health) = self.project_manager.get_health(&project.id).await? {
                projects_with_health.push(ProjectWithHealth {
                    project: (*project).clone(),
                    health,
                });
            }
        }

        projects_with_health.sort_by(|a, b| {
            b.health
                .score
                .partial_cmp(&a.health.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let average_health = if projects_with_health.is_empty() {
            100.0
        } else {
            projects_with_health
                .iter()
                .map(|p| p.health.score)
                .sum::<f32>()
                / projects_with_health.len() as f32
        };

        let healthiest: Vec<ProjectWithHealth> =
            projects_with_health.iter().take(3).cloned().collect();
        let least_healthy: Vec<ProjectWithHealth> =
            projects_with_health.iter().rev().take(3).cloned().collect();

        Ok(ProjectsSummary {
            active_count: active.len(),
            with_next_action,
            without_next_action,
            average_health,
            by_status,
            by_area,
            healthiest,
            least_healthy,
        })
    }

    async fn get_stalled_projects(&self, stalled_days: u32) -> Result<Vec<ProjectWithHealth>> {
        let stalled = self.project_manager.get_stalled(stalled_days).await?;

        let mut result = Vec::new();
        for project in stalled {
            if let Some(health) = self.project_manager.get_health(&project.id).await? {
                result.push(ProjectWithHealth { project, health });
            }
        }

        Ok(result)
    }

    async fn get_waiting_for_summary(&self) -> Result<WaitingForSummary> {
        let pending = self
            .waiting_manager
            .list(WaitingFilter {
                status: Some(WaitingStatus::Pending),
                ..Default::default()
            })
            .await?;

        let overdue = self.waiting_manager.get_overdue().await?;

        let mut by_person: HashMap<String, usize> = HashMap::new();
        for item in &pending {
            *by_person.entry(item.delegated_to.clone()).or_default() += 1;
        }

        // Items needing follow-up (no follow-up in 7 days)
        let needing_follow_up = self.waiting_manager.get_needing_follow_up(7).await?;

        Ok(WaitingForSummary {
            pending_count: pending.len(),
            overdue_count: overdue.len(),
            by_person,
            needing_follow_up,
            overdue_items: overdue,
        })
    }

    async fn get_upcoming_summary(&self, lookahead_days: u32) -> Result<UpcomingSummary> {
        let now = Utc::now();
        let end = now + Duration::days(lookahead_days as i64);

        let tasks = self
            .task_manager
            .list(TaskFilter {
                status: Some(TaskStatus::Next),
                due_before: Some(end),
                ..Default::default()
            })
            .await?;

        let mut deadlines: Vec<UpcomingDeadline> = Vec::new();
        let mut by_day: HashMap<String, usize> = HashMap::new();

        for task in &tasks {
            if let Some(due) = task.due_date {
                let days_until = (due - now).num_days();
                let day = due.format("%Y-%m-%d").to_string();
                *by_day.entry(day.clone()).or_default() += 1;

                deadlines.push(UpcomingDeadline {
                    description: task.description.clone(),
                    due_date: due,
                    days_until,
                    item_type: "task".to_string(),
                    project: task.project_id.clone(),
                });
            }
        }

        deadlines.sort_by_key(|d| d.due_date);

        // Find busy days (more than 3 items due)
        let busy_days: Vec<String> = by_day
            .iter()
            .filter(|(_, count)| **count > 3)
            .map(|(day, _)| day.clone())
            .collect();

        Ok(UpcomingSummary {
            tasks_due: tasks,
            deadlines,
            busy_days,
        })
    }

    async fn get_someday_maybe_summary(&self) -> Result<SomedayMaybeSummary> {
        let all_items = self.someday_manager.list(SomedayFilter::default()).await?;

        let due_for_review = self.someday_manager.get_due_for_review().await?;

        let mut by_category: HashMap<String, usize> = HashMap::new();
        for item in &all_items {
            if let Some(ref cat) = item.category {
                *by_category.entry(cat.clone()).or_default() += 1;
            } else {
                *by_category.entry("Uncategorized".to_string()).or_default() += 1;
            }
        }

        // Suggest activations for items that have been on the list a while
        let suggested: Vec<SomedayItem> = all_items
            .iter()
            .filter(|i| {
                let age_days = (Utc::now() - i.created_at).num_days();
                age_days > 30 && i.trigger.is_some()
            })
            .take(3)
            .cloned()
            .collect();

        Ok(SomedayMaybeSummary {
            total_count: all_items.len(),
            due_for_review,
            by_category,
            suggested_activations: suggested,
        })
    }

    async fn get_areas_summary(&self) -> Result<AreasSummary> {
        let store = self.store.read().await;

        let area_entities = store.find_entities_by_type(EntityType::Area, 100).await?;

        let projects = self.project_manager.list(ProjectFilter::default()).await?;

        drop(store);

        let tasks = self.task_manager.list(TaskFilter::default()).await?;

        let mut areas: Vec<AreaStatus> = Vec::new();
        let mut attention_distribution: HashMap<String, f32> = HashMap::new();
        let total_activity = projects.len() + tasks.len();

        for area_entity in area_entities {
            let area_name = area_entity.name.clone();

            let area_projects: Vec<&Project> = projects
                .iter()
                .filter(|p| p.area.as_ref() == Some(&area_name))
                .collect();

            let area_tasks: Vec<&Task> = tasks
                .iter()
                .filter(|t| {
                    if let Some(ref proj_id) = t.project_id {
                        area_projects.iter().any(|p| &p.id == proj_id)
                    } else {
                        false
                    }
                })
                .collect();

            let last_activity = area_projects
                .iter()
                .map(|p| p.updated_at)
                .chain(area_tasks.iter().map(|t| t.updated_at))
                .max();

            let activity_count = area_projects.len() + area_tasks.len();
            let attention_score = if total_activity > 0 {
                activity_count as f32 / total_activity as f32
            } else {
                0.0
            };

            attention_distribution.insert(area_name.clone(), attention_score);

            areas.push(AreaStatus {
                name: area_name,
                active_projects: area_projects.len(),
                pending_tasks: area_tasks.len(),
                last_activity,
                attention_score,
            });
        }

        // Find neglected areas (no activity in 14 days)
        let threshold = Utc::now() - Duration::days(14);
        let neglected: Vec<String> = areas
            .iter()
            .filter(|a| a.last_activity.map(|d| d < threshold).unwrap_or(true))
            .map(|a| a.name.clone())
            .collect();

        Ok(AreasSummary {
            areas,
            neglected,
            attention_distribution,
        })
    }

    async fn calculate_system_health(
        &self,
        inbox: &InboxStatus,
        projects: &ProjectsSummary,
        waiting: &WaitingForSummary,
        someday: &SomedayMaybeSummary,
    ) -> SystemHealth {
        let inbox_clear = inbox.at_zero || inbox.unprocessed_count < 5;
        let projects_healthy = projects.without_next_action == 0;
        let waiting_current = waiting.overdue_count == 0;
        let tasks_current = true; // Would need to check overdue tasks
        let someday_reviewed = someday.due_for_review.is_empty();
        let areas_balanced = true; // Would need more analysis

        let mut score = 0.0f32;
        if inbox_clear {
            score += 20.0;
        }
        if projects_healthy {
            score += 25.0;
        }
        if waiting_current {
            score += 20.0;
        }
        if tasks_current {
            score += 15.0;
        }
        if someday_reviewed {
            score += 10.0;
        }
        if areas_balanced {
            score += 10.0;
        }

        let rating = match score as u32 {
            90..=100 => HealthRating::Excellent,
            70..=89 => HealthRating::Good,
            50..=69 => HealthRating::NeedsAttention,
            _ => HealthRating::Critical,
        };

        let summary = match rating {
            HealthRating::Excellent => "Your GTD system is in excellent shape!".to_string(),
            HealthRating::Good => {
                "Your GTD system is healthy with minor areas to address.".to_string()
            }
            HealthRating::NeedsAttention => {
                "Your GTD system needs some attention - review the recommendations.".to_string()
            }
            HealthRating::Critical => {
                "Your GTD system needs immediate attention to get back on track.".to_string()
            }
        };

        SystemHealth {
            score,
            rating,
            factors: SystemHealthFactors {
                inbox_clear,
                projects_healthy,
                waiting_current,
                tasks_current,
                someday_reviewed,
                areas_balanced,
            },
            summary,
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn generate_recommendations(
        &self,
        inbox: &InboxStatus,
        projects: &ProjectsSummary,
        stalled: &[ProjectWithHealth],
        waiting: &WaitingForSummary,
        upcoming: &UpcomingSummary,
        someday: &SomedayMaybeSummary,
        quick_wins: &[Task],
    ) -> Vec<Recommendation> {
        let mut recs = Vec::new();

        // Inbox recommendations
        if !inbox.at_zero {
            recs.push(Recommendation {
                priority: 1,
                category: "Inbox".to_string(),
                action: format!(
                    "Process {} inbox items to get to inbox zero",
                    inbox.unprocessed_count
                ),
                reason: "A clear inbox is essential for trusted GTD system".to_string(),
                related_item_id: None,
            });
        }

        if let Some(days) = inbox.oldest_item_days {
            if days > 3 {
                recs.push(Recommendation {
                    priority: 2,
                    category: "Inbox".to_string(),
                    action: format!("Address items that have been in inbox for {} days", days),
                    reason: "Old inbox items indicate processing backlog".to_string(),
                    related_item_id: None,
                });
            }
        }

        // Project recommendations
        if projects.without_next_action > 0 {
            recs.push(Recommendation {
                priority: 1,
                category: "Projects".to_string(),
                action: format!(
                    "Define next actions for {} projects without them",
                    projects.without_next_action
                ),
                reason: "Every active project needs a clear next action".to_string(),
                related_item_id: None,
            });
        }

        for stalled_project in stalled.iter().take(3) {
            recs.push(Recommendation {
                priority: 2,
                category: "Projects".to_string(),
                action: format!("Review stalled project: {}", stalled_project.project.name),
                reason: "Project has had no activity recently".to_string(),
                related_item_id: Some(stalled_project.project.id.clone()),
            });
        }

        // Waiting for recommendations
        for overdue in waiting.overdue_items.iter().take(3) {
            recs.push(Recommendation {
                priority: 1,
                category: "Waiting For".to_string(),
                action: format!(
                    "Follow up with {} on: {}",
                    overdue.delegated_to, overdue.description
                ),
                reason: "Item is overdue".to_string(),
                related_item_id: Some(overdue.id.clone()),
            });
        }

        for follow_up in waiting.needing_follow_up.iter().take(2) {
            recs.push(Recommendation {
                priority: 3,
                category: "Waiting For".to_string(),
                action: format!(
                    "Check in with {} about: {}",
                    follow_up.delegated_to, follow_up.description
                ),
                reason: "No follow-up recorded recently".to_string(),
                related_item_id: Some(follow_up.id.clone()),
            });
        }

        // Upcoming/deadline recommendations
        for deadline in upcoming
            .deadlines
            .iter()
            .filter(|d| d.days_until <= 2)
            .take(3)
        {
            recs.push(Recommendation {
                priority: 1,
                category: "Deadlines".to_string(),
                action: format!("Complete before deadline: {}", deadline.description),
                reason: format!("Due in {} day(s)", deadline.days_until),
                related_item_id: None,
            });
        }

        // Quick wins
        if !quick_wins.is_empty() {
            recs.push(Recommendation {
                priority: 4,
                category: "Quick Wins".to_string(),
                action: format!("Knock out {} quick 2-minute tasks", quick_wins.len()),
                reason: "Quick wins build momentum and clear mental space".to_string(),
                related_item_id: None,
            });
        }

        // Someday/maybe
        if !someday.due_for_review.is_empty() {
            recs.push(Recommendation {
                priority: 4,
                category: "Someday/Maybe".to_string(),
                action: format!(
                    "Review {} someday/maybe items due for reconsideration",
                    someday.due_for_review.len()
                ),
                reason: "Regular review keeps this list relevant".to_string(),
                related_item_id: None,
            });
        }

        // Sort by priority
        recs.sort_by_key(|r| r.priority);

        recs
    }
}

// ============================================================================
// Default Implementations
// ============================================================================

#[allow(clippy::derivable_impls)]
impl Default for CompletedTasksSummary {
    fn default() -> Self {
        Self {
            total_completed: 0,
            by_day: HashMap::new(),
            by_project: HashMap::new(),
            by_context: HashMap::new(),
            notable_completions: Vec::new(),
        }
    }
}

impl Default for ProjectsSummary {
    fn default() -> Self {
        Self {
            active_count: 0,
            with_next_action: 0,
            without_next_action: 0,
            average_health: 100.0,
            by_status: HashMap::new(),
            by_area: HashMap::new(),
            healthiest: Vec::new(),
            least_healthy: Vec::new(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for WaitingForSummary {
    fn default() -> Self {
        Self {
            pending_count: 0,
            overdue_count: 0,
            by_person: HashMap::new(),
            needing_follow_up: Vec::new(),
            overdue_items: Vec::new(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for UpcomingSummary {
    fn default() -> Self {
        Self {
            tasks_due: Vec::new(),
            deadlines: Vec::new(),
            busy_days: Vec::new(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for SomedayMaybeSummary {
    fn default() -> Self {
        Self {
            total_count: 0,
            due_for_review: Vec::new(),
            by_category: HashMap::new(),
            suggested_activations: Vec::new(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for AreasSummary {
    fn default() -> Self {
        Self {
            areas: Vec::new(),
            neglected: Vec::new(),
            attention_distribution: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_manager() -> ReviewManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        ReviewManager::new(store)
    }

    #[tokio::test]
    async fn test_generate_weekly_review() {
        let manager = create_test_manager().await;

        let report = manager
            .generate_weekly_review(WeeklyReviewParams::default())
            .await
            .unwrap();

        assert!(report.period.start < report.period.end);
        assert!(report.generated_at <= Utc::now());
    }

    #[tokio::test]
    async fn test_generate_daily_review() {
        let manager = create_test_manager().await;

        let report = manager.generate_daily_review().await.unwrap();

        assert!(report.generated_at <= Utc::now());
    }

    #[tokio::test]
    async fn test_system_health_calculation() {
        let manager = create_test_manager().await;

        let inbox = InboxStatus {
            total_count: 0,
            unprocessed_count: 0,
            at_zero: true,
            oldest_item_days: None,
        };

        let projects = ProjectsSummary::default();
        let waiting = WaitingForSummary::default();
        let someday = SomedayMaybeSummary::default();

        let health = manager
            .calculate_system_health(&inbox, &projects, &waiting, &someday)
            .await;

        assert!(health.score >= 0.0 && health.score <= 100.0);
    }

    #[tokio::test]
    async fn test_period_calculation() {
        let manager = create_test_manager().await;
        let now = Utc::now();

        let period = manager.calculate_period(now);

        assert!(period.start <= period.end);
        assert_eq!((period.end - period.start).num_days(), 6);
    }
}

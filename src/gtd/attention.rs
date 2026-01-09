//! Attention Economics for GTD.
//!
//! This module tracks where your attention goes across projects and areas,
//! providing insights into focus allocation and balance.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{ProjectFilter, Task, TaskFilter, TaskStatus};
use crate::gtd::{ProjectManager, TaskManager};
use crate::ontology::{EmbeddedOntologyStore, EntityType, OntologyStore};
use schemars::JsonSchema;

// ============================================================================
// Attention Types
// ============================================================================

/// Parameters for attention analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AttentionParams {
    /// Start of the analysis period.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub period_start: Option<DateTime<Utc>>,
    /// End of the analysis period.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub period_end: Option<DateTime<Utc>>,
    /// Number of days to analyze (default: 30).
    #[serde(default = "default_period_days")]
    pub period_days: u32,
    /// Include detailed breakdown by project.
    #[serde(default = "default_true")]
    pub include_projects: bool,
    /// Include attention trends over time.
    #[serde(default = "default_true")]
    pub include_trends: bool,
    /// Areas to focus on (empty = all).
    #[serde(default)]
    pub focus_areas: Vec<String>,
}

fn default_period_days() -> u32 {
    30
}

fn default_true() -> bool {
    true
}

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            period_start: None,
            period_end: None,
            period_days: 30,
            include_projects: true,
            include_trends: true,
            focus_areas: Vec::new(),
        }
    }
}

/// Complete attention metrics report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMetrics {
    /// Period analyzed.
    pub period: AttentionPeriod,
    /// Attention distribution by area.
    pub by_area: HashMap<String, AttentionScore>,
    /// Attention distribution by project.
    pub by_project: HashMap<String, AttentionScore>,
    /// Attention distribution by context.
    pub by_context: HashMap<String, AttentionScore>,
    /// Detected imbalances.
    pub imbalances: Vec<AttentionImbalance>,
    /// Daily attention trends.
    pub trends: Vec<DailyAttention>,
    /// Focus depth score (how focused vs fragmented).
    pub focus_depth: FocusDepth,
    /// Recommendations for attention allocation.
    pub recommendations: Vec<String>,
    /// When this report was generated.
    pub generated_at: DateTime<Utc>,
}

/// Time period for attention analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPeriod {
    /// Start of period.
    pub start: DateTime<Utc>,
    /// End of period.
    pub end: DateTime<Utc>,
    /// Number of days.
    pub days: u32,
}

/// Attention score for an entity (area, project, context).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionScore {
    /// Name/identifier.
    pub name: String,
    /// Number of tasks completed.
    pub tasks_completed: usize,
    /// Number of tasks created.
    pub tasks_created: usize,
    /// Number of tasks currently active.
    pub active_tasks: usize,
    /// Estimated time invested (if tracked) in minutes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_invested_minutes: Option<u32>,
    /// Documents/items created or modified.
    pub documents_touched: usize,
    /// Relative attention score (0.0-1.0).
    pub relative_score: f32,
    /// Trend compared to previous period.
    pub trend: AttentionTrend,
    /// Last activity timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_activity: Option<DateTime<Utc>>,
}

/// Trend direction for attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionTrend {
    /// Increasing attention.
    Increasing,
    /// Stable attention.
    Stable,
    /// Decreasing attention.
    Decreasing,
    /// No previous data to compare.
    New,
}

/// An imbalance in attention allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionImbalance {
    /// Area or project with imbalance.
    pub entity_name: String,
    /// Entity type (area, project).
    pub entity_type: String,
    /// Expected attention based on priority/importance.
    pub expected_attention: f32,
    /// Actual attention received.
    pub actual_attention: f32,
    /// Gap (positive = over-attended, negative = under-attended).
    pub gap: f32,
    /// Severity level.
    pub severity: ImbalanceSeverity,
    /// Suggested action.
    pub suggestion: String,
}

/// Severity of an attention imbalance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImbalanceSeverity {
    /// Minor imbalance.
    Low,
    /// Moderate imbalance.
    Medium,
    /// Significant imbalance.
    High,
    /// Critical imbalance.
    Critical,
}

/// Daily attention breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyAttention {
    /// The date.
    pub date: String,
    /// Tasks completed that day.
    pub tasks_completed: usize,
    /// Tasks created that day.
    pub tasks_created: usize,
    /// Primary focus areas that day.
    pub focus_areas: Vec<String>,
    /// Primary focus projects that day.
    pub focus_projects: Vec<String>,
    /// Focus depth for the day.
    pub focus_depth: f32,
}

/// Focus depth analysis - how focused vs fragmented attention is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusDepth {
    /// Overall focus score (0.0-1.0, higher = more focused).
    pub score: f32,
    /// Rating category.
    pub rating: FocusRating,
    /// Average areas touched per day.
    pub areas_per_day: f32,
    /// Average projects touched per day.
    pub projects_per_day: f32,
    /// Context switches detected.
    pub context_switches: usize,
    /// Description of focus pattern.
    pub description: String,
}

/// Focus rating categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusRating {
    /// Deep focus (few context switches, concentrated work).
    DeepFocus,
    /// Good focus.
    Focused,
    /// Moderate focus.
    Moderate,
    /// Fragmented attention.
    Fragmented,
    /// Highly scattered attention.
    Scattered,
}

// ============================================================================
// Attention Manager
// ============================================================================

/// Manages attention economics analysis.
pub struct AttentionManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    task_manager: TaskManager,
    project_manager: ProjectManager,
}

impl AttentionManager {
    /// Create a new attention manager.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self {
            task_manager: TaskManager::new(store.clone()),
            project_manager: ProjectManager::new(store.clone()),
            store,
        }
    }

    /// Analyze attention metrics.
    pub async fn analyze(&self, params: AttentionParams) -> Result<AttentionMetrics> {
        let now = Utc::now();
        let period_end = params.period_end.unwrap_or(now);
        let period_start = params
            .period_start
            .unwrap_or_else(|| period_end - Duration::days(params.period_days as i64));

        let period = AttentionPeriod {
            start: period_start,
            end: period_end,
            days: params.period_days,
        };

        // Gather all tasks for the period
        let all_tasks = self.task_manager.list(TaskFilter::default()).await?;
        let all_projects = self.project_manager.list(ProjectFilter::default()).await?;

        // Filter tasks by period
        let completed_in_period: Vec<&Task> = all_tasks
            .iter()
            .filter(|t| {
                t.completed_at
                    .map(|c| c >= period_start && c <= period_end)
                    .unwrap_or(false)
            })
            .collect();

        let created_in_period: Vec<&Task> = all_tasks
            .iter()
            .filter(|t| t.created_at >= period_start && t.created_at <= period_end)
            .collect();

        // Calculate attention by area
        let by_area = self
            .calculate_area_attention(&completed_in_period, &created_in_period, &all_projects)
            .await?;

        // Calculate attention by project
        let by_project = self
            .calculate_project_attention(&completed_in_period, &created_in_period, &all_tasks)
            .await;

        // Calculate attention by context
        let by_context = self
            .calculate_context_attention(&completed_in_period, &created_in_period)
            .await;

        // Detect imbalances
        let imbalances = self.detect_imbalances(&by_area, &by_project).await;

        // Calculate daily trends
        let trends = if params.include_trends {
            self.calculate_daily_trends(&all_tasks, period_start, period_end)
                .await
        } else {
            Vec::new()
        };

        // Calculate focus depth
        let focus_depth = self.calculate_focus_depth(&trends, &by_area, &by_project);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&imbalances, &focus_depth);

        Ok(AttentionMetrics {
            period,
            by_area,
            by_project,
            by_context,
            imbalances,
            trends,
            focus_depth,
            recommendations,
            generated_at: Utc::now(),
        })
    }

    /// Get a quick attention summary.
    pub async fn quick_summary(&self) -> Result<AttentionMetrics> {
        self.analyze(AttentionParams {
            period_days: 7,
            include_trends: false,
            ..Default::default()
        })
        .await
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    async fn calculate_area_attention(
        &self,
        completed: &[&Task],
        created: &[&Task],
        projects: &[crate::gtd::types::Project],
    ) -> Result<HashMap<String, AttentionScore>> {
        let store = self.store.read().await;
        let area_entities = store.find_entities_by_type(EntityType::Area, 100).await?;
        drop(store);

        let mut scores: HashMap<String, AttentionScore> = HashMap::new();

        // Build project -> area mapping
        let mut project_to_area: HashMap<String, String> = HashMap::new();
        for project in projects {
            if let Some(ref area) = project.area {
                project_to_area.insert(project.id.clone(), area.clone());
            }
        }

        // Initialize all areas
        for area in &area_entities {
            scores.insert(
                area.name.clone(),
                AttentionScore {
                    name: area.name.clone(),
                    tasks_completed: 0,
                    tasks_created: 0,
                    active_tasks: 0,
                    time_invested_minutes: None,
                    documents_touched: 0,
                    relative_score: 0.0,
                    trend: AttentionTrend::New,
                    last_activity: None,
                },
            );
        }

        // Count completed tasks by area
        for task in completed {
            if let Some(ref proj_id) = task.project_id {
                if let Some(area) = project_to_area.get(proj_id) {
                    if let Some(score) = scores.get_mut(area) {
                        score.tasks_completed += 1;
                        if let Some(mins) = task.estimated_minutes {
                            let current = score.time_invested_minutes.unwrap_or(0);
                            score.time_invested_minutes = Some(current + mins);
                        }
                        if task.completed_at > score.last_activity {
                            score.last_activity = task.completed_at;
                        }
                    }
                }
            }
        }

        // Count created tasks by area
        for task in created {
            if let Some(ref proj_id) = task.project_id {
                if let Some(area) = project_to_area.get(proj_id) {
                    if let Some(score) = scores.get_mut(area) {
                        score.tasks_created += 1;
                    }
                }
            }
        }

        // Calculate relative scores
        let total_completed: usize = scores.values().map(|s| s.tasks_completed).sum();
        if total_completed > 0 {
            for score in scores.values_mut() {
                score.relative_score = score.tasks_completed as f32 / total_completed as f32;
            }
        }

        Ok(scores)
    }

    async fn calculate_project_attention(
        &self,
        completed: &[&Task],
        created: &[&Task],
        all_tasks: &[Task],
    ) -> HashMap<String, AttentionScore> {
        let mut scores: HashMap<String, AttentionScore> = HashMap::new();

        // Count completed tasks by project
        for task in completed {
            if let Some(ref proj_id) = task.project_id {
                let score = scores
                    .entry(proj_id.clone())
                    .or_insert_with(|| AttentionScore {
                        name: proj_id.clone(),
                        tasks_completed: 0,
                        tasks_created: 0,
                        active_tasks: 0,
                        time_invested_minutes: None,
                        documents_touched: 0,
                        relative_score: 0.0,
                        trend: AttentionTrend::New,
                        last_activity: None,
                    });
                score.tasks_completed += 1;
                if let Some(mins) = task.estimated_minutes {
                    let current = score.time_invested_minutes.unwrap_or(0);
                    score.time_invested_minutes = Some(current + mins);
                }
                if task.completed_at > score.last_activity {
                    score.last_activity = task.completed_at;
                }
            }
        }

        // Count created tasks by project
        for task in created {
            if let Some(ref proj_id) = task.project_id {
                let score = scores
                    .entry(proj_id.clone())
                    .or_insert_with(|| AttentionScore {
                        name: proj_id.clone(),
                        tasks_completed: 0,
                        tasks_created: 0,
                        active_tasks: 0,
                        time_invested_minutes: None,
                        documents_touched: 0,
                        relative_score: 0.0,
                        trend: AttentionTrend::New,
                        last_activity: None,
                    });
                score.tasks_created += 1;
            }
        }

        // Count active tasks by project
        for task in all_tasks {
            if task.status != TaskStatus::Done {
                if let Some(ref proj_id) = task.project_id {
                    if let Some(score) = scores.get_mut(proj_id) {
                        score.active_tasks += 1;
                    }
                }
            }
        }

        // Calculate relative scores
        let total_completed: usize = scores.values().map(|s| s.tasks_completed).sum();
        if total_completed > 0 {
            for score in scores.values_mut() {
                score.relative_score = score.tasks_completed as f32 / total_completed as f32;
            }
        }

        scores
    }

    async fn calculate_context_attention(
        &self,
        completed: &[&Task],
        created: &[&Task],
    ) -> HashMap<String, AttentionScore> {
        let mut scores: HashMap<String, AttentionScore> = HashMap::new();

        // Count completed tasks by context
        for task in completed {
            for ctx in &task.contexts {
                let score = scores.entry(ctx.clone()).or_insert_with(|| AttentionScore {
                    name: ctx.clone(),
                    tasks_completed: 0,
                    tasks_created: 0,
                    active_tasks: 0,
                    time_invested_minutes: None,
                    documents_touched: 0,
                    relative_score: 0.0,
                    trend: AttentionTrend::New,
                    last_activity: None,
                });
                score.tasks_completed += 1;
                if let Some(mins) = task.estimated_minutes {
                    let current = score.time_invested_minutes.unwrap_or(0);
                    score.time_invested_minutes = Some(current + mins);
                }
                if task.completed_at > score.last_activity {
                    score.last_activity = task.completed_at;
                }
            }
        }

        // Count created tasks by context
        for task in created {
            for ctx in &task.contexts {
                let score = scores.entry(ctx.clone()).or_insert_with(|| AttentionScore {
                    name: ctx.clone(),
                    tasks_completed: 0,
                    tasks_created: 0,
                    active_tasks: 0,
                    time_invested_minutes: None,
                    documents_touched: 0,
                    relative_score: 0.0,
                    trend: AttentionTrend::New,
                    last_activity: None,
                });
                score.tasks_created += 1;
            }
        }

        // Calculate relative scores
        let total_completed: usize = scores.values().map(|s| s.tasks_completed).sum();
        if total_completed > 0 {
            for score in scores.values_mut() {
                score.relative_score = score.tasks_completed as f32 / total_completed as f32;
            }
        }

        scores
    }

    async fn detect_imbalances(
        &self,
        by_area: &HashMap<String, AttentionScore>,
        _by_project: &HashMap<String, AttentionScore>,
    ) -> Vec<AttentionImbalance> {
        let mut imbalances = Vec::new();

        // Detect area imbalances
        // For now, we assume all areas should get equal attention
        // In a real system, this would be based on priority settings
        let area_count = by_area.len();
        if area_count > 0 {
            let expected_per_area = 1.0 / area_count as f32;

            for (name, score) in by_area {
                let gap = score.relative_score - expected_per_area;
                let abs_gap = gap.abs();

                if abs_gap > 0.15 {
                    // More than 15% deviation
                    let severity = if abs_gap > 0.4 {
                        ImbalanceSeverity::Critical
                    } else if abs_gap > 0.3 {
                        ImbalanceSeverity::High
                    } else if abs_gap > 0.2 {
                        ImbalanceSeverity::Medium
                    } else {
                        ImbalanceSeverity::Low
                    };

                    let suggestion = if gap < 0.0 {
                        format!(
                            "Allocate more attention to '{}' - it's receiving less focus than expected",
                            name
                        )
                    } else {
                        format!(
                            "Consider if '{}' is receiving too much focus relative to other areas",
                            name
                        )
                    };

                    imbalances.push(AttentionImbalance {
                        entity_name: name.clone(),
                        entity_type: "area".to_string(),
                        expected_attention: expected_per_area,
                        actual_attention: score.relative_score,
                        gap,
                        severity,
                        suggestion,
                    });
                }
            }
        }

        // Sort by severity (critical first)
        imbalances.sort_by(|a, b| {
            let severity_ord = |s: &ImbalanceSeverity| match s {
                ImbalanceSeverity::Critical => 0,
                ImbalanceSeverity::High => 1,
                ImbalanceSeverity::Medium => 2,
                ImbalanceSeverity::Low => 3,
            };
            severity_ord(&a.severity).cmp(&severity_ord(&b.severity))
        });

        imbalances
    }

    async fn calculate_daily_trends(
        &self,
        tasks: &[Task],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<DailyAttention> {
        let mut trends = Vec::new();
        let mut current = start;

        while current <= end {
            let day_start = current;
            let day_end = current + Duration::days(1);
            let date_str = current.format("%Y-%m-%d").to_string();

            // Tasks completed on this day
            let completed: Vec<&Task> = tasks
                .iter()
                .filter(|t| {
                    t.completed_at
                        .map(|c| c >= day_start && c < day_end)
                        .unwrap_or(false)
                })
                .collect();

            // Tasks created on this day
            let created: Vec<&Task> = tasks
                .iter()
                .filter(|t| t.created_at >= day_start && t.created_at < day_end)
                .collect();

            // Get unique areas and projects touched
            let areas: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut projects: std::collections::HashSet<String> = std::collections::HashSet::new();

            for task in &completed {
                if let Some(ref proj) = task.project_id {
                    projects.insert(proj.clone());
                }
            }

            // Calculate focus depth for the day
            let focus_depth = if projects.is_empty() {
                1.0 // No projects = fully focused (or no work)
            } else {
                1.0 / projects.len() as f32
            };

            trends.push(DailyAttention {
                date: date_str,
                tasks_completed: completed.len(),
                tasks_created: created.len(),
                focus_areas: areas.into_iter().collect(),
                focus_projects: projects.into_iter().collect(),
                focus_depth,
            });

            current = day_end;
        }

        trends
    }

    fn calculate_focus_depth(
        &self,
        trends: &[DailyAttention],
        _by_area: &HashMap<String, AttentionScore>,
        _by_project: &HashMap<String, AttentionScore>,
    ) -> FocusDepth {
        // Calculate averages from daily trends
        let total_days = trends.len().max(1) as f32;

        let total_areas: usize = trends.iter().map(|d| d.focus_areas.len()).sum();
        let total_projects: usize = trends.iter().map(|d| d.focus_projects.len()).sum();

        let areas_per_day = total_areas as f32 / total_days;
        let projects_per_day = total_projects as f32 / total_days;

        // Estimate context switches (simplified)
        let context_switches = trends
            .windows(2)
            .filter(|w| w[0].focus_projects != w[1].focus_projects)
            .count();

        // Calculate focus score
        // Lower areas/projects per day = higher focus
        // Fewer context switches = higher focus
        let score = {
            let area_factor = 1.0 / (1.0 + areas_per_day);
            let project_factor = 1.0 / (1.0 + projects_per_day / 2.0);
            let switch_factor = 1.0 / (1.0 + context_switches as f32 / total_days);
            (area_factor + project_factor + switch_factor) / 3.0
        };

        let rating = if score >= 0.7 {
            FocusRating::DeepFocus
        } else if score >= 0.5 {
            FocusRating::Focused
        } else if score >= 0.35 {
            FocusRating::Moderate
        } else if score >= 0.2 {
            FocusRating::Fragmented
        } else {
            FocusRating::Scattered
        };

        let description = match rating {
            FocusRating::DeepFocus => {
                "Excellent focus - concentrated work on few areas".to_string()
            }
            FocusRating::Focused => {
                "Good focus - balanced attention with minimal context switching".to_string()
            }
            FocusRating::Moderate => {
                "Moderate focus - some context switching but manageable".to_string()
            }
            FocusRating::Fragmented => {
                "Attention is fragmented across many areas - consider consolidating".to_string()
            }
            FocusRating::Scattered => {
                "Highly scattered attention - significant focus improvement needed".to_string()
            }
        };

        FocusDepth {
            score,
            rating,
            areas_per_day,
            projects_per_day,
            context_switches,
            description,
        }
    }

    fn generate_recommendations(
        &self,
        imbalances: &[AttentionImbalance],
        focus_depth: &FocusDepth,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        // Focus depth recommendations
        match focus_depth.rating {
            FocusRating::Fragmented | FocusRating::Scattered => {
                recs.push(format!(
                    "Your attention is spread across {:.1} projects per day on average. Consider batching similar work.",
                    focus_depth.projects_per_day
                ));
                recs.push(
                    "Try time-blocking to dedicate focused periods to specific areas.".to_string(),
                );
            }
            FocusRating::Moderate => {
                recs.push(
                    "Your focus is adequate but could improve. Consider reducing context switches."
                        .to_string(),
                );
            }
            _ => {}
        }

        // Imbalance recommendations
        for imbalance in imbalances.iter().take(3) {
            if matches!(
                imbalance.severity,
                ImbalanceSeverity::High | ImbalanceSeverity::Critical
            ) {
                recs.push(imbalance.suggestion.clone());
            }
        }

        // General recommendations
        if imbalances.is_empty()
            && matches!(
                focus_depth.rating,
                FocusRating::DeepFocus | FocusRating::Focused
            )
        {
            recs.push(
                "Your attention allocation looks healthy. Keep up the good work!".to_string(),
            );
        }

        recs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_manager() -> AttentionManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        AttentionManager::new(store)
    }

    #[tokio::test]
    async fn test_analyze_attention() {
        let manager = create_test_manager().await;
        let metrics = manager.analyze(AttentionParams::default()).await.unwrap();

        assert!(metrics.period.days == 30);
        assert!(metrics.generated_at <= Utc::now());
    }

    #[tokio::test]
    async fn test_quick_summary() {
        let manager = create_test_manager().await;
        let metrics = manager.quick_summary().await.unwrap();

        assert!(metrics.period.days == 7);
    }

    #[test]
    fn test_focus_rating() {
        let focus = FocusDepth {
            score: 0.8,
            rating: FocusRating::DeepFocus,
            areas_per_day: 1.0,
            projects_per_day: 2.0,
            context_switches: 2,
            description: "Test".to_string(),
        };

        assert_eq!(focus.rating, FocusRating::DeepFocus);
    }
}

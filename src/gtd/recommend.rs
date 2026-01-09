//! Task recommendation engine for GTD.
//!
//! This module provides intelligent task recommendations based on context,
//! energy level, available time, and other factors.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{
    EnergyLevel, Priority, Project, ProjectFilter, Task, TaskFilter, TaskStatus,
};
use crate::gtd::{ProjectManager, TaskManager};
use crate::ontology::EmbeddedOntologyStore;
use schemars::JsonSchema;

// ============================================================================
// Recommendation Types
// ============================================================================

/// Parameters for getting task recommendations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RecommendParams {
    /// Current context (e.g., "@home", "@work", "@phone").
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
    /// Focus on a specific project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus_project: Option<String>,
    /// Include tasks with any due date.
    #[serde(default = "default_true")]
    pub include_due_tasks: bool,
    /// Include tasks from projects.
    #[serde(default = "default_true")]
    pub include_project_tasks: bool,
    /// Include standalone tasks.
    #[serde(default = "default_true")]
    pub include_standalone_tasks: bool,
    /// Maximum number of recommendations.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_true() -> bool {
    true
}

fn default_limit() -> usize {
    10
}

impl Default for RecommendParams {
    fn default() -> Self {
        Self {
            current_context: None,
            energy_level: None,
            time_available: None,
            focus_area: None,
            focus_project: None,
            include_due_tasks: true,
            include_project_tasks: true,
            include_standalone_tasks: true,
            limit: 10,
        }
    }
}

/// A task recommendation with scoring and reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecommendation {
    /// The recommended task.
    pub task: Task,
    /// Overall recommendation score (0-100).
    pub score: f32,
    /// Breakdown of score components.
    pub score_breakdown: ScoreBreakdown,
    /// Human-readable reasons for this recommendation.
    pub reasons: Vec<String>,
    /// Whether this is a quick win.
    pub is_quick_win: bool,
    /// Related project if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<ProjectSummary>,
}

/// Breakdown of how the score was calculated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Points from context match.
    pub context_score: f32,
    /// Points from energy match.
    pub energy_score: f32,
    /// Points from time fit.
    pub time_score: f32,
    /// Points from priority.
    pub priority_score: f32,
    /// Points from due date urgency.
    pub urgency_score: f32,
    /// Points from project health (tasks from unhealthy projects get boost).
    pub project_health_score: f32,
    /// Penalty for blocked tasks.
    pub blocked_penalty: f32,
}

impl ScoreBreakdown {
    /// Calculate total score from components.
    pub fn total(&self) -> f32 {
        let sum = self.context_score
            + self.energy_score
            + self.time_score
            + self.priority_score
            + self.urgency_score
            + self.project_health_score;
        (sum - self.blocked_penalty).clamp(0.0, 100.0)
    }
}

/// Summary of a project for recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSummary {
    /// Project ID.
    pub id: String,
    /// Project name.
    pub name: String,
    /// Project health score.
    pub health_score: f32,
}

/// Response from recommendation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendResponse {
    /// Recommended tasks.
    pub recommendations: Vec<TaskRecommendation>,
    /// Summary of recommendation context.
    pub context: RecommendContext,
    /// Reasoning explanation.
    pub reasoning: String,
}

/// Context used for recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendContext {
    /// Context used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Energy level used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy: Option<EnergyLevel>,
    /// Time available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_minutes: Option<u32>,
    /// Total tasks considered.
    pub tasks_considered: usize,
    /// Tasks filtered out.
    pub tasks_filtered: usize,
}

// ============================================================================
// Score Weights
// ============================================================================

/// Weights for scoring components.
#[derive(Debug, Clone)]
pub struct ScoreWeights {
    /// Weight for context match (0-1).
    pub context: f32,
    /// Weight for energy match (0-1).
    pub energy: f32,
    /// Weight for time fit (0-1).
    pub time: f32,
    /// Weight for priority (0-1).
    pub priority: f32,
    /// Weight for urgency (0-1).
    pub urgency: f32,
    /// Weight for project health (0-1).
    pub project_health: f32,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            context: 0.25,
            energy: 0.15,
            time: 0.15,
            priority: 0.20,
            urgency: 0.20,
            project_health: 0.05,
        }
    }
}

// ============================================================================
// Recommendation Engine
// ============================================================================

/// Engine for generating task recommendations.
pub struct RecommendationEngine {
    #[allow(dead_code)]
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    task_manager: TaskManager,
    project_manager: ProjectManager,
    weights: ScoreWeights,
    /// Quick task threshold in minutes.
    quick_task_minutes: u32,
}

impl RecommendationEngine {
    /// Create a new recommendation engine.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self {
            task_manager: TaskManager::new(store.clone()),
            project_manager: ProjectManager::new(store.clone()),
            store,
            weights: ScoreWeights::default(),
            quick_task_minutes: 2,
        }
    }

    /// Set custom scoring weights.
    pub fn with_weights(mut self, weights: ScoreWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set quick task threshold.
    pub fn with_quick_threshold(mut self, minutes: u32) -> Self {
        self.quick_task_minutes = minutes;
        self
    }

    /// Get task recommendations based on parameters.
    pub async fn recommend(&self, params: RecommendParams) -> Result<RecommendResponse> {
        // Get all actionable tasks
        let mut tasks = self
            .task_manager
            .list(TaskFilter {
                status: Some(TaskStatus::Next),
                ..Default::default()
            })
            .await?;

        let total_tasks = tasks.len();

        // Filter by project focus if specified
        if let Some(ref project_id) = params.focus_project {
            tasks.retain(|t| t.project_id.as_ref() == Some(project_id));
        }

        // Filter by context if specified
        if let Some(ref ctx) = params.current_context {
            tasks.retain(|t| {
                t.contexts.is_empty() // Tasks with no context match any
                    || t.contexts.iter().any(|c| c.to_lowercase() == ctx.to_lowercase())
                    || t.contexts.contains(&"@anywhere".to_string())
            });
        }

        // Filter by time if specified
        if let Some(time_available) = params.time_available {
            tasks.retain(|t| {
                t.estimated_minutes
                    .map(|m| m <= time_available)
                    .unwrap_or(true) // Include if no estimate
            });
        }

        let filtered_tasks = total_tasks - tasks.len();

        // Get projects for context
        let projects = self.project_manager.list(ProjectFilter::default()).await?;
        let project_map: HashMap<String, Project> =
            projects.into_iter().map(|p| (p.id.clone(), p)).collect();

        // Get project health scores
        let mut health_scores: HashMap<String, f32> = HashMap::new();
        for project_id in project_map.keys() {
            if let Some(health) = self.project_manager.get_health(project_id).await? {
                health_scores.insert(project_id.clone(), health.score);
            }
        }

        // Score each task
        let mut recommendations: Vec<TaskRecommendation> = Vec::new();

        for task in tasks {
            let breakdown = self.calculate_score(&task, &params, &health_scores);
            let score = breakdown.total();
            let reasons = self.generate_reasons(&task, &params, &breakdown);
            let is_quick_win = task
                .estimated_minutes
                .map(|m| m <= self.quick_task_minutes)
                .unwrap_or(false);

            let project = task.project_id.as_ref().and_then(|pid| {
                project_map.get(pid).map(|p| ProjectSummary {
                    id: p.id.clone(),
                    name: p.name.clone(),
                    health_score: health_scores.get(pid).copied().unwrap_or(100.0),
                })
            });

            recommendations.push(TaskRecommendation {
                task,
                score,
                score_breakdown: breakdown,
                reasons,
                is_quick_win,
                project,
            });
        }

        // Sort by score (highest first)
        recommendations.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        recommendations.truncate(params.limit);

        // Generate reasoning summary
        let reasoning = self.generate_reasoning_summary(&params, &recommendations);

        Ok(RecommendResponse {
            recommendations,
            context: RecommendContext {
                context: params.current_context,
                energy: params.energy_level,
                time_minutes: params.time_available,
                tasks_considered: total_tasks,
                tasks_filtered: filtered_tasks,
            },
            reasoning,
        })
    }

    /// Get quick win recommendations (2-minute tasks).
    pub async fn quick_wins(&self, limit: usize) -> Result<Vec<TaskRecommendation>> {
        let tasks = self.task_manager.get_quick_wins().await?;

        let recommendations: Vec<TaskRecommendation> = tasks
            .into_iter()
            .take(limit)
            .map(|task| {
                let breakdown = ScoreBreakdown {
                    context_score: 20.0,
                    energy_score: 15.0,
                    time_score: 25.0, // High score for quick tasks
                    priority_score: 15.0,
                    urgency_score: 10.0,
                    project_health_score: 5.0,
                    blocked_penalty: 0.0,
                };

                TaskRecommendation {
                    task: task.clone(),
                    score: 90.0,
                    score_breakdown: breakdown,
                    reasons: vec![
                        "Quick 2-minute task".to_string(),
                        "Completing quick wins builds momentum".to_string(),
                    ],
                    is_quick_win: true,
                    project: None,
                }
            })
            .collect();

        Ok(recommendations)
    }

    /// Get context-specific recommendations.
    pub async fn for_context(
        &self,
        context: &str,
        limit: usize,
    ) -> Result<Vec<TaskRecommendation>> {
        self.recommend(RecommendParams {
            current_context: Some(context.to_string()),
            limit,
            ..Default::default()
        })
        .await
        .map(|r| r.recommendations)
    }

    /// Get energy-appropriate recommendations.
    pub async fn for_energy(
        &self,
        energy: EnergyLevel,
        limit: usize,
    ) -> Result<Vec<TaskRecommendation>> {
        self.recommend(RecommendParams {
            energy_level: Some(energy),
            limit,
            ..Default::default()
        })
        .await
        .map(|r| r.recommendations)
    }

    /// Get recommendations for available time.
    pub async fn for_time(&self, minutes: u32, limit: usize) -> Result<Vec<TaskRecommendation>> {
        self.recommend(RecommendParams {
            time_available: Some(minutes),
            limit,
            ..Default::default()
        })
        .await
        .map(|r| r.recommendations)
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    fn calculate_score(
        &self,
        task: &Task,
        params: &RecommendParams,
        project_health: &HashMap<String, f32>,
    ) -> ScoreBreakdown {
        let max_score = 100.0;

        // Context score (0-25 points by default)
        let context_score = if let Some(ref ctx) = params.current_context {
            if task.contexts.is_empty() || task.contexts.contains(&"@anywhere".to_string()) {
                self.weights.context * max_score * 0.5 // Partial match for any-context tasks
            } else if task
                .contexts
                .iter()
                .any(|c| c.to_lowercase() == ctx.to_lowercase())
            {
                self.weights.context * max_score // Full match
            } else {
                0.0 // No match
            }
        } else {
            self.weights.context * max_score * 0.5 // No context specified, partial score
        };

        // Energy score (0-15 points by default)
        let energy_score = if let Some(ref energy) = params.energy_level {
            let match_level = self.energy_match_score(&task.energy_level, energy);
            self.weights.energy * max_score * match_level
        } else {
            self.weights.energy * max_score * 0.5
        };

        // Time score (0-15 points by default)
        let time_score = if let Some(time_available) = params.time_available {
            if let Some(estimated) = task.estimated_minutes {
                if estimated <= time_available {
                    // Good fit: score based on how well it uses the time
                    let usage_ratio = estimated as f32 / time_available as f32;
                    let fit_score = if usage_ratio > 0.8 {
                        1.0 // Great fit
                    } else if usage_ratio > 0.5 {
                        0.8 // Good fit
                    } else {
                        0.6 // Okay fit (leaves room for more)
                    };
                    self.weights.time * max_score * fit_score
                } else {
                    0.0 // Won't fit
                }
            } else {
                self.weights.time * max_score * 0.5 // Unknown duration
            }
        } else {
            self.weights.time * max_score * 0.5
        };

        // Priority score (0-20 points by default)
        let priority_score = match task.priority {
            Priority::Critical => self.weights.priority * max_score,
            Priority::High => self.weights.priority * max_score * 0.8,
            Priority::Normal => self.weights.priority * max_score * 0.5,
            Priority::Low => self.weights.priority * max_score * 0.2,
        };

        // Urgency score based on due date (0-20 points by default)
        let urgency_score = if let Some(due) = task.due_date {
            let days_until = (due - Utc::now()).num_days();
            let urgency = if days_until < 0 {
                1.0 // Overdue - maximum urgency
            } else if days_until == 0 {
                0.95 // Due today
            } else if days_until == 1 {
                0.85 // Due tomorrow
            } else if days_until <= 3 {
                0.7
            } else if days_until <= 7 {
                0.5
            } else {
                0.3
            };
            self.weights.urgency * max_score * urgency
        } else {
            self.weights.urgency * max_score * 0.3 // No due date, lower urgency
        };

        // Project health score - boost tasks from unhealthy projects
        let project_health_score = if let Some(ref project_id) = task.project_id {
            if let Some(&health) = project_health.get(project_id) {
                // Inverse relationship: lower health = higher score
                let health_factor = (100.0 - health) / 100.0;
                self.weights.project_health * max_score * health_factor
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Blocked penalty
        let blocked_penalty = if !task.blocked_by.is_empty() {
            25.0 // Significant penalty for blocked tasks
        } else {
            0.0
        };

        ScoreBreakdown {
            context_score,
            energy_score,
            time_score,
            priority_score,
            urgency_score,
            project_health_score,
            blocked_penalty,
        }
    }

    fn energy_match_score(&self, task_energy: &EnergyLevel, user_energy: &EnergyLevel) -> f32 {
        match (task_energy, user_energy) {
            // Perfect matches
            (EnergyLevel::Low, EnergyLevel::Low) => 1.0,
            (EnergyLevel::Medium, EnergyLevel::Medium) => 1.0,
            (EnergyLevel::High, EnergyLevel::High) => 1.0,

            // Can do lower energy tasks with higher energy
            (EnergyLevel::Low, EnergyLevel::Medium) => 0.7,
            (EnergyLevel::Low, EnergyLevel::High) => 0.5,
            (EnergyLevel::Medium, EnergyLevel::High) => 0.8,

            // Harder to do high energy tasks when tired
            (EnergyLevel::High, EnergyLevel::Medium) => 0.3,
            (EnergyLevel::High, EnergyLevel::Low) => 0.1,
            (EnergyLevel::Medium, EnergyLevel::Low) => 0.4,
        }
    }

    fn generate_reasons(
        &self,
        task: &Task,
        params: &RecommendParams,
        breakdown: &ScoreBreakdown,
    ) -> Vec<String> {
        let mut reasons = Vec::new();

        // Context reason
        if let Some(ref ctx) = params.current_context {
            if task
                .contexts
                .iter()
                .any(|c| c.to_lowercase() == ctx.to_lowercase())
            {
                reasons.push(format!("Matches your {} context", ctx));
            } else if task.contexts.is_empty() {
                reasons.push("Can be done anywhere".to_string());
            }
        }

        // Time reason
        if let Some(time_available) = params.time_available {
            if let Some(estimated) = task.estimated_minutes {
                if estimated <= time_available {
                    reasons.push(format!(
                        "Fits in your {} minutes ({} min task)",
                        time_available, estimated
                    ));
                }
            }
        }

        // Energy reason
        if let Some(ref energy) = params.energy_level {
            if &task.energy_level == energy {
                reasons.push(format!("Matches your {:?} energy level", energy));
            } else if task.energy_level == EnergyLevel::Low {
                reasons.push("Low-energy task good for any time".to_string());
            }
        }

        // Priority reason
        match task.priority {
            Priority::Critical => reasons.push("Critical priority task".to_string()),
            Priority::High => reasons.push("High priority task".to_string()),
            _ => {}
        }

        // Due date reason
        if let Some(due) = task.due_date {
            let days_until = (due - Utc::now()).num_days();
            if days_until < 0 {
                reasons.push("Overdue - needs immediate attention".to_string());
            } else if days_until == 0 {
                reasons.push("Due today".to_string());
            } else if days_until == 1 {
                reasons.push("Due tomorrow".to_string());
            } else if days_until <= 3 {
                reasons.push(format!("Due in {} days", days_until));
            }
        }

        // Quick win reason
        if let Some(mins) = task.estimated_minutes {
            if mins <= self.quick_task_minutes {
                reasons.push(format!("Quick {}-minute task", mins));
            }
        }

        // Project health reason
        if breakdown.project_health_score > 0.0 {
            reasons.push("From a project that needs attention".to_string());
        }

        // Default reason if none generated
        if reasons.is_empty() {
            reasons.push("Active next action".to_string());
        }

        reasons
    }

    fn generate_reasoning_summary(
        &self,
        params: &RecommendParams,
        recommendations: &[TaskRecommendation],
    ) -> String {
        let mut parts = Vec::new();

        if recommendations.is_empty() {
            return "No tasks match your current criteria. Consider adjusting your filters."
                .to_string();
        }

        parts.push(format!("Found {} recommendations", recommendations.len()));

        if let Some(ref ctx) = params.current_context {
            parts.push(format!("for {} context", ctx));
        }

        if let Some(ref energy) = params.energy_level {
            parts.push(format!("at {:?} energy", energy));
        }

        if let Some(time) = params.time_available {
            parts.push(format!("with {} minutes available", time));
        }

        let quick_wins = recommendations.iter().filter(|r| r.is_quick_win).count();
        if quick_wins > 0 {
            parts.push(format!("including {} quick wins", quick_wins));
        }

        let urgent = recommendations
            .iter()
            .filter(|r| r.score_breakdown.urgency_score > 15.0)
            .count();
        if urgent > 0 {
            parts.push(format!("with {} urgent items", urgent));
        }

        parts.join(", ") + "."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_engine() -> RecommendationEngine {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        RecommendationEngine::new(store)
    }

    #[tokio::test]
    async fn test_recommend_empty() {
        let engine = create_test_engine().await;

        let response = engine.recommend(RecommendParams::default()).await.unwrap();

        assert!(response.recommendations.is_empty());
        assert_eq!(response.context.tasks_considered, 0);
    }

    #[tokio::test]
    async fn test_score_breakdown_total() {
        let breakdown = ScoreBreakdown {
            context_score: 20.0,
            energy_score: 15.0,
            time_score: 15.0,
            priority_score: 20.0,
            urgency_score: 20.0,
            project_health_score: 5.0,
            blocked_penalty: 0.0,
        };

        assert_eq!(breakdown.total(), 95.0);
    }

    #[tokio::test]
    async fn test_score_breakdown_with_penalty() {
        let breakdown = ScoreBreakdown {
            context_score: 20.0,
            energy_score: 15.0,
            time_score: 15.0,
            priority_score: 20.0,
            urgency_score: 20.0,
            project_health_score: 5.0,
            blocked_penalty: 25.0,
        };

        assert_eq!(breakdown.total(), 70.0);
    }

    #[test]
    fn test_energy_match_score() {
        let store = Arc::new(std::sync::RwLock::new(()));

        // Perfect matches should score 1.0
        assert_eq!(
            1.0,
            match (&EnergyLevel::Low, &EnergyLevel::Low) {
                (EnergyLevel::Low, EnergyLevel::Low) => 1.0,
                _ => 0.0,
            }
        );

        // Can do low energy when high energy
        assert_eq!(
            0.5,
            match (&EnergyLevel::Low, &EnergyLevel::High) {
                (EnergyLevel::Low, EnergyLevel::High) => 0.5,
                _ => 0.0,
            }
        );
    }
}

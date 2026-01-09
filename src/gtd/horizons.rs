//! GTD Horizons of Focus mapping.
//!
//! This module provides horizon-based visualization of GTD elements,
//! mapping tasks, projects, areas, goals, vision, and purpose across
//! David Allen's 6 horizons of focus.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{ProjectFilter, ProjectStatus, TaskFilter, TaskStatus};
use crate::gtd::{ProjectManager, TaskManager};
use crate::ontology::{EmbeddedOntologyStore, EntityType, OntologyStore};
use schemars::JsonSchema;

// ============================================================================
// Horizon Types
// ============================================================================

/// The six horizons of focus in GTD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HorizonLevel {
    /// Ground/Runway - Current actions and tasks.
    Runway,
    /// 10,000 feet - Current projects.
    H10k,
    /// 20,000 feet - Areas of focus and responsibility.
    H20k,
    /// 30,000 feet - 1-2 year goals.
    H30k,
    /// 40,000 feet - 3-5 year vision.
    H40k,
    /// 50,000 feet - Life purpose and principles.
    H50k,
}

impl HorizonLevel {
    /// Get the human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            HorizonLevel::Runway => "Runway (Actions)",
            HorizonLevel::H10k => "10,000 ft (Projects)",
            HorizonLevel::H20k => "20,000 ft (Areas)",
            HorizonLevel::H30k => "30,000 ft (Goals)",
            HorizonLevel::H40k => "40,000 ft (Vision)",
            HorizonLevel::H50k => "50,000 ft (Purpose)",
        }
    }

    /// Get the description.
    pub fn description(&self) -> &'static str {
        match self {
            HorizonLevel::Runway => "Current actions and next steps you can do right now",
            HorizonLevel::H10k => {
                "Outcomes requiring multiple actions, with clear completion criteria"
            }
            HorizonLevel::H20k => {
                "Key areas of life/work you maintain - no end point, just standards"
            }
            HorizonLevel::H30k => "1-2 year outcomes that define success in your key areas",
            HorizonLevel::H40k => "3-5 year vision of what success looks like",
            HorizonLevel::H50k => "Core purpose, values, and principles that guide everything",
        }
    }

    /// Get the altitude metaphor.
    pub fn altitude(&self) -> &'static str {
        match self {
            HorizonLevel::Runway => "Ground Level",
            HorizonLevel::H10k => "10,000 ft",
            HorizonLevel::H20k => "20,000 ft",
            HorizonLevel::H30k => "30,000 ft",
            HorizonLevel::H40k => "40,000 ft",
            HorizonLevel::H50k => "50,000 ft",
        }
    }
}

/// Parameters for horizon mapping.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HorizonParams {
    /// Horizons to include (empty = all).
    #[serde(default)]
    pub horizons: Vec<HorizonLevel>,
    /// Include item counts.
    #[serde(default = "default_true")]
    pub include_counts: bool,
    /// Include health metrics.
    #[serde(default = "default_true")]
    pub include_health: bool,
    /// Include alignment analysis.
    #[serde(default = "default_true")]
    pub include_alignment: bool,
    /// Maximum items per horizon.
    #[serde(default = "default_items_per_horizon")]
    pub items_per_horizon: usize,
    /// Filter by area.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub area: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_items_per_horizon() -> usize {
    10
}

impl Default for HorizonParams {
    fn default() -> Self {
        Self {
            horizons: Vec::new(),
            include_counts: true,
            include_health: true,
            include_alignment: true,
            items_per_horizon: 10,
            area: None,
        }
    }
}

/// Complete horizon map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonMap {
    /// Horizons with their items.
    pub horizons: HashMap<String, Horizon>,
    /// Alignment analysis between levels.
    pub alignment: AlignmentAnalysis,
    /// Overall system health across horizons.
    pub overall_health: HorizonHealth,
    /// Recommendations for improving horizon clarity.
    pub recommendations: Vec<HorizonRecommendation>,
    /// When this map was generated.
    pub generated_at: DateTime<Utc>,
}

/// A single horizon with its items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Horizon {
    /// The horizon level.
    pub level: HorizonLevel,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Items at this horizon.
    pub items: Vec<HorizonItem>,
    /// Total count of items (may be more than shown).
    pub total_count: usize,
    /// Health score for this horizon (0-100).
    pub health_score: f32,
    /// Status summary.
    pub status: HorizonStatus,
}

/// An item at a horizon level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonItem {
    /// Unique identifier.
    pub id: String,
    /// Name/description.
    pub name: String,
    /// Item type (task, project, area, goal, etc.).
    pub item_type: String,
    /// Current status.
    pub status: String,
    /// Parent item at higher horizon (for alignment).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Parent name for display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_name: Option<String>,
    /// Child count at lower horizon.
    pub child_count: usize,
    /// Is this item aligned with higher horizons?
    pub is_aligned: bool,
    /// Health score (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_score: Option<f32>,
    /// Last activity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_activity: Option<DateTime<Utc>>,
}

/// Status summary for a horizon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonStatus {
    /// Is this horizon well-defined?
    pub is_defined: bool,
    /// Active item count.
    pub active_count: usize,
    /// Items needing attention.
    pub needs_attention: usize,
    /// Summary message.
    pub summary: String,
}

/// Alignment analysis between horizons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentAnalysis {
    /// Overall alignment score (0-100).
    pub alignment_score: f32,
    /// Orphaned items (no connection to higher horizon).
    pub orphaned_items: Vec<OrphanedItem>,
    /// Gaps in the hierarchy.
    pub gaps: Vec<AlignmentGap>,
    /// Well-aligned chains.
    pub strong_alignments: Vec<AlignmentChain>,
}

/// An item that's not connected to higher horizons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrphanedItem {
    /// The item.
    pub item: HorizonItem,
    /// Its horizon level.
    pub horizon: HorizonLevel,
    /// Which higher horizon it's missing connection to.
    pub missing_connection_to: HorizonLevel,
}

/// A gap in the horizon hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentGap {
    /// Description of the gap.
    pub description: String,
    /// Lower horizon with items.
    pub lower_horizon: HorizonLevel,
    /// Higher horizon missing definition.
    pub higher_horizon: HorizonLevel,
    /// Suggested action.
    pub suggestion: String,
}

/// A well-aligned chain from purpose to action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentChain {
    /// Chain of items from highest to lowest horizon.
    pub chain: Vec<ChainLink>,
    /// Strength of alignment (0-1).
    pub strength: f32,
}

/// A link in an alignment chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainLink {
    /// Horizon level.
    pub horizon: HorizonLevel,
    /// Item name.
    pub name: String,
    /// Item ID.
    pub id: String,
}

/// Overall health across horizons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonHealth {
    /// Overall score (0-100).
    pub score: f32,
    /// Health by horizon.
    pub by_horizon: HashMap<String, f32>,
    /// Rating.
    pub rating: String,
    /// Summary.
    pub summary: String,
}

/// A recommendation for horizon improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonRecommendation {
    /// Priority (1-5, 1 is highest).
    pub priority: u8,
    /// Target horizon.
    pub horizon: HorizonLevel,
    /// Recommendation.
    pub recommendation: String,
    /// Reason.
    pub reason: String,
}

// ============================================================================
// Horizon Manager
// ============================================================================

/// Manages GTD horizon mapping.
pub struct HorizonManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    task_manager: TaskManager,
    project_manager: ProjectManager,
}

impl HorizonManager {
    /// Create a new horizon manager.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self {
            task_manager: TaskManager::new(store.clone()),
            project_manager: ProjectManager::new(store.clone()),
            store,
        }
    }

    /// Generate a complete horizon map.
    pub async fn map(&self, params: HorizonParams) -> Result<HorizonMap> {
        let include_all = params.horizons.is_empty();

        let mut horizons: HashMap<String, Horizon> = HashMap::new();

        // Build each horizon
        if include_all || params.horizons.contains(&HorizonLevel::Runway) {
            let runway = self.build_runway_horizon(&params).await?;
            horizons.insert("runway".to_string(), runway);
        }

        if include_all || params.horizons.contains(&HorizonLevel::H10k) {
            let h10k = self.build_projects_horizon(&params).await?;
            horizons.insert("h10k".to_string(), h10k);
        }

        if include_all || params.horizons.contains(&HorizonLevel::H20k) {
            let h20k = self.build_areas_horizon(&params).await?;
            horizons.insert("h20k".to_string(), h20k);
        }

        if include_all || params.horizons.contains(&HorizonLevel::H30k) {
            let h30k = self.build_goals_horizon(&params).await?;
            horizons.insert("h30k".to_string(), h30k);
        }

        if include_all || params.horizons.contains(&HorizonLevel::H40k) {
            let h40k = self.build_vision_horizon(&params).await?;
            horizons.insert("h40k".to_string(), h40k);
        }

        if include_all || params.horizons.contains(&HorizonLevel::H50k) {
            let h50k = self.build_purpose_horizon(&params).await?;
            horizons.insert("h50k".to_string(), h50k);
        }

        // Analyze alignment
        let alignment = if params.include_alignment {
            self.analyze_alignment(&horizons).await
        } else {
            AlignmentAnalysis {
                alignment_score: 100.0,
                orphaned_items: Vec::new(),
                gaps: Vec::new(),
                strong_alignments: Vec::new(),
            }
        };

        // Calculate overall health
        let overall_health = if params.include_health {
            self.calculate_overall_health(&horizons).await
        } else {
            HorizonHealth {
                score: 100.0,
                by_horizon: HashMap::new(),
                rating: "Not calculated".to_string(),
                summary: "Health metrics not included".to_string(),
            }
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(&horizons, &alignment);

        Ok(HorizonMap {
            horizons,
            alignment,
            overall_health,
            recommendations,
            generated_at: Utc::now(),
        })
    }

    /// Get a quick horizon overview.
    pub async fn overview(&self) -> Result<HorizonMap> {
        self.map(HorizonParams {
            items_per_horizon: 5,
            ..Default::default()
        })
        .await
    }

    // ========================================================================
    // Horizon Builders
    // ========================================================================

    async fn build_runway_horizon(&self, params: &HorizonParams) -> Result<Horizon> {
        let tasks = self
            .task_manager
            .list(TaskFilter {
                status: Some(TaskStatus::Next),
                limit: params.items_per_horizon,
                ..Default::default()
            })
            .await?;

        let total_count = self
            .task_manager
            .list(TaskFilter {
                status: Some(TaskStatus::Next),
                ..Default::default()
            })
            .await?
            .len();

        let items: Vec<HorizonItem> = tasks
            .iter()
            .map(|t| HorizonItem {
                id: t.id.clone(),
                name: t.description.clone(),
                item_type: "task".to_string(),
                status: format!("{:?}", t.status),
                parent_id: t.project_id.clone(),
                parent_name: None, // Would need to look up
                child_count: 0,
                is_aligned: t.project_id.is_some(),
                health_score: None,
                last_activity: Some(t.updated_at),
            })
            .collect();

        let needs_attention = items.iter().filter(|i| !i.is_aligned).count();

        Ok(Horizon {
            level: HorizonLevel::Runway,
            name: HorizonLevel::Runway.display_name().to_string(),
            description: HorizonLevel::Runway.description().to_string(),
            items,
            total_count,
            health_score: if total_count > 0 { 80.0 } else { 50.0 },
            status: HorizonStatus {
                is_defined: total_count > 0,
                active_count: total_count,
                needs_attention,
                summary: format!("{} active next actions", total_count),
            },
        })
    }

    async fn build_projects_horizon(&self, params: &HorizonParams) -> Result<Horizon> {
        let mut filter = ProjectFilter {
            status: Some(ProjectStatus::Active),
            limit: params.items_per_horizon,
            ..Default::default()
        };

        if let Some(ref area) = params.area {
            filter.area = Some(area.clone());
        }

        let projects = self.project_manager.list(filter).await?;

        let total_count = self
            .project_manager
            .list(ProjectFilter {
                status: Some(ProjectStatus::Active),
                ..Default::default()
            })
            .await?
            .len();

        let items: Vec<HorizonItem> = projects
            .iter()
            .map(|p| HorizonItem {
                id: p.id.clone(),
                name: p.name.clone(),
                item_type: "project".to_string(),
                status: format!("{:?}", p.status),
                parent_id: p.area.clone(),
                parent_name: p.area.clone(),
                child_count: 0, // Would count tasks
                is_aligned: p.area.is_some(),
                health_score: p.health_score,
                last_activity: Some(p.updated_at),
            })
            .collect();

        let needs_attention = items.iter().filter(|i| !i.is_aligned).count();

        Ok(Horizon {
            level: HorizonLevel::H10k,
            name: HorizonLevel::H10k.display_name().to_string(),
            description: HorizonLevel::H10k.description().to_string(),
            items,
            total_count,
            health_score: if total_count > 0 { 75.0 } else { 50.0 },
            status: HorizonStatus {
                is_defined: total_count > 0,
                active_count: total_count,
                needs_attention,
                summary: format!("{} active projects", total_count),
            },
        })
    }

    async fn build_areas_horizon(&self, params: &HorizonParams) -> Result<Horizon> {
        let store = self.store.read().await;
        let areas = store.find_entities_by_type(EntityType::Area, 100).await?;
        drop(store);

        let items: Vec<HorizonItem> = areas
            .iter()
            .take(params.items_per_horizon)
            .map(|a| HorizonItem {
                id: a.id.clone(),
                name: a.name.clone(),
                item_type: "area".to_string(),
                status: "active".to_string(),
                parent_id: None,
                parent_name: None,
                child_count: 0,
                is_aligned: true, // Areas don't need parent alignment
                health_score: None,
                last_activity: Some(a.updated_at),
            })
            .collect();

        let total_count = areas.len();
        let needs_attention = 0; // Areas don't typically need attention flagging

        Ok(Horizon {
            level: HorizonLevel::H20k,
            name: HorizonLevel::H20k.display_name().to_string(),
            description: HorizonLevel::H20k.description().to_string(),
            items,
            total_count,
            health_score: if total_count > 0 { 80.0 } else { 40.0 },
            status: HorizonStatus {
                is_defined: total_count > 0,
                active_count: total_count,
                needs_attention,
                summary: if total_count > 0 {
                    format!("{} defined areas of focus", total_count)
                } else {
                    "No areas of focus defined".to_string()
                },
            },
        })
    }

    async fn build_goals_horizon(&self, params: &HorizonParams) -> Result<Horizon> {
        let store = self.store.read().await;
        let goals = store.find_entities_by_type(EntityType::Goal, 100).await?;
        drop(store);

        let items: Vec<HorizonItem> = goals
            .iter()
            .take(params.items_per_horizon)
            .map(|g| HorizonItem {
                id: g.id.clone(),
                name: g.name.clone(),
                item_type: "goal".to_string(),
                status: "active".to_string(),
                parent_id: None,
                parent_name: None,
                child_count: 0,
                is_aligned: true,
                health_score: None,
                last_activity: Some(g.updated_at),
            })
            .collect();

        let total_count = goals.len();

        Ok(Horizon {
            level: HorizonLevel::H30k,
            name: HorizonLevel::H30k.display_name().to_string(),
            description: HorizonLevel::H30k.description().to_string(),
            items,
            total_count,
            health_score: if total_count > 0 { 70.0 } else { 30.0 },
            status: HorizonStatus {
                is_defined: total_count > 0,
                active_count: total_count,
                needs_attention: 0,
                summary: if total_count > 0 {
                    format!("{} goals defined", total_count)
                } else {
                    "No 1-2 year goals defined".to_string()
                },
            },
        })
    }

    async fn build_vision_horizon(&self, params: &HorizonParams) -> Result<Horizon> {
        let store = self.store.read().await;
        let visions = store.find_entities_by_type(EntityType::Vision, 100).await?;
        drop(store);

        let items: Vec<HorizonItem> = visions
            .iter()
            .take(params.items_per_horizon)
            .map(|v| HorizonItem {
                id: v.id.clone(),
                name: v.name.clone(),
                item_type: "vision".to_string(),
                status: "active".to_string(),
                parent_id: None,
                parent_name: None,
                child_count: 0,
                is_aligned: true,
                health_score: None,
                last_activity: Some(v.updated_at),
            })
            .collect();

        let total_count = visions.len();

        Ok(Horizon {
            level: HorizonLevel::H40k,
            name: HorizonLevel::H40k.display_name().to_string(),
            description: HorizonLevel::H40k.description().to_string(),
            items,
            total_count,
            health_score: if total_count > 0 { 70.0 } else { 20.0 },
            status: HorizonStatus {
                is_defined: total_count > 0,
                active_count: total_count,
                needs_attention: 0,
                summary: if total_count > 0 {
                    format!("{} vision statements", total_count)
                } else {
                    "No 3-5 year vision defined".to_string()
                },
            },
        })
    }

    async fn build_purpose_horizon(&self, params: &HorizonParams) -> Result<Horizon> {
        let store = self.store.read().await;
        let purposes = store
            .find_entities_by_type(EntityType::Purpose, 100)
            .await?;
        drop(store);

        let items: Vec<HorizonItem> = purposes
            .iter()
            .take(params.items_per_horizon)
            .map(|p| HorizonItem {
                id: p.id.clone(),
                name: p.name.clone(),
                item_type: "purpose".to_string(),
                status: "active".to_string(),
                parent_id: None,
                parent_name: None,
                child_count: 0,
                is_aligned: true,
                health_score: None,
                last_activity: Some(p.updated_at),
            })
            .collect();

        let total_count = purposes.len();

        Ok(Horizon {
            level: HorizonLevel::H50k,
            name: HorizonLevel::H50k.display_name().to_string(),
            description: HorizonLevel::H50k.description().to_string(),
            items,
            total_count,
            health_score: if total_count > 0 { 70.0 } else { 10.0 },
            status: HorizonStatus {
                is_defined: total_count > 0,
                active_count: total_count,
                needs_attention: 0,
                summary: if total_count > 0 {
                    format!("{} purpose/principles defined", total_count)
                } else {
                    "No life purpose or principles defined".to_string()
                },
            },
        })
    }

    // ========================================================================
    // Analysis Helpers
    // ========================================================================

    async fn analyze_alignment(&self, horizons: &HashMap<String, Horizon>) -> AlignmentAnalysis {
        let mut orphaned_items = Vec::new();
        let mut gaps = Vec::new();

        // Check runway items for project alignment
        if let Some(runway) = horizons.get("runway") {
            for item in &runway.items {
                if !item.is_aligned {
                    orphaned_items.push(OrphanedItem {
                        item: item.clone(),
                        horizon: HorizonLevel::Runway,
                        missing_connection_to: HorizonLevel::H10k,
                    });
                }
            }
        }

        // Check projects for area alignment
        if let Some(h10k) = horizons.get("h10k") {
            for item in &h10k.items {
                if !item.is_aligned {
                    orphaned_items.push(OrphanedItem {
                        item: item.clone(),
                        horizon: HorizonLevel::H10k,
                        missing_connection_to: HorizonLevel::H20k,
                    });
                }
            }
        }

        // Check for horizon gaps
        let has_areas = horizons
            .get("h20k")
            .map(|h| h.total_count > 0)
            .unwrap_or(false);
        let has_goals = horizons
            .get("h30k")
            .map(|h| h.total_count > 0)
            .unwrap_or(false);
        let has_vision = horizons
            .get("h40k")
            .map(|h| h.total_count > 0)
            .unwrap_or(false);
        let has_purpose = horizons
            .get("h50k")
            .map(|h| h.total_count > 0)
            .unwrap_or(false);

        if !has_areas {
            gaps.push(AlignmentGap {
                description: "No areas of focus defined".to_string(),
                lower_horizon: HorizonLevel::H10k,
                higher_horizon: HorizonLevel::H20k,
                suggestion: "Define your key areas of responsibility and focus".to_string(),
            });
        }

        if has_areas && !has_goals {
            gaps.push(AlignmentGap {
                description: "Areas defined but no goals".to_string(),
                lower_horizon: HorizonLevel::H20k,
                higher_horizon: HorizonLevel::H30k,
                suggestion: "Set 1-2 year goals for your key areas".to_string(),
            });
        }

        if has_goals && !has_vision {
            gaps.push(AlignmentGap {
                description: "Goals defined but no vision".to_string(),
                lower_horizon: HorizonLevel::H30k,
                higher_horizon: HorizonLevel::H40k,
                suggestion: "Clarify your 3-5 year vision to guide goal setting".to_string(),
            });
        }

        if has_vision && !has_purpose {
            gaps.push(AlignmentGap {
                description: "Vision defined but no purpose".to_string(),
                lower_horizon: HorizonLevel::H40k,
                higher_horizon: HorizonLevel::H50k,
                suggestion: "Articulate your core purpose and principles".to_string(),
            });
        }

        // Calculate alignment score
        let orphan_penalty = (orphaned_items.len() as f32 * 5.0).min(30.0);
        let gap_penalty = (gaps.len() as f32 * 15.0).min(40.0);
        let alignment_score = (100.0 - orphan_penalty - gap_penalty).max(0.0);

        AlignmentAnalysis {
            alignment_score,
            orphaned_items,
            gaps,
            strong_alignments: Vec::new(), // Would need deeper analysis
        }
    }

    async fn calculate_overall_health(&self, horizons: &HashMap<String, Horizon>) -> HorizonHealth {
        let mut by_horizon: HashMap<String, f32> = HashMap::new();
        let mut total_score = 0.0;
        let mut count = 0;

        for (name, horizon) in horizons {
            by_horizon.insert(name.clone(), horizon.health_score);
            total_score += horizon.health_score;
            count += 1;
        }

        let score = if count > 0 {
            total_score / count as f32
        } else {
            0.0
        };

        let (rating, summary) = if score >= 80.0 {
            (
                "Excellent".to_string(),
                "Your horizon system is well-defined and aligned".to_string(),
            )
        } else if score >= 60.0 {
            (
                "Good".to_string(),
                "Your horizons are mostly defined - some areas need attention".to_string(),
            )
        } else if score >= 40.0 {
            (
                "Needs Work".to_string(),
                "Several horizons need definition or better alignment".to_string(),
            )
        } else {
            (
                "Requires Attention".to_string(),
                "Your horizon system needs significant development".to_string(),
            )
        };

        HorizonHealth {
            score,
            by_horizon,
            rating,
            summary,
        }
    }

    fn generate_recommendations(
        &self,
        horizons: &HashMap<String, Horizon>,
        alignment: &AlignmentAnalysis,
    ) -> Vec<HorizonRecommendation> {
        let mut recs = Vec::new();

        // Recommendations for gaps
        for gap in &alignment.gaps {
            recs.push(HorizonRecommendation {
                priority: 2,
                horizon: gap.higher_horizon,
                recommendation: gap.suggestion.clone(),
                reason: gap.description.clone(),
            });
        }

        // Recommendations for orphaned items
        if !alignment.orphaned_items.is_empty() {
            let orphan_count = alignment.orphaned_items.len();
            recs.push(HorizonRecommendation {
                priority: 3,
                horizon: HorizonLevel::Runway,
                recommendation: format!(
                    "Connect {} orphaned items to their parent projects/areas",
                    orphan_count
                ),
                reason: "Items without parent alignment may lack context".to_string(),
            });
        }

        // Horizon-specific recommendations
        for horizon in horizons.values() {
            if !horizon.status.is_defined {
                recs.push(HorizonRecommendation {
                    priority: 1,
                    horizon: horizon.level,
                    recommendation: format!("Define items at the {} level", horizon.name),
                    reason: format!("{} is empty", horizon.name),
                });
            }
        }

        // Sort by priority
        recs.sort_by_key(|r| r.priority);

        recs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_manager() -> HorizonManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        HorizonManager::new(store)
    }

    #[tokio::test]
    async fn test_horizon_map() {
        let manager = create_test_manager().await;
        let map = manager.map(HorizonParams::default()).await.unwrap();

        assert!(!map.horizons.is_empty());
        assert!(map.generated_at <= Utc::now());
    }

    #[tokio::test]
    async fn test_horizon_overview() {
        let manager = create_test_manager().await;
        let map = manager.overview().await.unwrap();

        assert!(!map.horizons.is_empty());
    }

    #[test]
    fn test_horizon_level_display() {
        assert_eq!(HorizonLevel::Runway.display_name(), "Runway (Actions)");
        assert_eq!(HorizonLevel::H50k.altitude(), "50,000 ft");
    }
}

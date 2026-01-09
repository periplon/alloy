//! Getting Things Done (GTD) module for task and project management.
//!
//! This module provides the core GTD functionality including:
//!
//! - **Projects**: Outcomes requiring multiple actions
//! - **Tasks**: Single next actions
//! - **Waiting For**: Delegated items
//! - **Someday/Maybe**: Deferred items for future consideration
//! - **Inbox Processing**: Intelligent classification of new items
//! - **Recommendations**: Context and energy-aware task suggestions
//! - **Weekly Review**: Comprehensive GTD system review
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     GTD Layer                                │
//! │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐      │
//! │  │ ProjectManager│ │  TaskManager  │ │WaitingManager │      │
//! │  └───────────────┘ └───────────────┘ └───────────────┘      │
//! │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐      │
//! │  │SomedayManager │ │ InboxManager  │ │ ReviewManager │      │
//! │  └───────────────┘ └───────────────┘ └───────────────┘      │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │           RecommendationEngine                          ││
//! │  └─────────────────────────────────────────────────────────┘│
//! │                           │                                  │
//! │                           ▼                                  │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │              Ontology Store                              ││
//! │  │  (Entities: Project, Task, WaitingFor, SomedayMaybe)    ││
//! │  └─────────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use alloy::gtd::{ProjectManager, TaskManager, Project, Task, InboxManager, ReviewManager};
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//!
//! // Create managers with a shared ontology store
//! let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
//! let projects = ProjectManager::new(store.clone());
//! let tasks = TaskManager::new(store.clone());
//! let inbox = InboxManager::new(store.clone());
//! let review = ReviewManager::new(store.clone());
//!
//! // Create a project
//! let project = Project::new("Website Redesign")
//!     .with_outcome("Launch new website with improved UX");
//! let project = projects.create(project).await?;
//!
//! // Create a task for the project
//! let task = Task::new("Draft wireframes")
//!     .with_project(&project.id)
//!     .with_context("@computer");
//! let task = tasks.create(task).await?;
//!
//! // Get recommendations based on context
//! let recs = tasks.recommend(RecommendParams {
//!     current_context: Some("@computer".to_string()),
//!     energy_level: Some(EnergyLevel::High),
//!     ..Default::default()
//! }).await?;
//!
//! // Process inbox items
//! let result = inbox.process(ProcessInboxParams::default()).await?;
//!
//! // Generate weekly review
//! let report = review.generate_weekly_review(WeeklyReviewParams::default()).await?;
//! ```

pub mod attention;
pub mod commitments;
pub mod dependencies;
pub mod horizons;
pub mod inbox;
mod projects;
pub mod recommend;
pub mod review;
mod someday;
mod tasks;
pub mod types;
mod waiting;

pub use inbox::{
    ClassificationResult, InboxItem, InboxManager, ProcessInboxParams, ProcessInboxResponse,
    ProcessingSummary, ProjectSuggestion, SuggestedGtdType,
};
pub use projects::ProjectManager;
// The recommend module provides an advanced recommendation engine.
// Use recommend::RecommendParams (not types::RecommendParams) for the full engine.
pub use recommend::RecommendParams as AdvancedRecommendParams;
pub use recommend::TaskRecommendation as DetailedTaskRecommendation;
pub use recommend::{
    RecommendContext, RecommendResponse, RecommendationEngine, ScoreBreakdown, ScoreWeights,
};
pub use review::{
    AreaStatus, AreasSummary, CompletedTasksSummary, HealthRating, InboxStatus, ProjectWithHealth,
    ProjectsSummary, Recommendation, ReviewManager, ReviewPeriod, ReviewSection,
    SomedayMaybeSummary, SystemHealth, SystemHealthFactors, UpcomingDeadline, UpcomingSummary,
    WaitingForSummary, WeeklyReviewParams, WeeklyReviewReport,
};
pub use someday::SomedayManager;
pub use tasks::TaskManager;
pub use types::*;
pub use waiting::WaitingManager;

// Advanced features (Phase 8)
pub use attention::{
    AttentionImbalance, AttentionManager, AttentionMetrics, AttentionParams, AttentionScore,
    AttentionTrend, DailyAttention, FocusDepth, FocusRating, ImbalanceSeverity,
};
pub use commitments::{
    Commitment, CommitmentDirection, CommitmentExtractionResult, CommitmentFilter,
    CommitmentManager, CommitmentStatus, CommitmentSummary, PersonCommitments,
};
pub use dependencies::{
    BlockedItem, BlockerInfo, BlockerItem, BlockerSeverity, CriticalPath, DependencyEdge,
    DependencyGraph, DependencyManager, DependencyNode, DependencyParams, DependencyType,
    GraphStats, NodeType, OutputFormat,
};
pub use horizons::{
    AlignmentAnalysis, AlignmentChain, AlignmentGap, ChainLink, Horizon, HorizonHealth,
    HorizonItem, HorizonLevel, HorizonManager, HorizonMap, HorizonParams, HorizonRecommendation,
    HorizonStatus, OrphanedItem,
};

//! Getting Things Done (GTD) module for task and project management.
//!
//! This module provides the core GTD functionality including:
//!
//! - **Projects**: Outcomes requiring multiple actions
//! - **Tasks**: Single next actions
//! - **Waiting For**: Delegated items
//! - **Someday/Maybe**: Deferred items for future consideration
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                     GTD Layer                            │
//! │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐  │
//! │  │ ProjectManager│ │  TaskManager  │ │WaitingManager │  │
//! │  └───────────────┘ └───────────────┘ └───────────────┘  │
//! │  ┌───────────────┐                                      │
//! │  │SomedayManager │                                      │
//! │  └───────────────┘                                      │
//! │                           │                              │
//! │                           ▼                              │
//! │  ┌─────────────────────────────────────────────────────┐│
//! │  │              Ontology Store                         ││
//! │  │  (Entities: Project, Task, WaitingFor, SomedayMaybe)││
//! │  └─────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use alloy::gtd::{ProjectManager, TaskManager, Project, Task};
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//!
//! // Create managers with a shared ontology store
//! let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
//! let projects = ProjectManager::new(store.clone());
//! let tasks = TaskManager::new(store.clone());
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
//! ```

mod projects;
mod someday;
mod tasks;
pub mod types;
mod waiting;

pub use projects::ProjectManager;
pub use someday::SomedayManager;
pub use tasks::TaskManager;
pub use types::*;
pub use waiting::WaitingManager;

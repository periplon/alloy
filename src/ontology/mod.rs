//! Ontology module for semantic entity and relationship management.
//!
//! This module provides the foundation for the GTD and Knowledge Graph system,
//! enabling structured storage and querying of entities (people, projects, tasks,
//! concepts) and their relationships.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Ontology Layer                        │
//! │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐  │
//! │  │ Entity Store  │ │ Relationship  │ │ Query Engine  │  │
//! │  │ (typed nodes) │ │    Graph      │ │ (traversal)   │  │
//! │  └───────────────┘ └───────────────┘ └───────────────┘  │
//! │                                                          │
//! │  ┌─────────────────────────────────────────────────────┐│
//! │  │           Entity Extraction Pipeline                ││
//! │  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  ││
//! │  │  │  Temporal   │ │   Action    │ │     NER       │  ││
//! │  │  │   Parser    │ │  Detector   │ │ (Local/LLM)   │  ││
//! │  │  └─────────────┘ └─────────────┘ └───────────────┘  ││
//! │  └─────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Modules
//!
//! - `types`: Core entity and relationship type definitions
//! - `store`: Storage backends for the ontology
//! - `extraction`: Entity extraction pipeline (temporal, NER, actions, relations)

pub mod extraction;
mod store;
mod types;

pub use store::{EmbeddedOntologyStore, OntologyStore};
pub use types::*;

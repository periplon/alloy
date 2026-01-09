//! Search module for hybrid search orchestration.
//!
//! This module provides hybrid search capabilities that combine:
//! - Vector similarity search (semantic matching)
//! - Full-text BM25 search (keyword matching)
//! - Configurable fusion algorithms (RRF, DBSF)
//!
//! # Architecture
//!
//! ```text
//! Query Text
//!     │
//!     ├─────────────────┬───────────────────┐
//!     ▼                 ▼                   ▼
//! Embedding        Text Search         Filters
//!     │                 │                   │
//!     ▼                 ▼                   │
//! Vector Search    BM25 Search             │
//!     │                 │                   │
//!     └────────┬────────┘                   │
//!              ▼                            │
//!         Fusion (RRF/DBSF)                 │
//!              │                            │
//!              └────────────────────────────┘
//!                        │
//!                        ▼
//!                 Ranked Results
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use alloy::search::{HybridSearchOrchestrator, HybridQuery, SearchFilter, FusionAlgorithm};
//!
//! // Create orchestrator
//! let orchestrator = HybridSearchOrchestrator::with_defaults(storage, embedder);
//!
//! // Build query with filters
//! let query = HybridQuery::new("machine learning")
//!     .limit(10)
//!     .vector_weight(0.6)  // Favor semantic search
//!     .filter(SearchFilter::new().source("docs").file_type("application/pdf"));
//!
//! // Execute search
//! let response = orchestrator.search(query).await?;
//!
//! for result in response.results {
//!     println!("{}: {:.3}", result.document_id, result.score);
//! }
//! ```

mod fusion;
mod hybrid;

pub use fusion::*;
pub use hybrid::*;

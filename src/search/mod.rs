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
//!              ├─── Query Expansion ────────┤
//!              │                            │
//!              ├─── Reranking ──────────────┤
//!              │                            │
//!              └────────────────────────────┘
//!                        │
//!                        ▼
//!                 Ranked Results
//!
//!     [Optional: Clustering for exploration]
//! ```
//!
//! # Search Enhancement Features
//!
//! ## Reranking
//!
//! Improve search precision by re-scoring top candidates:
//! - Score-based: Uses embedding similarity
//! - Cross-encoder: Uses cross-encoder models (API-based)
//! - LLM-based: Uses language models for relevance scoring
//!
//! ## Query Expansion
//!
//! Improve recall by augmenting queries with related terms:
//! - Embedding-based: Find semantically similar terms
//! - Synonym-based: Use string similarity and word stems
//! - Pseudo-relevance feedback: Expand using terms from top results
//!
//! ## Semantic Clustering
//!
//! Group similar documents for exploration:
//! - K-Means: Classic centroid-based clustering
//! - DBSCAN: Density-based clustering with automatic cluster count
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

mod clustering;
mod expansion;
mod fusion;
mod hybrid;
mod reranker;

pub use clustering::*;
pub use expansion::*;
pub use fusion::*;
pub use hybrid::*;
pub use reranker::*;

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
//!
//! ## Query Expansion Example
//!
//! ```rust,ignore
//! // Enable query expansion to find more relevant results
//! let query = HybridQuery::new("database error")
//!     .expand_query(true)        // Enable expansion for this query
//!     .max_expansions(5)         // Add up to 5 related terms
//!     .rerank(true)              // Enable reranking for better precision
//!     .limit(20);
//!
//! // Search will expand "database error" with related terms like
//! // "exception", "failure", "storage", etc.
//! let response = orchestrator.search(query).await?;
//!
//! // Check if expansion was applied
//! if response.stats.query_expanded {
//!     println!("Expanded query: {:?}", response.stats.expanded_query);
//! }
//! ```

mod clustering;
mod clustering_cache;
mod clustering_viz;
mod cross_encoder;
mod expansion;
mod fusion;
mod hybrid;
mod reranker;

pub use clustering::*;
pub use clustering_cache::*;
pub use clustering_viz::*;
pub use cross_encoder::*;
pub use expansion::*;
pub use fusion::*;
pub use hybrid::*;
pub use reranker::*;

//! Alloy: Hybrid Document Indexing MCP Server
//!
//! A Rust MCP server for indexing local and S3 files with hybrid search
//! combining vector similarity and full-text matching.

pub mod config;
pub mod coordinator;
pub mod embedding;
pub mod error;
pub mod mcp;
pub mod processing;
pub mod search;
pub mod sources;
pub mod storage;

pub use config::Config;
pub use coordinator::{IndexCoordinator, IndexCoordinatorBuilder, IndexProgress, IndexedSource};
pub use error::{AlloyError, Result};
pub use mcp::{run_server, AlloyServer};

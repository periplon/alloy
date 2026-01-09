//! Alloy: Hybrid Document Indexing MCP Server
//!
//! A Rust MCP server for indexing local and S3 files with hybrid search
//! combining vector similarity and full-text matching.

pub mod backup;
pub mod cache;
pub mod config;
pub mod coordinator;
pub mod embedding;
pub mod error;
pub mod mcp;
pub mod metrics;
pub mod processing;
pub mod search;
pub mod sources;
pub mod storage;
pub mod versioning;

pub use backup::{BackupManager, BackupResult, ExportFormat, ExportOptions, RestoreResult};
pub use cache::{CacheStats, CachedEmbedder, QueryCache};
pub use config::Config;
pub use coordinator::{IndexCoordinator, IndexCoordinatorBuilder, IndexProgress, IndexedSource};
pub use error::{AlloyError, Result};
pub use mcp::{run_server, AlloyServer};
pub use metrics::{get_metrics, HealthCheck, HealthState, HealthStatus, Metrics, MetricsSnapshot};

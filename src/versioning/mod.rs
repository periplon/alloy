//! Document versioning module.
//!
//! Provides version control for indexed documents:
//! - Version history tracking
//! - Content diffing between versions
//! - Version restoration
//! - Retention policies
//! - Chunk-level versioning for embeddings
//! - Compression support (gzip, zstd)

mod diff;
mod manager;
mod retention;
mod storage;

pub use diff::{compute_diff, compute_unified_diff, DiffChange, DiffHunk, DiffResult, DiffStats, UnifiedDiff};
pub use manager::{VersionDiff, VersionManager, VersioningConfig};
pub use retention::{RetentionEnforcer, RetentionPattern, RetentionPolicy};
pub use storage::{
    compression, ChangeType, ChunkVersion, CleanupStats, CompressionMethod, DeltaOperation,
    DocumentVersion, FileVersionStorage, InMemoryVersionStorage, VersionMetadata, VersionStorage,
    VersionStorageType,
};

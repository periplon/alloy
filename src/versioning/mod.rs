//! Document versioning module.
//!
//! Provides version control for indexed documents:
//! - Version history tracking
//! - Content diffing between versions
//! - Version restoration
//! - Retention policies

mod diff;
mod manager;
mod retention;
mod storage;

pub use diff::{compute_diff, DiffResult, DiffStats, UnifiedDiff};
pub use manager::{VersionManager, VersioningConfig};
pub use retention::{RetentionPattern, RetentionPolicy};
pub use storage::{
    ChangeType, DocumentVersion, FileVersionStorage, VersionMetadata, VersionStorage,
    VersionStorageType,
};

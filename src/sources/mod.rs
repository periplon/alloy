//! Source module for document sources (local files, S3).
//!
//! This module provides abstractions and implementations for various
//! document sources that can be indexed by Alloy.
//!
//! # Sources
//!
//! - [`LocalSource`]: Scans local filesystem directories with glob pattern matching
//!   and optional file watching for real-time change detection.
//!
//! - [`S3Source`]: Scans S3 buckets with prefix filtering and pattern matching,
//!   with optional polling for change detection.
//!
//! # Example
//!
//! ```ignore
//! use alloy::sources::{LocalSource, LocalSourceConfig, Source};
//! use std::path::PathBuf;
//!
//! // Create a local source
//! let config = LocalSourceConfig {
//!     path: PathBuf::from("/path/to/docs"),
//!     patterns: vec!["**/*.md".to_string(), "**/*.txt".to_string()],
//!     exclude_patterns: vec!["**/node_modules/**".to_string()],
//!     watch: false,
//!     follow_symlinks: false,
//! };
//!
//! let source = LocalSource::new(config)?;
//! let items = source.scan().await?;
//! ```

mod local;
mod s3;
mod traits;

pub use local::LocalSource;
pub use s3::{parse_s3_uri, S3Source};
pub use traits::*;

use crate::error::Result;
use std::sync::Arc;

/// A manager for multiple document sources.
pub struct SourceManager {
    sources: Vec<Arc<dyn Source>>,
}

impl SourceManager {
    /// Create a new empty source manager.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Add a source to the manager.
    pub fn add_source<S: Source + 'static>(&mut self, source: S) {
        self.sources.push(Arc::new(source));
    }

    /// Add a boxed source to the manager.
    pub fn add_boxed_source(&mut self, source: Arc<dyn Source>) {
        self.sources.push(source);
    }

    /// Get all registered sources.
    pub fn sources(&self) -> &[Arc<dyn Source>] {
        &self.sources
    }

    /// Find the source that handles a given URI.
    pub fn find_source(&self, uri: &str) -> Option<Arc<dyn Source>> {
        self.sources.iter().find(|s| s.handles(uri)).cloned()
    }

    /// Scan all sources and return all items.
    pub async fn scan_all(&self) -> Result<Vec<SourceItem>> {
        let mut all_items = Vec::new();

        for source in &self.sources {
            match source.scan().await {
                Ok(items) => {
                    all_items.extend(items);
                }
                Err(e) => {
                    tracing::error!("Error scanning source {}: {}", source.id(), e);
                }
            }
        }

        Ok(all_items)
    }

    /// Get statistics for all sources.
    pub fn stats(&self) -> Vec<(String, SourceStats)> {
        self.sources
            .iter()
            .map(|s| (s.id().to_string(), s.stats()))
            .collect()
    }

    /// Remove a source by ID.
    pub fn remove_source(&mut self, id: &str) -> bool {
        let initial_len = self.sources.len();
        self.sources.retain(|s| s.id() != id);
        self.sources.len() != initial_len
    }
}

impl Default for SourceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_dir() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        File::create(temp_dir.path().join("test.txt"))
            .unwrap()
            .write_all(b"test content")
            .unwrap();
        temp_dir
    }

    #[tokio::test]
    async fn test_source_manager() {
        let temp_dir = create_test_dir();

        let config = LocalSourceConfig {
            path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let source = LocalSource::new(config).unwrap();
        let source_id = source.id().to_string();

        let mut manager = SourceManager::new();
        manager.add_source(source);

        assert_eq!(manager.sources().len(), 1);

        // Test scan_all
        let items = manager.scan_all().await.unwrap();
        assert!(!items.is_empty());

        // Test find_source
        let file_path = temp_dir.path().join("test.txt");
        let found = manager.find_source(file_path.to_str().unwrap());
        assert!(found.is_some());

        // Test stats
        let stats = manager.stats();
        assert_eq!(stats.len(), 1);

        // Test remove_source
        assert!(manager.remove_source(&source_id));
        assert_eq!(manager.sources().len(), 0);
    }

    #[tokio::test]
    async fn test_source_manager_find_correct_source() {
        let temp_dir1 = TempDir::new().unwrap();
        let temp_dir2 = TempDir::new().unwrap();

        File::create(temp_dir1.path().join("file1.txt"))
            .unwrap()
            .write_all(b"1")
            .unwrap();
        File::create(temp_dir2.path().join("file2.txt"))
            .unwrap()
            .write_all(b"2")
            .unwrap();

        let source1 = LocalSource::new(LocalSourceConfig {
            path: temp_dir1.path().to_path_buf(),
            ..Default::default()
        })
        .unwrap();

        let source2 = LocalSource::new(LocalSourceConfig {
            path: temp_dir2.path().to_path_buf(),
            ..Default::default()
        })
        .unwrap();

        let source1_id = source1.id().to_string();
        let source2_id = source2.id().to_string();

        let mut manager = SourceManager::new();
        manager.add_source(source1);
        manager.add_source(source2);

        let file1_path = temp_dir1.path().join("file1.txt");
        let found1 = manager.find_source(file1_path.to_str().unwrap());
        assert_eq!(found1.unwrap().id(), source1_id);

        let file2_path = temp_dir2.path().join("file2.txt");
        let found2 = manager.find_source(file2_path.to_str().unwrap());
        assert_eq!(found2.unwrap().id(), source2_id);
    }
}

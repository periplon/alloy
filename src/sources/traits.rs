//! Source trait definitions and common types.

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

use crate::error::Result;

/// Represents an item from a source (file, S3 object, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceItem {
    /// Unique identifier for this item
    pub id: String,
    /// URI or path to the item
    pub uri: String,
    /// MIME type
    pub mime_type: String,
    /// Size in bytes
    pub size: u64,
    /// Last modified time
    pub modified: DateTime<Utc>,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl SourceItem {
    /// Create a new source item from a local file path.
    pub fn from_path(path: &std::path::Path, metadata: &std::fs::Metadata) -> Result<Self> {
        let uri = path.to_string_lossy().to_string();
        let id = uuid::Uuid::new_v4().to_string();

        let mime_type = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();

        let modified = metadata
            .modified()
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(|_| Utc::now());

        Ok(Self {
            id,
            uri,
            mime_type,
            size: metadata.len(),
            modified,
            metadata: serde_json::json!({}),
        })
    }

    /// Create a new source item from an S3 object.
    pub fn from_s3_object(
        bucket: &str,
        key: &str,
        size: u64,
        last_modified: Option<DateTime<Utc>>,
        e_tag: Option<&str>,
    ) -> Self {
        let uri = format!("s3://{}/{}", bucket, key);
        let id = uuid::Uuid::new_v4().to_string();

        let mime_type = mime_guess::from_path(key)
            .first_or_octet_stream()
            .to_string();

        let metadata = serde_json::json!({
            "bucket": bucket,
            "key": key,
            "etag": e_tag,
        });

        Self {
            id,
            uri,
            mime_type,
            size,
            modified: last_modified.unwrap_or_else(Utc::now),
            metadata,
        }
    }
}

/// Events from source watchers.
#[derive(Debug, Clone)]
pub enum SourceEvent {
    /// A new item was added
    Created(SourceItem),
    /// An existing item was modified
    Modified(SourceItem),
    /// An item was deleted
    Deleted(String),
}

impl fmt::Display for SourceEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceEvent::Created(item) => write!(f, "Created: {}", item.uri),
            SourceEvent::Modified(item) => write!(f, "Modified: {}", item.uri),
            SourceEvent::Deleted(uri) => write!(f, "Deleted: {}", uri),
        }
    }
}

/// Configuration for a local file source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSourceConfig {
    /// Base path for the source
    pub path: PathBuf,
    /// Glob patterns to include (e.g., "**/*.md", "*.txt")
    pub patterns: Vec<String>,
    /// Glob patterns to exclude (e.g., "**/node_modules/**", "*.tmp")
    pub exclude_patterns: Vec<String>,
    /// Whether to watch for changes
    pub watch: bool,
    /// Whether to follow symlinks
    pub follow_symlinks: bool,
}

impl Default for LocalSourceConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("."),
            patterns: vec!["**/*".to_string()],
            exclude_patterns: vec![
                "**/node_modules/**".to_string(),
                "**/.git/**".to_string(),
                "**/target/**".to_string(),
                "**/__pycache__/**".to_string(),
                "**/venv/**".to_string(),
                "**/.venv/**".to_string(),
            ],
            watch: false,
            follow_symlinks: false,
        }
    }
}

/// Configuration for an S3 source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3SourceConfig {
    /// S3 bucket name
    pub bucket: String,
    /// Prefix to filter objects (e.g., "documents/")
    pub prefix: Option<String>,
    /// AWS region
    pub region: Option<String>,
    /// Custom endpoint URL (for MinIO, LocalStack, etc.)
    pub endpoint_url: Option<String>,
    /// Polling interval in seconds for change detection
    pub poll_interval_secs: u64,
    /// File patterns to include (matched against object keys)
    pub patterns: Vec<String>,
    /// File patterns to exclude
    pub exclude_patterns: Vec<String>,
}

impl Default for S3SourceConfig {
    fn default() -> Self {
        Self {
            bucket: String::new(),
            prefix: None,
            region: None,
            endpoint_url: None,
            poll_interval_secs: 300, // 5 minutes
            patterns: vec!["*".to_string()],
            exclude_patterns: vec![],
        }
    }
}

/// Trait for document sources.
#[async_trait]
pub trait Source: Send + Sync {
    /// Get the source identifier (unique name/path).
    fn id(&self) -> &str;

    /// Scan the source and return all items.
    async fn scan(&self) -> Result<Vec<SourceItem>>;

    /// Fetch the content of a specific item.
    async fn fetch(&self, uri: &str) -> Result<Bytes>;

    /// Check if a specific URI belongs to this source.
    fn handles(&self, uri: &str) -> bool;

    /// Get a stream of change events (if watching is supported).
    fn supports_watch(&self) -> bool {
        false
    }

    /// Get source statistics.
    fn stats(&self) -> SourceStats {
        SourceStats::default()
    }
}

/// Statistics for a source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceStats {
    /// Total number of items scanned
    pub total_items: usize,
    /// Total size in bytes
    pub total_bytes: u64,
    /// Last scan time
    pub last_scan: Option<DateTime<Utc>>,
    /// Number of items currently being watched
    pub watched_items: usize,
}

/// A boxed source for dynamic dispatch.
pub type BoxedSource = Box<dyn Source>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::TempDir;

    #[test]
    fn test_source_item_from_path() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        File::create(&file_path).unwrap();

        let metadata = std::fs::metadata(&file_path).unwrap();
        let item = SourceItem::from_path(&file_path, &metadata).unwrap();

        assert!(item.uri.contains("test.txt"));
        assert_eq!(item.mime_type, "text/plain");
    }

    #[test]
    fn test_source_item_from_s3() {
        let item = SourceItem::from_s3_object(
            "my-bucket",
            "documents/file.pdf",
            1024,
            Some(Utc::now()),
            Some("abc123"),
        );

        assert_eq!(item.uri, "s3://my-bucket/documents/file.pdf");
        assert_eq!(item.mime_type, "application/pdf");
        assert_eq!(item.size, 1024);
    }

    #[test]
    fn test_source_event_display() {
        let item = SourceItem::from_s3_object("bucket", "key", 0, None, None);
        let event = SourceEvent::Created(item);
        let display = format!("{}", event);
        assert!(display.contains("Created"));
        assert!(display.contains("s3://bucket/key"));
    }

    #[test]
    fn test_local_source_config_default() {
        let config = LocalSourceConfig::default();
        assert!(config
            .exclude_patterns
            .iter()
            .any(|p| p.contains("node_modules")));
        assert!(!config.watch);
    }

    #[test]
    fn test_s3_source_config_default() {
        let config = S3SourceConfig::default();
        assert_eq!(config.poll_interval_secs, 300);
        assert!(config.prefix.is_none());
    }
}

//! Local file source implementation with glob patterns and file watching.

use async_trait::async_trait;
use bytes::Bytes;
use chrono::Utc;
use glob::Pattern;
use notify::{
    event::{CreateKind, ModifyKind, RemoveKind},
    Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::traits::{LocalSourceConfig, Source, SourceEvent, SourceItem, SourceStats};
use crate::error::{Result, SourceError};

/// A local filesystem source that scans files using glob patterns.
pub struct LocalSource {
    /// Unique identifier for this source
    id: String,
    /// Configuration
    config: LocalSourceConfig,
    /// Compiled include patterns
    include_patterns: Vec<Pattern>,
    /// Compiled exclude patterns
    exclude_patterns: Vec<Pattern>,
    /// Cached items by URI
    items_cache: Arc<RwLock<HashMap<String, SourceItem>>>,
    /// Statistics
    stats: Arc<RwLock<SourceStats>>,
    /// Event sender for watch events
    event_sender: Option<mpsc::Sender<SourceEvent>>,
    /// Watcher handle (kept alive to maintain watching)
    #[allow(dead_code)]
    watcher: Option<RecommendedWatcher>,
}

impl LocalSource {
    /// Create a new local source from configuration.
    pub fn new(config: LocalSourceConfig) -> Result<Self> {
        let canonical_path = config
            .path
            .canonicalize()
            .map_err(|e| SourceError::PathNotFound(format!("{}: {}", config.path.display(), e)))?;

        let id = canonical_path.to_string_lossy().to_string();

        // Compile include patterns
        let include_patterns: Vec<Pattern> = config
            .patterns
            .iter()
            .filter_map(|p| match Pattern::new(p) {
                Ok(pattern) => Some(pattern),
                Err(e) => {
                    warn!("Invalid include pattern '{}': {}", p, e);
                    None
                }
            })
            .collect();

        // Compile exclude patterns
        let exclude_patterns: Vec<Pattern> = config
            .exclude_patterns
            .iter()
            .filter_map(|p| match Pattern::new(p) {
                Ok(pattern) => Some(pattern),
                Err(e) => {
                    warn!("Invalid exclude pattern '{}': {}", p, e);
                    None
                }
            })
            .collect();

        Ok(Self {
            id,
            config: LocalSourceConfig {
                path: canonical_path,
                ..config
            },
            include_patterns,
            exclude_patterns,
            items_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SourceStats::default())),
            event_sender: None,
            watcher: None,
        })
    }

    /// Create a new local source with file watching enabled.
    pub fn with_watcher(config: LocalSourceConfig) -> Result<(Self, mpsc::Receiver<SourceEvent>)> {
        let mut source = Self::new(LocalSourceConfig {
            watch: true,
            ..config
        })?;

        let (tx, rx) = mpsc::channel(1000);
        source.event_sender = Some(tx.clone());

        // Set up the file watcher
        let watcher = source.setup_watcher(tx)?;
        source.watcher = Some(watcher);

        Ok((source, rx))
    }

    /// Set up the file watcher.
    fn setup_watcher(&self, tx: mpsc::Sender<SourceEvent>) -> Result<RecommendedWatcher> {
        let path = self.config.path.clone();
        let include_patterns = self.include_patterns.clone();
        let exclude_patterns = self.exclude_patterns.clone();
        let follow_symlinks = self.config.follow_symlinks;

        let watcher = notify::recommended_watcher(move |res: std::result::Result<Event, _>| {
            match res {
                Ok(event) => {
                    Self::handle_notify_event(
                        &event,
                        &path,
                        &include_patterns,
                        &exclude_patterns,
                        follow_symlinks,
                        &tx,
                    );
                }
                Err(e) => {
                    error!("Watch error: {:?}", e);
                }
            }
        })
        .map_err(|e| SourceError::Watch(e.to_string()))?;

        Ok(watcher)
    }

    /// Start watching the source directory.
    pub fn start_watching(&mut self) -> Result<()> {
        if let Some(ref mut watcher) = self.watcher {
            watcher
                .watch(&self.config.path, RecursiveMode::Recursive)
                .map_err(|e| SourceError::Watch(e.to_string()))?;
            info!("Started watching: {}", self.config.path.display());
        }
        Ok(())
    }

    /// Stop watching the source directory.
    pub fn stop_watching(&mut self) -> Result<()> {
        if let Some(ref mut watcher) = self.watcher {
            watcher
                .unwatch(&self.config.path)
                .map_err(|e| SourceError::Watch(e.to_string()))?;
            info!("Stopped watching: {}", self.config.path.display());
        }
        Ok(())
    }

    /// Handle a notify event and convert to SourceEvent.
    fn handle_notify_event(
        event: &Event,
        base_path: &Path,
        include_patterns: &[Pattern],
        exclude_patterns: &[Pattern],
        _follow_symlinks: bool,
        tx: &mpsc::Sender<SourceEvent>,
    ) {
        for path in &event.paths {
            // Skip directories
            if path.is_dir() {
                continue;
            }

            // Get relative path for pattern matching
            let rel_path = path
                .strip_prefix(base_path)
                .unwrap_or(path)
                .to_string_lossy();

            // Check patterns
            if !Self::matches_patterns(&rel_path, include_patterns, exclude_patterns) {
                debug!("Skipping path (pattern mismatch): {}", path.display());
                continue;
            }

            let source_event = match &event.kind {
                EventKind::Create(CreateKind::File) => {
                    if let Ok(metadata) = std::fs::metadata(path) {
                        if let Ok(item) = SourceItem::from_path(path, &metadata) {
                            Some(SourceEvent::Created(item))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                EventKind::Modify(ModifyKind::Data(_) | ModifyKind::Any) => {
                    if let Ok(metadata) = std::fs::metadata(path) {
                        if let Ok(item) = SourceItem::from_path(path, &metadata) {
                            Some(SourceEvent::Modified(item))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                EventKind::Remove(RemoveKind::File) => {
                    Some(SourceEvent::Deleted(path.to_string_lossy().to_string()))
                }
                _ => None,
            };

            if let Some(evt) = source_event {
                debug!("Source event: {}", evt);
                if let Err(e) = tx.blocking_send(evt) {
                    error!("Failed to send source event: {}", e);
                }
            }
        }
    }

    /// Check if a path matches include patterns and doesn't match exclude patterns.
    fn matches_patterns(
        rel_path: &str,
        include_patterns: &[Pattern],
        exclude_patterns: &[Pattern],
    ) -> bool {
        // Check if excluded first
        for pattern in exclude_patterns {
            if pattern.matches(rel_path) || pattern.matches_path(Path::new(rel_path)) {
                return false;
            }
        }

        // Then check if included (empty means include all)
        if include_patterns.is_empty() {
            return true;
        }

        for pattern in include_patterns {
            if pattern.matches(rel_path) || pattern.matches_path(Path::new(rel_path)) {
                return true;
            }
        }

        false
    }

    /// Recursively scan a directory for files.
    fn scan_directory(&self, dir: &Path) -> Result<Vec<SourceItem>> {
        let mut items = Vec::new();

        let entries = std::fs::read_dir(dir).map_err(|e| {
            SourceError::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {}", dir.display(), e),
            ))
        })?;

        for entry in entries.flatten() {
            let path = entry.path();

            // Handle symlinks
            let metadata = if self.config.follow_symlinks {
                std::fs::metadata(&path)
            } else {
                std::fs::symlink_metadata(&path)
            };

            let metadata = match metadata {
                Ok(m) => m,
                Err(e) => {
                    debug!("Cannot read metadata for {}: {}", path.display(), e);
                    continue;
                }
            };

            if metadata.is_dir() {
                // Skip symlinked directories if not following
                if !self.config.follow_symlinks && metadata.file_type().is_symlink() {
                    continue;
                }
                // Recurse into directory
                match self.scan_directory(&path) {
                    Ok(sub_items) => items.extend(sub_items),
                    Err(e) => {
                        debug!("Error scanning directory {}: {}", path.display(), e);
                    }
                }
            } else if metadata.is_file() {
                // Check patterns
                let rel_path = path
                    .strip_prefix(&self.config.path)
                    .unwrap_or(&path)
                    .to_string_lossy();

                if Self::matches_patterns(&rel_path, &self.include_patterns, &self.exclude_patterns)
                {
                    match SourceItem::from_path(&path, &metadata) {
                        Ok(item) => {
                            debug!("Found file: {}", item.uri);
                            items.push(item);
                        }
                        Err(e) => {
                            debug!("Error creating source item for {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }

        Ok(items)
    }
}

#[async_trait]
impl Source for LocalSource {
    fn id(&self) -> &str {
        &self.id
    }

    async fn scan(&self) -> Result<Vec<SourceItem>> {
        info!("Scanning local source: {}", self.config.path.display());

        // Perform the scan synchronously in a blocking task
        let path = self.config.path.clone();
        let include_patterns = self.include_patterns.clone();
        let exclude_patterns = self.exclude_patterns.clone();
        let follow_symlinks = self.config.follow_symlinks;

        let items = tokio::task::spawn_blocking(move || {
            let source = LocalSource {
                id: String::new(),
                config: LocalSourceConfig {
                    path: path.clone(),
                    patterns: vec![],
                    exclude_patterns: vec![],
                    watch: false,
                    follow_symlinks,
                },
                include_patterns,
                exclude_patterns,
                items_cache: Arc::new(RwLock::new(HashMap::new())),
                stats: Arc::new(RwLock::new(SourceStats::default())),
                event_sender: None,
                watcher: None,
            };
            source.scan_directory(&path)
        })
        .await
        .map_err(|e| SourceError::Io(std::io::Error::other(e.to_string())))??;

        // Update cache and stats
        {
            let mut cache = self.items_cache.write().unwrap();
            cache.clear();
            for item in &items {
                cache.insert(item.uri.clone(), item.clone());
            }
        }

        {
            let mut stats = self.stats.write().unwrap();
            stats.total_items = items.len();
            stats.total_bytes = items.iter().map(|i| i.size).sum();
            stats.last_scan = Some(Utc::now());
            if self.config.watch {
                stats.watched_items = items.len();
            }
        }

        info!(
            "Scanned {} files ({} bytes) from {}",
            items.len(),
            items.iter().map(|i| i.size).sum::<u64>(),
            self.config.path.display()
        );

        Ok(items)
    }

    async fn fetch(&self, uri: &str) -> Result<Bytes> {
        let path = PathBuf::from(uri);

        // Canonicalize both paths for comparison (handles symlinks like /var -> /private/var)
        let canonical_uri = path.canonicalize().map_err(|e| {
            SourceError::Io(std::io::Error::new(e.kind(), format!("{}: {}", uri, e)))
        })?;

        // Verify the path is under our source directory
        if !canonical_uri.starts_with(&self.config.path) {
            return Err(SourceError::AccessDenied(format!(
                "Path {} is outside source directory {}",
                uri,
                self.config.path.display()
            ))
            .into());
        }

        let content = tokio::fs::read(&path).await.map_err(|e| {
            SourceError::Io(std::io::Error::new(e.kind(), format!("{}: {}", uri, e)))
        })?;

        Ok(Bytes::from(content))
    }

    fn handles(&self, uri: &str) -> bool {
        // Handle local paths that are under our source directory
        if uri.starts_with("s3://") {
            return false;
        }

        let path = PathBuf::from(uri);
        // Try to canonicalize for proper comparison (handles symlinks)
        if let Ok(canonical) = path.canonicalize() {
            return canonical.starts_with(&self.config.path);
        }
        // Fallback to simple prefix check
        path.starts_with(&self.config.path)
    }

    fn supports_watch(&self) -> bool {
        self.config.watch && self.watcher.is_some()
    }

    fn stats(&self) -> SourceStats {
        self.stats.read().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_dir() -> TempDir {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        File::create(temp_dir.path().join("file1.txt"))
            .unwrap()
            .write_all(b"content1")
            .unwrap();
        File::create(temp_dir.path().join("file2.md"))
            .unwrap()
            .write_all(b"# Markdown")
            .unwrap();

        // Create subdirectory
        fs::create_dir(temp_dir.path().join("subdir")).unwrap();
        File::create(temp_dir.path().join("subdir/file3.rs"))
            .unwrap()
            .write_all(b"fn main() {}")
            .unwrap();

        // Create excluded directory
        fs::create_dir(temp_dir.path().join("node_modules")).unwrap();
        File::create(temp_dir.path().join("node_modules/package.json"))
            .unwrap()
            .write_all(b"{}")
            .unwrap();

        temp_dir
    }

    #[tokio::test]
    async fn test_local_source_scan() {
        let temp_dir = create_test_dir();

        let config = LocalSourceConfig {
            path: temp_dir.path().to_path_buf(),
            patterns: vec!["**/*".to_string()],
            exclude_patterns: vec!["**/node_modules/**".to_string()],
            watch: false,
            follow_symlinks: false,
        };

        let source = LocalSource::new(config).unwrap();
        let items = source.scan().await.unwrap();

        // Should find 3 files (excluding node_modules)
        assert_eq!(items.len(), 3);

        // Verify file types
        let uris: Vec<&str> = items.iter().map(|i| i.uri.as_str()).collect();
        assert!(uris.iter().any(|u| u.ends_with("file1.txt")));
        assert!(uris.iter().any(|u| u.ends_with("file2.md")));
        assert!(uris.iter().any(|u| u.ends_with("file3.rs")));
        assert!(!uris.iter().any(|u| u.contains("node_modules")));
    }

    #[tokio::test]
    async fn test_local_source_fetch() {
        let temp_dir = create_test_dir();

        let config = LocalSourceConfig {
            path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let source = LocalSource::new(config).unwrap();
        let file_path = temp_dir.path().join("file1.txt");

        let content = source.fetch(file_path.to_str().unwrap()).await.unwrap();
        assert_eq!(content.as_ref(), b"content1");
    }

    #[tokio::test]
    async fn test_local_source_handles() {
        let temp_dir = create_test_dir();

        let config = LocalSourceConfig {
            path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let source = LocalSource::new(config).unwrap();

        assert!(source.handles(temp_dir.path().join("file1.txt").to_str().unwrap()));
        assert!(!source.handles("/some/other/path"));
        assert!(!source.handles("s3://bucket/key"));
    }

    #[test]
    fn test_pattern_matching() {
        let include = vec![Pattern::new("**/*.rs").unwrap()];
        let exclude = vec![Pattern::new("**/target/**").unwrap()];

        assert!(LocalSource::matches_patterns("src/main.rs", &include, &exclude));
        assert!(!LocalSource::matches_patterns("target/debug/main.rs", &include, &exclude));
        assert!(!LocalSource::matches_patterns("src/main.txt", &include, &exclude));
    }

    #[tokio::test]
    async fn test_local_source_stats() {
        let temp_dir = create_test_dir();

        let config = LocalSourceConfig {
            path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let source = LocalSource::new(config).unwrap();

        // Before scan
        let stats = source.stats();
        assert_eq!(stats.total_items, 0);

        // After scan
        source.scan().await.unwrap();
        let stats = source.stats();
        assert!(stats.total_items > 0);
        assert!(stats.last_scan.is_some());
    }

    #[test]
    fn test_glob_patterns_compile() {
        let config = LocalSourceConfig {
            path: PathBuf::from("."),
            patterns: vec![
                "**/*.md".to_string(),
                "**/*.txt".to_string(),
                "*.json".to_string(),
            ],
            exclude_patterns: vec![
                "**/node_modules/**".to_string(),
                "**/.git/**".to_string(),
            ],
            watch: false,
            follow_symlinks: false,
        };

        // This should not panic
        let _ = LocalSource::new(config);
    }
}

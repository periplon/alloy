//! S3 source implementation with scanning and change polling.

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::Client;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use glob::Pattern;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use super::traits::{S3SourceConfig, Source, SourceEvent, SourceItem, SourceStats};
use crate::error::{Result, SourceError};

/// S3 object state for change detection.
#[derive(Debug, Clone)]
struct S3ObjectState {
    size: u64,
    last_modified: DateTime<Utc>,
    etag: Option<String>,
}

/// An S3 bucket source that scans objects and polls for changes.
pub struct S3Source {
    /// Unique identifier for this source
    id: String,
    /// Configuration
    config: S3SourceConfig,
    /// S3 client
    client: Client,
    /// Compiled include patterns
    include_patterns: Vec<Pattern>,
    /// Compiled exclude patterns
    exclude_patterns: Vec<Pattern>,
    /// Cached object states for change detection
    object_states: Arc<RwLock<HashMap<String, S3ObjectState>>>,
    /// Statistics
    stats: Arc<RwLock<SourceStats>>,
    /// Polling task handle
    poll_handle: Option<tokio::task::JoinHandle<()>>,
}

impl S3Source {
    /// Create a new S3 source from configuration.
    pub async fn new(config: S3SourceConfig) -> Result<Self> {
        if config.bucket.is_empty() {
            return Err(SourceError::InvalidUri("Bucket name cannot be empty".to_string()).into());
        }

        let id = format!(
            "s3://{}/{}",
            config.bucket,
            config.prefix.as_deref().unwrap_or("")
        );

        // Build AWS SDK config
        let mut aws_config_builder = aws_config::defaults(BehaviorVersion::latest())
            .credentials_provider(
                aws_config::default_provider::credentials::DefaultCredentialsChain::builder()
                    .build()
                    .await,
            );

        // Set region if specified
        if let Some(ref region) = config.region {
            aws_config_builder = aws_config_builder.region(Region::new(region.clone()));
        }

        let aws_config = aws_config_builder.load().await;

        // Build S3 client
        let mut s3_config_builder = aws_sdk_s3::config::Builder::from(&aws_config);

        // Set custom endpoint if specified (for MinIO, LocalStack, etc.)
        if let Some(ref endpoint) = config.endpoint_url {
            s3_config_builder = s3_config_builder
                .endpoint_url(endpoint)
                .force_path_style(true);
        }

        let client = Client::from_conf(s3_config_builder.build());

        // Compile patterns
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
            config,
            client,
            include_patterns,
            exclude_patterns,
            object_states: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SourceStats::default())),
            poll_handle: None,
        })
    }

    /// Start polling for changes and return a channel for events.
    pub fn start_polling(&mut self) -> mpsc::Receiver<SourceEvent> {
        let (tx, rx) = mpsc::channel(1000);

        let client = self.client.clone();
        let bucket = self.config.bucket.clone();
        let prefix = self.config.prefix.clone();
        let poll_interval = Duration::from_secs(self.config.poll_interval_secs);
        let include_patterns = self.include_patterns.clone();
        let exclude_patterns = self.exclude_patterns.clone();
        let object_states = self.object_states.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(poll_interval);

            loop {
                interval.tick().await;

                match Self::poll_changes(
                    &client,
                    &bucket,
                    prefix.as_deref(),
                    &include_patterns,
                    &exclude_patterns,
                    &object_states,
                )
                .await
                {
                    Ok(events) => {
                        for event in events {
                            if let Err(e) = tx.send(event).await {
                                error!("Failed to send S3 event: {}", e);
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        error!("S3 polling error: {}", e);
                    }
                }
            }
        });

        self.poll_handle = Some(handle);
        rx
    }

    /// Stop the polling task.
    pub fn stop_polling(&mut self) {
        if let Some(handle) = self.poll_handle.take() {
            handle.abort();
            info!("Stopped S3 polling for {}", self.id);
        }
    }

    /// Poll for changes and return events.
    async fn poll_changes(
        client: &Client,
        bucket: &str,
        prefix: Option<&str>,
        include_patterns: &[Pattern],
        exclude_patterns: &[Pattern],
        object_states: &Arc<RwLock<HashMap<String, S3ObjectState>>>,
    ) -> Result<Vec<SourceEvent>> {
        let mut events = Vec::new();

        // List all objects
        let mut paginator = client
            .list_objects_v2()
            .bucket(bucket)
            .set_prefix(prefix.map(String::from))
            .into_paginator()
            .send();

        let mut current_objects: HashMap<String, S3ObjectState> = HashMap::new();

        while let Some(page) = paginator.next().await {
            let page = page.map_err(|e| SourceError::S3(e.to_string()))?;

            for object in page.contents() {
                let key = object.key().unwrap_or_default();

                // Check patterns
                if !Self::matches_patterns(key, include_patterns, exclude_patterns) {
                    continue;
                }

                let size = object.size().unwrap_or(0) as u64;
                let last_modified = object
                    .last_modified()
                    .and_then(|dt| DateTime::from_timestamp(dt.secs(), dt.subsec_nanos()))
                    .unwrap_or_else(Utc::now);
                let etag = object.e_tag().map(String::from);

                current_objects.insert(
                    key.to_string(),
                    S3ObjectState {
                        size,
                        last_modified,
                        etag: etag.clone(),
                    },
                );

                // Check for new or modified objects
                let previous_states = object_states.read().unwrap();
                let uri = format!("s3://{}/{}", bucket, key);

                if let Some(prev) = previous_states.get(key) {
                    // Check if modified (different size, time, or etag)
                    if prev.size != size || prev.last_modified != last_modified || prev.etag != etag
                    {
                        let item = SourceItem::from_s3_object(
                            bucket,
                            key,
                            size,
                            Some(last_modified),
                            etag.as_deref(),
                        );
                        events.push(SourceEvent::Modified(item));
                        debug!("S3 object modified: {}", uri);
                    }
                } else {
                    // New object
                    let item = SourceItem::from_s3_object(
                        bucket,
                        key,
                        size,
                        Some(last_modified),
                        etag.as_deref(),
                    );
                    events.push(SourceEvent::Created(item));
                    debug!("S3 object created: {}", uri);
                }
            }
        }

        // Check for deleted objects
        {
            let previous_states = object_states.read().unwrap();
            for key in previous_states.keys() {
                if !current_objects.contains_key(key) {
                    let uri = format!("s3://{}/{}", bucket, key);
                    events.push(SourceEvent::Deleted(uri.clone()));
                    debug!("S3 object deleted: {}", uri);
                }
            }
        }

        // Update states
        {
            let mut states = object_states.write().unwrap();
            *states = current_objects;
        }

        Ok(events)
    }

    /// Check if an object key matches patterns.
    fn matches_patterns(
        key: &str,
        include_patterns: &[Pattern],
        exclude_patterns: &[Pattern],
    ) -> bool {
        // Check exclusions first
        for pattern in exclude_patterns {
            if pattern.matches(key) {
                return false;
            }
        }

        // If no include patterns, include all
        if include_patterns.is_empty() {
            return true;
        }

        // Check inclusions
        for pattern in include_patterns {
            if pattern.matches(key) {
                return true;
            }
        }

        false
    }

    /// List objects in the S3 bucket.
    async fn list_objects(&self) -> Result<Vec<SourceItem>> {
        let mut items = Vec::new();

        let mut paginator = self
            .client
            .list_objects_v2()
            .bucket(&self.config.bucket)
            .set_prefix(self.config.prefix.clone())
            .into_paginator()
            .send();

        while let Some(page) = paginator.next().await {
            let page = page.map_err(|e| SourceError::S3(e.to_string()))?;

            for object in page.contents() {
                let key = object.key().unwrap_or_default();

                // Skip directories (keys ending with /)
                if key.ends_with('/') {
                    continue;
                }

                // Check patterns
                if !Self::matches_patterns(key, &self.include_patterns, &self.exclude_patterns) {
                    continue;
                }

                let size = object.size().unwrap_or(0) as u64;
                let last_modified = object
                    .last_modified()
                    .and_then(|dt| DateTime::from_timestamp(dt.secs(), dt.subsec_nanos()));
                let etag = object.e_tag();

                let item =
                    SourceItem::from_s3_object(&self.config.bucket, key, size, last_modified, etag);

                debug!("Found S3 object: {}", item.uri);
                items.push(item);
            }
        }

        Ok(items)
    }
}

#[async_trait]
impl Source for S3Source {
    fn id(&self) -> &str {
        &self.id
    }

    async fn scan(&self) -> Result<Vec<SourceItem>> {
        info!(
            "Scanning S3 source: s3://{}/{}",
            self.config.bucket,
            self.config.prefix.as_deref().unwrap_or("")
        );

        let items = self.list_objects().await?;

        // Update object states
        {
            let mut states = self.object_states.write().unwrap();
            states.clear();
            for item in &items {
                // Extract key from URI
                let key = item
                    .uri
                    .strip_prefix(&format!("s3://{}/", self.config.bucket))
                    .unwrap_or(&item.uri);

                states.insert(
                    key.to_string(),
                    S3ObjectState {
                        size: item.size,
                        last_modified: item.modified,
                        etag: item
                            .metadata
                            .get("etag")
                            .and_then(|v| v.as_str())
                            .map(String::from),
                    },
                );
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_items = items.len();
            stats.total_bytes = items.iter().map(|i| i.size).sum();
            stats.last_scan = Some(Utc::now());
        }

        info!(
            "Scanned {} objects ({} bytes) from s3://{}/{}",
            items.len(),
            items.iter().map(|i| i.size).sum::<u64>(),
            self.config.bucket,
            self.config.prefix.as_deref().unwrap_or("")
        );

        Ok(items)
    }

    async fn fetch(&self, uri: &str) -> Result<Bytes> {
        // Parse the S3 URI
        let (bucket, key) = parse_s3_uri(uri)?;

        // Verify bucket matches
        if bucket != self.config.bucket {
            return Err(SourceError::InvalidUri(format!(
                "Bucket '{}' does not match source bucket '{}'",
                bucket, self.config.bucket
            ))
            .into());
        }

        // Get the object
        let response = self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| SourceError::S3(e.to_string()))?;

        // Read the body
        let body = response
            .body
            .collect()
            .await
            .map_err(|e| SourceError::S3(e.to_string()))?;

        Ok(body.into_bytes())
    }

    fn handles(&self, uri: &str) -> bool {
        if !uri.starts_with("s3://") {
            return false;
        }

        // Check if it's in our bucket and prefix
        if let Ok((bucket, key)) = parse_s3_uri(uri) {
            if bucket != self.config.bucket {
                return false;
            }
            if let Some(ref prefix) = self.config.prefix {
                return key.starts_with(prefix);
            }
            return true;
        }

        false
    }

    fn supports_watch(&self) -> bool {
        self.poll_handle.is_some()
    }

    fn stats(&self) -> SourceStats {
        self.stats.read().unwrap().clone()
    }
}

impl Drop for S3Source {
    fn drop(&mut self) {
        self.stop_polling();
    }
}

/// Parse an S3 URI into (bucket, key).
pub fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    if !uri.starts_with("s3://") {
        return Err(SourceError::InvalidUri(format!("Not an S3 URI: {}", uri)).into());
    }

    let path = &uri[5..]; // Remove "s3://"
    let parts: Vec<&str> = path.splitn(2, '/').collect();

    if parts.is_empty() || parts[0].is_empty() {
        return Err(SourceError::InvalidUri("Missing bucket name".to_string()).into());
    }

    let bucket = parts[0].to_string();
    let key = parts.get(1).unwrap_or(&"").to_string();

    Ok((bucket, key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_uri() {
        let (bucket, key) = parse_s3_uri("s3://my-bucket/path/to/file.txt").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.txt");
    }

    #[test]
    fn test_parse_s3_uri_no_key() {
        let (bucket, key) = parse_s3_uri("s3://my-bucket/").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "");
    }

    #[test]
    fn test_parse_s3_uri_bucket_only() {
        let (bucket, key) = parse_s3_uri("s3://my-bucket").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "");
    }

    #[test]
    fn test_parse_s3_uri_invalid() {
        assert!(parse_s3_uri("https://s3.amazonaws.com/bucket/key").is_err());
        assert!(parse_s3_uri("/local/path").is_err());
        assert!(parse_s3_uri("s3://").is_err());
    }

    #[test]
    fn test_s3_pattern_matching() {
        let include = vec![
            Pattern::new("*.pdf").unwrap(),
            Pattern::new("docs/*").unwrap(),
        ];
        let exclude = vec![Pattern::new("*.tmp").unwrap()];

        assert!(S3Source::matches_patterns(
            "document.pdf",
            &include,
            &exclude
        ));
        assert!(S3Source::matches_patterns(
            "docs/readme.md",
            &include,
            &exclude
        ));
        assert!(!S3Source::matches_patterns("image.png", &include, &exclude));
        assert!(!S3Source::matches_patterns(
            "document.pdf.tmp",
            &include,
            &exclude
        ));
    }

    #[test]
    fn test_s3_pattern_matching_empty_include() {
        let include: Vec<Pattern> = vec![];
        let exclude = vec![Pattern::new("*.tmp").unwrap()];

        // With empty include, everything is included except excludes
        assert!(S3Source::matches_patterns(
            "anything.txt",
            &include,
            &exclude
        ));
        assert!(!S3Source::matches_patterns("file.tmp", &include, &exclude));
    }

    #[test]
    fn test_s3_source_config_default() {
        let config = S3SourceConfig::default();
        assert!(config.bucket.is_empty());
        assert_eq!(config.poll_interval_secs, 300);
        assert!(config.prefix.is_none());
    }

    // Note: Integration tests requiring actual S3 access should be in tests/integration/
}

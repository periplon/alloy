//! Webhook dispatcher for notifying external systems on index events.
//!
//! Supports configurable webhooks that fire on events like:
//! - `document.indexed` - Document was successfully indexed
//! - `document.updated` - Document was updated (re-indexed)
//! - `document.deleted` - Document was removed from index
//! - `source.added` - New source was added
//! - `source.removed` - Source was removed
//! - `index.error` - An error occurred during indexing

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// Webhook event types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebhookEvent {
    /// Document was indexed successfully.
    DocumentIndexed,
    /// Document was updated.
    DocumentUpdated,
    /// Document was deleted.
    DocumentDeleted,
    /// Source was added.
    SourceAdded,
    /// Source was removed.
    SourceRemoved,
    /// Indexing error occurred.
    IndexError,
    /// Backup was created.
    BackupCreated,
    /// Search was performed.
    SearchPerformed,
}

impl WebhookEvent {
    /// Parse event type from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "document.indexed" | "document_indexed" => Some(Self::DocumentIndexed),
            "document.updated" | "document_updated" => Some(Self::DocumentUpdated),
            "document.deleted" | "document_deleted" => Some(Self::DocumentDeleted),
            "source.added" | "source_added" => Some(Self::SourceAdded),
            "source.removed" | "source_removed" => Some(Self::SourceRemoved),
            "index.error" | "index_error" => Some(Self::IndexError),
            "backup.created" | "backup_created" => Some(Self::BackupCreated),
            "search.performed" | "search_performed" => Some(Self::SearchPerformed),
            _ => None,
        }
    }

    /// Get the event type as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DocumentIndexed => "document.indexed",
            Self::DocumentUpdated => "document.updated",
            Self::DocumentDeleted => "document.deleted",
            Self::SourceAdded => "source.added",
            Self::SourceRemoved => "source.removed",
            Self::IndexError => "index.error",
            Self::BackupCreated => "backup.created",
            Self::SearchPerformed => "search.performed",
        }
    }
}

impl std::fmt::Display for WebhookEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Webhook configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Unique identifier for this webhook.
    #[serde(default = "generate_webhook_id")]
    pub id: String,
    /// Target URL to send webhook payloads.
    pub url: String,
    /// Events to subscribe to.
    pub events: Vec<String>,
    /// Secret for HMAC signature verification.
    #[serde(default)]
    pub secret: Option<String>,
    /// Number of retry attempts on failure.
    #[serde(default = "default_retry_count")]
    pub retry_count: usize,
    /// Timeout in seconds for webhook requests.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Whether this webhook is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional description for this webhook.
    #[serde(default)]
    pub description: Option<String>,
    /// Custom headers to include in webhook requests.
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

fn generate_webhook_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn default_retry_count() -> usize {
    3
}

fn default_timeout() -> u64 {
    30
}

fn default_true() -> bool {
    true
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            id: generate_webhook_id(),
            url: String::new(),
            events: Vec::new(),
            secret: None,
            retry_count: 3,
            timeout_secs: 30,
            enabled: true,
            description: None,
            headers: HashMap::new(),
        }
    }
}

/// Webhook payload sent to external systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayload {
    /// Event type.
    pub event: String,
    /// Unique event ID.
    pub event_id: String,
    /// Timestamp when the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Event-specific data.
    pub data: serde_json::Value,
    /// Server version.
    pub version: String,
}

impl WebhookPayload {
    /// Create a new webhook payload.
    pub fn new(event: &WebhookEvent, data: impl Serialize) -> Self {
        Self {
            event: event.as_str().to_string(),
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            data: serde_json::to_value(data).unwrap_or(serde_json::Value::Null),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Compute HMAC-SHA256 signature for the payload.
    pub fn compute_signature(&self, secret: &str) -> String {
        use hmac::{Hmac, Mac};
        type HmacSha256 = Hmac<Sha256>;

        let payload_json = serde_json::to_string(self).unwrap_or_default();
        let mut mac =
            HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can take any size key");
        mac.update(payload_json.as_bytes());
        let result = mac.finalize();
        format!("sha256={}", hex::encode(result.into_bytes()))
    }
}

/// Result of a webhook delivery attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDeliveryResult {
    /// Webhook ID.
    pub webhook_id: String,
    /// Event type.
    pub event: String,
    /// Whether delivery was successful.
    pub success: bool,
    /// HTTP status code (if applicable).
    pub status_code: Option<u16>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Number of attempts made.
    pub attempts: usize,
    /// Duration of the request in milliseconds.
    pub duration_ms: u64,
    /// Timestamp of the delivery attempt.
    pub timestamp: DateTime<Utc>,
}

/// Webhook delivery statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebhookStats {
    /// Total events dispatched.
    pub events_dispatched: u64,
    /// Successful deliveries.
    pub successful_deliveries: u64,
    /// Failed deliveries.
    pub failed_deliveries: u64,
    /// Total retries.
    pub total_retries: u64,
    /// Average delivery time in milliseconds.
    pub avg_delivery_time_ms: f64,
}

/// Webhook dispatcher that manages and sends webhooks.
pub struct WebhookDispatcher {
    /// HTTP client for making requests.
    client: Client,
    /// Configured webhooks.
    webhooks: RwLock<Vec<WebhookConfig>>,
    /// Delivery statistics.
    stats: RwLock<WebhookStats>,
    /// Recent delivery results (for debugging).
    recent_deliveries: RwLock<Vec<WebhookDeliveryResult>>,
    /// Maximum number of recent deliveries to keep.
    max_recent_deliveries: usize,
    /// Event sender for async processing.
    event_tx: Option<mpsc::UnboundedSender<(WebhookEvent, serde_json::Value)>>,
}

impl WebhookDispatcher {
    /// Create a new webhook dispatcher with the given configurations.
    pub fn new(configs: Vec<WebhookConfig>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent(format!("Alloy/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .unwrap_or_default();

        Self {
            client,
            webhooks: RwLock::new(configs),
            stats: RwLock::new(WebhookStats::default()),
            recent_deliveries: RwLock::new(Vec::new()),
            max_recent_deliveries: 100,
            event_tx: None,
        }
    }

    /// Create a webhook dispatcher with async processing.
    pub fn with_async_processing(
        configs: Vec<WebhookConfig>,
    ) -> (
        Self,
        mpsc::UnboundedReceiver<(WebhookEvent, serde_json::Value)>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut dispatcher = Self::new(configs);
        dispatcher.event_tx = Some(tx);
        (dispatcher, rx)
    }

    /// Add a new webhook configuration.
    pub async fn add_webhook(&self, config: WebhookConfig) -> String {
        let id = config.id.clone();
        let mut webhooks = self.webhooks.write().await;
        webhooks.push(config);
        info!("Added webhook: {}", id);
        id
    }

    /// Remove a webhook by ID.
    pub async fn remove_webhook(&self, id: &str) -> bool {
        let mut webhooks = self.webhooks.write().await;
        let len_before = webhooks.len();
        webhooks.retain(|w| w.id != id);
        let removed = webhooks.len() < len_before;
        if removed {
            info!("Removed webhook: {}", id);
        }
        removed
    }

    /// Update a webhook configuration.
    pub async fn update_webhook(&self, id: &str, config: WebhookConfig) -> bool {
        let mut webhooks = self.webhooks.write().await;
        for webhook in webhooks.iter_mut() {
            if webhook.id == id {
                *webhook = config;
                info!("Updated webhook: {}", id);
                return true;
            }
        }
        false
    }

    /// List all configured webhooks.
    pub async fn list_webhooks(&self) -> Vec<WebhookConfig> {
        let webhooks = self.webhooks.read().await;
        webhooks.clone()
    }

    /// Get a webhook by ID.
    pub async fn get_webhook(&self, id: &str) -> Option<WebhookConfig> {
        let webhooks = self.webhooks.read().await;
        webhooks.iter().find(|w| w.id == id).cloned()
    }

    /// Dispatch an event to all matching webhooks.
    pub async fn dispatch(&self, event: &WebhookEvent, data: impl Serialize) {
        let data_value = serde_json::to_value(&data).unwrap_or(serde_json::Value::Null);

        // If async processing is enabled, just queue the event
        if let Some(tx) = &self.event_tx {
            if tx.send((event.clone(), data_value.clone())).is_ok() {
                debug!("Queued webhook event: {}", event);
                return;
            }
        }

        // Otherwise, dispatch synchronously
        self.dispatch_sync(event, data_value).await;
    }

    /// Dispatch an event synchronously.
    async fn dispatch_sync(&self, event: &WebhookEvent, data: serde_json::Value) {
        let webhooks = self.webhooks.read().await;

        // Find matching webhooks
        let matching: Vec<_> = webhooks
            .iter()
            .filter(|w| {
                w.enabled
                    && w.events
                        .iter()
                        .any(|e| WebhookEvent::parse(e) == Some(event.clone()))
            })
            .cloned()
            .collect();

        drop(webhooks);

        if matching.is_empty() {
            debug!("No webhooks configured for event: {}", event);
            return;
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.events_dispatched += 1;
        }

        // Create payload
        let payload = WebhookPayload {
            event: event.as_str().to_string(),
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            data,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        // Dispatch to all matching webhooks
        for webhook in matching {
            self.deliver(&webhook, &payload).await;
        }
    }

    /// Deliver a payload to a specific webhook with retries.
    async fn deliver(&self, webhook: &WebhookConfig, payload: &WebhookPayload) {
        let start_time = std::time::Instant::now();
        let mut attempts = 0;
        let mut last_error: Option<String> = None;
        let mut last_status: Option<u16> = None;

        let payload_json = serde_json::to_string(payload).unwrap_or_default();

        for attempt in 0..=webhook.retry_count {
            attempts = attempt + 1;

            let mut request = self
                .client
                .post(&webhook.url)
                .header("Content-Type", "application/json")
                .header("X-Alloy-Event", &payload.event)
                .header("X-Alloy-Event-ID", &payload.event_id)
                .header("X-Alloy-Timestamp", payload.timestamp.to_rfc3339())
                .timeout(Duration::from_secs(webhook.timeout_secs));

            // Add signature if secret is configured
            if let Some(secret) = &webhook.secret {
                let signature = payload.compute_signature(secret);
                request = request.header("X-Alloy-Signature", signature);
            }

            // Add custom headers
            for (key, value) in &webhook.headers {
                request = request.header(key, value);
            }

            match request.body(payload_json.clone()).send().await {
                Ok(response) => {
                    let status = response.status();
                    last_status = Some(status.as_u16());

                    if status.is_success() {
                        debug!(
                            "Webhook delivery successful: {} -> {} (status: {})",
                            payload.event, webhook.url, status
                        );

                        let duration_ms = start_time.elapsed().as_millis() as u64;
                        self.record_delivery(WebhookDeliveryResult {
                            webhook_id: webhook.id.clone(),
                            event: payload.event.clone(),
                            success: true,
                            status_code: Some(status.as_u16()),
                            error: None,
                            attempts,
                            duration_ms,
                            timestamp: Utc::now(),
                        })
                        .await;

                        return;
                    } else {
                        let error_body = response.text().await.unwrap_or_default();
                        last_error = Some(format!("HTTP {}: {}", status, error_body));
                        warn!(
                            "Webhook delivery failed (attempt {}): {} -> {} - {}",
                            attempt + 1,
                            payload.event,
                            webhook.url,
                            last_error.as_ref().unwrap()
                        );
                    }
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    warn!(
                        "Webhook delivery error (attempt {}): {} -> {} - {}",
                        attempt + 1,
                        payload.event,
                        webhook.url,
                        e
                    );
                }
            }

            // Wait before retry (exponential backoff)
            if attempt < webhook.retry_count {
                let delay = Duration::from_millis(100 * 2u64.pow(attempt as u32));
                tokio::time::sleep(delay).await;

                let mut stats = self.stats.write().await;
                stats.total_retries += 1;
            }
        }

        // All retries exhausted
        error!(
            "Webhook delivery failed after {} attempts: {} -> {}",
            attempts, payload.event, webhook.url
        );

        let duration_ms = start_time.elapsed().as_millis() as u64;
        self.record_delivery(WebhookDeliveryResult {
            webhook_id: webhook.id.clone(),
            event: payload.event.clone(),
            success: false,
            status_code: last_status,
            error: last_error,
            attempts,
            duration_ms,
            timestamp: Utc::now(),
        })
        .await;
    }

    /// Record a delivery result.
    async fn record_delivery(&self, result: WebhookDeliveryResult) {
        // Update stats
        {
            let mut stats = self.stats.write().await;
            if result.success {
                stats.successful_deliveries += 1;
            } else {
                stats.failed_deliveries += 1;
            }

            // Update average delivery time
            let total = stats.successful_deliveries + stats.failed_deliveries;
            stats.avg_delivery_time_ms = (stats.avg_delivery_time_ms * (total - 1) as f64
                + result.duration_ms as f64)
                / total as f64;
        }

        // Store in recent deliveries
        {
            let mut recent = self.recent_deliveries.write().await;
            recent.push(result);
            while recent.len() > self.max_recent_deliveries {
                recent.remove(0);
            }
        }
    }

    /// Get delivery statistics.
    pub async fn stats(&self) -> WebhookStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Get recent delivery results.
    pub async fn recent_deliveries(&self) -> Vec<WebhookDeliveryResult> {
        let recent = self.recent_deliveries.read().await;
        recent.clone()
    }

    /// Test a webhook by sending a test event.
    pub async fn test_webhook(&self, webhook_id: &str) -> Option<WebhookDeliveryResult> {
        let webhook = self.get_webhook(webhook_id).await?;

        let test_data = serde_json::json!({
            "test": true,
            "message": "This is a test webhook delivery from Alloy"
        });

        let payload = WebhookPayload {
            event: "test".to_string(),
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            data: test_data,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        let start_time = std::time::Instant::now();
        let payload_json = serde_json::to_string(&payload).unwrap_or_default();

        let mut request = self
            .client
            .post(&webhook.url)
            .header("Content-Type", "application/json")
            .header("X-Alloy-Event", "test")
            .header("X-Alloy-Event-ID", &payload.event_id)
            .timeout(Duration::from_secs(webhook.timeout_secs));

        if let Some(secret) = &webhook.secret {
            let signature = payload.compute_signature(secret);
            request = request.header("X-Alloy-Signature", signature);
        }

        for (key, value) in &webhook.headers {
            request = request.header(key, value);
        }

        let result = match request.body(payload_json).send().await {
            Ok(response) => {
                let status = response.status();
                let error = if !status.is_success() {
                    Some(response.text().await.unwrap_or_default())
                } else {
                    None
                };

                WebhookDeliveryResult {
                    webhook_id: webhook_id.to_string(),
                    event: "test".to_string(),
                    success: status.is_success(),
                    status_code: Some(status.as_u16()),
                    error,
                    attempts: 1,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                    timestamp: Utc::now(),
                }
            }
            Err(e) => WebhookDeliveryResult {
                webhook_id: webhook_id.to_string(),
                event: "test".to_string(),
                success: false,
                status_code: None,
                error: Some(e.to_string()),
                attempts: 1,
                duration_ms: start_time.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            },
        };

        Some(result)
    }
}

/// Arc wrapper for sharing the dispatcher.
pub type SharedWebhookDispatcher = Arc<WebhookDispatcher>;

/// Create a shared webhook dispatcher from configuration.
pub fn create_dispatcher(configs: Vec<WebhookConfig>) -> SharedWebhookDispatcher {
    Arc::new(WebhookDispatcher::new(configs))
}

/// Background task for processing webhook events asynchronously.
pub async fn run_webhook_processor(
    dispatcher: SharedWebhookDispatcher,
    mut rx: mpsc::UnboundedReceiver<(WebhookEvent, serde_json::Value)>,
) {
    info!("Webhook processor started");

    while let Some((event, data)) = rx.recv().await {
        dispatcher.dispatch_sync(&event, data).await;
    }

    info!("Webhook processor stopped");
}

// ============================================================================
// Convenience functions for dispatching events
// ============================================================================

/// Event data for document indexed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentIndexedData {
    pub document_id: String,
    pub source_id: String,
    pub path: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub chunk_count: usize,
}

/// Event data for document deleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentDeletedData {
    pub document_id: String,
    pub source_id: String,
    pub path: String,
}

/// Event data for source added.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAddedData {
    pub source_id: String,
    pub source_type: String,
    pub path: String,
    pub document_count: usize,
}

/// Event data for source removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRemovedData {
    pub source_id: String,
    pub documents_removed: usize,
}

/// Event data for index error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexErrorData {
    pub source_id: String,
    pub path: String,
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_event_parsing() {
        assert_eq!(
            WebhookEvent::parse("document.indexed"),
            Some(WebhookEvent::DocumentIndexed)
        );
        assert_eq!(
            WebhookEvent::parse("source.added"),
            Some(WebhookEvent::SourceAdded)
        );
        assert_eq!(WebhookEvent::parse("invalid"), None);
    }

    #[test]
    fn test_payload_signature() {
        let payload = WebhookPayload::new(
            &WebhookEvent::DocumentIndexed,
            serde_json::json!({"test": true}),
        );
        let signature = payload.compute_signature("secret123");
        assert!(signature.starts_with("sha256="));
    }

    #[tokio::test]
    async fn test_dispatcher_creation() {
        let config = WebhookConfig {
            url: "https://example.com/webhook".to_string(),
            events: vec!["document.indexed".to_string()],
            ..Default::default()
        };

        let dispatcher = WebhookDispatcher::new(vec![config]);
        let webhooks = dispatcher.list_webhooks().await;
        assert_eq!(webhooks.len(), 1);
    }

    #[tokio::test]
    async fn test_add_remove_webhook() {
        let dispatcher = WebhookDispatcher::new(vec![]);

        let config = WebhookConfig {
            id: "test-webhook".to_string(),
            url: "https://example.com/webhook".to_string(),
            events: vec!["document.indexed".to_string()],
            ..Default::default()
        };

        let id = dispatcher.add_webhook(config).await;
        assert_eq!(id, "test-webhook");

        let webhooks = dispatcher.list_webhooks().await;
        assert_eq!(webhooks.len(), 1);

        let removed = dispatcher.remove_webhook("test-webhook").await;
        assert!(removed);

        let webhooks = dispatcher.list_webhooks().await;
        assert!(webhooks.is_empty());
    }
}

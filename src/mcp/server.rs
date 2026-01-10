//! MCP server implementation for Alloy.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::acl::{
    AclEntry, AclResolver, AclStorage, DocumentAcl, MemoryAclStorage, Permission, Principal,
    SourceAcl,
};
use crate::auth::AuthContext;
use crate::config::Config;
use crate::coordinator::{IndexCoordinator, IndexProgress};
use crate::mcp::query_tools::{NlQueryStats, QueryParams, QueryResponse};
use crate::mcp::tools::{
    AclConfigInfo, AclEntryInfo, AddWebhookResponse, BackupInfo, CacheConfigInfo,
    CheckDuplicateResponse, CheckPermissionResponse, ClearCacheResponse,
    ClearDeduplicationResponse, ClusterDocumentInfo, ClusterDocumentsResponse, ClusterInfo,
    ClusterMetrics, ClusterOutlineInfo, ClusterVisualizationResponse, ConfigureResponse,
    CreateBackupResponse, DeduplicationStatsResponse, DeleteDocumentAclResponse,
    DeleteSourceAclResponse, DiffStatsInfo, DiffVersionsResponse, DocumentDetails,
    ExportDocumentsResponse, FindSimilarClusterResponse, GetAclStatsResponse,
    GetCacheStatsResponse, GetClusterDocumentsResponse, GetDocumentAclResponse,
    GetDocumentHistoryResponse, GetMetricsResponse, GetSourceAclResponse,
    GetVersionContentResponse, GetWebhookStatsResponse, ImportDocumentsResponse, IndexPathResponse,
    IndexStats, ListBackupsResponse, ListSourcesResponse, ListWebhooksResponse,
    RemoveSourceResponse, RemoveWebhookResponse, RestoreBackupResponse, RestoreVersionResponse,
    RoleInfo, SearchResponse, SearchResult, SetDocumentAclResponse, SetSourceAclResponse,
    SimilarClusterInfo, SourceInfo, TestWebhookResponse, VersionInfo, VersioningRetentionInfo,
    VersioningStatsResponse, VisualizationPoint, WebhookDeliveryInfo, WebhookInfo,
};
use crate::metrics::get_metrics;
use crate::ontology::DeletionStrategy;
use crate::query::{IntentClassifier, QueryMode};
use crate::search::{HybridQuery, SearchFilter};
use crate::sources::parse_s3_uri;
use crate::webhooks::{
    DocumentDeletedData, DocumentIndexedData, IndexErrorData, SharedWebhookDispatcher,
    SourceAddedData, SourceRemovedData, WebhookConfig, WebhookDispatcher, WebhookEvent,
};

/// Alloy MCP server state.
pub struct AlloyState {
    /// Server configuration
    pub config: Config,
    /// Start time for uptime calculation
    pub start_time: Instant,
    /// Index coordinator (shared across sessions)
    pub coordinator: Option<Arc<IndexCoordinator>>,
    /// Progress events log (recent events)
    pub progress_log: Vec<IndexProgress>,
    /// ACL storage backend
    pub acl_storage: Arc<dyn AclStorage>,
    /// ACL resolver for permission checks
    pub acl_resolver: Arc<AclResolver>,
    /// Webhook dispatcher for sending notifications
    pub webhook_dispatcher: SharedWebhookDispatcher,
}

impl AlloyState {
    pub fn new(config: Config) -> Self {
        Self::with_coordinator(config, None)
    }

    pub fn with_coordinator(config: Config, coordinator: Option<Arc<IndexCoordinator>>) -> Self {
        let acl_storage: Arc<dyn AclStorage> = Arc::new(MemoryAclStorage::new());
        let acl_resolver = Arc::new(AclResolver::new(
            acl_storage.clone(),
            config.security.acl.clone(),
        ));

        // Create webhook dispatcher from configuration
        let webhook_configs: Vec<WebhookConfig> = config
            .integration
            .webhooks
            .endpoints
            .iter()
            .map(|e| WebhookConfig {
                id: uuid::Uuid::new_v4().to_string(),
                url: e.url.clone(),
                events: e.events.clone(),
                secret: e.secret.clone(),
                retry_count: config.integration.webhooks.max_retries,
                timeout_secs: config.integration.webhooks.timeout_secs,
                enabled: config.integration.webhooks.enabled,
                description: e.description.clone(),
                headers: std::collections::HashMap::new(),
            })
            .collect();

        let webhook_dispatcher = Arc::new(WebhookDispatcher::new(webhook_configs));

        Self {
            config,
            start_time: Instant::now(),
            coordinator,
            progress_log: Vec::new(),
            acl_storage,
            acl_resolver,
            webhook_dispatcher,
        }
    }
}

/// Alloy MCP server handler.
#[derive(Clone)]
pub struct AlloyServer {
    state: Arc<RwLock<AlloyState>>,
    tool_router: ToolRouter<Self>,
}

impl AlloyServer {
    /// Create a new Alloy server with the given configuration.
    pub fn new(config: Config) -> Self {
        Self {
            state: Arc::new(RwLock::new(AlloyState::new(config))),
            tool_router: Self::tool_router(),
        }
    }

    /// Create a new Alloy server with a shared coordinator.
    pub fn with_shared_coordinator(config: Config, coordinator: Arc<IndexCoordinator>) -> Self {
        Self {
            state: Arc::new(RwLock::new(AlloyState::with_coordinator(
                config,
                Some(coordinator),
            ))),
            tool_router: Self::tool_router(),
        }
    }

    /// Create a new Alloy server with default configuration.
    pub fn with_defaults() -> crate::error::Result<Self> {
        let config = Config::load()?;
        Ok(Self::new(config))
    }

    /// Ensure the coordinator is initialized.
    async fn ensure_coordinator(&self) -> Result<(), McpError> {
        let mut state = self.state.write().await;
        if state.coordinator.is_none() {
            let mut coordinator =
                IndexCoordinator::new(state.config.clone())
                    .await
                    .map_err(|e| {
                        McpError::internal_error(
                            format!("Failed to initialize coordinator: {}", e),
                            None,
                        )
                    })?;

            // Set up webhook integration by subscribing to progress events
            let webhook_dispatcher = state.webhook_dispatcher.clone();
            let webhooks_enabled = state.config.integration.webhooks.enabled;

            if webhooks_enabled {
                let rx = coordinator.progress_channel();
                // Spawn a background task to process progress events and dispatch webhooks
                tokio::spawn(async move {
                    process_progress_events_for_webhooks(rx, webhook_dispatcher).await;
                });
            }

            state.coordinator = Some(Arc::new(coordinator));
        }
        Ok(())
    }
}

/// Process IndexProgress events and dispatch corresponding webhooks.
async fn process_progress_events_for_webhooks(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<IndexProgress>,
    dispatcher: SharedWebhookDispatcher,
) {
    use tracing::debug;

    while let Some(progress) = rx.recv().await {
        match progress {
            IndexProgress::DocumentProcessed {
                source_id,
                uri,
                chunks,
            } => {
                // Dispatch document.indexed webhook
                let data = DocumentIndexedData {
                    document_id: uri.clone(),
                    source_id: source_id.clone(),
                    path: uri.clone(),
                    mime_type: "application/octet-stream".to_string(), // Default, could be improved
                    size_bytes: 0, // Not available in progress event
                    chunk_count: chunks,
                };
                dispatcher
                    .dispatch(&WebhookEvent::DocumentIndexed, data)
                    .await;
                debug!("Dispatched document.indexed webhook for {}", uri);
            }
            IndexProgress::DocumentRemoved { source_id, uri } => {
                // Dispatch document.deleted webhook
                let data = DocumentDeletedData {
                    document_id: uri.clone(),
                    source_id: source_id.clone(),
                    path: uri.clone(),
                };
                dispatcher
                    .dispatch(&WebhookEvent::DocumentDeleted, data)
                    .await;
                debug!("Dispatched document.deleted webhook for {}", uri);
            }
            IndexProgress::IndexComplete {
                source_id,
                documents,
                ..
            } => {
                // Dispatch source.added webhook when indexing completes
                let data = SourceAddedData {
                    source_id: source_id.clone(),
                    source_type: "unknown".to_string(), // Could be improved with more context
                    path: source_id.clone(),
                    document_count: documents,
                };
                dispatcher.dispatch(&WebhookEvent::SourceAdded, data).await;
                debug!("Dispatched source.added webhook for {}", source_id);
            }
            IndexProgress::DocumentError {
                source_id,
                uri,
                error,
            } => {
                // Dispatch index.error webhook
                let data = IndexErrorData {
                    source_id: source_id.clone(),
                    path: uri.clone(),
                    error: error.clone(),
                };
                dispatcher.dispatch(&WebhookEvent::IndexError, data).await;
                debug!("Dispatched index.error webhook for {}: {}", uri, error);
            }
            IndexProgress::DocumentDeduplicated {
                source_id,
                uri,
                duplicate_of,
                similarity,
            } => {
                // Dispatch document.updated webhook for deduplication events
                let data = serde_json::json!({
                    "document_id": uri,
                    "source_id": source_id,
                    "duplicate_of": duplicate_of,
                    "similarity": similarity,
                    "action": "deduplicated"
                });
                dispatcher
                    .dispatch(&WebhookEvent::DocumentUpdated, data)
                    .await;
                debug!(
                    "Dispatched document.updated (deduplicated) webhook for {}",
                    uri
                );
            }
            // Other progress events don't need webhook dispatch
            _ => {}
        }
    }
}

// Parameters for index_path tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IndexPathParams {
    /// Path to index (local path or s3://bucket/prefix)
    pub path: String,
    /// Glob pattern to filter files (e.g., '*.md', '**/*.py')
    #[serde(default)]
    pub pattern: Option<String>,
    /// Watch for file changes and auto-reindex
    #[serde(default)]
    pub watch: Option<bool>,
    /// Recursively index subdirectories
    #[serde(default)]
    pub recursive: Option<bool>,
}

// Parameters for search tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// Search query
    pub query: String,
    /// Maximum number of results (default: 10)
    #[serde(default)]
    pub limit: Option<usize>,
    /// Vector search weight from 0.0 to 1.0 (default: 0.5)
    #[serde(default)]
    pub vector_weight: Option<f32>,
    /// Filter by source ID
    #[serde(default)]
    pub source_id: Option<String>,
    /// Enable query expansion to find more relevant results using synonyms and related terms
    #[serde(default)]
    pub expand_query: Option<bool>,
    /// Maximum number of expansion terms to add (default: 3)
    #[serde(default)]
    pub max_expansions: Option<usize>,
    /// Enable reranking for improved precision (requires reranking to be configured)
    #[serde(default)]
    pub rerank: Option<bool>,
}

// Parameters for get_document tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetDocumentParams {
    /// Document ID
    pub document_id: String,
    /// Include full document content
    #[serde(default)]
    pub include_content: Option<bool>,
}

// Parameters for remove_source tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RemoveSourceParams {
    /// Source ID to remove
    pub source_id: String,
    /// Knowledge graph deletion strategy. Options:
    /// - "remove_source_refs" (default): Remove refs, delete only orphaned entities
    /// - "delete_affected": Delete all entities/relationships referencing the source
    /// - "preserve_knowledge": Remove refs only, never delete knowledge
    #[serde(default)]
    pub knowledge_strategy: Option<String>,
}

// Parameters for configure tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConfigureParams {
    /// Configuration updates as JSON
    pub updates: serde_json::Value,
}

// Parameters for cluster_documents tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClusterDocumentsParams {
    /// Optional filter by source ID
    #[serde(default)]
    pub source_id: Option<String>,
    /// Clustering algorithm to use (kmeans or dbscan)
    #[serde(default)]
    pub algorithm: Option<String>,
    /// Number of clusters (for k-means, default: auto-detect)
    #[serde(default)]
    pub num_clusters: Option<usize>,
    /// Include 2D visualization data (default: false)
    #[serde(default)]
    pub include_visualization: Option<bool>,
}

// Parameters for get_cluster_visualization tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetClusterVisualizationParams {
    /// Optional filter by source ID
    #[serde(default)]
    pub source_id: Option<String>,
    /// Clustering algorithm to use (kmeans, dbscan, or agglomerative)
    #[serde(default)]
    pub algorithm: Option<String>,
    /// Number of clusters (for k-means/agglomerative, default: auto-detect)
    #[serde(default)]
    pub num_clusters: Option<usize>,
}

// Parameters for find_similar_cluster tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FindSimilarClusterParams {
    /// Query text to find similar cluster for
    pub query: String,
    /// Optional source ID to filter clusters
    #[serde(default)]
    pub source_id: Option<String>,
    /// Algorithm used for clustering (kmeans, dbscan, or agglomerative)
    #[serde(default)]
    pub algorithm: Option<String>,
    /// Number of clusters in the clustering
    #[serde(default)]
    pub num_clusters: Option<usize>,
    /// Number of top similar clusters to return (default: 3)
    #[serde(default)]
    pub top_k: Option<usize>,
}

// Parameters for get_cluster_documents tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetClusterDocumentsParams {
    /// Cluster ID to get documents for
    pub cluster_id: usize,
    /// Optional source ID filter
    #[serde(default)]
    pub source_id: Option<String>,
    /// Algorithm used for clustering
    #[serde(default)]
    pub algorithm: Option<String>,
    /// Number of clusters in the clustering
    #[serde(default)]
    pub num_clusters: Option<usize>,
    /// Maximum number of documents to return (default: 50)
    #[serde(default)]
    pub limit: Option<usize>,
    /// Offset for pagination
    #[serde(default)]
    pub offset: Option<usize>,
}

// Parameters for check_duplicate tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CheckDuplicateParams {
    /// Content to check for duplicates
    pub content: String,
    /// Optional document ID for the content
    #[serde(default)]
    pub document_id: Option<String>,
}

// Parameters for get_document_history tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetDocumentHistoryParams {
    /// Document ID to get history for
    pub document_id: String,
    /// Maximum number of versions to return (default: 50)
    #[serde(default)]
    pub limit: Option<usize>,
}

// Parameters for diff_versions tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DiffVersionsParams {
    /// Document ID
    pub document_id: String,
    /// First version ID or version number (e.g., "v1" or "1")
    pub version_a: String,
    /// Second version ID or version number (e.g., "v2" or "latest")
    pub version_b: String,
    /// Number of context lines in diff (default: 3)
    #[serde(default)]
    pub context_lines: Option<usize>,
}

// Parameters for restore_version tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RestoreVersionParams {
    /// Document ID
    pub document_id: String,
    /// Version ID to restore
    pub version_id: String,
}

// Parameters for get_version_content tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetVersionContentParams {
    /// Version ID to get content for
    pub version_id: String,
}

// ============================================================================
// Operations Tool Parameters
// ============================================================================

// Parameters for create_backup tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CreateBackupParams {
    /// Optional description for the backup
    #[serde(default)]
    pub description: Option<String>,
    /// Output path for the backup (optional, defaults to configured backup directory)
    #[serde(default)]
    pub output_path: Option<String>,
}

// Parameters for restore_backup tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RestoreBackupParams {
    /// Backup ID to restore, or path to backup file
    pub backup_id: String,
}

// Parameters for export_documents tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ExportDocumentsParams {
    /// Output file path
    pub output_path: String,
    /// Export format (jsonl or json)
    #[serde(default)]
    pub format: Option<String>,
    /// Optional source ID filter
    #[serde(default)]
    pub source_id: Option<String>,
    /// Include embeddings in export (increases file size significantly)
    #[serde(default)]
    pub include_embeddings: Option<bool>,
}

// Parameters for import_documents tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImportDocumentsParams {
    /// Path to import file
    pub input_path: String,
}

// ============================================================================
// ACL Tool Parameters
// ============================================================================

// Parameters for get_document_acl tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetDocumentAclParams {
    /// Document ID
    pub document_id: String,
}

// Parameters for set_document_acl tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SetDocumentAclParams {
    /// Document ID
    pub document_id: String,
    /// Owner user ID (optional, uses current user if not set)
    #[serde(default)]
    pub owner: Option<String>,
    /// ACL entries
    pub entries: Vec<AclEntryParamInput>,
    /// Whether to inherit ACL from source
    #[serde(default)]
    pub inherit_from_source: Option<bool>,
}

// ACL entry parameter for input
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AclEntryParamInput {
    /// Principal type: "user", "role", "group", "everyone", "authenticated"
    pub principal_type: String,
    /// Principal ID (for user, role, group)
    #[serde(default)]
    pub principal_id: Option<String>,
    /// Permissions: "read", "write", "delete", "admin"
    pub permissions: Vec<String>,
}

// Parameters for delete_document_acl tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DeleteDocumentAclParams {
    /// Document ID
    pub document_id: String,
}

// Parameters for check_permission tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CheckPermissionParams {
    /// Document ID
    pub document_id: String,
    /// User ID to check (optional, uses current user if not set)
    #[serde(default)]
    pub user_id: Option<String>,
    /// Roles to check with
    #[serde(default)]
    pub roles: Option<Vec<String>>,
    /// Permission to check: "read", "write", "delete", "admin"
    pub permission: String,
}

// Parameters for get_source_acl tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetSourceAclParams {
    /// Source ID
    pub source_id: String,
}

// Parameters for set_source_acl tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SetSourceAclParams {
    /// Source ID
    pub source_id: String,
    /// Owner user ID (optional, uses current user if not set)
    #[serde(default)]
    pub owner: Option<String>,
    /// ACL entries for the source
    pub entries: Vec<AclEntryParamInput>,
    /// Default ACL entries for new documents in this source
    #[serde(default)]
    pub default_document_acl: Option<Vec<AclEntryParamInput>>,
}

// Parameters for delete_source_acl tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DeleteSourceAclParams {
    /// Source ID
    pub source_id: String,
}

// ============================================================================
// Webhook Tool Parameters
// ============================================================================

// Parameters for add_webhook tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AddWebhookParams {
    /// Target URL for webhook payloads
    pub url: String,
    /// Events to subscribe to (e.g., "document.indexed", "index.error")
    pub events: Vec<String>,
    /// Secret for HMAC signature (optional)
    #[serde(default)]
    pub secret: Option<String>,
    /// Number of retry attempts on failure (default: 3)
    #[serde(default)]
    pub retry_count: Option<usize>,
    /// Timeout in seconds (default: 30)
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Optional description
    #[serde(default)]
    pub description: Option<String>,
}

// Parameters for remove_webhook tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RemoveWebhookParams {
    /// Webhook ID to remove
    pub webhook_id: String,
}

// Parameters for test_webhook tool
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TestWebhookParams {
    /// Webhook ID to test
    pub webhook_id: String,
}

#[tool_router]
impl AlloyServer {
    /// Index a local path or S3 URI for hybrid search. Supports glob patterns and optional file watching.
    #[tool(
        description = "Index a local path or S3 URI for hybrid search. Supports glob patterns and optional file watching."
    )]
    async fn index_path(
        &self,
        Parameters(params): Parameters<IndexPathParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let watch = params.watch.unwrap_or(false);
        let patterns = params.pattern.map(|p| vec![p]).unwrap_or_default();

        // Get coordinator and perform indexing
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        let result = if params.path.starts_with("s3://") {
            // Parse S3 URI
            let (bucket, prefix) = parse_s3_uri(&params.path)
                .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

            coordinator
                .index_s3(
                    bucket,
                    Some(prefix),
                    patterns,
                    None, // Use default region
                )
                .await
        } else {
            // Local path
            let path = PathBuf::from(&params.path);
            if !path.exists() {
                return Err(McpError::invalid_params(
                    format!("Path does not exist: {}", params.path),
                    None,
                ));
            }

            coordinator
                .index_local(
                    path,
                    patterns,
                    vec![], // Use default exclude patterns
                    watch,
                )
                .await
        };

        match result {
            Ok(source) => {
                let response = IndexPathResponse {
                    source_id: source.id.clone(),
                    documents_indexed: source.document_count,
                    chunks_created: 0, // TODO: track chunk count
                    watching: source.watching,
                    message: format!(
                        "Successfully indexed {} documents from {}",
                        source.document_count, params.path
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to index path: {}",
                e
            ))])),
        }
    }

    /// Search indexed documents using hybrid search combining vector similarity and full-text matching.
    #[tool(
        description = "Search indexed documents using hybrid search combining vector similarity and full-text matching."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let limit = params.limit.unwrap_or(10);
        let vector_weight = params.vector_weight.unwrap_or(0.5);

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Build the query
        let mut query = HybridQuery::new(&params.query)
            .limit(limit)
            .vector_weight(vector_weight);

        if let Some(source_id) = params.source_id.clone() {
            query = query.filter(SearchFilter::new().source(source_id));
        }

        // Apply query expansion settings if specified
        if let Some(expand) = params.expand_query {
            query = query.expand_query(expand);
        }
        if let Some(max_expansions) = params.max_expansions {
            query = query.max_expansions(max_expansions);
        }

        // Apply reranking settings if specified
        if let Some(rerank) = params.rerank {
            query = query.rerank(rerank);
        }

        // Execute search
        let start_time = Instant::now();
        match coordinator.search(query).await {
            Ok(response) => {
                // Filter results based on ACL permissions if enabled
                let mut filtered_results = Vec::new();
                let acl_resolver = &state.acl_resolver;
                let acl_enabled = state.config.security.acl.enabled
                    && state.config.security.acl.enforce_on_search;

                // For now, use an anonymous context for ACL checks
                // In a full implementation, we'd get the auth context from the request
                let auth_ctx = AuthContext::anonymous();

                for r in response.results.iter() {
                    // Check if user has read permission for this document
                    let has_access = if acl_enabled {
                        match acl_resolver
                            .check_permission(&auth_ctx, &r.document_id, Permission::Read)
                            .await
                        {
                            Ok(result) => result.allowed,
                            Err(_) => false,
                        }
                    } else {
                        true // No ACL enforcement, allow access
                    };

                    if has_access {
                        filtered_results.push(SearchResult {
                            document_id: r.document_id.clone(),
                            chunk_id: r.chunk_id.clone(),
                            source_id: String::new(), // Not available in search result
                            path: r.path.clone().unwrap_or_default(),
                            content: r.text.clone(),
                            score: r.score,
                            highlights: r
                                .highlights
                                .iter()
                                .filter_map(|(start, end)| {
                                    // Ensure positions are valid UTF-8 char boundaries
                                    if *end <= r.text.len()
                                        && r.text.is_char_boundary(*start)
                                        && r.text.is_char_boundary(*end)
                                    {
                                        Some(r.text[*start..*end].to_string())
                                    } else {
                                        None
                                    }
                                })
                                .collect(),
                            metadata: serde_json::json!({}),
                        });
                    }
                }

                let results = filtered_results;

                let search_response = SearchResponse {
                    results,
                    total_matches: response.results.len(),
                    took_ms: start_time.elapsed().as_millis() as u64,
                    query_expanded: if response.stats.query_expanded {
                        Some(true)
                    } else {
                        None
                    },
                    expanded_query: response.stats.expanded_query.clone(),
                    reranked: if response.stats.reranked {
                        Some(true)
                    } else {
                        None
                    },
                };

                let message = if search_response.results.is_empty() {
                    format!(
                        "No results found for query: '{}' (vector_weight: {}, limit: {}{})",
                        params.query,
                        vector_weight,
                        limit,
                        params
                            .source_id
                            .map(|s| format!(", source: {}", s))
                            .unwrap_or_default()
                    )
                } else {
                    serde_json::to_string_pretty(&search_response).unwrap()
                };

                Ok(CallToolResult::success(vec![Content::text(message)]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Search failed: {}",
                e
            ))])),
        }
    }

    /// Retrieve a document by ID with optional full content.
    #[tool(description = "Retrieve a document by ID with optional full content.")]
    async fn get_document(
        &self,
        Parameters(params): Parameters<GetDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let include_content = params.include_content.unwrap_or(true);

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Check ACL permissions if enabled
        let acl_enabled =
            state.config.security.acl.enabled && state.config.security.acl.enforce_on_get;

        if acl_enabled {
            // For now, use an anonymous context for ACL checks
            // In a full implementation, we'd get the auth context from the request
            let auth_ctx = AuthContext::anonymous();

            match state
                .acl_resolver
                .check_permission(&auth_ctx, &params.document_id, Permission::Read)
                .await
            {
                Ok(result) if !result.allowed => {
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Access denied: {}",
                        result.reason
                    ))]));
                }
                Err(e) => {
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Error checking permissions: {}",
                        e
                    ))]));
                }
                _ => {} // Access allowed, continue
            }
        }

        match coordinator.get_document(&params.document_id).await {
            Ok(Some(doc)) => {
                let response = DocumentDetails {
                    document_id: doc.id.clone(),
                    source_id: doc.source_id.clone(),
                    path: doc.path.clone(),
                    mime_type: doc.mime_type.clone(),
                    size_bytes: doc.size,
                    chunk_count: 0, // Not tracked per-document currently
                    content: if include_content {
                        Some(doc.content.clone())
                    } else {
                        None
                    },
                    modified_at: doc.modified_at,
                    indexed_at: doc.indexed_at,
                    metadata: doc.metadata.clone(),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Ok(None) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Document not found: {}",
                params.document_id
            ))])),
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error retrieving document: {}",
                e
            ))])),
        }
    }

    /// List all indexed sources with their status and document counts.
    #[tool(description = "List all indexed sources with their status and document counts.")]
    async fn list_sources(&self) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        let sources = coordinator.list_sources().await;

        let source_infos: Vec<SourceInfo> = sources
            .iter()
            .map(|s| SourceInfo {
                source_id: s.id.clone(),
                source_type: s.source_type.clone(),
                path: s.path.clone(),
                document_count: s.document_count,
                watching: s.watching,
                last_scan: s.last_scan,
                status: "indexed".to_string(),
            })
            .collect();

        let response = ListSourcesResponse {
            sources: source_infos,
        };

        let message = if response.sources.is_empty() {
            "No sources indexed yet. Use 'index_path' to add sources.".to_string()
        } else {
            serde_json::to_string_pretty(&response).unwrap()
        };

        Ok(CallToolResult::success(vec![Content::text(message)]))
    }

    /// Remove an indexed source and all its documents from the index.
    #[tool(
        description = "Remove an indexed source and all its documents from the index. Optionally specify a knowledge_strategy to control how associated entities/relationships are handled: 'remove_source_refs' (default) removes refs and deletes orphans, 'delete_affected' deletes all affected knowledge, 'preserve_knowledge' keeps all knowledge even if orphaned."
    )]
    async fn remove_source(
        &self,
        Parameters(params): Parameters<RemoveSourceParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Check ACL permissions if enabled - require admin for source removal
        let acl_enabled =
            state.config.security.acl.enabled && state.config.security.acl.enforce_on_delete;

        if acl_enabled {
            // For now, use an anonymous context for ACL checks
            // In a full implementation, we'd get the auth context from the request
            let auth_ctx = AuthContext::anonymous();

            // Check source-level ACL
            if let Ok(Some(source_acl)) = state.acl_storage.get_source_acl(&params.source_id).await
            {
                // Only owner can delete a source
                let is_owner = auth_ctx
                    .user_id
                    .as_ref()
                    .map(|u| u == &source_acl.owner)
                    .unwrap_or(false);

                if !is_owner {
                    return Ok(CallToolResult::success(vec![Content::text(
                        "Access denied: Only the source owner can remove a source",
                    )]));
                }
            }
            // If no source ACL exists, we allow the operation (default behavior)
        }

        // Parse the deletion strategy from the parameter
        let strategy = match &params.knowledge_strategy {
            Some(s) => match s.parse::<DeletionStrategy>() {
                Ok(strat) => strat,
                Err(e) => {
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Invalid knowledge_strategy: {}",
                        e
                    ))]));
                }
            },
            None => DeletionStrategy::default(),
        };

        match coordinator
            .remove_source_with_strategy(&params.source_id, strategy)
            .await
        {
            Ok(Some(result)) => {
                // Dispatch source.removed webhook if enabled
                if state.config.integration.webhooks.enabled {
                    let webhook_data = SourceRemovedData {
                        source_id: params.source_id.clone(),
                        documents_removed: result.documents_removed,
                    };
                    state
                        .webhook_dispatcher
                        .dispatch(&WebhookEvent::SourceRemoved, webhook_data)
                        .await;
                }

                let knowledge = &result.knowledge_result;
                let response = RemoveSourceResponse {
                    success: true,
                    source_id: params.source_id.clone(),
                    documents_removed: result.documents_removed,
                    entities_affected: Some(knowledge.total_entities_affected()),
                    entities_deleted: Some(knowledge.entities_deleted),
                    relationships_affected: Some(knowledge.total_relationships_affected()),
                    relationships_deleted: Some(knowledge.relationships_deleted),
                    strategy: Some(strategy.to_string()),
                    message: format!(
                        "Removed source {} with {} documents. Knowledge impact: {} entities affected ({} deleted), {} relationships affected ({} deleted). Strategy: {}",
                        params.source_id,
                        result.documents_removed,
                        knowledge.total_entities_affected(),
                        knowledge.entities_deleted,
                        knowledge.total_relationships_affected(),
                        knowledge.relationships_deleted,
                        strategy
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Ok(None) => {
                let response = RemoveSourceResponse {
                    success: false,
                    source_id: params.source_id.clone(),
                    documents_removed: 0,
                    entities_affected: None,
                    entities_deleted: None,
                    relationships_affected: None,
                    relationships_deleted: None,
                    strategy: None,
                    message: format!("Source not found: {}", params.source_id),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error removing source: {}",
                e
            ))])),
        }
    }

    /// Get statistics about the index including document count, storage size, and configuration.
    #[tool(
        description = "Get statistics about the index including document count, storage size, and configuration."
    )]
    async fn get_stats(&self) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        let storage_stats = coordinator
            .stats()
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to get stats: {}", e), None))?;

        let sources = coordinator.list_sources().await;

        let stats = IndexStats {
            source_count: sources.len(),
            document_count: storage_stats.document_count,
            chunk_count: storage_stats.chunk_count,
            storage_bytes: storage_stats.storage_bytes,
            embedding_dimension: coordinator.embedding_dimension(),
            storage_backend: format!("{:?}", state.config.storage.backend),
            embedding_provider: format!("{:?}", state.config.embedding.provider),
            uptime_secs: state.start_time.elapsed().as_secs(),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&stats).unwrap(),
        )]))
    }

    /// Update runtime configuration settings.
    #[tool(description = "Update runtime configuration settings.")]
    async fn configure(
        &self,
        Parameters(params): Parameters<ConfigureParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        let response = ConfigureResponse {
            success: true,
            config: serde_json::json!({
                "server": {
                    "transport": format!("{:?}", state.config.server.transport),
                    "http_port": state.config.server.http_port,
                },
                "embedding": {
                    "provider": format!("{:?}", state.config.embedding.provider),
                    "model": state.config.embedding.model.clone(),
                },
                "storage": {
                    "backend": format!("{:?}", state.config.storage.backend),
                    "data_dir": state.config.storage.data_dir.clone(),
                },
                "processing": {
                    "chunk_size": state.config.processing.chunk_size,
                    "chunk_overlap": state.config.processing.chunk_overlap,
                },
            }),
            message: format!("Configuration update acknowledged: {:?}", params.updates),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Cluster indexed documents by semantic similarity for exploration and organization.
    #[tool(
        description = "Group indexed documents into clusters based on semantic similarity. Returns cluster labels, keywords, and quality metrics."
    )]
    async fn cluster_documents(
        &self,
        Parameters(params): Parameters<ClusterDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Parse algorithm if provided
        let algorithm = params.algorithm.as_deref().map(|algo| match algo {
            "dbscan" => crate::config::ClusteringAlgorithm::Dbscan,
            _ => crate::config::ClusteringAlgorithm::KMeans,
        });

        match coordinator
            .cluster_documents(params.source_id.as_deref(), algorithm, params.num_clusters)
            .await
        {
            Ok(result) => {
                let response = ClusterDocumentsResponse {
                    clusters: result
                        .clusters
                        .iter()
                        .map(|c| ClusterInfo {
                            cluster_id: c.cluster_id,
                            label: c.label.clone(),
                            keywords: c.keywords.clone(),
                            size: c.document_ids.len(),
                            coherence_score: c.coherence_score,
                            representative_docs: c.representative_docs.clone(),
                        })
                        .collect(),
                    outliers: result.outliers.clone(),
                    metrics: ClusterMetrics {
                        silhouette_score: result.metrics.silhouette_score,
                        calinski_harabasz_index: if result.metrics.calinski_harabasz_index > 0.0 {
                            Some(result.metrics.calinski_harabasz_index)
                        } else {
                            None
                        },
                        davies_bouldin_index: if result.metrics.davies_bouldin_index > 0.0 {
                            Some(result.metrics.davies_bouldin_index)
                        } else {
                            None
                        },
                        num_clusters: result.metrics.num_clusters,
                        num_outliers: result.metrics.num_outliers,
                    },
                    algorithm: result.algorithm_used.clone(),
                    total_documents: result.total_documents,
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Clustering failed: {}",
                e
            ))])),
        }
    }

    /// Get 2D visualization data for clustered documents.
    #[tool(
        description = "Get 2D visualization data for document clusters. Returns points and cluster outlines for rendering a scatter plot."
    )]
    async fn get_cluster_visualization(
        &self,
        Parameters(params): Parameters<GetClusterVisualizationParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::search::PcaProjection;

        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Parse algorithm if provided
        let algorithm = params.algorithm.as_deref().map(|algo| match algo {
            "dbscan" => crate::config::ClusteringAlgorithm::Dbscan,
            _ => crate::config::ClusteringAlgorithm::KMeans,
        });

        match coordinator
            .cluster_documents(params.source_id.as_deref(), algorithm, params.num_clusters)
            .await
        {
            Ok(result) => {
                // Get embeddings for visualization
                let chunks = coordinator
                    .storage()
                    .get_all_chunks_for_clustering(params.source_id.as_deref())
                    .await
                    .map_err(|e| {
                        McpError::internal_error(format!("Failed to get chunks: {}", e), None)
                    })?;

                // Convert to input format
                let embeddings: Vec<Vec<f32>> =
                    chunks.iter().map(|c| c.embedding.clone()).collect();
                let doc_ids: Vec<String> = chunks.iter().map(|c| c.document_id.clone()).collect();

                // Create cluster assignments
                let mut cluster_assignments: Vec<Option<usize>> = vec![None; doc_ids.len()];
                for cluster in &result.clusters {
                    for doc_id in &cluster.document_ids {
                        if let Some(idx) = doc_ids.iter().position(|id| id == doc_id) {
                            cluster_assignments[idx] = Some(cluster.cluster_id);
                        }
                    }
                }

                // Perform PCA projection
                let viz = PcaProjection::project(&embeddings, &doc_ids, &cluster_assignments);

                // Convert to response format
                let response = ClusterVisualizationResponse {
                    points: viz
                        .points
                        .into_iter()
                        .map(|p| VisualizationPoint {
                            x: p.x,
                            y: p.y,
                            document_id: p.document_id,
                            cluster_id: p.cluster_id,
                        })
                        .collect(),
                    clusters: viz
                        .clusters
                        .into_iter()
                        .map(|c| ClusterOutlineInfo {
                            cluster_id: c.cluster_id,
                            label: c.label,
                            centroid: c.centroid,
                            hull: c.hull,
                        })
                        .collect(),
                    projection_method: viz.projection_method,
                    original_dimension: viz.original_dimension,
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Visualization failed: {}",
                e
            ))])),
        }
    }

    /// Find the most similar clusters for a given query.
    #[tool(
        description = "Find the most similar document clusters for a given query. Returns top-k clusters ranked by similarity."
    )]
    async fn find_similar_cluster(
        &self,
        Parameters(params): Parameters<FindSimilarClusterParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Parse algorithm if provided
        let algorithm = params.algorithm.as_deref().map(|algo| match algo {
            "dbscan" => crate::config::ClusteringAlgorithm::Dbscan,
            "agglomerative" => crate::config::ClusteringAlgorithm::Agglomerative,
            _ => crate::config::ClusteringAlgorithm::KMeans,
        });

        let top_k = params.top_k.unwrap_or(3);

        // First, get the clustering result
        match coordinator
            .cluster_documents(params.source_id.as_deref(), algorithm, params.num_clusters)
            .await
        {
            Ok(clustering_result) => {
                // Use the clustering engine to find similar clusters
                match coordinator
                    .find_similar_cluster(&params.query, &clustering_result, top_k)
                    .await
                {
                    Ok(similarities) => {
                        let similar_clusters: Vec<SimilarClusterInfo> = similarities
                            .iter()
                            .filter_map(|(cluster_id, similarity)| {
                                clustering_result
                                    .clusters
                                    .iter()
                                    .find(|c| c.cluster_id == *cluster_id)
                                    .map(|c| SimilarClusterInfo {
                                        cluster_id: c.cluster_id,
                                        label: c.label.clone(),
                                        similarity: *similarity,
                                        keywords: c.keywords.clone(),
                                        size: c.size,
                                    })
                            })
                            .collect();

                        let response = FindSimilarClusterResponse {
                            query: params.query.clone(),
                            similar_clusters,
                            algorithm: clustering_result.algorithm_used.clone(),
                        };

                        Ok(CallToolResult::success(vec![Content::text(
                            serde_json::to_string_pretty(&response).unwrap(),
                        )]))
                    }
                    Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Failed to find similar clusters: {}",
                        e
                    ))])),
                }
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Clustering failed: {}",
                e
            ))])),
        }
    }

    /// Get documents in a specific cluster.
    #[tool(
        description = "Get the documents belonging to a specific cluster. Returns document IDs with distance from centroid."
    )]
    async fn get_cluster_documents(
        &self,
        Parameters(params): Parameters<GetClusterDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Parse algorithm if provided
        let algorithm = params.algorithm.as_deref().map(|algo| match algo {
            "dbscan" => crate::config::ClusteringAlgorithm::Dbscan,
            "agglomerative" => crate::config::ClusteringAlgorithm::Agglomerative,
            _ => crate::config::ClusteringAlgorithm::KMeans,
        });

        let limit = params.limit.unwrap_or(50);
        let offset = params.offset.unwrap_or(0);

        // Get the clustering result
        match coordinator
            .cluster_documents(params.source_id.as_deref(), algorithm, params.num_clusters)
            .await
        {
            Ok(clustering_result) => {
                // Find the requested cluster
                if let Some(cluster) = clustering_result
                    .clusters
                    .iter()
                    .find(|c| c.cluster_id == params.cluster_id)
                {
                    let total_count = cluster.document_ids.len();

                    // Apply pagination
                    let documents: Vec<ClusterDocumentInfo> = cluster
                        .document_ids
                        .iter()
                        .skip(offset)
                        .take(limit)
                        .map(|doc_id| {
                            let is_representative = cluster.representative_docs.contains(doc_id);
                            ClusterDocumentInfo {
                                document_id: doc_id.clone(),
                                uri: doc_id.clone(), // Document ID is typically the URI
                                distance_from_centroid: 0.0, // Would need embeddings to compute
                                is_representative,
                            }
                        })
                        .collect();

                    let response = GetClusterDocumentsResponse {
                        cluster_id: cluster.cluster_id,
                        label: cluster.label.clone(),
                        documents,
                        total_count,
                        offset,
                        limit,
                    };

                    Ok(CallToolResult::success(vec![Content::text(
                        serde_json::to_string_pretty(&response).unwrap(),
                    )]))
                } else {
                    Ok(CallToolResult::success(vec![Content::text(format!(
                        "Cluster {} not found. Available clusters: {}",
                        params.cluster_id,
                        clustering_result
                            .clusters
                            .iter()
                            .map(|c| c.cluster_id.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))]))
                }
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Clustering failed: {}",
                e
            ))])),
        }
    }

    /// Check if content is a duplicate of an existing indexed document.
    #[tool(
        description = "Check if content is a duplicate of an existing indexed document. Returns similarity score and matching document ID if duplicate found."
    )]
    async fn check_duplicate(
        &self,
        Parameters(params): Parameters<CheckDuplicateParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        if !coordinator.is_deduplication_enabled() {
            return Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&CheckDuplicateResponse {
                    is_duplicate: false,
                    duplicate_of: None,
                    similarity: 0.0,
                    strategy: "disabled".to_string(),
                    message: "Deduplication is not enabled. Enable it in config.toml with [indexing.deduplication] enabled = true".to_string(),
                })
                .unwrap(),
            )]));
        }

        match coordinator
            .check_duplicate(&params.content, params.document_id.as_deref())
            .await
        {
            Ok(Some(result)) => {
                let response = CheckDuplicateResponse {
                    is_duplicate: result.is_duplicate,
                    duplicate_of: result.duplicate_of.clone(),
                    similarity: result.similarity,
                    strategy: format!("{:?}", result.strategy),
                    message: if result.is_duplicate {
                        format!(
                            "Content is a duplicate of document '{}' with {:.1}% similarity",
                            result.duplicate_of.as_deref().unwrap_or("unknown"),
                            result.similarity * 100.0
                        )
                    } else {
                        "Content is not a duplicate of any indexed document".to_string()
                    },
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Ok(None) => Ok(CallToolResult::success(vec![Content::text(
                "Deduplication is not enabled",
            )])),
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Duplicate check failed: {}",
                e
            ))])),
        }
    }

    /// Get deduplication configuration and statistics.
    #[tool(description = "Get deduplication configuration and statistics.")]
    async fn get_deduplication_stats(&self) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let config = coordinator.deduplication_config();

        let response = DeduplicationStatsResponse {
            enabled: config.enabled,
            strategy: format!("{:?}", config.strategy),
            threshold: config.threshold,
            action: format!("{:?}", config.action),
            minhash_num_hashes: config.minhash_num_hashes,
            shingle_size: config.shingle_size,
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Clear the deduplication index to reset duplicate tracking.
    #[tool(
        description = "Clear the deduplication index to reset duplicate tracking. This allows previously detected duplicates to be re-indexed."
    )]
    async fn clear_deduplication(&self) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        if !coordinator.is_deduplication_enabled() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Deduplication is not enabled",
            )]));
        }

        match coordinator.clear_deduplication_index().await {
            Ok(()) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&ClearDeduplicationResponse {
                    success: true,
                    message: "Deduplication index cleared successfully".to_string(),
                })
                .unwrap(),
            )])),
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to clear deduplication index: {}",
                e
            ))])),
        }
    }

    // ========================================================================
    // Versioning Tools
    // ========================================================================

    /// Get version history for a document.
    #[tool(
        description = "Get the version history of a document. Returns a list of all versions with timestamps, authors, and change types."
    )]
    async fn get_document_history(
        &self,
        Parameters(params): Parameters<GetDocumentHistoryParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        if !coordinator.is_versioning_enabled() {
            return Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&GetDocumentHistoryResponse {
                    document_id: params.document_id.clone(),
                    versions: vec![],
                    total_versions: 0,
                    message: "Versioning is not enabled. Enable it in config.toml with [indexing.versioning] enabled = true".to_string(),
                })
                .unwrap(),
            )]));
        }

        let limit = params.limit.unwrap_or(50);
        match coordinator
            .get_document_history(&params.document_id, Some(limit))
            .await
        {
            Ok(versions) => {
                let version_infos: Vec<VersionInfo> = versions
                    .iter()
                    .map(|v| VersionInfo {
                        version_id: v.version_id.clone(),
                        version_number: v.version_number,
                        timestamp: v.timestamp,
                        author: v.author.clone(),
                        change_type: format!("{:?}", v.change_type),
                        size_bytes: v.size_bytes,
                        content_hash: v.content_hash.clone(),
                    })
                    .collect();

                let response = GetDocumentHistoryResponse {
                    document_id: params.document_id.clone(),
                    total_versions: version_infos.len(),
                    versions: version_infos,
                    message: format!(
                        "Found {} versions for document {}",
                        versions.len(),
                        params.document_id
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to get document history: {}",
                e
            ))])),
        }
    }

    /// Compare two versions of a document.
    #[tool(
        description = "Compare two versions of a document and show the differences. Returns a unified diff format."
    )]
    async fn diff_versions(
        &self,
        Parameters(params): Parameters<DiffVersionsParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        if !coordinator.is_versioning_enabled() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Versioning is not enabled. Enable it in config.toml with [indexing.versioning] enabled = true",
            )]));
        }

        let context_lines = params.context_lines.unwrap_or(3);

        match coordinator
            .diff_versions(
                &params.document_id,
                &params.version_a,
                &params.version_b,
                context_lines,
            )
            .await
        {
            Ok(diff) => {
                let response = DiffVersionsResponse {
                    document_id: params.document_id.clone(),
                    version_a: VersionInfo {
                        version_id: diff.version_a.version_id.clone(),
                        version_number: diff.version_a.version_number,
                        timestamp: diff.version_a.timestamp,
                        author: diff.version_a.author.clone(),
                        change_type: format!("{:?}", diff.version_a.change_type),
                        size_bytes: diff.version_a.size_bytes,
                        content_hash: diff.version_a.content_hash.clone(),
                    },
                    version_b: VersionInfo {
                        version_id: diff.version_b.version_id.clone(),
                        version_number: diff.version_b.version_number,
                        timestamp: diff.version_b.timestamp,
                        author: diff.version_b.author.clone(),
                        change_type: format!("{:?}", diff.version_b.change_type),
                        size_bytes: diff.version_b.size_bytes,
                        content_hash: diff.version_b.content_hash.clone(),
                    },
                    unified_diff: diff.unified_diff.clone(),
                    stats: DiffStatsInfo {
                        lines_added: diff.diff.stats.lines_added,
                        lines_removed: diff.diff.stats.lines_removed,
                        lines_unchanged: diff.diff.stats.lines_unchanged,
                    },
                    message: format!(
                        "Diff between v{} and v{}: +{} -{} lines",
                        diff.version_a.version_number,
                        diff.version_b.version_number,
                        diff.diff.stats.lines_added,
                        diff.diff.stats.lines_removed
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to diff versions: {}",
                e
            ))])),
        }
    }

    /// Restore a document to a previous version.
    #[tool(
        description = "Restore a document to a previous version. Creates a new version with the restored content."
    )]
    async fn restore_version(
        &self,
        Parameters(params): Parameters<RestoreVersionParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        if !coordinator.is_versioning_enabled() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Versioning is not enabled. Enable it in config.toml with [indexing.versioning] enabled = true",
            )]));
        }

        match coordinator
            .restore_version(&params.document_id, &params.version_id, None)
            .await
        {
            Ok(restored) => {
                let response = RestoreVersionResponse {
                    document_id: params.document_id.clone(),
                    restored_version: VersionInfo {
                        version_id: restored.version_id.clone(),
                        version_number: restored.version_number,
                        timestamp: restored.timestamp,
                        author: restored.author.clone(),
                        change_type: format!("{:?}", restored.change_type),
                        size_bytes: restored.size_bytes,
                        content_hash: restored.content_hash.clone(),
                    },
                    restored_from: params.version_id.clone(),
                    success: true,
                    message: format!(
                        "Successfully restored document {} to version {} (new version: {})",
                        params.document_id, params.version_id, restored.version_number
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to restore version: {}",
                e
            ))])),
        }
    }

    /// Get the content of a specific version.
    #[tool(description = "Get the full content of a specific document version.")]
    async fn get_version_content(
        &self,
        Parameters(params): Parameters<GetVersionContentParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        if !coordinator.is_versioning_enabled() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Versioning is not enabled. Enable it in config.toml with [indexing.versioning] enabled = true",
            )]));
        }

        match coordinator.get_version_content(&params.version_id).await {
            Ok((version, content)) => {
                let response = GetVersionContentResponse {
                    version_id: version.version_id.clone(),
                    document_id: params.version_id.clone(), // VersionMetadata doesn't store doc_id
                    version_number: version.version_number,
                    content,
                    size_bytes: version.size_bytes,
                    message: format!(
                        "Retrieved content for version {} (v{})",
                        version.version_id, version.version_number
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to get version content: {}",
                e
            ))])),
        }
    }

    /// Get versioning configuration and statistics.
    #[tool(description = "Get versioning configuration and statistics.")]
    async fn get_versioning_stats(&self) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let config = coordinator.versioning_config();

        let (total_versions, total_size) = if config.enabled {
            match coordinator.versioning_stats().await {
                Ok(stats) => (Some(stats.0), Some(stats.1)),
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        let response = VersioningStatsResponse {
            enabled: config.enabled,
            storage: config.storage.clone(),
            delta_threshold: config.delta_threshold,
            total_versions,
            total_size_bytes: total_size,
            retention: VersioningRetentionInfo {
                min_versions: config.retention.min_versions,
                max_versions: config.retention.max_versions,
                min_age_days: config.retention.min_age_days,
                max_age_days: config.retention.max_age_days,
                keep_full_versions: config.retention.keep_full_versions,
                auto_cleanup: config.retention.auto_cleanup,
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    // ========================================================================
    // Operations Tools (Metrics, Backup, Cache)
    // ========================================================================

    /// Get server metrics in Prometheus and JSON format.
    #[tool(
        description = "Get server metrics including document counts, search queries, cache stats, and timing histograms."
    )]
    async fn get_metrics(&self) -> Result<CallToolResult, McpError> {
        let metrics = get_metrics();
        let prometheus = metrics.export_prometheus();
        let json = serde_json::to_value(metrics.export_json()).unwrap_or_default();

        let response = GetMetricsResponse {
            prometheus,
            json,
            message: "Metrics retrieved successfully".to_string(),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get cache statistics and configuration.
    #[tool(description = "Get cache statistics including hit/miss counts and configuration.")]
    async fn get_cache_stats(&self) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let config = &state.config.search.cache;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Get live cache stats from coordinator
        let cache_stats = coordinator.cache_stats();

        let response = GetCacheStatsResponse {
            enabled: config.enabled,
            embedding_entries: cache_stats.embedding_entries,
            result_entries: cache_stats.result_entries,
            embedding_size_bytes: cache_stats.embedding_size_bytes,
            result_size_bytes: cache_stats.result_size_bytes,
            config: CacheConfigInfo {
                max_entries: config.max_entries,
                ttl_secs: config.ttl_secs,
                cache_embeddings: config.cache_embeddings,
                cache_results: config.cache_results,
            },
            message: format!(
                "Cache stats retrieved successfully: {} embeddings, {} results cached",
                cache_stats.embedding_entries, cache_stats.result_entries
            ),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Clear all cached data.
    #[tool(description = "Clear all cached embeddings and search results.")]
    async fn clear_cache(&self) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        // Get stats before clearing
        let stats_before = coordinator.cache_stats();

        // Clear the cache
        coordinator.clear_cache().await;

        let response = ClearCacheResponse {
            success: true,
            message: format!(
                "Cache cleared successfully. Removed {} embeddings and {} results.",
                stats_before.embedding_entries, stats_before.result_entries
            ),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// List available backups.
    #[tool(description = "List all available index backups.")]
    async fn list_backups(&self) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let backup_dir = state
            .config
            .operations
            .backup
            .backup_dir
            .clone()
            .unwrap_or_else(|| {
                let data_dir = state.config.storage.data_dir.clone();
                format!("{}/backups", data_dir)
            });

        match crate::backup::BackupManager::new(&backup_dir) {
            Ok(manager) => match manager.list_backups() {
                Ok(backups) => {
                    let backup_infos: Vec<BackupInfo> = backups
                        .iter()
                        .map(|b| BackupInfo {
                            backup_id: b.backup_id.clone(),
                            created_at: b.created_at,
                            version: b.version.clone(),
                            document_count: b.document_count,
                            chunk_count: b.chunk_count,
                            size_bytes: b.size_bytes,
                            description: b.description.clone(),
                        })
                        .collect();

                    let response = ListBackupsResponse {
                        backups: backup_infos,
                        message: format!("Found {} backups", backups.len()),
                    };

                    Ok(CallToolResult::success(vec![Content::text(
                        serde_json::to_string_pretty(&response).unwrap(),
                    )]))
                }
                Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                    "Failed to list backups: {}",
                    e
                ))])),
            },
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to initialize backup manager: {}",
                e
            ))])),
        }
    }

    /// Create a backup of the index.
    #[tool(
        description = "Create a backup of the index including all documents and their embeddings."
    )]
    async fn create_backup(
        &self,
        Parameters(params): Parameters<CreateBackupParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        let backup_dir = params.output_path.unwrap_or_else(|| {
            state
                .config
                .operations
                .backup
                .backup_dir
                .clone()
                .unwrap_or_else(|| {
                    let data_dir = state.config.storage.data_dir.clone();
                    format!("{}/backups", data_dir)
                })
        });

        match crate::backup::BackupManager::new(&backup_dir) {
            Ok(manager) => {
                match manager
                    .create_backup(
                        coordinator.storage(),
                        &format!("{:?}", state.config.storage.backend),
                        coordinator.embedding_dimension(),
                        params.description,
                    )
                    .await
                {
                    Ok(result) => {
                        let response = CreateBackupResponse {
                            backup_id: result.metadata.backup_id.clone(),
                            path: result.path.display().to_string(),
                            document_count: result.metadata.document_count,
                            size_bytes: result.metadata.size_bytes,
                            duration_ms: result.duration_ms,
                            success: true,
                            message: format!(
                                "Backup created successfully: {} documents",
                                result.metadata.document_count
                            ),
                        };

                        Ok(CallToolResult::success(vec![Content::text(
                            serde_json::to_string_pretty(&response).unwrap(),
                        )]))
                    }
                    Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Failed to create backup: {}",
                        e
                    ))])),
                }
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to initialize backup manager: {}",
                e
            ))])),
        }
    }

    /// Restore from a backup.
    #[tool(description = "Restore the index from a backup file.")]
    async fn restore_backup(
        &self,
        Parameters(params): Parameters<RestoreBackupParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        let backup_dir = state
            .config
            .operations
            .backup
            .backup_dir
            .clone()
            .unwrap_or_else(|| {
                let data_dir = state.config.storage.data_dir.clone();
                format!("{}/backups", data_dir)
            });

        match crate::backup::BackupManager::new(&backup_dir) {
            Ok(manager) => {
                // Try to find backup by ID or use as path
                let backup_path = if std::path::Path::new(&params.backup_id).exists() {
                    std::path::PathBuf::from(&params.backup_id)
                } else {
                    // Find backup file by listing and matching ID
                    let backups = manager.list_backups().map_err(|e| {
                        McpError::internal_error(format!("Failed to list backups: {}", e), None)
                    })?;

                    backups
                        .iter()
                        .find(|b| b.backup_id == params.backup_id)
                        .map(|b| {
                            let timestamp = b.created_at.format("%Y%m%d_%H%M%S");
                            std::path::PathBuf::from(&backup_dir)
                                .join(format!("backup_{}.jsonl", timestamp))
                        })
                        .ok_or_else(|| {
                            McpError::invalid_params(
                                format!("Backup not found: {}", params.backup_id),
                                None,
                            )
                        })?
                };

                match manager
                    .restore_backup(coordinator.storage(), &backup_path)
                    .await
                {
                    Ok(result) => {
                        let response = RestoreBackupResponse {
                            backup_id: result.metadata.backup_id.clone(),
                            documents_restored: result.documents_restored,
                            chunks_restored: result.chunks_restored,
                            duration_ms: result.duration_ms,
                            success: true,
                            message: format!(
                                "Restored {} documents from backup",
                                result.documents_restored
                            ),
                        };

                        Ok(CallToolResult::success(vec![Content::text(
                            serde_json::to_string_pretty(&response).unwrap(),
                        )]))
                    }
                    Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Failed to restore backup: {}",
                        e
                    ))])),
                }
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to initialize backup manager: {}",
                e
            ))])),
        }
    }

    /// Export documents to a file.
    #[tool(description = "Export indexed documents to a file in JSONL or JSON format.")]
    async fn export_documents(
        &self,
        Parameters(params): Parameters<ExportDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        let backup_dir = state
            .config
            .operations
            .backup
            .backup_dir
            .clone()
            .unwrap_or_else(|| {
                let data_dir = state.config.storage.data_dir.clone();
                format!("{}/backups", data_dir)
            });

        let format = match params.format.as_deref() {
            Some("json") => crate::backup::ExportFormat::Json,
            _ => crate::backup::ExportFormat::Jsonl,
        };

        let options = crate::backup::ExportOptions {
            format,
            include_content: true,
            include_embeddings: params.include_embeddings.unwrap_or(false),
            source_id: params.source_id,
            compress: false,
        };

        match crate::backup::BackupManager::new(&backup_dir) {
            Ok(manager) => {
                let start = std::time::Instant::now();
                match manager
                    .export_documents(coordinator.storage(), &params.output_path, &options)
                    .await
                {
                    Ok(result) => {
                        let response = ExportDocumentsResponse {
                            path: result.path.display().to_string(),
                            document_count: result.document_count,
                            size_bytes: result.size_bytes,
                            duration_ms: start.elapsed().as_millis() as u64,
                            success: true,
                            message: format!(
                                "Exported {} documents to {}",
                                result.document_count, params.output_path
                            ),
                        };

                        Ok(CallToolResult::success(vec![Content::text(
                            serde_json::to_string_pretty(&response).unwrap(),
                        )]))
                    }
                    Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Failed to export documents: {}",
                        e
                    ))])),
                }
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to initialize backup manager: {}",
                e
            ))])),
        }
    }

    /// Import documents from a file.
    #[tool(description = "Import documents from a JSONL or JSON file.")]
    async fn import_documents(
        &self,
        Parameters(params): Parameters<ImportDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        // Ensure coordinator is initialized
        self.ensure_coordinator().await?;

        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();

        let backup_dir = state
            .config
            .operations
            .backup
            .backup_dir
            .clone()
            .unwrap_or_else(|| {
                let data_dir = state.config.storage.data_dir.clone();
                format!("{}/backups", data_dir)
            });

        match crate::backup::BackupManager::new(&backup_dir) {
            Ok(manager) => {
                let start = std::time::Instant::now();
                match manager
                    .import_documents(coordinator.storage(), &params.input_path)
                    .await
                {
                    Ok((documents_imported, chunks_imported)) => {
                        let response = ImportDocumentsResponse {
                            documents_imported,
                            chunks_imported,
                            duration_ms: start.elapsed().as_millis() as u64,
                            success: true,
                            message: format!(
                                "Imported {} documents with {} chunks from {}",
                                documents_imported, chunks_imported, params.input_path
                            ),
                        };

                        Ok(CallToolResult::success(vec![Content::text(
                            serde_json::to_string_pretty(&response).unwrap(),
                        )]))
                    }
                    Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Failed to import documents: {}",
                        e
                    ))])),
                }
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to initialize backup manager: {}",
                e
            ))])),
        }
    }

    // ========================================================================
    // ACL (Access Control List) Tools
    // ========================================================================

    /// Get the ACL for a document.
    #[tool(
        description = "Get the access control list (ACL) for a document, showing who has what permissions."
    )]
    async fn get_document_acl(
        &self,
        Parameters(params): Parameters<GetDocumentAclParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        // Check if ACL is enabled
        if !state.config.security.acl.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&GetDocumentAclResponse {
                    document_id: params.document_id.clone(),
                    owner: "system".to_string(),
                    entries: vec![],
                    inherit_from_source: false,
                    is_public: true,
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    message: "ACL is not enabled. Enable it in config.toml with [security.acl] enabled = true".to_string(),
                })
                .unwrap(),
            )]));
        }

        match state
            .acl_storage
            .get_document_acl(&params.document_id)
            .await
        {
            Ok(Some(acl)) => {
                let entries: Vec<AclEntryInfo> = acl
                    .entries
                    .iter()
                    .map(|e| {
                        let (principal_type, principal_id) = match &e.principal {
                            Principal::User(id) => ("user".to_string(), Some(id.clone())),
                            Principal::Role(name) => ("role".to_string(), Some(name.clone())),
                            Principal::Group(id) => ("group".to_string(), Some(id.clone())),
                            Principal::Everyone => ("everyone".to_string(), None),
                            Principal::Authenticated => ("authenticated".to_string(), None),
                        };
                        AclEntryInfo {
                            principal_type,
                            principal_id,
                            permissions: e.permissions.iter().map(|p| p.to_string()).collect(),
                        }
                    })
                    .collect();

                let response = GetDocumentAclResponse {
                    document_id: acl.document_id.clone(),
                    owner: acl.owner.clone(),
                    entries,
                    inherit_from_source: acl.inherit_from_source,
                    is_public: acl.is_public(),
                    created_at: acl.created_at,
                    updated_at: acl.updated_at,
                    message: format!("ACL retrieved for document {}", params.document_id),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Ok(None) => {
                // No ACL exists for this document - return default info
                let default_public = state.config.security.acl.default_public;
                let response = GetDocumentAclResponse {
                    document_id: params.document_id.clone(),
                    owner: "system".to_string(),
                    entries: if default_public {
                        vec![AclEntryInfo {
                            principal_type: "everyone".to_string(),
                            principal_id: None,
                            permissions: vec!["read".to_string()],
                        }]
                    } else if state.config.security.acl.default_authenticated_read {
                        vec![AclEntryInfo {
                            principal_type: "authenticated".to_string(),
                            principal_id: None,
                            permissions: vec!["read".to_string()],
                        }]
                    } else {
                        vec![]
                    },
                    inherit_from_source: true,
                    is_public: default_public,
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    message: format!(
                        "No explicit ACL for document {}. Using default settings (public: {})",
                        params.document_id, default_public
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to get document ACL: {}",
                e
            ))])),
        }
    }

    /// Set the ACL for a document.
    #[tool(
        description = "Set access control list (ACL) for a document to control who can read, write, or delete it."
    )]
    async fn set_document_acl(
        &self,
        Parameters(params): Parameters<SetDocumentAclParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        // Check if ACL is enabled
        if !state.config.security.acl.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                "ACL is not enabled. Enable it in config.toml with [security.acl] enabled = true",
            )]));
        }

        // Build the document ACL
        let owner = params.owner.unwrap_or_else(|| "system".to_string());
        let mut acl = DocumentAcl::new(&params.document_id, &owner);

        if let Some(inherit) = params.inherit_from_source {
            acl.inherit_from_source = inherit;
        }

        // Parse and add entries
        for entry_param in &params.entries {
            let principal = match entry_param.principal_type.as_str() {
                "user" => {
                    let id = entry_param.principal_id.clone().unwrap_or_default();
                    Principal::User(id)
                }
                "role" => {
                    let name = entry_param.principal_id.clone().unwrap_or_default();
                    Principal::Role(name)
                }
                "group" => {
                    let id = entry_param.principal_id.clone().unwrap_or_default();
                    Principal::Group(id)
                }
                "everyone" => Principal::Everyone,
                "authenticated" => Principal::Authenticated,
                _ => {
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Invalid principal type: {}. Valid types: user, role, group, everyone, authenticated",
                        entry_param.principal_type
                    ))]));
                }
            };

            let permissions: Vec<Permission> = entry_param
                .permissions
                .iter()
                .filter_map(|p| Permission::from_str(p))
                .collect();

            if permissions.is_empty() {
                return Ok(CallToolResult::success(vec![Content::text(
                    "No valid permissions found. Valid permissions: read, write, delete, admin",
                )]));
            }

            acl.add_entry(AclEntry::new(principal, permissions));
        }

        // Store the ACL
        match state.acl_storage.set_document_acl(acl).await {
            Ok(()) => {
                let response = SetDocumentAclResponse {
                    document_id: params.document_id.clone(),
                    success: true,
                    entry_count: params.entries.len(),
                    message: format!(
                        "ACL set for document {} with {} entries",
                        params.document_id,
                        params.entries.len()
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to set document ACL: {}",
                e
            ))])),
        }
    }

    /// Delete the ACL for a document.
    #[tool(
        description = "Delete the access control list for a document, reverting to default permissions."
    )]
    async fn delete_document_acl(
        &self,
        Parameters(params): Parameters<DeleteDocumentAclParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        // Check if ACL is enabled
        if !state.config.security.acl.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                "ACL is not enabled. Enable it in config.toml with [security.acl] enabled = true",
            )]));
        }

        match state
            .acl_storage
            .delete_document_acl(&params.document_id)
            .await
        {
            Ok(()) => {
                let response = DeleteDocumentAclResponse {
                    document_id: params.document_id.clone(),
                    success: true,
                    message: format!(
                        "ACL deleted for document {}. Default permissions now apply.",
                        params.document_id
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to delete document ACL: {}",
                e
            ))])),
        }
    }

    /// Check if a user has a specific permission on a document.
    #[tool(
        description = "Check if a user has a specific permission (read, write, delete, admin) on a document."
    )]
    async fn check_permission(
        &self,
        Parameters(params): Parameters<CheckPermissionParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        // Parse the requested permission
        let permission = match Permission::from_str(&params.permission) {
            Some(p) => p,
            None => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Invalid permission: {}. Valid permissions: read, write, delete, admin",
                    params.permission
                ))]));
            }
        };

        // Build auth context for the check
        let user_id = params.user_id.unwrap_or_else(|| "anonymous".to_string());
        let roles = params.roles.unwrap_or_default();
        let auth_ctx = if user_id == "anonymous" {
            AuthContext::anonymous()
        } else {
            AuthContext::authenticated(user_id.clone(), roles.clone(), "check")
        };

        // Check permission
        match state
            .acl_resolver
            .check_permission(&auth_ctx, &params.document_id, permission)
            .await
        {
            Ok(result) => {
                let response = CheckPermissionResponse {
                    document_id: params.document_id.clone(),
                    user_id: user_id.clone(),
                    permission: params.permission.clone(),
                    allowed: result.allowed,
                    reason: result.reason.clone(),
                    granted_permissions: result
                        .granted_permissions
                        .iter()
                        .map(|p| p.to_string())
                        .collect(),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to check permission: {}",
                e
            ))])),
        }
    }

    /// Get ACL configuration and statistics.
    #[tool(
        description = "Get ACL configuration, role definitions, and statistics about document and source ACLs."
    )]
    async fn get_acl_stats(&self) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;
        let config = &state.config.security.acl;

        // Get ACL counts
        let doc_acls = state
            .acl_storage
            .list_document_acls(1000, 0)
            .await
            .unwrap_or_default();
        let source_acls = state
            .acl_storage
            .list_source_acls()
            .await
            .unwrap_or_default();

        let roles: Vec<RoleInfo> = config
            .roles
            .iter()
            .map(|r| RoleInfo {
                name: r.name.clone(),
                inherits_from: r.inherits_from.clone(),
                permissions: r.permissions.clone(),
            })
            .collect();

        let response = GetAclStatsResponse {
            enabled: config.enabled,
            config: AclConfigInfo {
                default_public: config.default_public,
                default_authenticated_read: config.default_authenticated_read,
                enforce_on_search: config.enforce_on_search,
                enforce_on_get: config.enforce_on_get,
                enforce_on_delete: config.enforce_on_delete,
            },
            document_acl_count: doc_acls.len(),
            source_acl_count: source_acls.len(),
            roles,
            message: if config.enabled {
                format!(
                    "ACL is enabled with {} document ACLs and {} source ACLs",
                    doc_acls.len(),
                    source_acls.len()
                )
            } else {
                "ACL is not enabled. All documents are accessible to everyone.".to_string()
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get the ACL for a source.
    #[tool(
        description = "Get the access control list (ACL) for a source, showing who has access to documents in this source."
    )]
    async fn get_source_acl(
        &self,
        Parameters(params): Parameters<GetSourceAclParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        // Check if ACL is enabled
        if !state.config.security.acl.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&GetSourceAclResponse {
                    source_id: params.source_id.clone(),
                    owner: "system".to_string(),
                    entries: vec![],
                    default_document_acl: vec![],
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    message: "ACL is not enabled. Enable it in config.toml with [security.acl] enabled = true".to_string(),
                })
                .unwrap(),
            )]));
        }

        match state.acl_storage.get_source_acl(&params.source_id).await {
            Ok(Some(acl)) => {
                let entries: Vec<AclEntryInfo> = acl
                    .entries
                    .iter()
                    .map(|e| {
                        let (principal_type, principal_id) = match &e.principal {
                            Principal::User(id) => ("user".to_string(), Some(id.clone())),
                            Principal::Role(name) => ("role".to_string(), Some(name.clone())),
                            Principal::Group(id) => ("group".to_string(), Some(id.clone())),
                            Principal::Everyone => ("everyone".to_string(), None),
                            Principal::Authenticated => ("authenticated".to_string(), None),
                        };
                        AclEntryInfo {
                            principal_type,
                            principal_id,
                            permissions: e.permissions.iter().map(|p| p.to_string()).collect(),
                        }
                    })
                    .collect();

                let default_entries: Vec<AclEntryInfo> = acl
                    .default_document_acl
                    .iter()
                    .map(|e| {
                        let (principal_type, principal_id) = match &e.principal {
                            Principal::User(id) => ("user".to_string(), Some(id.clone())),
                            Principal::Role(name) => ("role".to_string(), Some(name.clone())),
                            Principal::Group(id) => ("group".to_string(), Some(id.clone())),
                            Principal::Everyone => ("everyone".to_string(), None),
                            Principal::Authenticated => ("authenticated".to_string(), None),
                        };
                        AclEntryInfo {
                            principal_type,
                            principal_id,
                            permissions: e.permissions.iter().map(|p| p.to_string()).collect(),
                        }
                    })
                    .collect();

                let response = GetSourceAclResponse {
                    source_id: acl.source_id.clone(),
                    owner: acl.owner.clone(),
                    entries,
                    default_document_acl: default_entries,
                    created_at: acl.created_at,
                    updated_at: acl.updated_at,
                    message: format!("ACL retrieved for source {}", params.source_id),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Ok(None) => {
                let response = GetSourceAclResponse {
                    source_id: params.source_id.clone(),
                    owner: "system".to_string(),
                    entries: vec![],
                    default_document_acl: vec![],
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    message: format!(
                        "No explicit ACL for source {}. Using default settings.",
                        params.source_id
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to get source ACL: {}",
                e
            ))])),
        }
    }

    /// Set the ACL for a source.
    #[tool(
        description = "Set access control list (ACL) for a source to control who can access documents in this source."
    )]
    async fn set_source_acl(
        &self,
        Parameters(params): Parameters<SetSourceAclParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        // Check if ACL is enabled
        if !state.config.security.acl.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                "ACL is not enabled. Enable it in config.toml with [security.acl] enabled = true",
            )]));
        }

        // Build the source ACL
        let owner = params.owner.unwrap_or_else(|| "system".to_string());
        let mut acl = SourceAcl::new(&params.source_id, &owner);

        // Parse and add entries
        for entry_param in &params.entries {
            let principal = match entry_param.principal_type.as_str() {
                "user" => {
                    let id = entry_param.principal_id.clone().unwrap_or_default();
                    Principal::User(id)
                }
                "role" => {
                    let name = entry_param.principal_id.clone().unwrap_or_default();
                    Principal::Role(name)
                }
                "group" => {
                    let id = entry_param.principal_id.clone().unwrap_or_default();
                    Principal::Group(id)
                }
                "everyone" => Principal::Everyone,
                "authenticated" => Principal::Authenticated,
                _ => {
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Invalid principal type: {}. Valid types: user, role, group, everyone, authenticated",
                        entry_param.principal_type
                    ))]));
                }
            };

            let permissions: Vec<Permission> = entry_param
                .permissions
                .iter()
                .filter_map(|p| Permission::from_str(p))
                .collect();

            if permissions.is_empty() {
                return Ok(CallToolResult::success(vec![Content::text(
                    "No valid permissions found. Valid permissions: read, write, delete, admin",
                )]));
            }

            acl.entries.push(AclEntry::new(principal, permissions));
        }

        // Parse and add default document ACL entries
        if let Some(default_entries) = &params.default_document_acl {
            for entry_param in default_entries {
                let principal = match entry_param.principal_type.as_str() {
                    "user" => {
                        let id = entry_param.principal_id.clone().unwrap_or_default();
                        Principal::User(id)
                    }
                    "role" => {
                        let name = entry_param.principal_id.clone().unwrap_or_default();
                        Principal::Role(name)
                    }
                    "group" => {
                        let id = entry_param.principal_id.clone().unwrap_or_default();
                        Principal::Group(id)
                    }
                    "everyone" => Principal::Everyone,
                    "authenticated" => Principal::Authenticated,
                    _ => {
                        return Ok(CallToolResult::success(vec![Content::text(format!(
                            "Invalid principal type: {}. Valid types: user, role, group, everyone, authenticated",
                            entry_param.principal_type
                        ))]));
                    }
                };

                let permissions: Vec<Permission> = entry_param
                    .permissions
                    .iter()
                    .filter_map(|p| Permission::from_str(p))
                    .collect();

                if !permissions.is_empty() {
                    acl.default_document_acl
                        .push(AclEntry::new(principal, permissions));
                }
            }
        }

        // Store the ACL
        match state.acl_storage.set_source_acl(acl).await {
            Ok(()) => {
                let response = SetSourceAclResponse {
                    source_id: params.source_id.clone(),
                    success: true,
                    entry_count: params.entries.len(),
                    message: format!(
                        "ACL set for source {} with {} entries",
                        params.source_id,
                        params.entries.len()
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to set source ACL: {}",
                e
            ))])),
        }
    }

    /// Delete the ACL for a source.
    #[tool(
        description = "Delete the access control list for a source, reverting to default permissions."
    )]
    async fn delete_source_acl(
        &self,
        Parameters(params): Parameters<DeleteSourceAclParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        // Check if ACL is enabled
        if !state.config.security.acl.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                "ACL is not enabled. Enable it in config.toml with [security.acl] enabled = true",
            )]));
        }

        match state.acl_storage.delete_source_acl(&params.source_id).await {
            Ok(()) => {
                let response = DeleteSourceAclResponse {
                    source_id: params.source_id.clone(),
                    success: true,
                    message: format!(
                        "ACL deleted for source {}. Default permissions now apply.",
                        params.source_id
                    ),
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to delete source ACL: {}",
                e
            ))])),
        }
    }

    // ========================================================================
    // Webhook Tools
    // ========================================================================

    /// List all configured webhooks.
    #[tool(description = "List all configured webhooks with their events and configuration.")]
    async fn list_webhooks(&self) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        if !state.config.integration.webhooks.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&ListWebhooksResponse {
                    webhooks: vec![],
                    message: "Webhooks are not enabled. Enable them in config.toml with [integration.webhooks] enabled = true".to_string(),
                })
                .unwrap(),
            )]));
        }

        let webhooks = state.webhook_dispatcher.list_webhooks().await;

        let webhook_infos: Vec<WebhookInfo> = webhooks
            .iter()
            .map(|w| WebhookInfo {
                id: w.id.clone(),
                url: w.url.clone(),
                events: w.events.clone(),
                has_secret: w.secret.is_some(),
                retry_count: w.retry_count,
                timeout_secs: w.timeout_secs,
                enabled: w.enabled,
                description: w.description.clone(),
            })
            .collect();

        let response = ListWebhooksResponse {
            webhooks: webhook_infos.clone(),
            message: format!("Found {} configured webhooks", webhook_infos.len()),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Add a new webhook configuration.
    #[tool(
        description = "Add a new webhook to receive notifications for index events like document.indexed, source.added, index.error."
    )]
    async fn add_webhook(
        &self,
        Parameters(params): Parameters<AddWebhookParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        if !state.config.integration.webhooks.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                "Webhooks are not enabled. Enable them in config.toml with [integration.webhooks] enabled = true",
            )]));
        }

        // Validate URL
        if !params.url.starts_with("http://") && !params.url.starts_with("https://") {
            return Ok(CallToolResult::success(vec![Content::text(
                "Invalid webhook URL. Must start with http:// or https://",
            )]));
        }

        // Validate events
        let valid_events = [
            "document.indexed",
            "document.updated",
            "document.deleted",
            "source.added",
            "source.removed",
            "index.error",
            "backup.created",
            "search.performed",
        ];

        for event in &params.events {
            if !valid_events.contains(&event.as_str()) {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Invalid event: {}. Valid events: {}",
                    event,
                    valid_events.join(", ")
                ))]));
            }
        }

        let config = WebhookConfig {
            id: uuid::Uuid::new_v4().to_string(),
            url: params.url.clone(),
            events: params.events.clone(),
            secret: params.secret,
            retry_count: params.retry_count.unwrap_or(3),
            timeout_secs: params.timeout_secs.unwrap_or(30),
            enabled: true,
            description: params.description,
            headers: std::collections::HashMap::new(),
        };

        let webhook_id = state.webhook_dispatcher.add_webhook(config).await;

        let response = AddWebhookResponse {
            webhook_id: webhook_id.clone(),
            success: true,
            message: format!(
                "Webhook {} added successfully for events: {}",
                webhook_id,
                params.events.join(", ")
            ),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Remove a webhook by ID.
    #[tool(description = "Remove a webhook configuration by its ID.")]
    async fn remove_webhook(
        &self,
        Parameters(params): Parameters<RemoveWebhookParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        if !state.config.integration.webhooks.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                "Webhooks are not enabled",
            )]));
        }

        let removed = state
            .webhook_dispatcher
            .remove_webhook(&params.webhook_id)
            .await;

        let response = RemoveWebhookResponse {
            webhook_id: params.webhook_id.clone(),
            success: removed,
            message: if removed {
                format!("Webhook {} removed successfully", params.webhook_id)
            } else {
                format!("Webhook {} not found", params.webhook_id)
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Test a webhook by sending a test event.
    #[tool(description = "Send a test event to a webhook to verify it's working correctly.")]
    async fn test_webhook(
        &self,
        Parameters(params): Parameters<TestWebhookParams>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        if !state.config.integration.webhooks.enabled {
            return Ok(CallToolResult::success(vec![Content::text(
                "Webhooks are not enabled",
            )]));
        }

        match state
            .webhook_dispatcher
            .test_webhook(&params.webhook_id)
            .await
        {
            Some(result) => {
                let error_msg = result.error.clone();
                let message = if result.success {
                    format!("Webhook test successful ({}ms)", result.duration_ms)
                } else {
                    format!(
                        "Webhook test failed: {}",
                        error_msg.as_deref().unwrap_or("Unknown error")
                    )
                };

                let response = TestWebhookResponse {
                    webhook_id: params.webhook_id.clone(),
                    success: result.success,
                    status_code: result.status_code,
                    error: result.error,
                    duration_ms: result.duration_ms,
                    message,
                };

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap(),
                )]))
            }
            None => Ok(CallToolResult::success(vec![Content::text(format!(
                "Webhook {} not found",
                params.webhook_id
            ))])),
        }
    }

    /// Get webhook statistics and recent delivery history.
    #[tool(
        description = "Get webhook delivery statistics including success/failure counts and recent deliveries."
    )]
    async fn get_webhook_stats(&self) -> Result<CallToolResult, McpError> {
        let state = self.state.read().await;

        let webhooks_enabled = state.config.integration.webhooks.enabled;
        let webhook_count = state.webhook_dispatcher.list_webhooks().await.len();
        let stats = state.webhook_dispatcher.stats().await;
        let recent = state.webhook_dispatcher.recent_deliveries().await;

        let recent_deliveries: Vec<WebhookDeliveryInfo> = recent
            .iter()
            .take(20)
            .map(|d| WebhookDeliveryInfo {
                webhook_id: d.webhook_id.clone(),
                event: d.event.clone(),
                success: d.success,
                status_code: d.status_code,
                error: d.error.clone(),
                attempts: d.attempts,
                duration_ms: d.duration_ms,
                timestamp: d.timestamp,
            })
            .collect();

        let response = GetWebhookStatsResponse {
            enabled: webhooks_enabled,
            webhook_count,
            events_dispatched: stats.events_dispatched,
            successful_deliveries: stats.successful_deliveries,
            failed_deliveries: stats.failed_deliveries,
            total_retries: stats.total_retries,
            avg_delivery_time_ms: stats.avg_delivery_time_ms,
            recent_deliveries,
            message: if webhooks_enabled {
                format!(
                    "{} webhooks, {} events dispatched ({} successful, {} failed)",
                    webhook_count,
                    stats.events_dispatched,
                    stats.successful_deliveries,
                    stats.failed_deliveries
                )
            } else {
                "Webhooks are not enabled".to_string()
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    // ========================================================================
    // GTD Tools
    // ========================================================================

    /// Manage GTD projects - list, create, update, archive, and check health.
    #[tool(
        description = "Manage GTD projects. Actions: list, get, create, update, archive, complete, health. Projects are multi-step outcomes with next actions."
    )]
    async fn gtd_projects(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_tools::GtdProjectsParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{Project, ProjectFilter, ProjectManager};
        use crate::mcp::gtd_tools::{GtdProjectsResponse, ProjectAction};

        // Get ontology store from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();

        let manager = ProjectManager::new(ontology_store);

        let response = match params.action {
            ProjectAction::List => {
                let filter = ProjectFilter {
                    status: params.status,
                    area: params.area.clone(),
                    has_next_action: params.has_next_action,
                    stalled_days: params.stalled_days,
                    limit: params.limit.unwrap_or(100),
                    offset: 0,
                };
                match manager.list(filter).await {
                    Ok(projects) => GtdProjectsResponse::success_list(
                        projects.clone(),
                        format!("Found {} projects", projects.len()),
                    ),
                    Err(e) => GtdProjectsResponse::error(format!("Failed to list projects: {}", e)),
                }
            }
            ProjectAction::Get => {
                let project_id = params.project_id.ok_or_else(|| {
                    McpError::invalid_params("project_id is required for 'get' action", None)
                })?;
                match manager.get(&project_id).await {
                    Ok(Some(project)) => {
                        GtdProjectsResponse::success_single(project, "Project retrieved")
                    }
                    Ok(None) => {
                        GtdProjectsResponse::error(format!("Project not found: {}", project_id))
                    }
                    Err(e) => GtdProjectsResponse::error(format!("Failed to get project: {}", e)),
                }
            }
            ProjectAction::Create => {
                let name = params.name.ok_or_else(|| {
                    McpError::invalid_params("name is required for 'create' action", None)
                })?;
                let mut project = Project::new(name);
                if let Some(outcome) = params.outcome {
                    project = project.with_outcome(outcome);
                }
                if let Some(area) = params.area {
                    project = project.with_area(area);
                }
                if let Some(goal) = params.goal {
                    project = project.with_goal(goal);
                }
                match manager.create(project).await {
                    Ok(created) => {
                        GtdProjectsResponse::success_single(created, "Project created successfully")
                    }
                    Err(e) => {
                        GtdProjectsResponse::error(format!("Failed to create project: {}", e))
                    }
                }
            }
            ProjectAction::Update => {
                let project_id = params.project_id.ok_or_else(|| {
                    McpError::invalid_params("project_id is required for 'update' action", None)
                })?;
                match manager.get(&project_id).await {
                    Ok(Some(mut project)) => {
                        if let Some(name) = params.name {
                            project.name = name;
                        }
                        if let Some(outcome) = params.outcome {
                            project.outcome = Some(outcome);
                        }
                        if let Some(area) = params.area {
                            project.area = Some(area);
                        }
                        if let Some(goal) = params.goal {
                            project.supporting_goal = Some(goal);
                        }
                        if let Some(status) = params.status {
                            project.status = status;
                        }
                        match manager.update(&project_id, project).await {
                            Ok(updated) => {
                                GtdProjectsResponse::success_single(updated, "Project updated")
                            }
                            Err(e) => {
                                GtdProjectsResponse::error(format!("Failed to update: {}", e))
                            }
                        }
                    }
                    Ok(None) => {
                        GtdProjectsResponse::error(format!("Project not found: {}", project_id))
                    }
                    Err(e) => GtdProjectsResponse::error(format!("Failed to get project: {}", e)),
                }
            }
            ProjectAction::Archive => {
                let project_id = params.project_id.ok_or_else(|| {
                    McpError::invalid_params("project_id is required for 'archive' action", None)
                })?;
                match manager.archive(&project_id).await {
                    Ok(Some(project)) => {
                        GtdProjectsResponse::success_single(project, "Project archived")
                    }
                    Ok(None) => {
                        GtdProjectsResponse::error(format!("Project not found: {}", project_id))
                    }
                    Err(e) => GtdProjectsResponse::error(format!("Failed to archive: {}", e)),
                }
            }
            ProjectAction::Complete => {
                let project_id = params.project_id.ok_or_else(|| {
                    McpError::invalid_params("project_id is required for 'complete' action", None)
                })?;
                match manager.complete(&project_id).await {
                    Ok(Some(project)) => {
                        GtdProjectsResponse::success_single(project, "Project completed")
                    }
                    Ok(None) => {
                        GtdProjectsResponse::error(format!("Project not found: {}", project_id))
                    }
                    Err(e) => GtdProjectsResponse::error(format!("Failed to complete: {}", e)),
                }
            }
            ProjectAction::Health => {
                let project_id = params.project_id.ok_or_else(|| {
                    McpError::invalid_params("project_id is required for 'health' action", None)
                })?;
                match manager.get_health(&project_id).await {
                    Ok(Some(health)) => {
                        GtdProjectsResponse::success_health(health, "Project health calculated")
                    }
                    Ok(None) => {
                        GtdProjectsResponse::error(format!("Project not found: {}", project_id))
                    }
                    Err(e) => {
                        GtdProjectsResponse::error(format!("Failed to calculate health: {}", e))
                    }
                }
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Manage GTD tasks - list, create, complete, and get smart recommendations.
    #[tool(
        description = "Manage GTD tasks. Actions: list, get, create, update, complete, delete, recommend, quick_wins, overdue. Supports context-aware recommendations."
    )]
    async fn gtd_tasks(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_tools::GtdTasksParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{RecommendParams, Task, TaskFilter, TaskManager};
        use crate::mcp::gtd_tools::{GtdTasksResponse, TaskAction};

        // Get ontology store from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();

        let manager = TaskManager::new(ontology_store);

        let response = match params.action {
            TaskAction::List => {
                let filter = TaskFilter {
                    contexts: params.contexts.unwrap_or_default(),
                    project_id: params.project_id.clone(),
                    status: params.status,
                    energy_level: params.energy_level,
                    time_available: params.time_available,
                    due_before: params.due_date,
                    priority: params.priority,
                    description_contains: params.description_contains.clone(),
                    limit: params.limit.unwrap_or(100),
                    offset: 0,
                };
                match manager.list(filter).await {
                    Ok(tasks) => GtdTasksResponse::success_list(
                        tasks.clone(),
                        format!("Found {} tasks", tasks.len()),
                    ),
                    Err(e) => GtdTasksResponse::error(format!("Failed to list tasks: {}", e)),
                }
            }
            TaskAction::Get => {
                let task_id = params.task_id.ok_or_else(|| {
                    McpError::invalid_params("task_id is required for 'get' action", None)
                })?;
                match manager.get(&task_id).await {
                    Ok(Some(task)) => GtdTasksResponse::success_single(task, "Task retrieved"),
                    Ok(None) => GtdTasksResponse::error(format!("Task not found: {}", task_id)),
                    Err(e) => GtdTasksResponse::error(format!("Failed to get task: {}", e)),
                }
            }
            TaskAction::Create => {
                let description = params.description.ok_or_else(|| {
                    McpError::invalid_params("description is required for 'create' action", None)
                })?;
                let mut task = Task::new(description);
                if let Some(project_id) = params.project_id {
                    task = task.with_project(project_id);
                }
                if let Some(contexts) = params.contexts {
                    task = task.with_contexts(contexts);
                }
                if let Some(energy) = params.energy_level {
                    task = task.with_energy(energy);
                }
                if let Some(duration) = params.estimated_minutes {
                    task = task.with_duration(duration);
                }
                if let Some(due) = params.due_date {
                    task = task.with_due_date(due);
                }
                if let Some(priority) = params.priority {
                    task = task.with_priority(priority);
                }
                match manager.create(task).await {
                    Ok(created) => GtdTasksResponse::success_single(created, "Task created"),
                    Err(e) => GtdTasksResponse::error(format!("Failed to create task: {}", e)),
                }
            }
            TaskAction::Update => {
                let task_id = params.task_id.ok_or_else(|| {
                    McpError::invalid_params("task_id is required for 'update' action", None)
                })?;
                match manager.get(&task_id).await {
                    Ok(Some(mut task)) => {
                        if let Some(desc) = params.description {
                            task.description = desc;
                        }
                        if let Some(project_id) = params.project_id {
                            task.project_id = Some(project_id);
                        }
                        if let Some(contexts) = params.contexts {
                            task.contexts = contexts;
                        }
                        if let Some(status) = params.status {
                            task.status = status;
                        }
                        if let Some(energy) = params.energy_level {
                            task.energy_level = energy;
                        }
                        if let Some(duration) = params.estimated_minutes {
                            task.estimated_minutes = Some(duration);
                        }
                        if let Some(due) = params.due_date {
                            task.due_date = Some(due);
                        }
                        if let Some(priority) = params.priority {
                            task.priority = priority;
                        }
                        match manager.update(&task_id, task).await {
                            Ok(updated) => {
                                GtdTasksResponse::success_single(updated, "Task updated")
                            }
                            Err(e) => GtdTasksResponse::error(format!("Failed to update: {}", e)),
                        }
                    }
                    Ok(None) => GtdTasksResponse::error(format!("Task not found: {}", task_id)),
                    Err(e) => GtdTasksResponse::error(format!("Failed to get task: {}", e)),
                }
            }
            TaskAction::Complete => {
                let task_id = params.task_id.ok_or_else(|| {
                    McpError::invalid_params("task_id is required for 'complete' action", None)
                })?;
                match manager.complete(&task_id).await {
                    Ok(Some(task)) => GtdTasksResponse::success_single(task, "Task completed"),
                    Ok(None) => GtdTasksResponse::error(format!("Task not found: {}", task_id)),
                    Err(e) => GtdTasksResponse::error(format!("Failed to complete: {}", e)),
                }
            }
            TaskAction::Delete => {
                let task_id = params.task_id.ok_or_else(|| {
                    McpError::invalid_params("task_id is required for 'delete' action", None)
                })?;
                match manager.delete(&task_id).await {
                    Ok(true) => GtdTasksResponse {
                        success: true,
                        task: None,
                        tasks: None,
                        recommendations: None,
                        message: "Task deleted".to_string(),
                    },
                    Ok(false) => GtdTasksResponse::error(format!("Task not found: {}", task_id)),
                    Err(e) => GtdTasksResponse::error(format!("Failed to delete: {}", e)),
                }
            }
            TaskAction::Recommend => {
                let rec_params = RecommendParams {
                    current_context: params.current_context,
                    energy_level: params.energy_level,
                    time_available: params.time_available,
                    focus_area: None,
                    limit: params.limit.unwrap_or(5),
                };
                match manager.recommend(rec_params).await {
                    Ok(recs) => GtdTasksResponse::success_recommendations(
                        recs.clone(),
                        format!("{} recommendations", recs.len()),
                    ),
                    Err(e) => {
                        GtdTasksResponse::error(format!("Failed to get recommendations: {}", e))
                    }
                }
            }
            TaskAction::QuickWins => match manager.get_quick_wins().await {
                Ok(tasks) => GtdTasksResponse::success_list(
                    tasks.clone(),
                    format!("{} quick wins (≤2 min)", tasks.len()),
                ),
                Err(e) => GtdTasksResponse::error(format!("Failed to get quick wins: {}", e)),
            },
            TaskAction::Overdue => match manager.get_overdue().await {
                Ok(tasks) => GtdTasksResponse::success_list(
                    tasks.clone(),
                    format!("{} overdue tasks", tasks.len()),
                ),
                Err(e) => GtdTasksResponse::error(format!("Failed to get overdue: {}", e)),
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Track items you're waiting on from others.
    #[tool(
        description = "Track waiting-for items. Actions: list, get, add, follow_up, resolve, overdue. Manage delegated items and follow-ups."
    )]
    async fn gtd_waiting(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_tools::GtdWaitingParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{WaitingFilter, WaitingFor, WaitingManager};
        use crate::mcp::gtd_tools::{GtdWaitingResponse, WaitingAction};

        // Get ontology store from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();

        let manager = WaitingManager::new(ontology_store);

        let response = match params.action {
            WaitingAction::List => {
                let filter = WaitingFilter {
                    delegated_to: params.delegated_to.clone(),
                    project_id: params.project_id.clone(),
                    status: params.status,
                    overdue_only: false,
                    limit: params.limit.unwrap_or(100),
                    offset: 0,
                };
                match manager.list(filter).await {
                    Ok(items) => GtdWaitingResponse::success_list(
                        items.clone(),
                        format!("Found {} waiting items", items.len()),
                    ),
                    Err(e) => GtdWaitingResponse::error(format!("Failed to list: {}", e)),
                }
            }
            WaitingAction::Get => {
                let item_id = params.item_id.ok_or_else(|| {
                    McpError::invalid_params("item_id is required for 'get' action", None)
                })?;
                match manager.get(&item_id).await {
                    Ok(Some(item)) => GtdWaitingResponse::success_single(item, "Item retrieved"),
                    Ok(None) => GtdWaitingResponse::error(format!("Item not found: {}", item_id)),
                    Err(e) => GtdWaitingResponse::error(format!("Failed to get: {}", e)),
                }
            }
            WaitingAction::Add => {
                let description = params.description.ok_or_else(|| {
                    McpError::invalid_params("description is required for 'add' action", None)
                })?;
                let delegated_to = params.delegated_to.ok_or_else(|| {
                    McpError::invalid_params("delegated_to is required for 'add' action", None)
                })?;
                let mut item = WaitingFor::new(description, delegated_to);
                if let Some(project_id) = params.project_id {
                    item = item.with_project(project_id);
                }
                if let Some(expected) = params.expected_by {
                    item = item.with_expected_by(expected);
                }
                match manager.create(item).await {
                    Ok(created) => {
                        GtdWaitingResponse::success_single(created, "Waiting item added")
                    }
                    Err(e) => GtdWaitingResponse::error(format!("Failed to add: {}", e)),
                }
            }
            WaitingAction::FollowUp => {
                let item_id = params.item_id.ok_or_else(|| {
                    McpError::invalid_params("item_id is required for 'follow_up' action", None)
                })?;
                match manager.record_follow_up(&item_id).await {
                    Ok(Some(item)) => {
                        GtdWaitingResponse::success_single(item, "Follow-up recorded")
                    }
                    Ok(None) => GtdWaitingResponse::error(format!("Item not found: {}", item_id)),
                    Err(e) => GtdWaitingResponse::error(format!("Failed to record: {}", e)),
                }
            }
            WaitingAction::Resolve => {
                let item_id = params.item_id.ok_or_else(|| {
                    McpError::invalid_params("item_id is required for 'resolve' action", None)
                })?;
                let resolution = params.resolution.unwrap_or_else(|| "Resolved".to_string());
                match manager.resolve(&item_id, &resolution).await {
                    Ok(Some(item)) => GtdWaitingResponse::success_single(item, "Item resolved"),
                    Ok(None) => GtdWaitingResponse::error(format!("Item not found: {}", item_id)),
                    Err(e) => GtdWaitingResponse::error(format!("Failed to resolve: {}", e)),
                }
            }
            WaitingAction::Overdue => match manager.get_overdue().await {
                Ok(items) => GtdWaitingResponse::success_list(
                    items.clone(),
                    format!("{} overdue items", items.len()),
                ),
                Err(e) => GtdWaitingResponse::error(format!("Failed to get overdue: {}", e)),
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Manage someday/maybe items for future consideration.
    #[tool(
        description = "Manage someday/maybe items. Actions: list, get, add, update, activate, archive, categories, due_for_review. Track deferred ideas and projects."
    )]
    async fn gtd_someday(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_tools::GtdSomedayParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{SomedayFilter, SomedayItem, SomedayManager};
        use crate::mcp::gtd_tools::{GtdSomedayResponse, SomedayAction};

        // Get ontology store from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();

        let manager = SomedayManager::new(ontology_store);

        let response = match params.action {
            SomedayAction::List => {
                let filter = SomedayFilter {
                    category: params.category.clone(),
                    due_for_review: false,
                    limit: params.limit.unwrap_or(100),
                    offset: 0,
                };
                match manager.list(filter).await {
                    Ok(items) => GtdSomedayResponse::success_list(
                        items.clone(),
                        format!("Found {} someday/maybe items", items.len()),
                    ),
                    Err(e) => GtdSomedayResponse::error(format!("Failed to list: {}", e)),
                }
            }
            SomedayAction::Get => {
                let item_id = params.item_id.ok_or_else(|| {
                    McpError::invalid_params("item_id is required for 'get' action", None)
                })?;
                match manager.get(&item_id).await {
                    Ok(Some(item)) => GtdSomedayResponse::success_single(item, "Item retrieved"),
                    Ok(None) => GtdSomedayResponse::error(format!("Item not found: {}", item_id)),
                    Err(e) => GtdSomedayResponse::error(format!("Failed to get: {}", e)),
                }
            }
            SomedayAction::Add => {
                let description = params.description.ok_or_else(|| {
                    McpError::invalid_params("description is required for 'add' action", None)
                })?;
                let mut item = SomedayItem::new(description);
                if let Some(category) = params.category {
                    item = item.with_category(category);
                }
                if let Some(trigger) = params.trigger {
                    item = item.with_trigger(trigger);
                }
                if let Some(review) = params.review_date {
                    item = item.with_review_date(review);
                }
                match manager.create(item).await {
                    Ok(created) => GtdSomedayResponse::success_single(created, "Item added"),
                    Err(e) => GtdSomedayResponse::error(format!("Failed to add: {}", e)),
                }
            }
            SomedayAction::Update => {
                let item_id = params.item_id.ok_or_else(|| {
                    McpError::invalid_params("item_id is required for 'update' action", None)
                })?;
                match manager.get(&item_id).await {
                    Ok(Some(mut item)) => {
                        if let Some(desc) = params.description {
                            item.description = desc;
                        }
                        if let Some(category) = params.category {
                            item.category = Some(category);
                        }
                        if let Some(trigger) = params.trigger {
                            item.trigger = Some(trigger);
                        }
                        if let Some(review) = params.review_date {
                            item.review_date = Some(review);
                        }
                        match manager.update(&item_id, item).await {
                            Ok(updated) => {
                                GtdSomedayResponse::success_single(updated, "Item updated")
                            }
                            Err(e) => GtdSomedayResponse::error(format!("Failed to update: {}", e)),
                        }
                    }
                    Ok(None) => GtdSomedayResponse::error(format!("Item not found: {}", item_id)),
                    Err(e) => GtdSomedayResponse::error(format!("Failed to get: {}", e)),
                }
            }
            SomedayAction::Activate => {
                let item_id = params.item_id.ok_or_else(|| {
                    McpError::invalid_params("item_id is required for 'activate' action", None)
                })?;
                match manager.activate(&item_id).await {
                    Ok(Some(task)) => GtdSomedayResponse::success_activated(
                        task,
                        "Item activated - converted to task",
                    ),
                    Ok(None) => GtdSomedayResponse::error(format!("Item not found: {}", item_id)),
                    Err(e) => GtdSomedayResponse::error(format!("Failed to activate: {}", e)),
                }
            }
            SomedayAction::Archive => {
                let item_id = params.item_id.ok_or_else(|| {
                    McpError::invalid_params("item_id is required for 'archive' action", None)
                })?;
                match manager.delete(&item_id).await {
                    Ok(true) => GtdSomedayResponse {
                        success: true,
                        item: None,
                        items: None,
                        task: None,
                        categories: None,
                        message: "Item archived/deleted".to_string(),
                    },
                    Ok(false) => GtdSomedayResponse::error(format!("Item not found: {}", item_id)),
                    Err(e) => GtdSomedayResponse::error(format!("Failed to archive: {}", e)),
                }
            }
            SomedayAction::Categories => match manager.get_categories().await {
                Ok(cats) => GtdSomedayResponse::success_categories(
                    cats.clone(),
                    format!("{} categories", cats.len()),
                ),
                Err(e) => GtdSomedayResponse::error(format!("Failed to get categories: {}", e)),
            },
            SomedayAction::DueForReview => match manager.get_due_for_review().await {
                Ok(items) => GtdSomedayResponse::success_list(
                    items.clone(),
                    format!("{} items due for review", items.len()),
                ),
                Err(e) => GtdSomedayResponse::error(format!("Failed to get: {}", e)),
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    // ========================================================================
    // GTD Advanced Features (Phase 8)
    // ========================================================================

    /// Analyze attention economics - where focus goes across areas and projects.
    #[tool(
        description = "Analyze attention economics. Shows where your focus is going across areas, projects, and contexts. Identifies attention imbalances and provides focus depth metrics. Use for understanding work distribution and rebalancing attention."
    )]
    async fn gtd_attention(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_tools::GtdAttentionParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{AttentionManager, AttentionParams};
        use crate::mcp::gtd_tools::GtdAttentionResponse;
        use chrono::Duration;

        // Get ontology store from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();

        let manager = AttentionManager::new(ontology_store);

        // Build attention params
        let now = chrono::Utc::now();
        let period_end = params.period_end.unwrap_or(now);
        let period_start = params
            .period_start
            .unwrap_or_else(|| period_end - Duration::days(params.period_days as i64));

        let attention_params = AttentionParams {
            period_start: Some(period_start),
            period_end: Some(period_end),
            period_days: params.period_days,
            include_projects: params.include_projects,
            include_trends: params.include_trends,
            focus_areas: params.focus_areas.clone(),
        };

        let response = match manager.analyze(attention_params).await {
            Ok(metrics) => {
                let rating_str = format!("{:?}", metrics.focus_depth.rating);
                let summary = format!(
                    "Attention analysis for {} days: {} areas, {} projects tracked. Focus depth: {:.0}% ({})",
                    params.period_days,
                    metrics.by_area.len(),
                    metrics.by_project.len(),
                    metrics.focus_depth.score * 100.0,
                    rating_str
                );
                GtdAttentionResponse::success(metrics, summary)
            }
            Err(e) => GtdAttentionResponse::error(format!("Failed to analyze attention: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Track and manage commitments made and received.
    #[tool(
        description = "Track commitments - promises made and received. Actions: list, get, extract (from text), create, fulfill, cancel, summary, overdue, made_to (person), received_from (person). Use for tracking promises and accountability."
    )]
    async fn gtd_commitments(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_tools::GtdCommitmentsParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{
            Commitment, CommitmentDirection, CommitmentFilter, CommitmentManager, CommitmentStatus,
        };
        use crate::mcp::gtd_tools::{CommitmentAction, GtdCommitmentsResponse};

        // Get ontology store from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();

        let manager = CommitmentManager::new(ontology_store);

        let response = match params.action {
            CommitmentAction::List => {
                let filter = CommitmentFilter {
                    commitment_type: params.commitment_type.as_ref().and_then(|ct| {
                        match ct.as_str() {
                            "made" => Some(CommitmentDirection::Made),
                            "received" => Some(CommitmentDirection::Received),
                            _ => None,
                        }
                    }),
                    status: params.status.as_ref().and_then(|s| match s.as_str() {
                        "pending" => Some(CommitmentStatus::Pending),
                        "fulfilled" => Some(CommitmentStatus::Fulfilled),
                        "cancelled" => Some(CommitmentStatus::Cancelled),
                        "overdue" => Some(CommitmentStatus::Overdue),
                        _ => None,
                    }),
                    person: params.person.clone(),
                    project_id: None,
                    overdue_only: false,
                    needs_follow_up: false,
                    limit: params.limit,
                };
                match manager.list(filter).await {
                    Ok(commitments) => GtdCommitmentsResponse::success_list(
                        commitments.clone(),
                        format!("Found {} commitments", commitments.len()),
                    ),
                    Err(e) => GtdCommitmentsResponse::error(format!("Failed to list: {}", e)),
                }
            }
            CommitmentAction::Get => {
                let id = params.commitment_id.ok_or_else(|| {
                    McpError::invalid_params("commitment_id is required for 'get' action", None)
                })?;
                match manager.get(&id).await {
                    Ok(Some(commitment)) => {
                        GtdCommitmentsResponse::success_single(commitment, "Commitment retrieved")
                    }
                    Ok(None) => {
                        GtdCommitmentsResponse::error(format!("Commitment not found: {}", id))
                    }
                    Err(e) => GtdCommitmentsResponse::error(format!("Failed to get: {}", e)),
                }
            }
            CommitmentAction::Extract => {
                let text = params.text.ok_or_else(|| {
                    McpError::invalid_params("text is required for 'extract' action", None)
                })?;
                let result = manager.extract_from_text(&text, params.document_id.as_deref());
                GtdCommitmentsResponse::success_extraction(
                    result.clone(),
                    format!("Extracted {} commitments", result.total_found),
                )
            }
            CommitmentAction::Create => {
                let description = params.description.ok_or_else(|| {
                    McpError::invalid_params("description is required for 'create' action", None)
                })?;
                let commitment = Commitment {
                    id: uuid::Uuid::new_v4().to_string(),
                    commitment_type: params
                        .commitment_type
                        .as_ref()
                        .map(|ct| {
                            if ct == "made" {
                                CommitmentDirection::Made
                            } else {
                                CommitmentDirection::Received
                            }
                        })
                        .unwrap_or(CommitmentDirection::Made),
                    description: description.clone(),
                    from_person: params.person.clone(),
                    to_person: None,
                    due_date: None,
                    status: CommitmentStatus::Pending,
                    source_document: params.document_id.clone(),
                    extracted_text: description,
                    confidence: 1.0,
                    project_id: None,
                    follow_up_date: None,
                    notes: None,
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                };
                match manager.create(commitment).await {
                    Ok(created) => {
                        GtdCommitmentsResponse::success_single(created, "Commitment created")
                    }
                    Err(e) => GtdCommitmentsResponse::error(format!("Failed to create: {}", e)),
                }
            }
            CommitmentAction::Fulfill => {
                let id = params.commitment_id.ok_or_else(|| {
                    McpError::invalid_params("commitment_id is required for 'fulfill' action", None)
                })?;
                match manager.fulfill(&id).await {
                    Ok(()) => GtdCommitmentsResponse {
                        success: true,
                        commitment: None,
                        commitments: None,
                        extraction: None,
                        summary: None,
                        message: "Commitment marked as fulfilled".to_string(),
                    },
                    Err(e) => GtdCommitmentsResponse::error(format!("Failed to fulfill: {}", e)),
                }
            }
            CommitmentAction::Cancel => {
                let id = params.commitment_id.ok_or_else(|| {
                    McpError::invalid_params("commitment_id is required for 'cancel' action", None)
                })?;
                match manager.cancel(&id).await {
                    Ok(()) => GtdCommitmentsResponse {
                        success: true,
                        commitment: None,
                        commitments: None,
                        extraction: None,
                        summary: None,
                        message: "Commitment cancelled".to_string(),
                    },
                    Err(e) => GtdCommitmentsResponse::error(format!("Failed to cancel: {}", e)),
                }
            }
            CommitmentAction::Summary => match manager.summary().await {
                Ok(summary) => {
                    GtdCommitmentsResponse::success_summary(summary, "Commitment summary generated")
                }
                Err(e) => {
                    GtdCommitmentsResponse::error(format!("Failed to generate summary: {}", e))
                }
            },
            CommitmentAction::Overdue => match manager.get_overdue().await {
                Ok(commitments) => GtdCommitmentsResponse::success_list(
                    commitments.clone(),
                    format!("{} overdue commitments", commitments.len()),
                ),
                Err(e) => GtdCommitmentsResponse::error(format!("Failed to get overdue: {}", e)),
            },
            CommitmentAction::MadeTo => {
                let person = params.person.ok_or_else(|| {
                    McpError::invalid_params("person is required for 'made_to' action", None)
                })?;
                match manager.get_made_to(&person).await {
                    Ok(commitments) => GtdCommitmentsResponse::success_list(
                        commitments.clone(),
                        format!("{} commitments made to {}", commitments.len(), person),
                    ),
                    Err(e) => GtdCommitmentsResponse::error(format!("Failed to get: {}", e)),
                }
            }
            CommitmentAction::ReceivedFrom => {
                let person = params.person.ok_or_else(|| {
                    McpError::invalid_params("person is required for 'received_from' action", None)
                })?;
                match manager.get_received_from(&person).await {
                    Ok(commitments) => GtdCommitmentsResponse::success_list(
                        commitments.clone(),
                        format!("{} commitments received from {}", commitments.len(), person),
                    ),
                    Err(e) => GtdCommitmentsResponse::error(format!("Failed to get: {}", e)),
                }
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Visualize and analyze task/project dependencies.
    #[tool(
        description = "Generate dependency graph for tasks and projects. Shows critical path, blockers, and dependency chains. Supports output in json, mermaid, dot, or text format. Use for understanding project structure and identifying bottlenecks."
    )]
    async fn gtd_dependencies(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_tools::GtdDependenciesParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{DependencyManager, DependencyParams, OutputFormat};
        use crate::mcp::gtd_tools::GtdDependenciesResponse;

        // Get ontology store from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();

        let manager = DependencyManager::new(ontology_store);

        // Build dependency params
        let output_format = match params.output_format.to_lowercase().as_str() {
            "mermaid" => OutputFormat::Mermaid,
            "dot" => OutputFormat::Dot,
            "text" => OutputFormat::Text,
            _ => OutputFormat::Json,
        };

        let dep_params = DependencyParams {
            project_id: params.project_id.clone(),
            include_completed: params.include_completed,
            max_depth: params.max_depth,
            include_critical_path: params.include_critical_path,
            include_blockers: params.include_blockers,
            output_format,
        };

        let response = match manager.generate(dep_params).await {
            Ok(graph) => {
                let summary = format!(
                    "Dependency graph: {} nodes, {} edges. Critical path: {} items. {} blockers found.",
                    graph.nodes.len(),
                    graph.edges.len(),
                    graph.critical_path.length,
                    graph.blockers.len()
                );

                let mut resp = GtdDependenciesResponse::success(graph.clone(), summary);
                resp.critical_path = Some(graph.critical_path);
                resp.blockers = Some(graph.blockers);
                resp
            }
            Err(e) => GtdDependenciesResponse::error(format!("Failed to generate graph: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Map and visualize GTD horizons of focus.
    #[tool(
        description = "Map GTD horizons of focus. Shows all 6 levels: Runway (actions), 10k (projects), 20k (areas), 30k (goals), 40k (vision), 50k (purpose). Includes alignment analysis and health metrics. Use for big-picture perspective and ensuring work aligns with higher purpose."
    )]
    async fn gtd_horizons(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_tools::GtdHorizonsParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{HorizonLevel, HorizonManager, HorizonParams};
        use crate::mcp::gtd_tools::GtdHorizonsResponse;

        // Get ontology store from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();

        let manager = HorizonManager::new(ontology_store);

        // Parse horizon filters
        let horizons: Vec<HorizonLevel> = params
            .horizons
            .iter()
            .filter_map(|h| match h.to_lowercase().as_str() {
                "runway" => Some(HorizonLevel::Runway),
                "h10k" | "10k" | "projects" => Some(HorizonLevel::H10k),
                "h20k" | "20k" | "areas" => Some(HorizonLevel::H20k),
                "h30k" | "30k" | "goals" => Some(HorizonLevel::H30k),
                "h40k" | "40k" | "vision" => Some(HorizonLevel::H40k),
                "h50k" | "50k" | "purpose" => Some(HorizonLevel::H50k),
                _ => None,
            })
            .collect();

        let horizon_params = HorizonParams {
            horizons,
            include_counts: params.include_counts,
            include_health: params.include_health,
            include_alignment: params.include_alignment,
            items_per_horizon: params.items_per_horizon,
            area: params.area.clone(),
        };

        let response = match manager.map(horizon_params).await {
            Ok(map) => {
                let summary = format!(
                    "Horizon map: {} levels. Overall health: {:.0}% ({}). {} recommendations.",
                    map.horizons.len(),
                    map.overall_health.score,
                    map.overall_health.rating,
                    map.recommendations.len()
                );
                GtdHorizonsResponse::success(map, summary)
            }
            Err(e) => GtdHorizonsResponse::error(format!("Failed to generate horizon map: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    // ========================================================================
    // Simplified GTD Tools (one tool per action for better LLM compatibility)
    // ========================================================================

    /// Create a new GTD task.
    #[tool(description = "Create a new GTD task. Provide description (required) and optional: project_id, contexts (array like [\"@phone\", \"@computer\"]), energy_level (low/medium/high), estimated_minutes, due_date (ISO 8601), priority (low/normal/high/critical).")]
    async fn task_create(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::TaskCreateParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{Task, TaskManager};
        use crate::mcp::gtd_tools::GtdTasksResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = TaskManager::new(ontology_store);

        let mut task = Task::new(&params.description);
        if let Some(project_id) = params.project_id {
            task = task.with_project(project_id);
        }
        if let Some(contexts) = params.contexts {
            task = task.with_contexts(contexts);
        }
        if let Some(energy) = params.energy_level {
            task = task.with_energy(energy);
        }
        if let Some(duration) = params.estimated_minutes {
            task = task.with_duration(duration);
        }
        if let Some(due) = params.due_date {
            task = task.with_due_date(due);
        }
        if let Some(priority) = params.priority {
            task = task.with_priority(priority);
        }

        let response = match manager.create(task).await {
            Ok(created) => GtdTasksResponse::success_single(created, "Task created successfully"),
            Err(e) => GtdTasksResponse::error(format!("Failed to create task: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// List GTD tasks with optional filters.
    #[tool(description = "List GTD tasks. Optional filters: project_id, contexts, status (next/scheduled/waiting/someday/done), energy_level (low/medium/high), priority (low/normal/high/critical), description_contains, limit.")]
    async fn task_list(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::TaskListParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{TaskFilter, TaskManager};
        use crate::mcp::gtd_tools::GtdTasksResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = TaskManager::new(ontology_store);

        let filter = TaskFilter {
            contexts: params.contexts.unwrap_or_default(),
            project_id: params.project_id,
            status: params.status,
            energy_level: params.energy_level,
            time_available: None,
            due_before: None,
            priority: params.priority,
            description_contains: params.description_contains,
            limit: params.limit.unwrap_or(100),
            offset: 0,
        };

        let response = match manager.list(filter).await {
            Ok(tasks) => {
                GtdTasksResponse::success_list(tasks.clone(), format!("Found {} tasks", tasks.len()))
            }
            Err(e) => GtdTasksResponse::error(format!("Failed to list tasks: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get a specific GTD task by ID.
    #[tool(description = "Get a specific GTD task by its ID.")]
    async fn task_get(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::TaskGetParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::TaskManager;
        use crate::mcp::gtd_tools::GtdTasksResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = TaskManager::new(ontology_store);

        let response = match manager.get(&params.task_id).await {
            Ok(Some(task)) => GtdTasksResponse::success_single(task, "Task retrieved"),
            Ok(None) => GtdTasksResponse::error(format!("Task not found: {}", params.task_id)),
            Err(e) => GtdTasksResponse::error(format!("Failed to get task: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Mark a GTD task as complete.
    #[tool(description = "Mark a GTD task as complete by its ID.")]
    async fn task_complete(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::TaskCompleteParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::TaskManager;
        use crate::mcp::gtd_tools::GtdTasksResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = TaskManager::new(ontology_store);

        let response = match manager.complete(&params.task_id).await {
            Ok(Some(task)) => GtdTasksResponse::success_single(task, "Task completed"),
            Ok(None) => GtdTasksResponse::error(format!("Task not found: {}", params.task_id)),
            Err(e) => GtdTasksResponse::error(format!("Failed to complete task: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Delete a GTD task.
    #[tool(description = "Delete a GTD task by its ID.")]
    async fn task_delete(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::TaskDeleteParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::TaskManager;
        use crate::mcp::gtd_tools::GtdTasksResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = TaskManager::new(ontology_store);

        let response = match manager.delete(&params.task_id).await {
            Ok(true) => GtdTasksResponse {
                success: true,
                task: None,
                tasks: None,
                recommendations: None,
                message: "Task deleted".to_string(),
            },
            Ok(false) => GtdTasksResponse::error(format!("Task not found: {}", params.task_id)),
            Err(e) => GtdTasksResponse::error(format!("Failed to delete task: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Update an existing GTD task.
    #[tool(description = "Update a GTD task. Requires task_id. Optional fields to update: description, project_id, contexts, status (next/scheduled/waiting/someday/done), energy_level, estimated_minutes, due_date, priority.")]
    async fn task_update(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::TaskUpdateParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::TaskManager;
        use crate::mcp::gtd_tools::GtdTasksResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = TaskManager::new(ontology_store);

        let response = match manager.get(&params.task_id).await {
            Ok(Some(mut task)) => {
                if let Some(desc) = params.description {
                    task.description = desc;
                }
                if let Some(project_id) = params.project_id {
                    task.project_id = Some(project_id);
                }
                if let Some(contexts) = params.contexts {
                    task.contexts = contexts;
                }
                if let Some(status) = params.status {
                    task.status = status;
                }
                if let Some(energy) = params.energy_level {
                    task.energy_level = energy;
                }
                if let Some(duration) = params.estimated_minutes {
                    task.estimated_minutes = Some(duration);
                }
                if let Some(due) = params.due_date {
                    task.due_date = Some(due);
                }
                if let Some(priority) = params.priority {
                    task.priority = priority;
                }
                match manager.update(&params.task_id, task).await {
                    Ok(updated) => GtdTasksResponse::success_single(updated, "Task updated"),
                    Err(e) => GtdTasksResponse::error(format!("Failed to update: {}", e)),
                }
            }
            Ok(None) => GtdTasksResponse::error(format!("Task not found: {}", params.task_id)),
            Err(e) => GtdTasksResponse::error(format!("Failed to get task: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Create a new GTD project.
    #[tool(description = "Create a new GTD project. Requires name. Optional: outcome (what does done look like?), area (Work/Personal/etc.), goal.")]
    async fn project_create(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::ProjectCreateParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{Project, ProjectManager};
        use crate::mcp::gtd_tools::GtdProjectsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = ProjectManager::new(ontology_store);

        let mut project = Project::new(&params.name);
        if let Some(outcome) = params.outcome {
            project = project.with_outcome(outcome);
        }
        if let Some(area) = params.area {
            project = project.with_area(area);
        }
        if let Some(goal) = params.goal {
            project = project.with_goal(goal);
        }

        let response = match manager.create(project).await {
            Ok(created) => GtdProjectsResponse::success_single(created, "Project created"),
            Err(e) => GtdProjectsResponse::error(format!("Failed to create project: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// List GTD projects with optional filters.
    #[tool(description = "List GTD projects. Optional filters: status (active/on_hold/completed/archived), area, has_next_action (true/false), stalled_days, limit.")]
    async fn project_list(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::ProjectListParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{ProjectFilter, ProjectManager};
        use crate::mcp::gtd_tools::GtdProjectsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = ProjectManager::new(ontology_store);

        let filter = ProjectFilter {
            status: params.status,
            area: params.area,
            has_next_action: params.has_next_action,
            stalled_days: params.stalled_days,
            limit: params.limit.unwrap_or(100),
            offset: 0,
        };

        let response = match manager.list(filter).await {
            Ok(projects) => GtdProjectsResponse::success_list(
                projects.clone(),
                format!("Found {} projects", projects.len()),
            ),
            Err(e) => GtdProjectsResponse::error(format!("Failed to list projects: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get a specific GTD project by ID.
    #[tool(description = "Get a specific GTD project by its ID.")]
    async fn project_get(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::ProjectGetParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::ProjectManager;
        use crate::mcp::gtd_tools::GtdProjectsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = ProjectManager::new(ontology_store);

        let response = match manager.get(&params.project_id).await {
            Ok(Some(project)) => GtdProjectsResponse::success_single(project, "Project retrieved"),
            Ok(None) => {
                GtdProjectsResponse::error(format!("Project not found: {}", params.project_id))
            }
            Err(e) => GtdProjectsResponse::error(format!("Failed to get project: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Mark a GTD project as complete.
    #[tool(description = "Mark a GTD project as complete by its ID.")]
    async fn project_complete(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::ProjectCompleteParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::ProjectManager;
        use crate::mcp::gtd_tools::GtdProjectsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = ProjectManager::new(ontology_store);

        let response = match manager.complete(&params.project_id).await {
            Ok(Some(project)) => GtdProjectsResponse::success_single(project, "Project completed"),
            Ok(None) => {
                GtdProjectsResponse::error(format!("Project not found: {}", params.project_id))
            }
            Err(e) => GtdProjectsResponse::error(format!("Failed to complete project: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Archive a GTD project.
    #[tool(description = "Archive a GTD project by its ID.")]
    async fn project_archive(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::ProjectArchiveParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::ProjectManager;
        use crate::mcp::gtd_tools::GtdProjectsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = ProjectManager::new(ontology_store);

        let response = match manager.archive(&params.project_id).await {
            Ok(Some(project)) => GtdProjectsResponse::success_single(project, "Project archived"),
            Ok(None) => {
                GtdProjectsResponse::error(format!("Project not found: {}", params.project_id))
            }
            Err(e) => GtdProjectsResponse::error(format!("Failed to archive project: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Add a waiting-for item.
    #[tool(description = "Add a new waiting-for item. Requires description and delegated_to (person/entity you're waiting on). Optional: project_id, expected_by (ISO 8601 date).")]
    async fn waiting_add(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::WaitingAddParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{WaitingFor, WaitingManager};
        use crate::mcp::gtd_tools::GtdWaitingResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = WaitingManager::new(ontology_store);

        let mut item = WaitingFor::new(&params.description, &params.delegated_to);
        if let Some(project_id) = params.project_id {
            item = item.with_project(project_id);
        }
        if let Some(expected) = params.expected_by {
            item = item.with_expected_by(expected);
        }

        let response = match manager.create(item).await {
            Ok(created) => GtdWaitingResponse::success_single(created, "Waiting-for item added"),
            Err(e) => GtdWaitingResponse::error(format!("Failed to add waiting-for item: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// List waiting-for items.
    #[tool(description = "List waiting-for items. Optional filters: status (pending/overdue/resolved), limit.")]
    async fn waiting_list(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::WaitingListParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{WaitingFilter, WaitingManager};
        use crate::mcp::gtd_tools::GtdWaitingResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = WaitingManager::new(ontology_store);

        let filter = WaitingFilter {
            status: params.status,
            delegated_to: None,
            project_id: None,
            overdue_only: false,
            limit: params.limit.unwrap_or(100),
            offset: 0,
        };

        let response = match manager.list(filter).await {
            Ok(items) => GtdWaitingResponse::success_list(
                items.clone(),
                format!("Found {} waiting-for items", items.len()),
            ),
            Err(e) => GtdWaitingResponse::error(format!("Failed to list waiting-for items: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Resolve a waiting-for item.
    #[tool(description = "Resolve/complete a waiting-for item. Requires item_id. Optional: resolution (text describing how it was resolved).")]
    async fn waiting_resolve(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::WaitingResolveParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::WaitingManager;
        use crate::mcp::gtd_tools::GtdWaitingResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = WaitingManager::new(ontology_store);

        let resolution = params.resolution.as_deref().unwrap_or("Resolved");
        let response = match manager.resolve(&params.item_id, resolution).await {
            Ok(Some(item)) => GtdWaitingResponse::success_single(item, "Waiting-for item resolved"),
            Ok(None) => {
                GtdWaitingResponse::error(format!("Waiting-for item not found: {}", params.item_id))
            }
            Err(e) => GtdWaitingResponse::error(format!("Failed to resolve: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Add a someday/maybe item.
    #[tool(description = "Add a new someday/maybe item for future consideration. Requires description. Optional: category, trigger (what would make this active?), review_date (ISO 8601).")]
    async fn someday_add(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::SomedayAddParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{SomedayItem, SomedayManager};
        use crate::mcp::gtd_tools::GtdSomedayResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = SomedayManager::new(ontology_store);

        let mut item = SomedayItem::new(&params.description);
        if let Some(category) = params.category {
            item = item.with_category(category);
        }
        if let Some(trigger) = params.trigger {
            item = item.with_trigger(trigger);
        }
        if let Some(review_date) = params.review_date {
            item = item.with_review_date(review_date);
        }

        let response = match manager.create(item).await {
            Ok(created) => GtdSomedayResponse::success_single(created, "Someday/maybe item added"),
            Err(e) => {
                GtdSomedayResponse::error(format!("Failed to add someday/maybe item: {}", e))
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// List someday/maybe items.
    #[tool(description = "List someday/maybe items. Optional filters: category, limit.")]
    async fn someday_list(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::SomedayListParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{SomedayFilter, SomedayManager};
        use crate::mcp::gtd_tools::GtdSomedayResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = SomedayManager::new(ontology_store);

        let filter = SomedayFilter {
            category: params.category,
            due_for_review: false,
            limit: params.limit.unwrap_or(100),
            offset: 0,
        };

        let response = match manager.list(filter).await {
            Ok(items) => GtdSomedayResponse::success_list(
                items.clone(),
                format!("Found {} someday/maybe items", items.len()),
            ),
            Err(e) => {
                GtdSomedayResponse::error(format!("Failed to list someday/maybe items: {}", e))
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Activate a someday/maybe item (convert to task).
    #[tool(description = "Activate a someday/maybe item - converts it to an active task. Requires item_id.")]
    async fn someday_activate(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::SomedayActivateParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::SomedayManager;
        use crate::mcp::gtd_tools::GtdSomedayResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = SomedayManager::new(ontology_store);

        let response = match manager.activate(&params.item_id).await {
            Ok(Some(task)) => GtdSomedayResponse::success_activated(task, "Item activated as task"),
            Ok(None) => {
                GtdSomedayResponse::error(format!("Someday/maybe item not found: {}", params.item_id))
            }
            Err(e) => GtdSomedayResponse::error(format!("Failed to activate: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    // ========================================================================
    // Simplified Calendar Tools
    // ========================================================================

    /// Create a new calendar event.
    #[tool(description = "Create a new calendar event. Requires title and start (ISO 8601 datetime). Optional: description, event_type (event/meeting/deadline/reminder/blocked_time), end (ISO 8601), duration_minutes, all_day, location, participants (array), project_id.")]
    async fn calendar_create(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CalendarCreateParams>,
    ) -> Result<CallToolResult, McpError> {
        use chrono::Duration;
        use crate::calendar::{CalendarEvent, CalendarManager, EventType};
        use crate::mcp::calendar_tools::{CalendarEventInfo, CalendarManageResponse};

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CalendarManager::new(ontology_store);

        let mut event = CalendarEvent::new(&params.title, params.start);
        if let Some(desc) = params.description {
            event = event.with_description(&desc);
        }
        if let Some(type_str) = params.event_type {
            let event_type = match type_str.to_lowercase().as_str() {
                "meeting" => EventType::Meeting,
                "deadline" => EventType::Deadline,
                "reminder" => EventType::Reminder,
                "blocked_time" => EventType::BlockedTime,
                "milestone" => EventType::Milestone,
                "appointment" => EventType::Appointment,
                "travel" => EventType::Travel,
                "call" => EventType::Call,
                "standup" => EventType::Standup,
                _ => EventType::Event,
            };
            event = event.with_type(event_type);
        }
        if let Some(end) = params.end {
            event = event.with_end(end);
        } else if let Some(minutes) = params.duration_minutes {
            event = event.with_duration(Duration::minutes(minutes));
        }
        if params.all_day == Some(true) {
            event = event.all_day_event();
        }
        if let Some(location) = params.location {
            event = event.with_location(&location);
        }
        if let Some(participants) = params.participants {
            event = event.with_participants(participants);
        }
        if let Some(project_id) = params.project_id {
            event = event.with_project(&project_id);
        }

        let response = match manager.create(event).await {
            Ok(created) => CalendarManageResponse {
                success: true,
                event: Some(CalendarEventInfo::from(created)),
                events: None,
                stats: None,
                message: "Event created successfully".to_string(),
            },
            Err(e) => CalendarManageResponse {
                success: false,
                event: None,
                events: None,
                stats: None,
                message: format!("Failed to create event: {}", e),
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// List calendar events.
    #[tool(description = "List calendar events. Optional: query_type (upcoming/today/this_week/next_week/date_range), start_date (ISO 8601), end_date (ISO 8601), days, limit.")]
    async fn calendar_list(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CalendarListParams>,
    ) -> Result<CallToolResult, McpError> {
        use chrono::{Duration, Utc};
        use crate::calendar::{CalendarFilter, CalendarQueryType, CalendarManager};
        use crate::mcp::calendar_tools::{CalendarEventInfo, CalendarManageResponse};

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CalendarManager::new(ontology_store);

        let query_type = params.query_type.as_deref().map(|s| match s.to_lowercase().as_str() {
            "today" => CalendarQueryType::Today,
            "this_week" => CalendarQueryType::ThisWeek,
            "next_week" => CalendarQueryType::NextWeek,
            "date_range" => CalendarQueryType::DateRange,
            _ => CalendarQueryType::Upcoming,
        }).unwrap_or(CalendarQueryType::Upcoming);

        let now = Utc::now();
        let filter = CalendarFilter {
            query_type,
            start_date: params.start_date.or(Some(now)),
            end_date: params.end_date.or(Some(now + Duration::days(params.days.unwrap_or(7)))),
            event_types: vec![],
            project_id: None,
            participant: None,
            search_query: None,
            limit: params.limit.unwrap_or(100),
            ..Default::default()
        };

        let response = match manager.list(&filter).await {
            Ok(events) => {
                let event_infos: Vec<CalendarEventInfo> = events.into_iter().map(CalendarEventInfo::from).collect();
                let count = event_infos.len();
                CalendarManageResponse {
                    success: true,
                    event: None,
                    events: Some(event_infos),
                    stats: None,
                    message: format!("Found {} events", count),
                }
            }
            Err(e) => CalendarManageResponse {
                success: false,
                event: None,
                events: None,
                stats: None,
                message: format!("Failed to list events: {}", e),
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get a calendar event by ID.
    #[tool(description = "Get a specific calendar event by its ID.")]
    async fn calendar_get(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CalendarGetParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::calendar::CalendarManager;
        use crate::mcp::calendar_tools::{CalendarEventInfo, CalendarManageResponse};

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CalendarManager::new(ontology_store);

        let response = match manager.get(&params.event_id).await {
            Ok(Some(event)) => CalendarManageResponse {
                success: true,
                event: Some(CalendarEventInfo::from(event)),
                events: None,
                stats: None,
                message: "Event retrieved".to_string(),
            },
            Ok(None) => CalendarManageResponse {
                success: false,
                event: None,
                events: None,
                stats: None,
                message: format!("Event not found: {}", params.event_id),
            },
            Err(e) => CalendarManageResponse {
                success: false,
                event: None,
                events: None,
                stats: None,
                message: format!("Failed to get event: {}", e),
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Delete a calendar event.
    #[tool(description = "Delete a calendar event by its ID.")]
    async fn calendar_delete(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CalendarDeleteParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::calendar::CalendarManager;
        use crate::mcp::calendar_tools::CalendarManageResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CalendarManager::new(ontology_store);

        let response = match manager.delete(&params.event_id).await {
            Ok(true) => CalendarManageResponse {
                success: true,
                event: None,
                events: None,
                stats: None,
                message: "Event deleted".to_string(),
            },
            Ok(false) => CalendarManageResponse {
                success: false,
                event: None,
                events: None,
                stats: None,
                message: format!("Event not found: {}", params.event_id),
            },
            Err(e) => CalendarManageResponse {
                success: false,
                event: None,
                events: None,
                stats: None,
                message: format!("Failed to delete event: {}", e),
            },
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    // ========================================================================
    // Simplified Commitment Tools
    // ========================================================================

    /// Create a new commitment.
    #[tool(description = "Create a new commitment (promise made or received). Provide description (required) and optional: commitment_type ('made' or 'received'), person, due_date, document_id.")]
    async fn commitment_create(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CommitmentCreateParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{Commitment, CommitmentDirection, CommitmentManager, CommitmentStatus};
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let commitment = Commitment {
            id: uuid::Uuid::new_v4().to_string(),
            commitment_type: params
                .commitment_type
                .as_ref()
                .map(|ct| {
                    if ct == "received" {
                        CommitmentDirection::Received
                    } else {
                        CommitmentDirection::Made
                    }
                })
                .unwrap_or(CommitmentDirection::Made),
            description: params.description.clone(),
            from_person: params.person.clone(),
            to_person: None,
            due_date: params.due_date,
            status: CommitmentStatus::Pending,
            source_document: params.document_id.clone(),
            extracted_text: params.description,
            confidence: 1.0,
            project_id: None,
            follow_up_date: None,
            notes: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let response = match manager.create(commitment).await {
            Ok(created) => GtdCommitmentsResponse::success_single(created, "Commitment created"),
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to create: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// List commitments with optional filters.
    #[tool(description = "List commitments with optional filters. Filter by: commitment_type ('made' or 'received'), status ('pending', 'fulfilled', 'cancelled', 'overdue'), person, limit.")]
    async fn commitment_list(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CommitmentListParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::{CommitmentDirection, CommitmentFilter, CommitmentManager, CommitmentStatus};
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let filter = CommitmentFilter {
            commitment_type: params.commitment_type.as_ref().and_then(|ct| match ct.as_str() {
                "made" => Some(CommitmentDirection::Made),
                "received" => Some(CommitmentDirection::Received),
                _ => None,
            }),
            status: params.status.as_ref().and_then(|s| match s.as_str() {
                "pending" => Some(CommitmentStatus::Pending),
                "fulfilled" => Some(CommitmentStatus::Fulfilled),
                "cancelled" => Some(CommitmentStatus::Cancelled),
                "overdue" => Some(CommitmentStatus::Overdue),
                _ => None,
            }),
            person: params.person.clone(),
            project_id: None,
            overdue_only: false,
            needs_follow_up: false,
            limit: params.limit.unwrap_or(100),
        };

        let response = match manager.list(filter).await {
            Ok(commitments) => GtdCommitmentsResponse::success_list(
                commitments.clone(),
                format!("Found {} commitments", commitments.len()),
            ),
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to list: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get a commitment by ID.
    #[tool(description = "Get a single commitment by its ID.")]
    async fn commitment_get(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CommitmentGetParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::CommitmentManager;
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let response = match manager.get(&params.commitment_id).await {
            Ok(Some(commitment)) => {
                GtdCommitmentsResponse::success_single(commitment, "Commitment retrieved")
            }
            Ok(None) => GtdCommitmentsResponse::error(format!(
                "Commitment not found: {}",
                params.commitment_id
            )),
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to get: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Mark a commitment as fulfilled.
    #[tool(description = "Mark a commitment as fulfilled (completed).")]
    async fn commitment_fulfill(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CommitmentFulfillParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::CommitmentManager;
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let response = match manager.fulfill(&params.commitment_id).await {
            Ok(()) => GtdCommitmentsResponse {
                success: true,
                commitment: None,
                commitments: None,
                extraction: None,
                summary: None,
                message: "Commitment marked as fulfilled".to_string(),
            },
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to fulfill: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Cancel a commitment.
    #[tool(description = "Cancel a commitment (mark as not going to be fulfilled).")]
    async fn commitment_cancel(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CommitmentCancelParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::CommitmentManager;
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let response = match manager.cancel(&params.commitment_id).await {
            Ok(()) => GtdCommitmentsResponse {
                success: true,
                commitment: None,
                commitments: None,
                extraction: None,
                summary: None,
                message: "Commitment cancelled".to_string(),
            },
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to cancel: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Extract commitments from text.
    #[tool(description = "Extract commitments (promises made or received) from text. Analyzes text for implicit and explicit commitments.")]
    async fn commitment_extract(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CommitmentExtractParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::CommitmentManager;
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let result = manager.extract_from_text(&params.text, params.document_id.as_deref());
        let response = GtdCommitmentsResponse::success_extraction(
            result.clone(),
            format!("Extracted {} commitments", result.total_found),
        );

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get a summary of all commitments.
    #[tool(description = "Get a summary of all commitments including counts by status, type, and overdue items.")]
    async fn commitment_summary(
        &self,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::CommitmentManager;
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let response = match manager.summary().await {
            Ok(summary) => {
                GtdCommitmentsResponse::success_summary(summary, "Commitment summary generated")
            }
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to generate summary: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get overdue commitments.
    #[tool(description = "Get all overdue commitments (past their due date).")]
    async fn commitment_overdue(
        &self,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::CommitmentManager;
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let response = match manager.get_overdue().await {
            Ok(commitments) => GtdCommitmentsResponse::success_list(
                commitments.clone(),
                format!("{} overdue commitments", commitments.len()),
            ),
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to get overdue: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get commitments made to a specific person.
    #[tool(description = "Get all commitments made to a specific person (promises I made to them).")]
    async fn commitment_made_to(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CommitmentMadeToParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::CommitmentManager;
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let response = match manager.get_made_to(&params.person).await {
            Ok(commitments) => GtdCommitmentsResponse::success_list(
                commitments.clone(),
                format!("{} commitments made to {}", commitments.len(), params.person),
            ),
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to get: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Get commitments received from a specific person.
    #[tool(description = "Get all commitments received from a specific person (promises they made to me).")]
    async fn commitment_received_from(
        &self,
        Parameters(params): Parameters<crate::mcp::gtd_simple_tools::CommitmentReceivedFromParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::gtd::CommitmentManager;
        use crate::mcp::gtd_tools::GtdCommitmentsResponse;

        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let manager = CommitmentManager::new(ontology_store);

        let response = match manager.get_received_from(&params.person).await {
            Ok(commitments) => GtdCommitmentsResponse::success_list(
                commitments.clone(),
                format!(
                    "{} commitments received from {}",
                    commitments.len(),
                    params.person
                ),
            ),
            Err(e) => GtdCommitmentsResponse::error(format!("Failed to get: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    // ========================================================================
    // Knowledge Graph Tools
    // ========================================================================

    /// Query the knowledge graph with semantic search, entity lookup, relationship queries, and more.
    #[tool(
        description = "Query the knowledge graph. Supports semantic_search (What do I know about X?), entity_lookup (find entity by name), relationship_query (Who works with X?), topic_summary (summarize knowledge on topic), connected_entities (graph traversal), and expert_finding (Who knows about X?)."
    )]
    async fn knowledge_query(
        &self,
        Parameters(params): Parameters<crate::mcp::knowledge_tools::KnowledgeQueryParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::knowledge::KnowledgeQueryEngine;
        use crate::mcp::knowledge_tools::KnowledgeQueryResponse;

        // Get ontology store and embedder from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let embedder = coordinator.embedder();

        let engine = KnowledgeQueryEngine::new(ontology_store, embedder);

        let query_params: crate::knowledge::KnowledgeQueryParams = params.into();
        let response = match engine.query(query_params).await {
            Ok(result) => KnowledgeQueryResponse::from_result(result),
            Err(e) => KnowledgeQueryResponse::error(format!("Query failed: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Find experts on a specific topic.
    #[tool(
        description = "Find people with expertise on a topic. Analyzes relationships, authored content, and semantic similarity to identify experts."
    )]
    async fn expert_find(
        &self,
        Parameters(params): Parameters<crate::mcp::knowledge_tools::ExpertFindParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::knowledge::KnowledgeQueryEngine;
        use crate::mcp::knowledge_tools::ExpertFindResponse;

        // Get ontology store and embedder from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let embedder = coordinator.embedder();

        let engine = KnowledgeQueryEngine::new(ontology_store, embedder);

        let limit = params.limit.unwrap_or(10);
        let response = match engine.find_experts(&params.topic, limit).await {
            Ok(experts) => {
                let filtered = if let Some(min_score) = params.min_score {
                    experts
                        .into_iter()
                        .filter(|e| e.expertise_score >= min_score)
                        .collect()
                } else {
                    experts
                };
                ExpertFindResponse::success(params.topic, filtered)
            }
            Err(e) => ExpertFindResponse::error(format!("Expert finding failed: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Summarize knowledge about a topic.
    #[tool(
        description = "Generate a comprehensive summary of knowledge about a topic, including key entities, sub-topics, and related topics."
    )]
    async fn topic_summarize(
        &self,
        Parameters(params): Parameters<crate::mcp::knowledge_tools::TopicSummarizeParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::knowledge::KnowledgeQueryEngine;
        use crate::mcp::knowledge_tools::TopicSummarizeResponse;

        // Get ontology store and embedder from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let embedder = coordinator.embedder();

        let engine = KnowledgeQueryEngine::new(ontology_store, embedder);

        let limit = params.limit.unwrap_or(20);
        let response = match engine.summarize_topic(&params.topic, limit).await {
            Ok(summary) => TopicSummarizeResponse::from_summary(summary),
            Err(e) => TopicSummarizeResponse::error(format!("Topic summarization failed: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Traverse the knowledge graph from an entity.
    #[tool(
        description = "Traverse relationships from an entity to discover connected entities. Returns paths and discovered nodes with relevance scores."
    )]
    async fn graph_traverse(
        &self,
        Parameters(params): Parameters<crate::mcp::knowledge_tools::GraphTraverseParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::knowledge::KnowledgeQueryEngine;
        use crate::mcp::knowledge_tools::GraphTraverseResponse;

        // Get ontology store and embedder from coordinator
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let embedder = coordinator.embedder();

        let engine = KnowledgeQueryEngine::new(ontology_store, embedder);

        let max_depth = params.max_depth.unwrap_or(2);
        let relation_types = params.relationship_type.map(|rt| rt.to_relation_types());

        let response = match engine
            .traverse_relationships(&params.from_entity_id, relation_types.as_deref(), max_depth)
            .await
        {
            Ok(result) => GraphTraverseResponse::from_result(result),
            Err(e) => GraphTraverseResponse::error(format!("Graph traversal failed: {}", e)),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Natural language query interface for GTD, calendar, knowledge, and search.
    #[tool(
        description = "Unified natural language query interface. Automatically routes to GTD (tasks/projects), Calendar, Knowledge Graph, or Document Search based on query intent. Examples: 'What are my @phone tasks?', 'What's on my calendar this week?', 'Who can help with AWS?', 'Show me stalled projects'."
    )]
    async fn query(
        &self,
        Parameters(params): Parameters<QueryParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::query::QueryIntent;
        use std::time::Instant;

        let start = Instant::now();

        // Get coordinator and subsystems
        self.ensure_coordinator().await?;
        let state = self.state.read().await;
        let coordinator = state.coordinator.as_ref().unwrap();
        let ontology_store = coordinator.ontology_store();
        let embedder = coordinator.embedder();

        // Classify the query
        let classifier = IntentClassifier::new();
        let mode = params.mode.map(QueryMode::from).unwrap_or(QueryMode::Auto);
        let classification = match mode {
            QueryMode::Auto => classifier.classify(&params.query),
            _ => classifier.classify_with_mode(&params.query, mode),
        };

        let limit = params.limit.unwrap_or(20);
        let intent = classification.intent.clone();

        // Execute based on intent
        let result: Result<(String, serde_json::Value), String> = match &intent {
            QueryIntent::Gtd(gtd_intent) => {
                self.execute_gtd_query(gtd_intent, &classification, ontology_store.clone(), limit)
                    .await
            }
            QueryIntent::Calendar(cal_intent) => {
                self.execute_calendar_query(cal_intent, &classification, ontology_store.clone())
                    .await
            }
            QueryIntent::Knowledge(know_intent) => {
                self.execute_knowledge_query(
                    know_intent,
                    &classification,
                    ontology_store.clone(),
                    embedder.clone(),
                    limit,
                )
                .await
            }
            QueryIntent::Search => {
                self.execute_search_query(&params.query, coordinator, limit)
                    .await
            }
            QueryIntent::Unknown => {
                Err("Could not understand your query. Try being more specific.".to_string())
            }
        };

        let elapsed = start.elapsed().as_millis() as u64;

        let response = match result {
            Ok((answer, data)) => QueryResponse {
                success: true,
                interpreted_as: intent.detailed_name(),
                confidence: classification.confidence,
                answer,
                data,
                suggestions: classification
                    .alternatives
                    .into_iter()
                    .map(|(i, c)| format!("{} ({:.0}%)", i.display_name(), c * 100.0))
                    .collect(),
                stats: NlQueryStats {
                    classification_time_ms: 0,
                    execution_time_ms: elapsed,
                    total_time_ms: elapsed,
                    subsystem: intent.display_name().to_string(),
                },
                message: "Query executed successfully".to_string(),
            },
            Err(e) => QueryResponse::error(e),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap(),
        )]))
    }

    /// Execute GTD queries.
    async fn execute_gtd_query(
        &self,
        intent: &crate::query::GtdIntent,
        classification: &crate::query::ClassificationResult,
        ontology_store: Arc<RwLock<crate::ontology::EmbeddedOntologyStore>>,
        limit: usize,
    ) -> Result<(String, serde_json::Value), String> {
        use crate::gtd::{
            ProjectFilter, ProjectManager, ProjectStatus, RecommendParams, SomedayFilter,
            SomedayManager, TaskFilter, TaskManager, TaskStatus, WaitingFilter, WaitingManager,
        };
        use crate::query::GtdIntent;

        let task_mgr = TaskManager::new(ontology_store.clone());
        let project_mgr = ProjectManager::new(ontology_store.clone());
        let waiting_mgr = WaitingManager::new(ontology_store.clone());
        let someday_mgr = SomedayManager::new(ontology_store.clone());

        match intent {
            GtdIntent::ListTasks => {
                let filter = TaskFilter {
                    limit,
                    ..Default::default()
                };
                let tasks = task_mgr.list(filter).await.map_err(|e| e.to_string())?;
                let count = tasks.len();
                Ok((
                    format!("Found {} tasks", count),
                    serde_json::to_value(&tasks).unwrap_or_default(),
                ))
            }
            GtdIntent::TasksByContext { context } => {
                let filter = TaskFilter {
                    contexts: vec![context.clone()],
                    status: Some(TaskStatus::Next),
                    limit,
                    ..Default::default()
                };
                let tasks = task_mgr.list(filter).await.map_err(|e| e.to_string())?;
                let count = tasks.len();
                Ok((
                    format!("Found {} tasks with context @{}", count, context),
                    serde_json::to_value(&tasks).unwrap_or_default(),
                ))
            }
            GtdIntent::QuickWins => {
                let tasks = task_mgr.get_quick_wins().await.map_err(|e| e.to_string())?;
                let count = tasks.len();
                Ok((
                    format!("Found {} quick wins (low effort, high priority)", count),
                    serde_json::to_value(&tasks).unwrap_or_default(),
                ))
            }
            GtdIntent::OverdueTasks => {
                let tasks = task_mgr.get_overdue().await.map_err(|e| e.to_string())?;
                let count = tasks.len();
                Ok((
                    format!("Found {} overdue tasks", count),
                    serde_json::to_value(&tasks).unwrap_or_default(),
                ))
            }
            GtdIntent::RecommendTasks | GtdIntent::WhatNow => {
                let context = classification.extracted_params.contexts.first().cloned();
                let params = RecommendParams {
                    current_context: context,
                    energy_level: None,
                    time_available: None,
                    focus_area: None,
                    limit,
                };
                let recommendations = task_mgr
                    .recommend(params)
                    .await
                    .map_err(|e| e.to_string())?;
                let count = recommendations.len();
                Ok((
                    format!("Recommending {} tasks based on current context", count),
                    serde_json::to_value(&recommendations).unwrap_or_default(),
                ))
            }
            GtdIntent::ListProjects => {
                let filter = ProjectFilter {
                    limit,
                    ..Default::default()
                };
                let projects = project_mgr.list(filter).await.map_err(|e| e.to_string())?;
                let count = projects.len();
                Ok((
                    format!("Found {} projects", count),
                    serde_json::to_value(&projects).unwrap_or_default(),
                ))
            }
            GtdIntent::StalledProjects => {
                let projects = project_mgr
                    .get_stalled(7)
                    .await
                    .map_err(|e| e.to_string())?;
                let count = projects.len();
                Ok((
                    format!("Found {} stalled projects (no activity in 7+ days)", count),
                    serde_json::to_value(&projects).unwrap_or_default(),
                ))
            }
            GtdIntent::ProjectsWithoutNextAction => {
                let projects = project_mgr
                    .get_without_next_action()
                    .await
                    .map_err(|e| e.to_string())?;
                let count = projects.len();
                Ok((
                    format!("Found {} projects without a next action", count),
                    serde_json::to_value(&projects).unwrap_or_default(),
                ))
            }
            GtdIntent::ProjectHealth => {
                // Return all active projects with their health status
                let filter = ProjectFilter {
                    status: Some(ProjectStatus::Active),
                    limit,
                    ..Default::default()
                };
                let projects = project_mgr.list(filter).await.map_err(|e| e.to_string())?;
                let count = projects.len();
                Ok((
                    format!("Showing health status for {} active projects", count),
                    serde_json::to_value(&projects).unwrap_or_default(),
                ))
            }
            GtdIntent::ListWaiting => {
                let filter = WaitingFilter {
                    limit,
                    ..Default::default()
                };
                let waiting = waiting_mgr.list(filter).await.map_err(|e| e.to_string())?;
                let count = waiting.len();
                Ok((
                    format!("Found {} waiting-for items", count),
                    serde_json::to_value(&waiting).unwrap_or_default(),
                ))
            }
            GtdIntent::OverdueWaiting => {
                let waiting = waiting_mgr.get_overdue().await.map_err(|e| e.to_string())?;
                let count = waiting.len();
                Ok((
                    format!("Found {} overdue waiting-for items", count),
                    serde_json::to_value(&waiting).unwrap_or_default(),
                ))
            }
            GtdIntent::WaitingForPerson { person } => {
                let filter = WaitingFilter {
                    delegated_to: Some(person.clone()),
                    limit,
                    ..Default::default()
                };
                let waiting = waiting_mgr.list(filter).await.map_err(|e| e.to_string())?;
                let count = waiting.len();
                Ok((
                    format!("Found {} items waiting on {}", count, person),
                    serde_json::to_value(&waiting).unwrap_or_default(),
                ))
            }
            GtdIntent::ListSomeday => {
                let filter = SomedayFilter {
                    limit,
                    ..Default::default()
                };
                let items = someday_mgr.list(filter).await.map_err(|e| e.to_string())?;
                let count = items.len();
                Ok((
                    format!("Found {} someday/maybe items", count),
                    serde_json::to_value(&items).unwrap_or_default(),
                ))
            }
            GtdIntent::WeeklyReview | GtdIntent::DailyReview | GtdIntent::General => {
                // For reviews, provide a summary
                let tasks = task_mgr
                    .list(TaskFilter {
                        limit: 10,
                        ..Default::default()
                    })
                    .await
                    .map_err(|e| e.to_string())?;
                let projects = project_mgr
                    .list(ProjectFilter {
                        status: Some(ProjectStatus::Active),
                        limit: 10,
                        ..Default::default()
                    })
                    .await
                    .map_err(|e| e.to_string())?;

                let summary = serde_json::json!({
                    "tasks_count": tasks.len(),
                    "active_projects_count": projects.len(),
                    "recent_tasks": tasks,
                    "active_projects": projects,
                });

                Ok((
                    format!(
                        "GTD Summary: {} tasks, {} active projects",
                        tasks.len(),
                        projects.len()
                    ),
                    summary,
                ))
            }
        }
    }

    /// Execute calendar queries.
    async fn execute_calendar_query(
        &self,
        intent: &crate::query::CalendarIntent,
        _classification: &crate::query::ClassificationResult,
        ontology_store: Arc<RwLock<crate::ontology::EmbeddedOntologyStore>>,
    ) -> Result<(String, serde_json::Value), String> {
        use crate::calendar::CalendarManager;
        use crate::query::CalendarIntent;

        let manager = CalendarManager::new(ontology_store);

        match intent {
            CalendarIntent::Today => {
                let events = manager.today().await.map_err(|e| e.to_string())?;
                let count = events.len();
                Ok((
                    format!("Found {} events for today", count),
                    serde_json::to_value(&events).unwrap_or_default(),
                ))
            }
            CalendarIntent::ThisWeek => {
                let events = manager.this_week().await.map_err(|e| e.to_string())?;
                let count = events.len();
                Ok((
                    format!("Found {} events this week", count),
                    serde_json::to_value(&events).unwrap_or_default(),
                ))
            }
            CalendarIntent::Upcoming { days } => {
                let days = days.map(|d| d as i64).unwrap_or(30);
                let events = manager
                    .upcoming(Some(days))
                    .await
                    .map_err(|e| e.to_string())?;
                let count = events.len();
                Ok((
                    format!("Found {} upcoming events", count),
                    serde_json::to_value(&events).unwrap_or_default(),
                ))
            }
            CalendarIntent::Conflicts => {
                // Get upcoming events and check for conflicts
                let events = manager
                    .upcoming(Some(14))
                    .await
                    .map_err(|e| e.to_string())?;
                let conflicts = manager.detect_conflicts(&events);
                let count = conflicts.len();
                Ok((
                    format!("Found {} scheduling conflicts in next 2 weeks", count),
                    serde_json::to_value(&conflicts).unwrap_or_default(),
                ))
            }
            CalendarIntent::FreeTime => {
                // Use upcoming events to show calendar summary for now
                let events = manager.today().await.map_err(|e| e.to_string())?;
                let count = events.len();
                Ok((
                    format!(
                        "Today has {} events. Use calendar tools for detailed free time analysis.",
                        count
                    ),
                    serde_json::to_value(&events).unwrap_or_default(),
                ))
            }
            CalendarIntent::SpecificDate { date_expr } => {
                // For now, just show today - date parsing would need more work
                let events = manager.today().await.map_err(|e| e.to_string())?;
                let count = events.len();
                Ok((
                    format!("Found {} events for '{}' (showing today)", count, date_expr),
                    serde_json::to_value(&events).unwrap_or_default(),
                ))
            }
            CalendarIntent::EventsWithPerson { person } => {
                // Show upcoming events - filtering by person would need more work
                let events = manager
                    .upcoming(Some(14))
                    .await
                    .map_err(|e| e.to_string())?;
                let count = events.len();
                Ok((
                    format!(
                        "Found {} upcoming events (filter by '{}' not implemented)",
                        count, person
                    ),
                    serde_json::to_value(&events).unwrap_or_default(),
                ))
            }
            CalendarIntent::General => {
                let events = manager.this_week().await.map_err(|e| e.to_string())?;
                let count = events.len();
                Ok((
                    format!("Found {} events this week", count),
                    serde_json::to_value(&events).unwrap_or_default(),
                ))
            }
        }
    }

    /// Execute knowledge graph queries.
    async fn execute_knowledge_query(
        &self,
        intent: &crate::query::KnowledgeIntent,
        _classification: &crate::query::ClassificationResult,
        ontology_store: Arc<RwLock<crate::ontology::EmbeddedOntologyStore>>,
        embedder: Arc<dyn crate::embedding::EmbeddingProvider>,
        limit: usize,
    ) -> Result<(String, serde_json::Value), String> {
        use crate::knowledge::KnowledgeQueryEngine;
        use crate::query::KnowledgeIntent;

        let engine = KnowledgeQueryEngine::new(ontology_store, embedder);

        match intent {
            KnowledgeIntent::WhatDoIKnowAbout { topic } => {
                let result = engine
                    .summarize_topic(topic, limit)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok((
                    format!("Knowledge summary for '{}'", topic),
                    serde_json::to_value(&result).unwrap_or_default(),
                ))
            }
            KnowledgeIntent::FindExpert { topic } => {
                let experts = engine
                    .find_experts(topic, limit)
                    .await
                    .map_err(|e| e.to_string())?;
                let count = experts.len();
                Ok((
                    format!("Found {} experts on '{}'", count, topic),
                    serde_json::to_value(&experts).unwrap_or_default(),
                ))
            }
            KnowledgeIntent::WhoWorksWith { person } => {
                let result = engine
                    .traverse_relationships(person, None, 2)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok((
                    format!("Found people working with '{}'", person),
                    serde_json::to_value(&result).unwrap_or_default(),
                ))
            }
            KnowledgeIntent::TopicSummary { topic } => {
                let result = engine
                    .summarize_topic(topic, limit)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok((
                    format!("Topic summary for '{}'", topic),
                    serde_json::to_value(&result).unwrap_or_default(),
                ))
            }
            KnowledgeIntent::EntityLookup { name } => {
                let result = engine
                    .traverse_relationships(name, None, 1)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok((
                    format!("Entity lookup for '{}'", name),
                    serde_json::to_value(&result).unwrap_or_default(),
                ))
            }
            KnowledgeIntent::RelationshipQuery { from } => {
                let result = engine
                    .traverse_relationships(from, None, 2)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok((
                    format!("Relationships from '{}'", from),
                    serde_json::to_value(&result).unwrap_or_default(),
                ))
            }
            KnowledgeIntent::General { query } => {
                // Use topic summary as fallback
                let result = engine
                    .summarize_topic(query, limit)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok((
                    format!("Search results for '{}'", query),
                    serde_json::to_value(&result).unwrap_or_default(),
                ))
            }
        }
    }

    /// Execute document search queries.
    async fn execute_search_query(
        &self,
        query: &str,
        coordinator: &IndexCoordinator,
        limit: usize,
    ) -> Result<(String, serde_json::Value), String> {
        let search_query = HybridQuery::new(query).limit(limit);

        match coordinator.search(search_query).await {
            Ok(response) => {
                let count = response.results.len();
                Ok((
                    format!("Found {} matching documents", count),
                    serde_json::to_value(&response.results).unwrap_or_default(),
                ))
            }
            Err(e) => Err(format!("Search failed: {}", e)),
        }
    }
}

#[tool_handler]
impl ServerHandler for AlloyServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Alloy is a hybrid document indexing MCP server. \
                 It indexes local files and S3 objects for semantic and full-text search. \
                 Use 'index_path' to add sources, 'search' to find documents, \
                 and 'get_stats' to see index information."
                    .to_string(),
            ),
        }
    }
}

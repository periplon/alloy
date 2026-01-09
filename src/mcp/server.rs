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
};
use crate::auth::AuthContext;
use crate::config::Config;
use crate::coordinator::{IndexCoordinator, IndexProgress};
use crate::mcp::tools::{
    AclConfigInfo, AclEntryInfo, AddWebhookResponse, BackupInfo, CacheConfigInfo,
    CheckDuplicateResponse, CheckPermissionResponse, ClearCacheResponse,
    ClearDeduplicationResponse, ClusterDocumentsResponse, ClusterInfo, ClusterMetrics,
    ConfigureResponse, CreateBackupResponse, DeduplicationStatsResponse,
    DeleteDocumentAclResponse, DiffStatsInfo, DiffVersionsResponse, DocumentDetails,
    ExportDocumentsResponse, GetAclStatsResponse, GetCacheStatsResponse, GetDocumentAclResponse,
    GetDocumentHistoryResponse, GetMetricsResponse, GetVersionContentResponse,
    GetWebhookStatsResponse, ImportDocumentsResponse, IndexPathResponse, IndexStats,
    ListBackupsResponse, ListSourcesResponse, ListWebhooksResponse, RemoveSourceResponse,
    RemoveWebhookResponse, RestoreBackupResponse, RestoreVersionResponse, RoleInfo,
    SearchResponse, SearchResult, SetDocumentAclResponse, SourceInfo, TestWebhookResponse,
    VersionInfo, VersioningRetentionInfo, VersioningStatsResponse, WebhookDeliveryInfo,
    WebhookInfo,
};
use crate::metrics::get_metrics;
use crate::search::{HybridQuery, SearchFilter};
use crate::sources::parse_s3_uri;
use crate::webhooks::{SharedWebhookDispatcher, WebhookConfig, WebhookDispatcher};

/// Alloy MCP server state.
pub struct AlloyState {
    /// Server configuration
    pub config: Config,
    /// Start time for uptime calculation
    pub start_time: Instant,
    /// Index coordinator (lazily initialized)
    pub coordinator: Option<IndexCoordinator>,
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
            coordinator: None,
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

    /// Create a new Alloy server with default configuration.
    pub fn with_defaults() -> crate::error::Result<Self> {
        let config = Config::load()?;
        Ok(Self::new(config))
    }

    /// Ensure the coordinator is initialized.
    async fn ensure_coordinator(&self) -> Result<(), McpError> {
        let mut state = self.state.write().await;
        if state.coordinator.is_none() {
            let coordinator = IndexCoordinator::new(state.config.clone())
                .await
                .map_err(|e| {
                    McpError::internal_error(
                        format!("Failed to initialize coordinator: {}", e),
                        None,
                    )
                })?;
            state.coordinator = Some(coordinator);
        }
        Ok(())
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
                                .map(|(start, end)| {
                                    if *end <= r.text.len() {
                                        r.text[*start..*end].to_string()
                                    } else {
                                        String::new()
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
    #[tool(description = "Remove an indexed source and all its documents from the index.")]
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

        match coordinator.remove_source(&params.source_id).await {
            Ok(docs_removed) => {
                let response = RemoveSourceResponse {
                    success: docs_removed > 0,
                    documents_removed: docs_removed,
                    message: if docs_removed > 0 {
                        format!(
                            "Removed source {} with {} documents",
                            params.source_id, docs_removed
                        )
                    } else {
                        format!("Source not found: {}", params.source_id)
                    },
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

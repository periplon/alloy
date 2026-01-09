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

use crate::config::Config;
use crate::coordinator::{IndexCoordinator, IndexProgress};
use crate::mcp::tools::{
    CheckDuplicateResponse, ClearDeduplicationResponse, ClusterDocumentsResponse, ClusterInfo,
    ClusterMetrics, ConfigureResponse, DeduplicationStatsResponse, DiffStatsInfo,
    DiffVersionsResponse, DocumentDetails, GetDocumentHistoryResponse, GetVersionContentResponse,
    IndexPathResponse, IndexStats, ListSourcesResponse, RemoveSourceResponse,
    RestoreVersionResponse, SearchResponse, SearchResult, SourceInfo, VersionInfo,
    VersioningRetentionInfo, VersioningStatsResponse,
};
use crate::search::{HybridQuery, SearchFilter};
use crate::sources::parse_s3_uri;

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
}

impl AlloyState {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            start_time: Instant::now(),
            coordinator: None,
            progress_log: Vec::new(),
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
                let results: Vec<SearchResult> = response
                    .results
                    .iter()
                    .map(|r| {
                        SearchResult {
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
                        }
                    })
                    .collect();

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

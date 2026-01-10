//! REST API request handlers.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::coordinator::{IndexCoordinator, SourceStatus};
use crate::search::{HybridQuery, SearchFilter};

/// Application state shared across handlers.
pub struct ApiState {
    /// Index coordinator for operations.
    pub coordinator: Arc<IndexCoordinator>,
}

impl ApiState {
    /// Create new API state.
    pub fn new(coordinator: Arc<IndexCoordinator>) -> Self {
        Self { coordinator }
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Index path request.
#[derive(Debug, Clone, Deserialize)]
pub struct IndexRequest {
    /// Path to index (local path or S3 URI).
    pub path: String,
    /// Glob pattern to filter files.
    #[serde(default)]
    pub pattern: Option<String>,
    /// Watch for file changes.
    #[serde(default)]
    pub watch: bool,
}

/// Index response.
#[derive(Debug, Clone, Serialize)]
pub struct IndexResponse {
    pub source_id: String,
    pub documents_indexed: usize,
    pub watching: bool,
    pub message: String,
    /// Status of the indexing operation.
    pub status: String,
}

/// Search query parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchQuery {
    /// Search query text.
    pub q: String,
    /// Maximum results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Vector weight (0.0 to 1.0).
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,
    /// Filter by source ID.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Enable query expansion.
    #[serde(default)]
    pub expand: bool,
    /// Enable reranking.
    #[serde(default)]
    pub rerank: bool,
}

fn default_limit() -> usize {
    10
}

fn default_vector_weight() -> f32 {
    0.5
}

/// Search result item.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResultItem {
    pub document_id: String,
    pub chunk_id: String,
    pub path: String,
    pub content: String,
    pub score: f32,
}

/// Search response.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub total: usize,
    pub took_ms: u64,
}

/// Document response.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentResponse {
    pub id: String,
    pub source_id: String,
    pub path: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub content: Option<String>,
    pub modified_at: chrono::DateTime<chrono::Utc>,
    pub indexed_at: chrono::DateTime<chrono::Utc>,
}

/// Source info response.
#[derive(Debug, Clone, Serialize)]
pub struct SourceResponse {
    pub id: String,
    pub source_type: String,
    pub path: String,
    pub document_count: usize,
    pub watching: bool,
    pub last_scan: chrono::DateTime<chrono::Utc>,
    /// Current status of the source (ready, indexing, failed).
    pub status: String,
    /// Error message if status is failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Sources list response.
#[derive(Debug, Clone, Serialize)]
pub struct SourcesListResponse {
    pub sources: Vec<SourceResponse>,
    pub total: usize,
}

/// Stats response.
#[derive(Debug, Clone, Serialize)]
pub struct StatsResponse {
    pub source_count: usize,
    pub document_count: usize,
    pub chunk_count: usize,
    pub storage_bytes: u64,
    pub embedding_dimension: usize,
    pub uptime_seconds: u64,
}

/// Error response.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

/// Remove source response.
#[derive(Debug, Clone, Serialize)]
pub struct RemoveSourceResponse {
    pub success: bool,
    pub documents_removed: usize,
    pub message: String,
}

// ============================================================================
// Handler Functions
// ============================================================================

/// POST /api/v1/index - Index a path.
///
/// This immediately registers the source and starts indexing in the background.
/// The response indicates that indexing has started, not that it's complete.
pub async fn index_handler(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<IndexRequest>,
) -> impl IntoResponse {
    let patterns = request.pattern.clone().map(|p| vec![p]).unwrap_or_default();

    // Determine source type and validate
    let (source_type, validated_path) = if request.path.starts_with("s3://") {
        // Parse and validate S3 URI
        match crate::sources::parse_s3_uri(&request.path) {
            Ok(_) => ("s3", request.path.clone()),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: e.to_string(),
                        code: "invalid_s3_uri".to_string(),
                    }),
                )
                    .into_response();
            }
        }
    } else {
        let path = std::path::PathBuf::from(&request.path);
        if !path.exists() {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Path does not exist: {}", request.path),
                    code: "path_not_found".to_string(),
                }),
            )
                .into_response();
        }
        ("local", request.path.clone())
    };

    // Register the source immediately (with "indexing" status)
    let pending_source = match state
        .coordinator
        .register_pending_source(source_type, &validated_path, request.watch)
        .await
    {
        Ok(source) => source,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: "registration_failed".to_string(),
                }),
            )
                .into_response();
        }
    };

    let source_id = pending_source.id.clone();

    // Spawn background indexing task
    let coordinator = state.coordinator.clone();
    let path = validated_path.clone();
    let watch = request.watch;

    tokio::spawn(async move {
        let result = if path.starts_with("s3://") {
            match crate::sources::parse_s3_uri(&path) {
                Ok((bucket, prefix)) => {
                    coordinator
                        .index_s3(bucket, Some(prefix), patterns, None)
                        .await
                }
                Err(e) => Err(e),
            }
        } else {
            let path_buf = std::path::PathBuf::from(&path);
            coordinator
                .index_local(path_buf, patterns, vec![], watch)
                .await
        };

        // Update source status based on result
        match result {
            Ok(indexed) => {
                let _ = coordinator
                    .update_source_status(
                        &indexed.id,
                        SourceStatus::Ready,
                        Some(indexed.document_count),
                        None,
                    )
                    .await;
                tracing::info!(
                    "Background indexing complete for {}: {} documents",
                    indexed.id,
                    indexed.document_count
                );
            }
            Err(e) => {
                let _ = coordinator
                    .update_source_status(&path, SourceStatus::Failed, None, Some(e.to_string()))
                    .await;
                tracing::error!("Background indexing failed for {}: {}", path, e);
            }
        }
    });

    // Return immediately with "indexing" status
    (
        StatusCode::ACCEPTED,
        Json(IndexResponse {
            source_id,
            documents_indexed: 0,
            watching: request.watch,
            message: "Indexing started in background".to_string(),
            status: "indexing".to_string(),
        }),
    )
        .into_response()
}

/// GET /api/v1/search - Search documents.
pub async fn search_handler(
    State(state): State<Arc<ApiState>>,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    let mut query = HybridQuery::new(&params.q)
        .limit(params.limit)
        .vector_weight(params.vector_weight)
        .expand_query(params.expand)
        .rerank(params.rerank);

    if let Some(source_id) = params.source_id {
        query = query.filter(SearchFilter::new().source(source_id));
    }

    match state.coordinator.search(query).await {
        Ok(response) => {
            let results: Vec<SearchResultItem> = response
                .results
                .into_iter()
                .map(|r| SearchResultItem {
                    document_id: r.document_id,
                    chunk_id: r.chunk_id,
                    path: r.path.unwrap_or_default(),
                    content: r.text,
                    score: r.score,
                })
                .collect();

            let total = results.len();

            (
                StatusCode::OK,
                Json(SearchResponse {
                    results,
                    total,
                    took_ms: start.elapsed().as_millis() as u64,
                }),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
                code: "search_failed".to_string(),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/documents/:id - Get a document.
pub async fn get_document_handler(
    State(state): State<Arc<ApiState>>,
    Path(document_id): Path<String>,
    Query(params): Query<DocumentQueryParams>,
) -> impl IntoResponse {
    match state.coordinator.get_document(&document_id).await {
        Ok(Some(doc)) => (
            StatusCode::OK,
            Json(DocumentResponse {
                id: doc.id,
                source_id: doc.source_id,
                path: doc.path,
                mime_type: doc.mime_type,
                size_bytes: doc.size,
                content: if params.include_content.unwrap_or(true) {
                    Some(doc.content)
                } else {
                    None
                },
                modified_at: doc.modified_at,
                indexed_at: doc.indexed_at,
            }),
        )
            .into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Document not found: {}", document_id),
                code: "not_found".to_string(),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
                code: "get_failed".to_string(),
            }),
        )
            .into_response(),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DocumentQueryParams {
    pub include_content: Option<bool>,
}

/// GET /api/v1/sources - List all sources.
pub async fn list_sources_handler(State(state): State<Arc<ApiState>>) -> impl IntoResponse {
    let sources = state.coordinator.list_sources().await;

    let source_responses: Vec<SourceResponse> = sources
        .into_iter()
        .map(|s| SourceResponse {
            id: s.id,
            source_type: s.source_type,
            path: s.path,
            document_count: s.document_count,
            watching: s.watching,
            last_scan: s.last_scan,
            status: s.status.to_string(),
            error: s.error,
        })
        .collect();

    let total = source_responses.len();

    (
        StatusCode::OK,
        Json(SourcesListResponse {
            sources: source_responses,
            total,
        }),
    )
}

/// DELETE /api/v1/sources/:id - Remove a source.
pub async fn remove_source_handler(
    State(state): State<Arc<ApiState>>,
    Path(source_id): Path<String>,
) -> impl IntoResponse {
    match state.coordinator.remove_source(&source_id).await {
        Ok(Some(docs_removed)) => (
            StatusCode::OK,
            Json(RemoveSourceResponse {
                success: true,
                documents_removed: docs_removed,
                message: format!(
                    "Source {} removed with {} documents",
                    source_id, docs_removed
                ),
            }),
        )
            .into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Source not found: {}", source_id),
                code: "not_found".to_string(),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
                code: "remove_failed".to_string(),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/stats - Get index statistics.
pub async fn stats_handler(State(state): State<Arc<ApiState>>) -> impl IntoResponse {
    let sources = state.coordinator.list_sources().await;

    match state.coordinator.stats().await {
        Ok(stats) => (
            StatusCode::OK,
            Json(StatsResponse {
                source_count: sources.len(),
                document_count: stats.document_count,
                chunk_count: stats.chunk_count,
                storage_bytes: stats.storage_bytes,
                embedding_dimension: state.coordinator.embedding_dimension(),
                uptime_seconds: 0, // Would need to track start time
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
                code: "stats_failed".to_string(),
            }),
        )
            .into_response(),
    }
}

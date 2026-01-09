//! REST API router and configuration.

use std::sync::Arc;

use axum::{
    http::{header, Method},
    routing::{delete, get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};

use crate::api::handlers::{
    get_document_handler, index_handler, list_sources_handler, remove_source_handler,
    search_handler, stats_handler, ApiState,
};
use crate::coordinator::IndexCoordinator;

/// REST API configuration.
#[derive(Debug, Clone)]
pub struct RestApiConfig {
    /// Enable CORS.
    pub enable_cors: bool,
    /// Allowed origins for CORS.
    pub cors_origins: Vec<String>,
    /// API prefix (e.g., "/api/v1").
    pub prefix: String,
}

impl Default for RestApiConfig {
    fn default() -> Self {
        Self {
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            prefix: "/api/v1".to_string(),
        }
    }
}

/// Create the REST API router.
///
/// Endpoints:
/// - POST   /api/v1/index          - Index a path
/// - GET    /api/v1/search?q=...   - Search documents
/// - GET    /api/v1/documents/:id  - Get a document
/// - GET    /api/v1/sources        - List sources
/// - DELETE /api/v1/sources/:id    - Remove a source
/// - GET    /api/v1/stats          - Get statistics
pub fn create_rest_router(coordinator: Arc<IndexCoordinator>, config: &RestApiConfig) -> Router {
    let state = Arc::new(ApiState::new(coordinator));

    let api_routes = Router::new()
        .route("/index", post(index_handler))
        .route("/search", get(search_handler))
        .route("/documents/:id", get(get_document_handler))
        .route("/sources", get(list_sources_handler))
        .route("/sources/:id", delete(remove_source_handler))
        .route("/stats", get(stats_handler))
        .with_state(state);

    // Build the full router with prefix
    let router = Router::new().nest(&config.prefix, api_routes);

    // Add CORS if enabled
    if config.enable_cors {
        let cors = CorsLayer::new()
            .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
            .allow_origin(Any);

        router.layer(cors)
    } else {
        router
    }
}

/// Create a combined router with both REST API and additional routes.
pub fn create_combined_router(
    coordinator: Arc<IndexCoordinator>,
    config: &RestApiConfig,
) -> Router {
    let rest_router = create_rest_router(coordinator, config);

    // Add API info route
    let info_route = Router::new().route("/api", get(api_info_handler));

    rest_router.merge(info_route)
}

/// API info handler.
async fn api_info_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "name": "Alloy REST API",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "REST API for Alloy hybrid document indexing",
        "endpoints": {
            "index": {
                "method": "POST",
                "path": "/api/v1/index",
                "description": "Index a local path or S3 URI"
            },
            "search": {
                "method": "GET",
                "path": "/api/v1/search",
                "description": "Search indexed documents",
                "params": {
                    "q": "Search query (required)",
                    "limit": "Maximum results (default: 10)",
                    "vector_weight": "Weight for vector search 0-1 (default: 0.5)",
                    "source_id": "Filter by source",
                    "expand": "Enable query expansion",
                    "rerank": "Enable reranking"
                }
            },
            "get_document": {
                "method": "GET",
                "path": "/api/v1/documents/:id",
                "description": "Get a document by ID"
            },
            "list_sources": {
                "method": "GET",
                "path": "/api/v1/sources",
                "description": "List all indexed sources"
            },
            "remove_source": {
                "method": "DELETE",
                "path": "/api/v1/sources/:id",
                "description": "Remove an indexed source"
            },
            "stats": {
                "method": "GET",
                "path": "/api/v1/stats",
                "description": "Get index statistics"
            }
        }
    }))
}

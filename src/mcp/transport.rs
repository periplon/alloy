//! Transport layer for MCP server.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Router;
use rmcp::transport::stdio;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::tower::{
    StreamableHttpServerConfig, StreamableHttpService,
};
use rmcp::ServiceExt;
use tracing::info;

use crate::config::{Config, TransportType};
use crate::mcp::AlloyServer;
use crate::metrics::{get_metrics, HealthCheck, HealthState, HealthStatus, ReadinessStatus};

/// Run the MCP server with stdio transport.
pub async fn run_stdio(server: AlloyServer) -> Result<()> {
    info!("Starting Alloy MCP server with stdio transport");

    let service = server.serve(stdio()).await?;

    info!("Alloy MCP server running...");
    service.waiting().await?;

    info!("Alloy MCP server shutting down");
    Ok(())
}

/// Run the MCP server with HTTP/SSE (Streamable HTTP) transport.
pub async fn run_http(config: Config, port: u16) -> Result<()> {
    info!(
        "Starting Alloy MCP server with HTTP/SSE transport on port {}",
        port
    );

    // Create session manager for handling multiple connections
    let session_manager = Arc::new(LocalSessionManager::default());

    // We need to clone config for the factory closure
    let config_for_factory = config.clone();

    // Build the streamable HTTP service with a factory function
    let http_config = StreamableHttpServerConfig::default();
    let http_service = StreamableHttpService::new(
        move || {
            // Factory creates a new server for each connection
            Ok(AlloyServer::new(config_for_factory.clone()))
        },
        session_manager,
        http_config,
    );

    // Build axum router - use fallback_service instead of nest for tower services
    let app = Router::new()
        .route("/health", axum::routing::get(health_handler))
        .route("/ready", axum::routing::get(readiness_handler))
        .route("/metrics", axum::routing::get(metrics_handler))
        .route("/", axum::routing::get(root_handler))
        .fallback_service(http_service);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Alloy MCP server listening on http://{}", addr);
    info!("MCP endpoint available at root path");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    info!("Alloy MCP server shutting down");
    Ok(())
}

/// Health check endpoint.
///
/// Returns detailed health status including individual component checks.
async fn health_handler() -> impl IntoResponse {
    let metrics = get_metrics();
    metrics.update_uptime();

    // Perform health checks
    let mut checks = Vec::new();
    let mut overall_status = HealthState::Healthy;

    // Basic server check - always healthy if we can respond
    checks.push(HealthCheck::healthy("server"));

    // Metrics system check
    checks.push(HealthCheck::healthy("metrics"));

    // Calculate overall status
    for check in &checks {
        match check.status {
            HealthState::Unhealthy => {
                overall_status = HealthState::Unhealthy;
                break;
            }
            HealthState::Degraded => {
                if overall_status == HealthState::Healthy {
                    overall_status = HealthState::Degraded;
                }
            }
            HealthState::Healthy => {}
        }
    }

    let health = HealthStatus {
        status: overall_status,
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: metrics.uptime_seconds.get(),
        checks,
    };

    let status_code = match overall_status {
        HealthState::Healthy | HealthState::Degraded => StatusCode::OK,
        HealthState::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };

    (status_code, axum::Json(health))
}

/// Readiness check endpoint.
///
/// Returns whether the server is ready to accept requests.
async fn readiness_handler() -> impl IntoResponse {
    // The server is ready if it can respond
    let status = ReadinessStatus::ready();

    if status.ready {
        (StatusCode::OK, axum::Json(status))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, axum::Json(status))
    }
}

/// Prometheus metrics endpoint.
///
/// Returns metrics in Prometheus text format.
async fn metrics_handler() -> impl IntoResponse {
    let metrics = get_metrics();
    let output = metrics.export_prometheus();

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; charset=utf-8",
        )],
        output,
    )
}

/// Root handler with basic info.
async fn root_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "name": "alloy",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Hybrid Document Indexing MCP Server",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics"
        }
    }))
}

/// Run the MCP server with the configured transport.
pub async fn run_server(
    server: AlloyServer,
    transport: TransportType,
    port: u16,
    config: Config,
) -> Result<()> {
    match transport {
        TransportType::Stdio => run_stdio(server).await,
        TransportType::Http => run_http(config, port).await,
    }
}

//! Transport layer for MCP server.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
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
        .route("/health", axum::routing::get(health_check))
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
async fn health_check() -> &'static str {
    "OK"
}

/// Root handler with basic info.
async fn root_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "name": "alloy",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Hybrid Document Indexing MCP Server",
        "endpoints": {
            "health": "/health"
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

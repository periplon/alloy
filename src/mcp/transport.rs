//! Transport layer for MCP server.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Router;
use axum_server::tls_rustls::RustlsConfig;
use rmcp::transport::stdio;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::tower::{
    StreamableHttpServerConfig, StreamableHttpService,
};
use rmcp::ServiceExt;
use tower::ServiceBuilder;
use tracing::{info, warn};

use crate::api::{create_rest_router, RestApiConfig};
use crate::auth::{AuthLayer, Authenticator};
use crate::config::{Config, TransportType};
use crate::coordinator::IndexCoordinator;
use crate::mcp::AlloyServer;
use crate::metrics::{get_metrics, HealthCheck, HealthState, HealthStatus, ReadinessStatus};
use crate::tls::{CertManager, TrustResult};
use crate::web::{create_web_ui_router, WebUiConfig};

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

    let app = build_http_app(config).await?;

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
        uptime_seconds: metrics.uptime_seconds.get() as u64,
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
            "metrics": "/metrics",
            "auth_status": "/auth/status",
            "rest_api": "/api/v1",
            "web_ui": "/ui"
        }
    }))
}

/// Auth status endpoint.
///
/// Returns authentication status from the request context.
async fn auth_status_handler(req: axum::extract::Request) -> impl IntoResponse {
    use crate::auth::AuthContext;

    // Try to get auth context from extensions
    let auth_ctx = req
        .extensions()
        .get::<AuthContext>()
        .cloned()
        .unwrap_or_else(AuthContext::anonymous);

    let response = serde_json::json!({
        "authenticated": !auth_ctx.anonymous,
        "user_id": auth_ctx.user_id,
        "roles": auth_ctx.roles,
        "groups": auth_ctx.groups,
        "method": auth_ctx.method
    });

    (StatusCode::OK, axum::Json(response))
}

/// Run the MCP server with HTTPS transport using auto-generated certificates.
pub async fn run_https(config: Config, port: u16) -> Result<()> {
    info!(
        "Starting Alloy MCP server with HTTPS transport on port {}",
        port
    );

    // Get data directory for certificate storage
    let data_dir = config.data_dir().context("Failed to get data directory")?;

    // Initialize certificate manager
    let cert_manager = CertManager::new(&data_dir).context("Failed to initialize certificate manager")?;

    // Check if we should use custom certificates
    let rustls_config = if let (Some(cert_file), Some(key_file)) = (
        &config.server.tls.cert_file,
        &config.server.tls.key_file,
    ) {
        info!("Using custom TLS certificates from {:?}", cert_file);
        RustlsConfig::from_pem_file(cert_file, key_file)
            .await
            .context("Failed to load custom TLS certificates")?
    } else {
        // Generate localhost certificate
        let certs = cert_manager
            .generate_localhost_cert()
            .context("Failed to generate localhost certificate")?;

        // Try to install CA to trust store if enabled
        if config.server.tls.auto_install_ca {
            match cert_manager.install_ca_to_trust_store() {
                TrustResult::Installed => {
                    info!("CA certificate installed to system trust store");
                }
                TrustResult::AlreadyInstalled => {
                    info!("CA certificate already in system trust store");
                }
                TrustResult::ManualRequired(reason) => {
                    warn!("Could not auto-install CA certificate: {}", reason);
                    cert_manager.print_trust_instructions();
                }
                TrustResult::Failed(err) => {
                    warn!("Failed to install CA certificate: {}", err);
                    cert_manager.print_trust_instructions();
                }
            }
        } else {
            info!("CA auto-installation disabled, manual trust required");
            cert_manager.print_trust_instructions();
        }

        info!("CA certificate: {}", cert_manager.ca_cert_path().display());

        RustlsConfig::from_pem(certs.cert_pem.into_bytes(), certs.key_pem.into_bytes())
            .await
            .context("Failed to configure TLS with generated certificates")?
    };

    // Build the application (same as HTTP)
    let app = build_http_app(config).await?;

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Alloy MCP server listening on https://{}", addr);
    info!("MCP endpoint available at root path");

    axum_server::bind_rustls(addr, rustls_config)
        .serve(app.into_make_service())
        .await
        .context("HTTPS server error")?;

    info!("Alloy MCP server shutting down");
    Ok(())
}

/// Build the HTTP/HTTPS application router.
async fn build_http_app(config: Config) -> Result<Router> {
    // Create session manager for handling multiple connections
    let session_manager = Arc::new(LocalSessionManager::default());

    // Create authenticator
    let authenticator = Authenticator::new(config.security.auth.clone());
    let auth_enabled = authenticator.is_enabled();

    if auth_enabled {
        info!(
            "Authentication enabled with method: {:?}",
            config.security.auth.method
        );
    } else {
        info!("Authentication disabled");
    }

    // Create shared coordinator for both REST API and MCP sessions
    let coordinator = match IndexCoordinator::new(config.clone()).await {
        Ok(c) => {
            info!("Index coordinator initialized");
            Arc::new(c)
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to create coordinator: {}", e));
        }
    };

    if config.integration.rest_api.enabled {
        info!("REST API enabled at {}", config.integration.rest_api.prefix);
    }

    // Clone for the factory closure - shared across all MCP sessions
    let config_for_factory = config.clone();
    let coordinator_for_factory = coordinator.clone();

    // Build the streamable HTTP service with a factory function
    let http_config = StreamableHttpServerConfig::default();
    let http_service = StreamableHttpService::new(
        move || {
            // Factory creates a new server for each connection, sharing the coordinator
            Ok(AlloyServer::with_shared_coordinator(
                config_for_factory.clone(),
                coordinator_for_factory.clone(),
            ))
        },
        session_manager,
        http_config,
    );

    // Build base routes
    let mut app = Router::new()
        .route("/health", axum::routing::get(health_handler))
        .route("/ready", axum::routing::get(readiness_handler))
        .route("/metrics", axum::routing::get(metrics_handler))
        .route("/", axum::routing::get(root_handler))
        .route("/auth/status", axum::routing::get(auth_status_handler));

    // Add REST API routes if enabled
    if config.integration.rest_api.enabled {
        let rest_config = RestApiConfig {
            enable_cors: config.integration.rest_api.enable_cors,
            cors_origins: config.integration.rest_api.cors_origins.clone(),
            prefix: config.integration.rest_api.prefix.clone(),
        };
        let rest_router = create_rest_router(coordinator, &rest_config);
        app = app.merge(rest_router);
    }

    // Add Web UI routes if enabled
    if config.integration.web_ui.enabled {
        info!(
            "Web UI enabled at {}",
            config.integration.web_ui.path_prefix
        );
        let web_config = WebUiConfig {
            enabled: true,
            path_prefix: config.integration.web_ui.path_prefix.clone(),
        };
        let web_router = create_web_ui_router(&web_config);
        app = app.merge(web_router);
    }

    // Apply auth middleware if enabled, then add MCP service as fallback
    let app = if auth_enabled {
        let auth_layer = AuthLayer::new(authenticator);
        app.fallback_service(
            ServiceBuilder::new()
                .layer(auth_layer)
                .service(http_service),
        )
    } else {
        app.fallback_service(http_service)
    };

    Ok(app)
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
        TransportType::Http => {
            if config.server.tls.enabled {
                run_https(config, port).await
            } else {
                run_http(config, port).await
            }
        }
    }
}

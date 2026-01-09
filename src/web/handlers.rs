//! Web UI route handlers.

use axum::{routing::get, Router};

use crate::web::static_files::{serve_css, serve_index, serve_js};

/// Web UI configuration.
#[derive(Debug, Clone)]
pub struct WebUiConfig {
    /// Enable the web UI.
    pub enabled: bool,
    /// Path prefix for web UI routes.
    pub path_prefix: String,
}

impl Default for WebUiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path_prefix: "/ui".to_string(),
        }
    }
}

/// Create the web UI router.
///
/// Routes:
/// - GET /ui           - Main dashboard
/// - GET /ui/style.css - CSS styles
/// - GET /ui/app.js    - JavaScript
/// - GET /ui/sources   - Sources page (served by index)
/// - GET /ui/search    - Search page (served by index)
pub fn create_web_ui_router(config: &WebUiConfig) -> Router {
    if !config.enabled {
        return Router::new();
    }

    let web_routes = Router::new()
        .route("/", get(serve_index))
        .route("/style.css", get(serve_css))
        .route("/app.js", get(serve_js))
        // These routes all serve the same SPA
        .route("/sources", get(serve_index))
        .route("/search", get(serve_index))
        .route("/documents", get(serve_index))
        .route("/settings", get(serve_index));

    Router::new().nest(&config.path_prefix, web_routes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WebUiConfig::default();
        assert!(config.enabled);
        assert_eq!(config.path_prefix, "/ui");
    }

    #[test]
    fn test_create_router() {
        let config = WebUiConfig::default();
        let _router = create_web_ui_router(&config);
        // Router creation should not panic
    }
}

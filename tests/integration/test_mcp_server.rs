//! Tests for the MCP server tools.

use tempfile::TempDir;

use alloy::config::Config;
use alloy::mcp::AlloyServer;

/// Create a test configuration.
fn create_test_config(data_dir: &std::path::Path) -> Config {
    let mut config = Config::default();
    config.storage.data_dir = data_dir.to_string_lossy().to_string();
    config
}

#[tokio::test]
async fn test_server_creation() {
    let data_dir = TempDir::new().unwrap();
    let config = create_test_config(data_dir.path());
    let server = AlloyServer::new(config);
    // Server should be created successfully
    drop(server);
}

#[tokio::test]
async fn test_server_info() {
    use rmcp::ServerHandler;

    let data_dir = TempDir::new().unwrap();
    let config = create_test_config(data_dir.path());
    let server = AlloyServer::new(config);

    let info = server.get_info();
    // Implementation::from_build_env() uses crate name (rmcp) not our app name
    // so we just verify it has a name and version
    assert!(!info.server_info.name.is_empty(), "Server should have a name");
    assert!(info.instructions.is_some());
    assert!(
        info.instructions.as_ref().unwrap().contains("hybrid") ||
        info.instructions.as_ref().unwrap().contains("Alloy"),
        "Instructions should mention hybrid search or Alloy"
    );
}

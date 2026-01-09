//! Integration tests for Alloy MCP Server.
//!
//! These tests verify the complete pipeline from indexing to search.
//! Most tests are marked as `#[ignore]` because they require downloading
//! the embedding model, which can be slow.
//!
//! Run ignored tests with:
//! ```bash
//! cargo test --test integration -- --ignored
//! ```

#[path = "integration/test_coordinator.rs"]
mod test_coordinator;

#[path = "integration/test_mcp_server.rs"]
mod test_mcp_server;

#[path = "integration/test_pipeline.rs"]
mod test_pipeline;

//! CLI command dispatcher.
//!
//! This module dispatches CLI commands to either local or remote execution.

use alloy::{
    mcp::{
        DocumentDetails, IndexPathResponse, IndexStats, ListSourcesResponse, RemoveSourceResponse,
        SearchResponse,
    },
    Config,
};
use anyhow::Result;

use super::{local, output, remote};

/// Execution mode for CLI commands.
#[derive(Clone)]
pub enum ExecutionMode {
    /// Execute locally via IndexCoordinator
    Local(Config),
    /// Execute remotely via MCP client
    Remote(String),
}

/// Run the index command.
pub async fn run_index(
    mode: ExecutionMode,
    path: String,
    pattern: Option<String>,
    watch: bool,
    json_output: bool,
) -> Result<()> {
    let result: IndexPathResponse = match mode {
        ExecutionMode::Local(config) => local::index(config, path, pattern, watch).await?,
        ExecutionMode::Remote(url) => remote::index(&url, path, pattern, watch).await?,
    };
    output::print_index_result(&result, json_output);
    Ok(())
}

/// Run the search command.
pub async fn run_search(
    mode: ExecutionMode,
    query: String,
    limit: usize,
    vector_weight: f32,
    source_id: Option<String>,
    json_output: bool,
) -> Result<()> {
    let result: SearchResponse = match mode {
        ExecutionMode::Local(config) => {
            local::search(config, query, limit, vector_weight, source_id).await?
        }
        ExecutionMode::Remote(url) => {
            remote::search(&url, query, limit, vector_weight, source_id).await?
        }
    };
    output::print_search_results(&result, json_output);
    Ok(())
}

/// Run the get-document command.
pub async fn run_get_document(
    mode: ExecutionMode,
    document_id: String,
    include_content: bool,
    json_output: bool,
) -> Result<()> {
    let result: Option<DocumentDetails> = match mode {
        ExecutionMode::Local(config) => {
            local::get_document(config, document_id, include_content).await?
        }
        ExecutionMode::Remote(url) => {
            remote::get_document(&url, document_id, include_content).await?
        }
    };
    output::print_document(&result, json_output);
    Ok(())
}

/// Run the list-sources command.
pub async fn run_list_sources(mode: ExecutionMode, json_output: bool) -> Result<()> {
    let result: ListSourcesResponse = match mode {
        ExecutionMode::Local(config) => local::list_sources(config).await?,
        ExecutionMode::Remote(url) => remote::list_sources(&url).await?,
    };
    output::print_sources(&result, json_output);
    Ok(())
}

/// Run the remove-source command.
pub async fn run_remove_source(
    mode: ExecutionMode,
    source_id: String,
    json_output: bool,
) -> Result<()> {
    let result: RemoveSourceResponse = match mode {
        ExecutionMode::Local(config) => local::remove_source(config, source_id).await?,
        ExecutionMode::Remote(url) => remote::remove_source(&url, source_id).await?,
    };
    output::print_remove_result(&result, json_output);
    Ok(())
}

/// Run the stats command.
pub async fn run_stats(mode: ExecutionMode, json_output: bool) -> Result<()> {
    let result: IndexStats = match mode {
        ExecutionMode::Local(config) => local::stats(config).await?,
        ExecutionMode::Remote(url) => remote::stats(&url).await?,
    };
    output::print_stats(&result, json_output);
    Ok(())
}

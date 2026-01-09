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
    Local(Box<Config>),
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
        ExecutionMode::Local(config) => local::index(*config, path, pattern, watch).await?,
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
            local::search(*config, query, limit, vector_weight, source_id).await?
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
            local::get_document(*config, document_id, include_content).await?
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
        ExecutionMode::Local(config) => local::list_sources(*config).await?,
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
        ExecutionMode::Local(config) => local::remove_source(*config, source_id).await?,
        ExecutionMode::Remote(url) => remote::remove_source(&url, source_id).await?,
    };
    output::print_remove_result(&result, json_output);
    Ok(())
}

/// Run the stats command.
pub async fn run_stats(mode: ExecutionMode, json_output: bool) -> Result<()> {
    let result: IndexStats = match mode {
        ExecutionMode::Local(config) => local::stats(*config).await?,
        ExecutionMode::Remote(url) => remote::stats(&url).await?,
    };
    output::print_stats(&result, json_output);
    Ok(())
}

/// Run the cluster command.
pub async fn run_cluster(
    mode: ExecutionMode,
    source_id: Option<String>,
    algorithm: Option<String>,
    num_clusters: Option<usize>,
    json_output: bool,
) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => {
            local::cluster(*config, source_id, algorithm, num_clusters).await?
        }
        ExecutionMode::Remote(url) => {
            remote::cluster(&url, source_id, algorithm, num_clusters).await?
        }
    };
    output::print_cluster_results(&result, json_output);
    Ok(())
}

/// Run the backup command.
pub async fn run_backup(
    mode: ExecutionMode,
    output_path: Option<String>,
    description: Option<String>,
    json_output: bool,
) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::backup(*config, output_path, description).await?,
        ExecutionMode::Remote(url) => remote::backup(&url, output_path, description).await?,
    };
    output::print_backup_result(&result, json_output);
    Ok(())
}

/// Run the restore command.
pub async fn run_restore(mode: ExecutionMode, input: String, json_output: bool) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::restore(*config, input).await?,
        ExecutionMode::Remote(url) => remote::restore(&url, input).await?,
    };
    output::print_restore_result(&result, json_output);
    Ok(())
}

/// Run the export command.
pub async fn run_export(
    mode: ExecutionMode,
    output_path: String,
    format: String,
    source_id: Option<String>,
    include_embeddings: bool,
    json_output: bool,
) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => {
            local::export(*config, output_path, format, source_id, include_embeddings).await?
        }
        ExecutionMode::Remote(url) => {
            remote::export(&url, output_path, format, source_id, include_embeddings).await?
        }
    };
    output::print_export_result(&result, json_output);
    Ok(())
}

/// Run the import command.
pub async fn run_import(mode: ExecutionMode, input_path: String, json_output: bool) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::import(*config, input_path).await?,
        ExecutionMode::Remote(url) => remote::import(&url, input_path).await?,
    };
    output::print_import_result(&result, json_output);
    Ok(())
}

/// Run the list backups command.
pub async fn run_list_backups(mode: ExecutionMode, json_output: bool) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::list_backups(*config).await?,
        ExecutionMode::Remote(url) => remote::list_backups(&url).await?,
    };
    output::print_list_backups_result(&result, json_output);
    Ok(())
}

// ============================================================================
// GTD Commands
// ============================================================================

use crate::{CalendarCommand, GtdCommand, KnowledgeCommand, OntologyCommand};

/// Run GTD commands.
pub async fn run_gtd(mode: ExecutionMode, command: GtdCommand, json_output: bool) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::gtd(*config, command).await?,
        ExecutionMode::Remote(url) => remote::gtd(&url, command).await?,
    };
    output::print_gtd_result(&result, json_output);
    Ok(())
}

/// Run Calendar commands.
pub async fn run_calendar(
    mode: ExecutionMode,
    command: CalendarCommand,
    json_output: bool,
) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::calendar(*config, command).await?,
        ExecutionMode::Remote(url) => remote::calendar(&url, command).await?,
    };
    output::print_calendar_result(&result, json_output);
    Ok(())
}

/// Run Knowledge commands.
pub async fn run_knowledge(
    mode: ExecutionMode,
    command: KnowledgeCommand,
    json_output: bool,
) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::knowledge(*config, command).await?,
        ExecutionMode::Remote(url) => remote::knowledge(&url, command).await?,
    };
    output::print_knowledge_result(&result, json_output);
    Ok(())
}

/// Run natural language query.
pub async fn run_query(
    mode: ExecutionMode,
    query: String,
    query_mode: Option<String>,
    json_output: bool,
) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::query(*config, query, query_mode).await?,
        ExecutionMode::Remote(url) => remote::query(&url, query, query_mode).await?,
    };
    output::print_query_result(&result, json_output);
    Ok(())
}

/// Run Ontology commands.
pub async fn run_ontology(
    mode: ExecutionMode,
    command: OntologyCommand,
    json_output: bool,
) -> Result<()> {
    let result = match mode {
        ExecutionMode::Local(config) => local::ontology(*config, command).await?,
        ExecutionMode::Remote(url) => remote::ontology(&url, command).await?,
    };
    output::print_ontology_result(&result, json_output);
    Ok(())
}

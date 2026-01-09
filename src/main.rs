//! Alloy MCP Server Entry Point

use alloy::{run_server, AlloyServer, Config};
use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod cli;

/// Alloy: Hybrid Document Indexing MCP Server
#[derive(Parser, Debug)]
#[command(name = "alloy")]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, global = true)]
    config: Option<String>,

    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,

    /// Connect to remote Alloy server via MCP protocol
    #[arg(short, long, global = true)]
    remote: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Index a local path or S3 URI
    Index {
        /// Path to index (local path or S3 URI like s3://bucket/prefix)
        path: String,
        /// Glob pattern to filter files (e.g., "*.md", "**/*.rs")
        #[arg(short, long)]
        pattern: Option<String>,
        /// Watch for changes and re-index
        #[arg(short, long)]
        watch: bool,
        /// Recursive indexing (default: true)
        #[arg(long, default_value = "true")]
        recursive: bool,
    },
    /// Search indexed documents
    Search {
        /// Search query
        query: String,
        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Weight for vector search (0.0 to 1.0)
        #[arg(short = 'w', long, default_value = "0.5")]
        vector_weight: f32,
        /// Filter by source ID
        #[arg(short, long)]
        source: Option<String>,
    },
    /// Get a document by ID
    Get {
        /// Document ID
        document_id: String,
        /// Don't include document content
        #[arg(long)]
        no_content: bool,
    },
    /// List indexed sources
    Sources,
    /// Remove an indexed source
    Remove {
        /// Source ID to remove
        source_id: String,
    },
    /// Show index statistics
    Stats,
    /// Cluster indexed documents by semantic similarity
    Cluster {
        /// Filter by source ID
        #[arg(short, long)]
        source: Option<String>,
        /// Clustering algorithm (kmeans or dbscan)
        #[arg(short, long, default_value = "kmeans")]
        algorithm: String,
        /// Number of clusters (for k-means, default: auto-detect)
        #[arg(short, long)]
        num_clusters: Option<usize>,
    },
    /// Create a backup of the index
    Backup {
        /// Output path for the backup file (defaults to configured backup directory)
        #[arg(short, long)]
        output: Option<String>,
        /// Description for the backup
        #[arg(short, long)]
        description: Option<String>,
    },
    /// Restore from a backup
    Restore {
        /// Path to backup file or backup ID
        input: String,
    },
    /// Export documents to a file
    Export {
        /// Output file path
        output: String,
        /// Export format (jsonl or json)
        #[arg(short, long, default_value = "jsonl")]
        format: String,
        /// Filter by source ID
        #[arg(short, long)]
        source: Option<String>,
        /// Include embeddings (increases file size significantly)
        #[arg(long)]
        include_embeddings: bool,
    },
    /// Import documents from a file
    Import {
        /// Path to import file
        input: String,
    },
    /// List available backups
    Backups,
    /// Run as MCP server (default behavior)
    Serve {
        /// Transport type (stdio or http)
        #[arg(short, long, default_value = "stdio")]
        transport: String,
        /// HTTP port (when using http transport)
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Enable JSON logging format
        #[arg(long)]
        json_logs: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // For CLI commands (non-serve), use minimal logging
    let is_serve = matches!(args.command, Some(Command::Serve { .. }) | None);

    if !is_serve {
        // Minimal logging for CLI commands
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_writer(std::io::stderr)
            .init();
    }

    // Determine execution mode
    let mode = if let Some(url) = &args.remote {
        cli::ExecutionMode::Remote(url.clone())
    } else {
        let config = if let Some(path) = &args.config {
            Config::from_file(path)?
        } else {
            Config::load()?
        };
        cli::ExecutionMode::Local(Box::new(config))
    };

    match args.command {
        Some(Command::Index {
            path,
            pattern,
            watch,
            recursive: _,
        }) => cli::run_index(mode, path, pattern, watch, args.json).await,
        Some(Command::Search {
            query,
            limit,
            vector_weight,
            source,
        }) => cli::run_search(mode, query, limit, vector_weight, source, args.json).await,
        Some(Command::Get {
            document_id,
            no_content,
        }) => cli::run_get_document(mode, document_id, !no_content, args.json).await,
        Some(Command::Sources) => cli::run_list_sources(mode, args.json).await,
        Some(Command::Remove { source_id }) => {
            cli::run_remove_source(mode, source_id, args.json).await
        }
        Some(Command::Stats) => cli::run_stats(mode, args.json).await,
        Some(Command::Cluster {
            source,
            algorithm,
            num_clusters,
        }) => cli::run_cluster(mode, source, Some(algorithm), num_clusters, args.json).await,
        Some(Command::Backup {
            output,
            description,
        }) => cli::run_backup(mode, output, description, args.json).await,
        Some(Command::Restore { input }) => cli::run_restore(mode, input, args.json).await,
        Some(Command::Export {
            output,
            format,
            source,
            include_embeddings,
        }) => cli::run_export(mode, output, format, source, include_embeddings, args.json).await,
        Some(Command::Import { input }) => cli::run_import(mode, input, args.json).await,
        Some(Command::Backups) => cli::run_list_backups(mode, args.json).await,
        Some(Command::Serve {
            transport,
            port,
            json_logs,
        }) => run_mcp_server(&args.config, &transport, port, json_logs).await,
        None => {
            // Default: run as MCP server with stdio
            run_mcp_server(&args.config, "stdio", 8080, false).await
        }
    }
}

/// Run the MCP server (extracted from original main)
async fn run_mcp_server(
    config_path: &Option<String>,
    transport: &str,
    port: u16,
    json_logs: bool,
) -> anyhow::Result<()> {
    // Initialize tracing for server mode
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if json_logs {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    tracing::info!("Starting Alloy MCP Server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let mut config = if let Some(config_path) = config_path {
        Config::from_file(config_path)?
    } else {
        Config::load()?
    };

    // Override transport from CLI args
    config.server.transport = match transport {
        "http" => alloy::config::TransportType::Http,
        _ => alloy::config::TransportType::Stdio,
    };
    config.server.http_port = port;

    tracing::info!(
        transport = ?config.server.transport,
        embedding_provider = ?config.embedding.provider,
        storage_backend = ?config.storage.backend,
        "Configuration loaded"
    );

    // Create and run server
    let server = AlloyServer::new(config.clone());
    run_server(
        server,
        config.server.transport,
        config.server.http_port,
        config,
    )
    .await?;

    Ok(())
}

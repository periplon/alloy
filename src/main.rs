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
        /// Transport type (stdio or http). If not specified, uses config file value.
        #[arg(short, long)]
        transport: Option<String>,
        /// HTTP port (when using http transport). If not specified, uses config file value.
        #[arg(short, long)]
        port: Option<u16>,
        /// Enable HTTPS with auto-generated certificates
        #[arg(long)]
        https: bool,
        /// Enable JSON logging format
        #[arg(long)]
        json_logs: bool,
    },

    /// GTD workflow management
    Gtd {
        #[command(subcommand)]
        action: GtdCommand,
    },

    /// Calendar queries and event management
    Calendar {
        #[command(subcommand)]
        action: CalendarCommand,
    },

    /// Knowledge graph queries
    Knowledge {
        #[command(subcommand)]
        action: KnowledgeCommand,
    },

    /// Natural language query
    Query {
        /// Query text
        query: String,
        /// Force query mode (auto, gtd, calendar, knowledge)
        #[arg(short, long)]
        mode: Option<String>,
    },

    /// Ontology management
    Ontology {
        #[command(subcommand)]
        action: OntologyCommand,
    },
}

/// GTD workflow subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum GtdCommand {
    /// List and manage projects
    Projects {
        /// Action: list, get, create, update, archive, complete
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Project ID (for get/update/archive/complete)
        #[arg(short, long)]
        id: Option<String>,

        // === Filters (for list) ===
        /// Filter by status: active, on_hold, completed, archived
        #[arg(long)]
        status: Option<String>,
        /// Filter by area
        #[arg(long)]
        area: Option<String>,
        /// Show only stalled projects (days without activity)
        #[arg(long)]
        stalled: Option<u32>,
        /// Show only projects without next actions
        #[arg(long)]
        no_next_action: bool,

        // === Create/Update fields ===
        /// Project name (for create/update)
        #[arg(short, long)]
        name: Option<String>,
        /// Desired outcome (for create/update)
        #[arg(long)]
        outcome: Option<String>,
        /// Area of focus (for create/update)
        #[arg(long)]
        set_area: Option<String>,
        /// Supporting goal ID (for create/update)
        #[arg(long)]
        goal: Option<String>,
    },

    /// List and manage tasks
    Tasks {
        /// Action: list, get, create, update, complete, recommend
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Task ID (for get/update/complete)
        #[arg(short, long)]
        id: Option<String>,

        // === Filters (for list/recommend) ===
        /// Filter by context (@home, @work, @computer, etc.)
        #[arg(long)]
        context: Option<Vec<String>>,
        /// Filter by energy level: low, medium, high
        #[arg(short, long)]
        energy: Option<String>,
        /// Filter by max duration (minutes)
        #[arg(short, long)]
        time: Option<u32>,
        /// Filter by project ID
        #[arg(short, long)]
        project: Option<String>,
        /// Filter by status: next, scheduled, waiting, someday, done
        #[arg(long)]
        status: Option<String>,
        /// Due before date (YYYY-MM-DD)
        #[arg(long)]
        due_before: Option<String>,
        /// Filter by description (case-insensitive substring match)
        #[arg(long)]
        description_contains: Option<String>,
        /// Result limit
        #[arg(short, long, default_value = "20")]
        limit: usize,

        // === Create/Update fields ===
        /// Task description (for create/update)
        #[arg(short, long)]
        description: Option<String>,
        /// Set contexts (for create/update, comma-separated)
        #[arg(long)]
        set_contexts: Option<String>,
        /// Set energy level (for create/update)
        #[arg(long)]
        set_energy: Option<String>,
        /// Set priority: low, medium, high, critical (for create/update)
        #[arg(long)]
        priority: Option<String>,
        /// Estimated duration in minutes (for create/update)
        #[arg(long)]
        duration: Option<u32>,
        /// Due date YYYY-MM-DD (for create/update)
        #[arg(long)]
        due: Option<String>,
        /// Scheduled date YYYY-MM-DD (for create/update)
        #[arg(long)]
        scheduled: Option<String>,
        /// Assign to project ID (for create/update)
        #[arg(long)]
        assign_project: Option<String>,
        /// Blocked by task ID (for create/update)
        #[arg(long)]
        blocked_by: Option<String>,
    },

    /// Track delegated items
    Waiting {
        /// Action: list, add, update, resolve, remind
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Waiting item ID (for update/resolve)
        #[arg(short, long)]
        id: Option<String>,

        // === Filters ===
        /// Filter by status: pending, overdue, resolved
        #[arg(long)]
        status: Option<String>,
        /// Filter by person
        #[arg(long)]
        person: Option<String>,
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        // === Create fields ===
        /// Description (for add)
        #[arg(short, long)]
        description: Option<String>,
        /// Delegated to person (for add)
        #[arg(long)]
        delegated_to: Option<String>,
        /// Expected by date YYYY-MM-DD (for add)
        #[arg(long)]
        expected_by: Option<String>,
        /// Related project ID (for add)
        #[arg(long)]
        for_project: Option<String>,
        /// Resolution note (for resolve)
        #[arg(long)]
        resolution: Option<String>,
    },

    /// Manage someday/maybe items
    Someday {
        /// Action: list, add, update, activate, archive
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Item ID (for update/activate/archive)
        #[arg(short, long)]
        id: Option<String>,

        // === Filters ===
        /// Filter by category
        #[arg(long)]
        category: Option<String>,

        // === Create/Update fields ===
        /// Description (for add/update)
        #[arg(short, long)]
        description: Option<String>,
        /// Set category (for add/update)
        #[arg(long)]
        set_category: Option<String>,
        /// Trigger condition (for add/update)
        #[arg(long)]
        trigger: Option<String>,
        /// Review date YYYY-MM-DD (for add/update)
        #[arg(long)]
        review_date: Option<String>,
    },

    /// Run weekly review
    Review {
        /// Week ending date (default: today)
        #[arg(long)]
        week_ending: Option<String>,
        /// Sections to include (comma-separated)
        #[arg(short, long)]
        sections: Option<String>,
    },

    /// View and manage GTD horizons
    Horizons {
        /// Action: list, add
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Specific level: runway, h10k, h20k, h30k, h40k, h50k
        #[arg(short, long)]
        level: Option<String>,

        // === Create fields (for areas, goals, vision, purpose) ===
        /// Name (for add)
        #[arg(short, long)]
        name: Option<String>,
        /// Description (for add)
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Track commitments
    Commitments {
        /// Action: list, add, resolve
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Commitment ID (for resolve)
        #[arg(short, long)]
        id: Option<String>,

        // === Filters ===
        /// Filter: made, received, all
        #[arg(short, long, default_value = "all")]
        filter: String,
        /// Show only pending
        #[arg(long)]
        pending: bool,

        // === Create fields ===
        /// Description (for add)
        #[arg(short, long)]
        description: Option<String>,
        /// Type: made, received (for add)
        #[arg(long)]
        commitment_type: Option<String>,
        /// To/From person (for add)
        #[arg(long)]
        person: Option<String>,
        /// Due date YYYY-MM-DD (for add)
        #[arg(long)]
        due: Option<String>,
        /// Resolution note (for resolve)
        #[arg(long)]
        resolution: Option<String>,
    },

    /// Analyze dependencies
    Dependencies {
        /// Action: list, add
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Project ID to analyze
        #[arg(short, long)]
        project: Option<String>,
        /// Show critical path
        #[arg(long)]
        critical_path: bool,

        // === Create dependency ===
        /// Task that is blocked (for add)
        #[arg(long)]
        blocked_task: Option<String>,
        /// Task that blocks (for add)
        #[arg(long)]
        blocking_task: Option<String>,
    },

    /// Attention economics
    Attention {
        /// Period: day, week, month
        #[arg(short, long, default_value = "week")]
        period: String,
        /// Group by: area, project
        #[arg(short, long, default_value = "area")]
        group_by: String,
    },

    /// Manage areas of focus
    Areas {
        /// Action: list, add, update, archive
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Area ID (for update/archive)
        #[arg(short, long)]
        id: Option<String>,
        /// Name (for add/update)
        #[arg(short, long)]
        name: Option<String>,
        /// Description (for add/update)
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Manage goals
    Goals {
        /// Action: list, add, update, complete, archive
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Goal ID (for update/complete/archive)
        #[arg(short, long)]
        id: Option<String>,
        /// Name (for add/update)
        #[arg(short, long)]
        name: Option<String>,
        /// Description (for add/update)
        #[arg(short, long)]
        description: Option<String>,
        /// Target date YYYY-MM-DD (for add/update)
        #[arg(long)]
        target_date: Option<String>,
        /// Related area ID (for add/update)
        #[arg(long)]
        area: Option<String>,
    },
}

/// Calendar subcommands
#[derive(Subcommand, Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum CalendarCommand {
    /// Today's events
    Today,

    /// This week's events
    Week,

    /// Events in date range
    Range {
        /// Start date (YYYY-MM-DD)
        #[arg(short, long)]
        start: String,
        /// End date (YYYY-MM-DD)
        #[arg(short, long)]
        end: String,
    },

    /// Find free time slots
    Free {
        /// Start date
        #[arg(short, long)]
        start: String,
        /// End date
        #[arg(short, long)]
        end: String,
        /// Minimum slot duration (minutes)
        #[arg(short, long, default_value = "30")]
        min_duration: u32,
    },

    /// Check for conflicts
    Conflicts,

    /// Upcoming events
    Upcoming {
        /// Number of events
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Manage calendar events
    Events {
        /// Action: list, get, add, update, delete
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Event ID (for get/update/delete)
        #[arg(short, long)]
        id: Option<String>,

        // === Create/Update fields ===
        /// Event title (for add/update)
        #[arg(short, long)]
        title: Option<String>,
        /// Event type: meeting, deadline, reminder, blocked_time (for add)
        #[arg(long)]
        event_type: Option<String>,
        /// Start datetime YYYY-MM-DDTHH:MM (for add/update)
        #[arg(long)]
        start: Option<String>,
        /// End datetime YYYY-MM-DDTHH:MM (for add/update)
        #[arg(long)]
        end: Option<String>,
        /// Location (for add/update)
        #[arg(long)]
        location: Option<String>,
        /// Participants (comma-separated, for add/update)
        #[arg(long)]
        participants: Option<String>,
        /// Related project ID (for add/update)
        #[arg(long)]
        project: Option<String>,
        /// Recurrence: daily, weekly, biweekly, monthly, yearly (for add)
        #[arg(long)]
        recurrence: Option<String>,
        /// Notes (for add/update)
        #[arg(long)]
        notes: Option<String>,

        // === Filters (for list) ===
        /// Filter by date range start
        #[arg(long)]
        from: Option<String>,
        /// Filter by date range end
        #[arg(long)]
        to: Option<String>,
        /// Filter by event type
        #[arg(long)]
        filter_type: Option<String>,
    },
}

/// Knowledge graph subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum KnowledgeCommand {
    /// Semantic search
    Search {
        /// Search query
        query: String,
        /// Filter by entity types (comma-separated)
        #[arg(short, long)]
        types: Option<String>,
        /// Result limit
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Look up entity
    Entity {
        /// Entity name
        name: String,
        /// Show relationships
        #[arg(short, long)]
        relationships: bool,
    },

    /// Find experts
    Expert {
        /// Topic to find experts for
        topic: String,
        /// Result limit
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },

    /// Topic summary
    Topic {
        /// Topic to summarize
        topic: String,
    },

    /// Connected entities (graph traversal)
    Connected {
        /// Starting entity name
        entity: String,
        /// Max depth
        #[arg(short, long, default_value = "2")]
        depth: usize,
    },
}

/// Ontology management subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum OntologyCommand {
    /// Ontology statistics
    Stats,

    /// Manage entities
    Entities {
        /// Action: list, get, add, update, delete, merge
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Entity ID (for get/update/delete/merge)
        #[arg(short, long)]
        id: Option<String>,

        // === Filters (for list) ===
        /// Filter by type: Person, Organization, Topic, Concept, Location, etc.
        #[arg(short, long)]
        entity_type: Option<String>,
        /// Search by name
        #[arg(long)]
        name_contains: Option<String>,
        /// Result limit
        #[arg(short, long, default_value = "50")]
        limit: usize,

        // === Create/Update fields ===
        /// Entity type (for add)
        #[arg(long)]
        set_type: Option<String>,
        /// Entity name (for add/update)
        #[arg(short, long)]
        name: Option<String>,
        /// Aliases (comma-separated, for add/update)
        #[arg(long)]
        aliases: Option<String>,
        /// Metadata JSON (for add/update)
        #[arg(long)]
        metadata: Option<String>,

        // === Merge ===
        /// Target entity ID to merge into (for merge)
        #[arg(long)]
        merge_into: Option<String>,
    },

    /// Manage relationships
    Relationships {
        /// Action: list, add, delete
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Relationship ID (for delete)
        #[arg(short, long)]
        id: Option<String>,

        // === Filters (for list) ===
        /// Filter by source entity ID
        #[arg(long)]
        source: Option<String>,
        /// Filter by target entity ID
        #[arg(long)]
        target: Option<String>,
        /// Filter by relationship type
        #[arg(long)]
        rel_type: Option<String>,
        /// Result limit
        #[arg(short, long, default_value = "50")]
        limit: usize,

        // === Create fields ===
        /// Source entity ID (for add)
        #[arg(long)]
        from_entity: Option<String>,
        /// Target entity ID (for add)
        #[arg(long)]
        to_entity: Option<String>,
        /// Relationship type (for add)
        #[arg(long)]
        set_type: Option<String>,
    },

    /// Extract entities from document
    Extract {
        /// Document ID or path
        document: String,
        /// Show confidence scores
        #[arg(long)]
        show_confidence: bool,
        /// Auto-add extracted entities to ontology
        #[arg(long)]
        auto_add: bool,
    },

    /// Add a person entity
    Person {
        /// Person name
        name: String,
        /// Organization they work for
        #[arg(long)]
        organization: Option<String>,
        /// Email address
        #[arg(long)]
        email: Option<String>,
        /// Topics they know about (comma-separated)
        #[arg(long)]
        topics: Option<String>,
        /// Aliases (comma-separated)
        #[arg(long)]
        aliases: Option<String>,
    },

    /// Add an organization entity
    Organization {
        /// Organization name
        name: String,
        /// Organization type (company, team, department, etc.)
        #[arg(long)]
        org_type: Option<String>,
        /// Parent organization
        #[arg(long)]
        parent: Option<String>,
        /// Aliases (comma-separated)
        #[arg(long)]
        aliases: Option<String>,
    },

    /// Add a topic entity
    Topic {
        /// Topic name
        name: String,
        /// Description
        #[arg(short, long)]
        description: Option<String>,
        /// Parent topic (for hierarchical topics)
        #[arg(long)]
        parent: Option<String>,
        /// Aliases (comma-separated)
        #[arg(long)]
        aliases: Option<String>,
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
            https,
            json_logs,
        }) => run_mcp_server(&args.config, transport, port, https, json_logs).await,

        // GTD commands
        Some(Command::Gtd { action }) => cli::run_gtd(mode, action, args.json).await,

        // Calendar commands
        Some(Command::Calendar { action }) => cli::run_calendar(mode, action, args.json).await,

        // Knowledge commands
        Some(Command::Knowledge { action }) => cli::run_knowledge(mode, action, args.json).await,

        // Natural language query
        Some(Command::Query {
            query,
            mode: query_mode,
        }) => cli::run_query(mode, query, query_mode, args.json).await,

        // Ontology commands
        Some(Command::Ontology { action }) => cli::run_ontology(mode, action, args.json).await,

        None => {
            // Default: run as MCP server using config file settings
            run_mcp_server(&args.config, None, None, false, false).await
        }
    }
}

/// Run the MCP server (extracted from original main)
async fn run_mcp_server(
    config_path: &Option<String>,
    transport: Option<String>,
    port: Option<u16>,
    https: bool,
    json_logs: bool,
) -> anyhow::Result<()> {
    // Install rustls crypto provider for HTTPS support
    if https {
        rustls::crypto::aws_lc_rs::default_provider()
            .install_default()
            .expect("Failed to install rustls crypto provider");
    }

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

    // Override transport from CLI args only if explicitly provided
    if let Some(ref t) = transport {
        config.server.transport = match t.as_str() {
            "http" => alloy::config::TransportType::Http,
            _ => alloy::config::TransportType::Stdio,
        };
    }
    if let Some(p) = port {
        config.server.http_port = p;
    }
    // Enable HTTPS if --https flag is provided
    if https {
        config.server.tls.enabled = true;
        // Also switch to HTTP transport if using stdio (HTTPS requires HTTP transport)
        if config.server.transport == alloy::config::TransportType::Stdio {
            config.server.transport = alloy::config::TransportType::Http;
        }
    }

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

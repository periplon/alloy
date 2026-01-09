# Alloy

A hybrid document indexing MCP server with vector search, full-text matching, and GTD intelligence.

Alloy indexes local files and S3 objects, enabling semantic search through vector embeddings combined with traditional full-text matching. Beyond search, Alloy includes a complete **semantic ontology layer** with GTD (Getting Things Done) methodology support, calendar intelligence, and knowledge graph capabilities. Built in Rust for performance and reliability.

## Features

### Core Search
- **Hybrid Search** - Combines vector similarity with full-text matching using configurable weights
- **Multiple Sources** - Index local directories and S3 buckets
- **File Watching** - Automatic re-indexing when files change
- **Flexible Embeddings** - Local (fastembed) or API-based (OpenAI-compatible) embeddings
- **Multiple Backends** - Embedded storage (LanceDB + Tantivy) or external Qdrant
- **Document Processing** - Text, PDF, DOCX, and images with optional OCR

### Ontology & Entity Extraction
- **18 Entity Types** - People, organizations, projects, tasks, topics, concepts, dates, commitments, and more
- **21 Relationship Types** - Semantic connections between entities (belongs_to, works_for, mentions, etc.)
- **Automatic Extraction** - Extract entities from documents using patterns, temporal parsing, and optional LLM
- **Knowledge Graph** - Query relationships, find experts, traverse connections

### GTD (Getting Things Done)
- **Complete GTD Workflow** - Inbox processing, projects, tasks, waiting-for, someday/maybe
- **Project Health Scoring** - Automatic health metrics for projects (next actions, activity, blockers)
- **Context-Aware Recommendations** - Get task suggestions based on context (@home, @work), energy level, and time available
- **Weekly Review Assistant** - Comprehensive review reports with actionable recommendations
- **Horizon Mapping** - Visualize commitments across all 6 GTD levels (runway to life purpose)
- **Commitment Tracking** - Automatically extract promises made and received from documents
- **Dependency Graphs** - Track task dependencies and identify blocking issues
- **Attention Economics** - Monitor where your attention goes across areas and projects

### Calendar Intelligence
- **Smart Date Extraction** - Parse natural language dates ("next Tuesday", "end of Q2", "every Monday")
- **Event Management** - CRUD operations for calendar events with recurrence support
- **Conflict Detection** - Identify scheduling conflicts automatically
- **Free Time Finding** - Query for available time slots

### Knowledge Queries
- **Semantic Search** - "What do I know about X?"
- **Expert Finding** - "Who knows about Kubernetes?" based on document analysis
- **Topic Summaries** - Consolidate knowledge on any topic
- **Graph Traversal** - Explore connected entities

## Quick Start

```bash
# Build
cargo build --release

# Index and search directly from CLI
alloy index ~/documents --pattern "*.md"
alloy search "rust async patterns"
alloy stats

# Or run as MCP server
alloy serve                              # stdio transport (default)
alloy serve --transport http --port 8080 # HTTP transport
```

## CLI Commands

Alloy provides direct CLI access to all functionality:

| Command | Description |
|---------|-------------|
| `alloy index <path>` | Index a local path or S3 URI |
| `alloy search <query>` | Search indexed documents |
| `alloy get <document_id>` | Retrieve a document by ID |
| `alloy sources` | List all indexed sources |
| `alloy remove <source_id>` | Remove an indexed source |
| `alloy stats` | Show index statistics |
| `alloy serve` | Run as MCP server |

### Global Flags

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to config file |
| `--json` | Output as JSON |
| `--remote <url>` | Connect to remote Alloy server via MCP |

### Remote Mode

Connect to a running Alloy server instead of using local storage:

```bash
# Start server on one machine
alloy serve --transport http --port 8080

# Query from another machine (or same machine)
alloy --remote http://localhost:8080 search "query"
alloy --remote http://server:8080 stats
alloy -r http://localhost:8080 sources
```

## MCP Tools

Once connected via MCP, use these tools:

### Document Indexing & Search
```
# Index a directory
index_path(path: "~/documents", pattern: "*.md", watch: true)

# Search documents
search(query: "rust async patterns", limit: 10)

# Natural language query (auto-routes to appropriate subsystem)
query(query: "What's blocking the website redesign project?")
```

### GTD Tools
```
# List active projects with health scores
gtd_projects(action: "list", filters: { status: "active" })

# Get task recommendations based on context
gtd_tasks(action: "recommend", filters: { contexts: ["@computer"], energy_level: "low" })

# Track delegated items
gtd_waiting(action: "list", filters: { status: "overdue" })

# Run weekly review
gtd_weekly_review(sections: ["projects_review", "waiting_for", "inbox_status"])
```

### Calendar & Knowledge
```
# Query calendar
calendar_query(query_type: "this_week")

# Find experts on a topic
knowledge_query(query: "Who knows about Kubernetes?", query_type: "expert_finding")
```

## Configuration

Copy `config.toml.example` to `config.toml` and adjust settings:

```toml
[embedding]
provider = "local"  # or "api"
model = "BAAI/bge-small-en-v1.5"

[storage]
backend = "embedded"  # or "qdrant"
data_dir = "~/.local/share/alloy"
```

## Documentation

See the [docs/](docs/) directory for detailed documentation:

- [Getting Started](docs/getting-started.md) - Installation and setup
- [CLI Reference](docs/cli.md) - Command-line interface
- [Configuration](docs/configuration.md) - All configuration options
- [Tools Reference](docs/tools.md) - MCP tools API
- [Usage Guide](docs/usage.md) - Patterns and examples
- [GTD Guide](docs/gtd.md) - Getting Things Done workflow and tools
- [Ontology & Knowledge](docs/ontology.md) - Entity extraction and knowledge queries

## Development

```bash
just help    # Show available commands
just build   # Build the project
just test    # Run tests
just lint    # Run clippy
```

## License

MIT

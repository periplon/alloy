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

### Core Commands

| Command | Description |
|---------|-------------|
| `alloy index <path>` | Index a local path or S3 URI |
| `alloy search <query>` | Search indexed documents |
| `alloy get <document_id>` | Retrieve a document by ID |
| `alloy sources` | List all indexed sources |
| `alloy remove <source_id>` | Remove an indexed source |
| `alloy stats` | Show index statistics |
| `alloy serve` | Run as MCP server |

### GTD & Productivity Commands

| Command | Description |
|---------|-------------|
| `alloy gtd projects` | List and manage GTD projects |
| `alloy gtd tasks` | List and manage tasks with context-aware recommendations |
| `alloy gtd waiting` | Track delegated items |
| `alloy gtd someday` | Manage someday/maybe items |
| `alloy gtd review` | Run weekly review |
| `alloy gtd horizons` | View GTD horizons (runway to life purpose) |
| `alloy gtd commitments` | Track commitments made and received |
| `alloy gtd dependencies` | Analyze task dependencies |
| `alloy gtd attention` | Attention economics breakdown |
| `alloy gtd areas` | Manage areas of focus |
| `alloy gtd goals` | Manage 1-2 year goals |

### Calendar Commands

| Command | Description |
|---------|-------------|
| `alloy calendar today` | Today's events |
| `alloy calendar week` | This week's events |
| `alloy calendar range` | Events in date range |
| `alloy calendar free` | Find free time slots |
| `alloy calendar conflicts` | Check for scheduling conflicts |
| `alloy calendar events` | Manage calendar events (CRUD) |

### Knowledge & Ontology Commands

| Command | Description |
|---------|-------------|
| `alloy knowledge search <query>` | Semantic knowledge search |
| `alloy knowledge expert <topic>` | Find experts on a topic |
| `alloy knowledge entity <name>` | Look up entity details |
| `alloy knowledge connected <entity>` | Graph traversal |
| `alloy ontology stats` | Ontology statistics |
| `alloy ontology entities` | Manage entities |
| `alloy ontology relationships` | Manage relationships |
| `alloy ontology person <name>` | Add a person entity |
| `alloy ontology organization <name>` | Add an organization |
| `alloy ontology topic <name>` | Add a topic |
| `alloy query <question>` | Natural language query |

### Global Flags

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to config file |
| `--json` | Output as JSON |
| `--remote <url>` | Connect to remote Alloy server via MCP |

### GTD CLI Examples

```bash
# Get context-aware task recommendations
alloy gtd tasks -a recommend --context @computer --energy low --time 30

# Filter tasks by description (case-insensitive)
alloy gtd tasks --description-contains "report"

# Create a new project
alloy gtd projects -a create \
  --name "Website Redesign" \
  --outcome "Launch new site with improved UX"

# Create a task with full metadata
alloy gtd tasks -a create \
  --description "Review homepage mockups" \
  --set-contexts "@computer,@work" \
  --priority high \
  --due 2024-02-01

# Run weekly review
alloy gtd review

# Track a delegated item
alloy gtd waiting -a add \
  --description "Design mockups from Sarah" \
  --delegated-to "Sarah" \
  --expected-by 2024-01-25

# View attention breakdown
alloy gtd attention --period week --group-by area
```

### Calendar & Event Examples

```bash
# View today's schedule
alloy calendar today

# Find free time slots
alloy calendar free --start 2024-01-15 --end 2024-01-19 --min-duration 60

# Create a meeting
alloy calendar events -a add \
  --title "Team Standup" \
  --event-type meeting \
  --start "2024-01-20T09:00" \
  --end "2024-01-20T09:30" \
  --recurrence weekly
```

### Ontology & Knowledge Examples

```bash
# Add a person with their expertise
alloy ontology person "Alice Chen" \
  --organization "Acme Corp" \
  --topics "Kubernetes,DevOps,AWS"

# Find experts on a topic
alloy knowledge expert "PostgreSQL"

# Natural language queries
alloy query "What should I work on now?"
alloy query "Who knows about Kubernetes?"
alloy query "What's blocking the website project?"
```

### Remote Mode

Connect to a running Alloy server instead of using local storage:

```bash
# Start server on one machine
alloy serve --transport http --port 8080

# Query from another machine (or same machine)
alloy --remote http://localhost:8080 search "query"
alloy --remote http://server:8080 gtd projects
alloy -r http://localhost:8080 query "What should I work on?"
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

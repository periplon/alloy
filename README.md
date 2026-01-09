# Alloy

A hybrid document indexing MCP server with vector and full-text search.

Alloy indexes local files and S3 objects, enabling semantic search through vector embeddings combined with traditional full-text matching. Built in Rust for performance and reliability.

## Features

- **Hybrid Search** - Combines vector similarity with full-text matching using configurable weights
- **Multiple Sources** - Index local directories and S3 buckets
- **File Watching** - Automatic re-indexing when files change
- **Flexible Embeddings** - Local (fastembed) or API-based (OpenAI-compatible) embeddings
- **Multiple Backends** - Embedded storage (LanceDB + Tantivy) or external Qdrant
- **Document Processing** - Text, PDF, DOCX, and images with optional OCR

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

```
# Index a directory
index_path(path: "~/documents", pattern: "*.md", watch: true)

# Search documents
search(query: "rust async patterns", limit: 10)

# List indexed sources
list_sources()

# Get index statistics
get_stats()
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

## Development

```bash
just help    # Show available commands
just build   # Build the project
just test    # Run tests
just lint    # Run clippy
```

## License

MIT

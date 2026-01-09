# Getting Started

This guide covers installing, configuring, and running Alloy for the first time.

## Prerequisites

- Rust 1.70+ (for building from source)
- An MCP-compatible client (Claude Desktop, etc.)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/alloy.git
cd alloy

# Build release binary
cargo build --release

# Binary will be at ./target/release/alloy
```

### Install to PATH

```bash
cargo install --path .
```

## First Run

### 1. Create Configuration (Optional)

Alloy works with sensible defaults, but you can customize behavior:

```bash
cp config.toml.example config.toml
```

Edit `config.toml` to adjust settings. See [Configuration](configuration.md) for details.

### 2. Index and Search (CLI Mode)

Use Alloy directly from the command line:

```bash
# Index a directory
alloy index ~/projects --pattern "*.md"

# Search documents
alloy search "how to configure"

# View index statistics
alloy stats
```

### 3. Run as MCP Server

**Stdio Transport** (for MCP clients like Claude Desktop):

```bash
alloy serve
# or simply:
alloy
```

**HTTP Transport** (for network access or remote CLI):

```bash
alloy serve --transport http --port 8080
```

### 4. Configure Your MCP Client

For Claude Desktop, add to your MCP configuration:

```json
{
  "mcpServers": {
    "alloy": {
      "command": "/path/to/alloy",
      "args": ["serve"]
    }
  }
}
```

## First Index

**From CLI:**

```bash
alloy index ~/projects --pattern "*.md"
alloy search "how to configure"
```

**From MCP client:**

```
index_path(path: "~/projects", pattern: "*.md")
search(query: "how to configure")
```

## Command Line Options

```
alloy [OPTIONS] [COMMAND]

Commands:
  index    Index a local path or S3 URI
  search   Search indexed documents
  get      Get a document by ID
  sources  List indexed sources
  remove   Remove an indexed source
  stats    Show index statistics
  serve    Run as MCP server (default)

Global Options:
  -c, --config <PATH>   Path to configuration file
      --json            Output as JSON
  -r, --remote <URL>    Connect to remote Alloy server via MCP
  -h, --help            Print help
  -V, --version         Print version
```

See [CLI Reference](cli.md) for detailed command documentation.

## Environment Variables

- `RUST_LOG` - Log level (e.g., `info`, `debug`, `alloy=debug`)
- `OPENAI_API_KEY` - API key for OpenAI-compatible embedding APIs
- `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - For S3 access

## Data Storage

By default, Alloy stores index data in:

- Linux: `~/.local/share/alloy/`
- macOS: `~/Library/Application Support/alloy/`
- Windows: `%APPDATA%\alloy\`

Override with `storage.data_dir` in config or the environment.

## Next Steps

- [Configuration](configuration.md) - Customize all settings
- [Tools Reference](tools.md) - Learn all available MCP tools
- [Usage Guide](usage.md) - Common patterns and examples

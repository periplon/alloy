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

### 2. Start the Server

**Stdio Transport** (for MCP clients like Claude Desktop):

```bash
alloy
```

**HTTP Transport** (for network access):

```bash
alloy --transport http --port 8080
```

### 3. Configure Your MCP Client

For Claude Desktop, add to your MCP configuration:

```json
{
  "mcpServers": {
    "alloy": {
      "command": "/path/to/alloy",
      "args": []
    }
  }
}
```

## First Index

Once connected, index your first directory:

```
index_path(path: "~/projects", pattern: "*.md")
```

Then search:

```
search(query: "how to configure")
```

## Command Line Options

```
alloy [OPTIONS]

Options:
  -c, --config <PATH>      Path to configuration file
  -t, --transport <TYPE>   Transport type: stdio (default) or http
  -p, --port <PORT>        HTTP port when using http transport (default: 8080)
      --json-logs          Enable JSON logging format
  -h, --help               Print help
  -V, --version            Print version
```

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

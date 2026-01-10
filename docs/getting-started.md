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

#### Stdio Transport (Local)

For local use with Claude Desktop via stdio:

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

#### HTTPS Transport (Required for Claude Desktop Remote)

Claude Desktop requires HTTPS for remote MCP servers. Alloy supports HTTPS with auto-generated certificates:

```bash
alloy serve --https --port 8081
```

However, Claude Desktop uses Node.js internally which doesn't trust the auto-generated CA certificate. There are two solutions:

**Option A: Cloudflare Tunnel (Recommended)**

Cloudflare Tunnel provides a free, secure HTTPS URL with a valid certificate:

```bash
# Install cloudflared
brew install cloudflared  # macOS
# or: sudo apt install cloudflared  # Debian/Ubuntu

# Start alloy without HTTPS (tunnel provides it)
alloy serve --transport http --port 8081

# In another terminal, create the tunnel
cloudflared tunnel --url http://localhost:8081
```

Cloudflared will output a URL like `https://random-name.trycloudflare.com`. Use this in Claude Desktop:

```json
{
  "mcpServers": {
    "alloy": {
      "url": "https://random-name.trycloudflare.com/mcp",
      "transport": "streamable-http"
    }
  }
}
```

For a persistent URL, set up a named tunnel with `cloudflared tunnel create`.

**Option B: ngrok**

```bash
# Install ngrok
brew install ngrok  # macOS

# Start alloy
alloy serve --transport http --port 8081

# Create tunnel
ngrok http 8081
```

Use the ngrok HTTPS URL in your Claude Desktop configuration.

**Option C: Trust the CA Certificate Manually**

If you want to use alloy's auto-generated certificates directly:

1. Start alloy with `--https` to generate the CA
2. The CA is stored at `~/.local/share/alloy/certs/alloy-ca.pem` (or `~/Library/Application Support/alloy/certs/` on macOS)
3. Add the CA to your system trust store with SSL trust
4. Set the `NODE_EXTRA_CA_CERTS` environment variable for Claude Desktop

On macOS:
```bash
# Add CA to system keychain with SSL trust
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ~/.local/share/alloy/certs/alloy-ca.pem
```

Note: Even with system trust, Claude Desktop may require `NODE_EXTRA_CA_CERTS` since it uses Node.js internally.

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

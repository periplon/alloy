# Configuration

Alloy can be configured via a TOML file. The server looks for configuration in:

1. Path specified via `--config` CLI argument
2. `./config.toml` in the current directory
3. `~/.config/alloy/config.toml`
4. Built-in defaults

## Complete Configuration Reference

```toml
[server]
# Transport type: "stdio" or "http"
transport = "stdio"

# HTTP port (only used when transport is "http")
http_port = 8080

# Maximum concurrent indexing tasks
max_concurrent_tasks = 4

[embedding]
# Provider type: "local" or "api"
provider = "local"

# Model name for local embeddings (fastembed)
# Options: "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", etc.
model = "BAAI/bge-small-en-v1.5"

[embedding.api]
# Base URL for embedding API (OpenAI-compatible)
base_url = "https://api.openai.com/v1"

# Model name for API embeddings
model = "text-embedding-3-small"

# API key (can also be set via OPENAI_API_KEY environment variable)
# api_key = "sk-..."

# Batch size for embedding requests
batch_size = 100

# Request timeout in seconds
timeout_secs = 30

[storage]
# Backend type: "embedded" or "qdrant"
backend = "embedded"

# Data directory for embedded storage
data_dir = "~/.local/share/alloy"

[storage.qdrant]
# Qdrant server URL
url = "http://localhost:6334"

# Collection name
collection = "alloy"

# API key (optional)
# api_key = "..."

[processing]
# Chunk size in tokens
chunk_size = 512

# Overlap between chunks in tokens
chunk_overlap = 64

[processing.image]
# Enable OCR for images
ocr = true

# Enable CLIP embeddings for images
clip = true

# Enable Vision API descriptions (requires API key)
vision_api = false
```

## Section Details

### Server

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transport` | string | `"stdio"` | Transport protocol. Use `stdio` for MCP clients, `http` for network access |
| `http_port` | integer | `8080` | Port for HTTP transport |
| `max_concurrent_tasks` | integer | `4` | Maximum parallel indexing operations |

### Embedding

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | string | `"local"` | Embedding provider: `local` (fastembed) or `api` (OpenAI-compatible) |
| `model` | string | `"BAAI/bge-small-en-v1.5"` | Model for local embeddings |

#### Local Embedding Models

Supported fastembed models:

- `BAAI/bge-small-en-v1.5` - 384 dimensions, fastest
- `BAAI/bge-base-en-v1.5` - 768 dimensions, balanced
- `BAAI/bge-large-en-v1.5` - 1024 dimensions, highest quality

#### API Embedding Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `base_url` | string | `"https://api.openai.com/v1"` | API endpoint URL |
| `model` | string | `"text-embedding-3-small"` | Model name |
| `api_key` | string | - | API key (or use `OPENAI_API_KEY` env var) |
| `batch_size` | integer | `100` | Documents per API request |
| `timeout_secs` | integer | `30` | Request timeout |

### Storage

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | string | `"embedded"` | Storage backend: `embedded` or `qdrant` |
| `data_dir` | string | platform-specific | Directory for embedded storage data |

#### Qdrant Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `url` | string | `"http://localhost:6334"` | Qdrant server URL |
| `collection` | string | `"alloy"` | Collection name |
| `api_key` | string | - | Optional API key |

### Processing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | `512` | Target chunk size in tokens |
| `chunk_overlap` | integer | `64` | Overlap between adjacent chunks |

#### Image Processing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ocr` | boolean | `true` | Extract text from images via OCR |
| `clip` | boolean | `true` | Generate CLIP embeddings for images |
| `vision_api` | boolean | `false` | Use Vision API for image descriptions |

## Environment Variables

Configuration can be supplemented with environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible services |
| `RUST_LOG` | Log level filter (e.g., `info`, `debug`) |
| `AWS_REGION` | Default AWS region for S3 |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |

## Examples

### Minimal Local Setup

```toml
[embedding]
provider = "local"

[storage]
backend = "embedded"
```

### OpenAI Embeddings with Qdrant

```toml
[embedding]
provider = "api"

[embedding.api]
base_url = "https://api.openai.com/v1"
model = "text-embedding-3-small"

[storage]
backend = "qdrant"

[storage.qdrant]
url = "http://localhost:6334"
```

### High-Quality Local Embeddings

```toml
[embedding]
provider = "local"
model = "BAAI/bge-large-en-v1.5"

[processing]
chunk_size = 256
chunk_overlap = 32
```

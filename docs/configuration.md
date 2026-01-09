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

# ============================================
# ONTOLOGY & ENTITY EXTRACTION
# ============================================

[ontology]
# Enable ontology and entity extraction features
enabled = true

# Storage backend: "embedded" (in-memory with persistence)
storage_backend = "embedded"

[ontology.extraction]
# Auto-extract entities when indexing documents
extract_on_index = true

# Minimum confidence threshold to store extracted entities (0.0-1.0)
confidence_threshold = 0.7

# Local extraction (pattern-based, always enabled)
[ontology.extraction.local]
# Enable temporal/date parsing
enable_temporal = true

# Enable regex-based entity detection
enable_patterns = true

# Use embeddings for entity clustering and deduplication
enable_embedding_ner = true

# LLM-based extraction (optional, higher accuracy)
[ontology.extraction.llm]
# Enable LLM-based extraction (requires API key)
enabled = false

# LLM provider: "openai", "anthropic", "local"
provider = "openai"

# Custom API endpoint (optional)
api_endpoint = ""

# Model to use for extraction
model = "gpt-4o-mini"

# Extract action items from prose
extract_tasks = true

# Extract entity relationships
extract_relationships = true

# Generate entity summaries
extract_summaries = true

# Maximum tokens per document for LLM extraction
max_tokens_per_doc = 4000

# Rate limit (requests per minute)
rate_limit_rpm = 60

# ============================================
# GTD (GETTING THINGS DONE)
# ============================================

[gtd]
# Enable GTD features
enabled = true

# GTD mode:
# - "inference_and_manual": Auto-extract from documents + manual creation
# - "manual_only": Only manual creation via tools
# - "inference_with_review": Extract with human review before adding
mode = "inference_and_manual"

# Default context tags
default_contexts = ["@home", "@work", "@phone", "@computer", "@errand", "@anywhere"]

# Day of week for weekly review reminders
weekly_review_day = "Sunday"

# Days without activity before project is considered stalled
stalled_project_days = 7

# Duration (minutes) for 2-minute rule quick tasks
quick_task_minutes = 2

# Automatically create projects from extracted items
auto_create_projects = true

# Link tasks to their source documents
auto_link_references = true

# ============================================
# CALENDAR INTELLIGENCE
# ============================================

[calendar]
# Enable calendar features
enabled = true

# Default timezone (IANA format)
default_timezone = "UTC"

# Working hours for free time calculations
working_hours_start = "09:00"
working_hours_end = "17:00"

# Auto-create calendar events from extracted dates
auto_create_events = true
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

### Ontology

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable ontology and entity extraction |
| `storage_backend` | string | `"embedded"` | Storage backend for entities |

#### Extraction Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `extract_on_index` | boolean | `true` | Auto-extract entities when indexing |
| `confidence_threshold` | float | `0.7` | Minimum confidence to store entities (0.0-1.0) |

#### Local Extraction

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_temporal` | boolean | `true` | Parse dates, times, and recurring patterns |
| `enable_patterns` | boolean | `true` | Regex-based entity detection |
| `enable_embedding_ner` | boolean | `true` | Use embeddings for entity clustering |

#### LLM Extraction (Optional)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable LLM-based extraction |
| `provider` | string | `"openai"` | LLM provider: `openai`, `anthropic`, `local` |
| `model` | string | `"gpt-4o-mini"` | Model for extraction |
| `extract_tasks` | boolean | `true` | Extract action items from prose |
| `extract_relationships` | boolean | `true` | Extract entity relationships |
| `max_tokens_per_doc` | integer | `4000` | Token limit per document |
| `rate_limit_rpm` | integer | `60` | Rate limit (requests per minute) |

### GTD

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable GTD features |
| `mode` | string | `"inference_and_manual"` | GTD mode (see below) |
| `default_contexts` | array | `["@home", "@work", ...]` | Available context tags |
| `weekly_review_day` | string | `"Sunday"` | Day for weekly review |
| `stalled_project_days` | integer | `7` | Days until project is stalled |
| `quick_task_minutes` | integer | `2` | 2-minute rule threshold |
| `auto_create_projects` | boolean | `true` | Auto-create projects from extraction |
| `auto_link_references` | boolean | `true` | Link tasks to source documents |

#### GTD Modes

| Mode | Description |
|------|-------------|
| `inference_and_manual` | Auto-extract from documents + manual creation via tools |
| `manual_only` | Only manual creation through GTD tools |
| `inference_with_review` | Extract with human review before adding |

### Calendar

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable calendar features |
| `default_timezone` | string | `"UTC"` | Default timezone (IANA format) |
| `working_hours_start` | string | `"09:00"` | Working hours start |
| `working_hours_end` | string | `"17:00"` | Working hours end |
| `auto_create_events` | boolean | `true` | Auto-create events from extracted dates |

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

### GTD-Focused Setup

```toml
[embedding]
provider = "local"

[storage]
backend = "embedded"

[ontology]
enabled = true

[ontology.extraction]
extract_on_index = true
confidence_threshold = 0.7

[ontology.extraction.local]
enable_temporal = true
enable_patterns = true

[gtd]
enabled = true
mode = "inference_and_manual"
default_contexts = ["@home", "@work", "@phone", "@computer", "@errand", "@waiting"]
stalled_project_days = 7
quick_task_minutes = 2
auto_create_projects = true

[calendar]
enabled = true
default_timezone = "America/New_York"
working_hours_start = "09:00"
working_hours_end = "18:00"
```

### LLM-Enhanced Extraction

```toml
[ontology]
enabled = true

[ontology.extraction.local]
enable_temporal = true
enable_patterns = true

[ontology.extraction.llm]
enabled = true
provider = "openai"
model = "gpt-4o-mini"
extract_tasks = true
extract_relationships = true
rate_limit_rpm = 30
```

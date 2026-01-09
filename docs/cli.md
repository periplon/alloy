# CLI Reference

Alloy provides a full-featured command-line interface for indexing and searching documents without requiring an MCP client connection.

## Overview

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

Options:
  -c, --config <PATH>   Path to configuration file
      --json            Output as JSON
  -r, --remote <URL>    Connect to remote Alloy server via MCP
  -h, --help            Print help
  -V, --version         Print version
```

## Global Options

### `--config <PATH>`

Specify a custom configuration file path. If not provided, Alloy looks for configuration in the standard locations (see [Configuration](configuration.md)).

```bash
alloy --config ~/myconfig.toml index ~/docs
```

### `--json`

Output results as JSON instead of human-readable format. Useful for scripting and automation.

```bash
alloy --json search "query" | jq '.results[0].path'
alloy --json stats > stats.json
```

### `--remote <URL>`

Connect to a remote Alloy server instead of using local storage. The remote server must be running with HTTP transport.

```bash
# Local server
alloy serve --transport http --port 8080

# Remote client
alloy --remote http://localhost:8080 search "query"
alloy -r http://server:8080 stats
```

## Commands

### index

Index a local directory or S3 URI for search.

```
alloy index [OPTIONS] <PATH>

Arguments:
  <PATH>  Path to index (local path or S3 URI like s3://bucket/prefix)

Options:
  -p, --pattern <PATTERN>  Glob pattern to filter files (e.g., "*.md")
  -w, --watch              Watch for changes and auto re-index
      --recursive          Recursive indexing (default: true)
```

**Examples:**

```bash
# Index all files in a directory
alloy index ~/documents

# Index only markdown files
alloy index ~/projects --pattern "*.md"

# Index with file watching
alloy index ~/notes --watch

# Index code files
alloy index ~/code/myapp --pattern "**/*.{rs,py,js,ts}"

# Index S3 bucket
alloy index s3://my-bucket/docs

# Index S3 with pattern
alloy index s3://my-bucket/documents --pattern "*.pdf"
```

**Output:**

```
Indexed 42 documents
Source ID: src_abc123def456
Chunks created: 156
Watching: false
Successfully indexed from ~/documents
```

### search

Search indexed documents using hybrid vector + full-text search.

```
alloy search [OPTIONS] <QUERY>

Arguments:
  <QUERY>  Search query text

Options:
  -l, --limit <LIMIT>            Maximum results (default: 10)
  -w, --vector-weight <WEIGHT>   Vector search weight 0.0-1.0 (default: 0.5)
  -s, --source <SOURCE_ID>       Filter by source ID
```

**Examples:**

```bash
# Basic search
alloy search "async error handling"

# Get more results
alloy search "configuration" --limit 20

# Semantic-focused search (conceptual queries)
alloy search "how to handle failures gracefully" --vector-weight 0.8

# Keyword-focused search (exact matches)
alloy search "RUST_LOG environment" --vector-weight 0.2

# Filter by source
alloy search "TODO" --source src_abc123

# JSON output for scripting
alloy --json search "query" | jq '.results[] | {path, score}'
```

**Output:**

```
Found 5 results (42ms)

1. [0.872] /home/user/docs/error-handling.md
   "Error handling in async Rust requires careful consideration of..."

2. [0.756] /home/user/docs/patterns.md
   "Common patterns for handling errors include Result types and..."

3. [0.698] /home/user/code/main.rs
   "fn handle_error(e: Error) -> Result<()> { ... }"
```

#### Vector Weight Guide

| Weight | Behavior |
|--------|----------|
| `0.0` | Pure full-text search (keyword matching) |
| `0.3` | Primarily keywords with some semantic |
| `0.5` | Balanced hybrid (default) |
| `0.7` | Primarily semantic with keyword boost |
| `1.0` | Pure vector search (semantic similarity) |

### get

Retrieve a document by its ID.

```
alloy get [OPTIONS] <DOCUMENT_ID>

Arguments:
  <DOCUMENT_ID>  Document ID

Options:
      --no-content  Don't include document content (metadata only)
```

**Examples:**

```bash
# Get document with content
alloy get doc_xyz789

# Get metadata only (faster)
alloy get doc_xyz789 --no-content

# JSON output
alloy --json get doc_xyz789 | jq '.path, .size_bytes'
```

**Output:**

```
Document ID: doc_xyz789
Source ID: src_abc123
Path: /home/user/docs/guide.md
MIME Type: text/markdown
Size: 4096 bytes
Chunks: 8
Modified: 2024-01-15T10:30:00Z
Indexed: 2024-01-15T10:35:00Z

--- Content ---
# Getting Started Guide

This guide covers...
```

### sources

List all indexed sources.

```
alloy sources
```

**Examples:**

```bash
# List sources
alloy sources

# JSON output
alloy --json sources | jq '.sources[] | {id: .source_id, docs: .document_count}'
```

**Output:**

```
SOURCE ID                                TYPE       DOCS     WATCH    PATH
----------------------------------------------------------------------------------------------------
src_abc123def456789012345678901234       local      42       yes      /home/user/docs
src_def456abc789012345678901234567       s3         100      no       s3://my-bucket/documents

Total: 2 sources
```

### remove

Remove an indexed source and all its documents.

```
alloy remove <SOURCE_ID>

Arguments:
  <SOURCE_ID>  Source ID to remove
```

**Examples:**

```bash
# Remove a source
alloy remove src_abc123def456

# Get source ID from sources list first
alloy sources
alloy remove src_abc123def456789012345678901234
```

**Output:**

```
Removed 42 documents
Removed source src_abc123def456 with 42 documents
```

### stats

Show index statistics.

```
alloy stats
```

**Examples:**

```bash
# View stats
alloy stats

# JSON output for monitoring
alloy --json stats | jq '{docs: .document_count, chunks: .chunk_count}'
```

**Output:**

```
Index Statistics
========================================
Sources:            3
Documents:          150
Chunks:             1200
Storage:            52428800 bytes
Embedding Dim:      384
Storage Backend:    Embedded
Embedding Provider: Local
Uptime:             3600s
```

### serve

Run Alloy as an MCP server.

```
alloy serve [OPTIONS]

Options:
  -t, --transport <TYPE>  Transport type: stdio or http (default: stdio)
  -p, --port <PORT>       HTTP port (default: 8080)
      --json-logs         Enable JSON logging format
```

**Examples:**

```bash
# Stdio transport (for MCP clients)
alloy serve

# HTTP transport (for network access)
alloy serve --transport http --port 8080

# With JSON logging
alloy serve --transport http --json-logs
```

## Remote Mode

Remote mode allows you to connect to a running Alloy server instead of using local storage. This is useful for:

- Querying a shared index from multiple machines
- Running searches without local index files
- Testing server deployments

### Setup

1. Start the server with HTTP transport:

```bash
alloy serve --transport http --port 8080
```

2. Use CLI with `--remote`:

```bash
alloy --remote http://localhost:8080 search "query"
alloy -r http://server.example.com:8080 stats
```

All commands except `serve` work with remote mode:

```bash
alloy -r http://localhost:8080 index ~/docs
alloy -r http://localhost:8080 search "query"
alloy -r http://localhost:8080 sources
alloy -r http://localhost:8080 stats
alloy -r http://localhost:8080 get doc_id
alloy -r http://localhost:8080 remove src_id
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |

## Scripting Examples

### Batch Indexing

```bash
#!/bin/bash
for dir in ~/projects/*/; do
    alloy index "$dir" --pattern "*.md"
done
```

### Search and Process Results

```bash
# Find files containing "TODO" and extract paths
alloy --json search "TODO" --limit 100 | \
    jq -r '.results[].path' | \
    sort -u

# Get total document count
alloy --json stats | jq '.document_count'
```

### Monitoring

```bash
# Watch stats every 5 seconds
watch -n 5 'alloy --json stats | jq "{docs: .document_count, chunks: .chunk_count}"'
```

### Remote Backup Check

```bash
# Verify remote server is healthy
alloy -r http://server:8080 --json stats | jq '.source_count > 0'
```

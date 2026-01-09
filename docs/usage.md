# Usage Guide

This guide covers common usage patterns and best practices for Alloy. Examples are shown for both CLI and MCP tool usage.

## Indexing Strategies

### Local Directories

Index a single directory:

**CLI:**
```bash
alloy index ~/documents
```

**MCP:**
```
index_path(path: "~/documents")
```

Index with a file filter:

**CLI:**
```bash
alloy index ~/projects --pattern "*.{md,txt,rst}"
```

**MCP:**
```
index_path(path: "~/projects", pattern: "*.{md,txt,rst}")
```

Index code repositories:

**CLI:**
```bash
alloy index ~/code/myproject --pattern "**/*.{rs,py,js,ts}"
```

**MCP:**
```
index_path(path: "~/code/myproject", pattern: "**/*.{rs,py,js,ts}")
```

### S3 Buckets

Index an entire bucket:

**CLI:**
```bash
alloy index s3://my-bucket
```

**MCP:**
```
index_path(path: "s3://my-bucket")
```

Index a specific prefix:

**CLI:**
```bash
alloy index s3://my-bucket/documents/2024
```

**MCP:**
```
index_path(path: "s3://my-bucket/documents/2024")
```

Index with pattern filter:

**CLI:**
```bash
alloy index s3://my-bucket/docs --pattern "*.pdf"
```

**MCP:**
```
index_path(path: "s3://my-bucket/docs", pattern: "*.pdf")
```

### File Watching

Enable automatic re-indexing when files change:

**CLI:**
```bash
alloy index ~/notes --watch
```

**MCP:**
```
index_path(path: "~/notes", watch: true)
```

File watching works for local directories only. Changes are detected and re-indexed automatically.

## Search Techniques

### Basic Search

Simple keyword search:

**CLI:**
```bash
alloy search "authentication flow"
```

**MCP:**
```
search(query: "authentication flow")
```

### Tuning Vector Weight

The `vector_weight` parameter controls the balance between semantic and keyword matching:

**Semantic-focused** (good for conceptual queries):

**CLI:**
```bash
alloy search "how to handle errors gracefully" --vector-weight 0.8
```

**MCP:**
```
search(query: "how to handle errors gracefully", vector_weight: 0.8)
```

**Keyword-focused** (good for exact matches):

**CLI:**
```bash
alloy search "RUST_LOG environment" --vector-weight 0.2
```

**MCP:**
```
search(query: "RUST_LOG environment", vector_weight: 0.2)
```

**Balanced** (default):

**CLI:**
```bash
alloy search "database connection pooling"
```

**MCP:**
```
search(query: "database connection pooling", vector_weight: 0.5)
```

### Filtering Results

Filter by source:

**CLI:**
```bash
alloy search "configuration" --source src_abc123
```

**MCP:**
```
search(query: "configuration", source_id: "src_abc123")
```

Limit results:

**CLI:**
```bash
alloy search "TODO" --limit 50
```

**MCP:**
```
search(query: "TODO", limit: 50)
```

### Search Tips

1. **Use natural language** for semantic search - "how do I configure logging" works better than "config log"

2. **Use exact terms** with low vector weight for code search - searching for function names, variable names, or error messages

3. **Increase limit** when exploring - use `limit: 50` to see more context

4. **Filter by source** when you know where to look - speeds up search and reduces noise

## Working with Results

### Understanding Scores

Scores range from 0.0 to 1.0:

- `0.9+` - Excellent match
- `0.7-0.9` - Good match
- `0.5-0.7` - Moderate relevance
- `<0.5` - Weak match

### Using Highlights

Search results include highlighted matching terms. Use these to quickly identify relevant sections.

### Fetching Full Documents

After finding relevant chunks, get the full document:

**CLI:**
```bash
alloy get doc_xyz789
```

**MCP:**
```
get_document(document_id: "doc_xyz789")
```

For metadata only (faster):

**CLI:**
```bash
alloy get doc_xyz789 --no-content
```

**MCP:**
```
get_document(document_id: "doc_xyz789", include_content: false)
```

## Managing Sources

### Listing Sources

View all indexed sources:

**CLI:**
```bash
alloy sources
```

**MCP:**
```
list_sources()
```

### Monitoring Index Health

Check index statistics:

**CLI:**
```bash
alloy stats
```

**MCP:**
```
get_stats()
```

Key metrics to monitor:

- `document_count` - Total indexed documents
- `chunk_count` - Total chunks (documents are split into chunks)
- `storage_bytes` - Storage space used

### Removing Sources

Remove a source and all its documents:

**CLI:**
```bash
alloy remove src_abc123
```

**MCP:**
```
remove_source(source_id: "src_abc123")
```

## Supported File Types

Alloy processes these file types:

| Type | Extensions | Processing |
|------|------------|------------|
| Text | `.txt`, `.md`, `.rst`, `.csv` | Direct text extraction |
| Code | `.rs`, `.py`, `.js`, `.ts`, `.go`, etc. | Syntax-aware chunking |
| Documents | `.pdf`, `.docx` | Document parsing |
| Images | `.png`, `.jpg`, `.jpeg` | OCR + CLIP embeddings |

## Performance Tips

### Chunking Configuration

Adjust chunk size based on your content:

- **Code**: Smaller chunks (256-384 tokens) preserve function boundaries
- **Prose**: Larger chunks (512-768 tokens) maintain context
- **Mixed**: Default (512 tokens) works well

### Embedding Provider Choice

**Local embeddings** (`provider = "local"`):

- No network latency
- No API costs
- Good for most use cases

**API embeddings** (`provider = "api"`):

- Higher quality for some models
- Requires network access
- Consider for production deployments

### Storage Backend Choice

**Embedded** (`backend = "embedded"`):

- Zero configuration
- Good for single-user scenarios
- Data stored locally

**Qdrant** (`backend = "qdrant"`):

- Better for large indexes
- Supports distributed deployment
- Requires running Qdrant server

## Troubleshooting

### No Results Found

1. Check that the source is indexed: `alloy sources` or `list_sources()`
2. Verify document count is non-zero: `alloy stats`
3. Try a simpler query
4. Adjust vector_weight (try both extremes: 0.1 and 0.9)

### Slow Indexing

1. Check `max_concurrent_tasks` in config
2. For large files, processing takes time
3. API embeddings have network latency

### Index Corruption

If the index becomes corrupted:

1. Stop the server
2. Delete the data directory (see [Configuration](configuration.md))
3. Restart and re-index

### Connection Issues

For HTTP transport:

1. Verify the port is available
2. Check firewall settings
3. Confirm the URL format: `http://localhost:8080`

For S3 sources:

1. Verify AWS credentials are set
2. Check bucket permissions
3. Confirm the S3 URI format: `s3://bucket/prefix`

## Remote Mode (CLI)

The CLI supports connecting to a remote Alloy server instead of using local storage.

### Setup

Start the server with HTTP transport:

```bash
alloy serve --transport http --port 8080
```

### Usage

Use the `--remote` (or `-r`) flag to connect:

```bash
# Search on remote server
alloy --remote http://localhost:8080 search "query"

# Check remote stats
alloy -r http://server.example.com:8080 stats

# List remote sources
alloy -r http://localhost:8080 sources

# Index via remote (files must be accessible from the server)
alloy -r http://localhost:8080 index /data/docs
```

### Use Cases

- **Shared indexes**: Multiple users can query the same index
- **Resource separation**: Run indexing on a powerful server, query from lightweight clients
- **CI/CD integration**: Query an index from build pipelines
- **Monitoring**: Check index health from management scripts

### JSON Output for Scripting

Combine with `--json` for automation:

```bash
# Get document count
alloy --json -r http://localhost:8080 stats | jq '.document_count'

# Extract search result paths
alloy --json -r http://localhost:8080 search "TODO" | jq -r '.results[].path'

# Monitor in a loop
watch -n 30 'alloy --json -r http://localhost:8080 stats | jq "{docs: .document_count}"'
```

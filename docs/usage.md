# Usage Guide

This guide covers common usage patterns and best practices for Alloy.

## Indexing Strategies

### Local Directories

Index a single directory:

```
index_path(path: "~/documents")
```

Index with a file filter:

```
index_path(path: "~/projects", pattern: "*.{md,txt,rst}")
```

Index code repositories:

```
index_path(path: "~/code/myproject", pattern: "**/*.{rs,py,js,ts}")
```

### S3 Buckets

Index an entire bucket:

```
index_path(path: "s3://my-bucket")
```

Index a specific prefix:

```
index_path(path: "s3://my-bucket/documents/2024")
```

Index with pattern filter:

```
index_path(path: "s3://my-bucket/docs", pattern: "*.pdf")
```

### File Watching

Enable automatic re-indexing when files change:

```
index_path(path: "~/notes", watch: true)
```

File watching works for local directories only. Changes are detected and re-indexed automatically.

## Search Techniques

### Basic Search

Simple keyword search:

```
search(query: "authentication flow")
```

### Tuning Vector Weight

The `vector_weight` parameter controls the balance between semantic and keyword matching:

**Semantic-focused** (good for conceptual queries):

```
search(query: "how to handle errors gracefully", vector_weight: 0.8)
```

**Keyword-focused** (good for exact matches):

```
search(query: "RUST_LOG environment", vector_weight: 0.2)
```

**Balanced** (default):

```
search(query: "database connection pooling", vector_weight: 0.5)
```

### Filtering Results

Filter by source:

```
search(query: "configuration", source_id: "src_abc123")
```

Limit results:

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

```
get_document(document_id: "doc_xyz789")
```

For metadata only (faster):

```
get_document(document_id: "doc_xyz789", include_content: false)
```

## Managing Sources

### Listing Sources

View all indexed sources:

```
list_sources()
```

### Monitoring Index Health

Check index statistics:

```
get_stats()
```

Key metrics to monitor:

- `document_count` - Total indexed documents
- `chunk_count` - Total chunks (documents are split into chunks)
- `storage_bytes` - Storage space used

### Removing Sources

Remove a source and all its documents:

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

1. Check that the source is indexed: `list_sources()`
2. Verify document count is non-zero
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

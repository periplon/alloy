# Tools Reference

Alloy exposes the following MCP tools for document indexing and search.

## index_path

Index a local path or S3 URI for hybrid search.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `path` | string | Yes | - | Local path or S3 URI (`s3://bucket/prefix`) |
| `pattern` | string | No | - | Glob pattern to filter files (e.g., `*.md`, `**/*.py`) |
| `watch` | boolean | No | `false` | Watch for file changes and auto-reindex |
| `recursive` | boolean | No | `true` | Recursively index subdirectories |

### Response

```json
{
  "source_id": "src_abc123",
  "documents_indexed": 42,
  "chunks_created": 156,
  "watching": true,
  "message": "Successfully indexed 42 documents from ~/projects"
}
```

### Examples

```
# Index markdown files
index_path(path: "~/docs", pattern: "*.md")

# Index with file watching
index_path(path: "~/projects/myapp", watch: true)

# Index S3 bucket
index_path(path: "s3://my-bucket/documents")

# Index specific file types
index_path(path: "~/code", pattern: "**/*.{rs,py,js}")
```

## search

Search indexed documents using hybrid search combining vector similarity and full-text matching.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `limit` | integer | No | `10` | Maximum number of results |
| `vector_weight` | float | No | `0.5` | Weight for vector search (0.0-1.0). Higher values favor semantic similarity |
| `source_id` | string | No | - | Filter results to a specific source |

### Response

```json
{
  "results": [
    {
      "document_id": "doc_xyz789",
      "chunk_id": "chunk_001",
      "source_id": "src_abc123",
      "path": "/home/user/docs/guide.md",
      "content": "Matching text content...",
      "score": 0.87,
      "highlights": ["matching", "text"],
      "metadata": {}
    }
  ],
  "total_matches": 15,
  "took_ms": 42
}
```

### Examples

```
# Basic search
search(query: "async rust error handling")

# More results with semantic focus
search(query: "machine learning", limit: 20, vector_weight: 0.8)

# Full-text focused search
search(query: "TODO fixme", vector_weight: 0.2)

# Search within a specific source
search(query: "configuration", source_id: "src_abc123")
```

### Vector Weight Guide

| Weight | Behavior |
|--------|----------|
| `0.0` | Pure full-text search (keyword matching) |
| `0.3` | Primarily keywords with some semantic |
| `0.5` | Balanced hybrid (default) |
| `0.7` | Primarily semantic with keyword boost |
| `1.0` | Pure vector search (semantic similarity) |

## get_document

Retrieve a document by ID with optional full content.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `document_id` | string | Yes | - | Document ID |
| `include_content` | boolean | No | `true` | Include full document content |

### Response

```json
{
  "document_id": "doc_xyz789",
  "source_id": "src_abc123",
  "path": "/home/user/docs/guide.md",
  "mime_type": "text/markdown",
  "size_bytes": 4096,
  "chunk_count": 8,
  "content": "Full document content...",
  "modified_at": "2024-01-15T10:30:00Z",
  "indexed_at": "2024-01-15T10:35:00Z",
  "metadata": {}
}
```

### Examples

```
# Get document with content
get_document(document_id: "doc_xyz789")

# Get metadata only
get_document(document_id: "doc_xyz789", include_content: false)
```

## list_sources

List all indexed sources with their status and document counts.

### Parameters

None.

### Response

```json
{
  "sources": [
    {
      "source_id": "src_abc123",
      "source_type": "local",
      "path": "/home/user/docs",
      "document_count": 42,
      "watching": true,
      "last_scan": "2024-01-15T10:30:00Z",
      "status": "indexed"
    },
    {
      "source_id": "src_def456",
      "source_type": "s3",
      "path": "s3://my-bucket/documents",
      "document_count": 100,
      "watching": false,
      "last_scan": "2024-01-14T08:00:00Z",
      "status": "indexed"
    }
  ]
}
```

## remove_source

Remove an indexed source and all its documents from the index.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `source_id` | string | Yes | - | Source ID to remove |

### Response

```json
{
  "success": true,
  "documents_removed": 42,
  "message": "Removed source src_abc123 with 42 documents"
}
```

### Examples

```
# Remove a source
remove_source(source_id: "src_abc123")
```

## get_stats

Get statistics about the index including document count, storage size, and configuration.

### Parameters

None.

### Response

```json
{
  "source_count": 3,
  "document_count": 150,
  "chunk_count": 1200,
  "storage_bytes": 52428800,
  "embedding_dimension": 384,
  "storage_backend": "Embedded",
  "embedding_provider": "Local",
  "uptime_secs": 3600
}
```

## configure

Update runtime configuration settings.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `updates` | object | Yes | - | Configuration updates as JSON |

### Response

```json
{
  "success": true,
  "config": {
    "server": { "transport": "Stdio", "http_port": 8080 },
    "embedding": { "provider": "Local", "model": "BAAI/bge-small-en-v1.5" },
    "storage": { "backend": "Embedded", "data_dir": "~/.local/share/alloy" },
    "processing": { "chunk_size": 512, "chunk_overlap": 64 }
  },
  "message": "Configuration update acknowledged"
}
```

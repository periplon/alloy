# Tools Reference

Alloy exposes MCP tools for document indexing, search, GTD workflow, calendar intelligence, and knowledge queries.

## index_path

Index a local path or S3 URI for hybrid search. Supports creating missing directories, indexing empty sources, per-source ontology extraction control, and custom naming.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `path` | string | Yes | - | Local path or S3 URI (`s3://bucket/prefix`) |
| `pattern` | string | No | - | Glob pattern to filter files (e.g., `*.md`, `**/*.py`) |
| `watch` | boolean | No | `false` | Watch for file changes and auto-reindex |
| `recursive` | boolean | No | `true` | Recursively index subdirectories |
| `create_if_missing` | boolean | No | `false` | Create the directory if it doesn't exist (local paths only) |
| `extract_ontology` | boolean | No | - | Override ontology extraction for this source (see notes) |
| `name` | string | No | - | Human-readable name for the source (for search and display) |
| `description` | string | No | - | Description of the source contents or purpose |

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

# Create directory if it doesn't exist and register as source
index_path(path: "~/new-docs", create_if_missing: true)

# Create and watch a new directory for incoming documents
index_path(path: "~/inbox", create_if_missing: true, watch: true)

# Enable ontology extraction for this source (even if globally disabled)
index_path(path: "~/meeting-notes", extract_ontology: true)

# Disable ontology extraction for this source (even if globally enabled)
index_path(path: "~/raw-data", extract_ontology: false)

# Index with a human-readable name and description
index_path(
  path: "~/projects/backend",
  name: "Backend API",
  description: "Rust backend service with REST API endpoints"
)

# Named S3 source with description
index_path(
  path: "s3://company-docs/engineering",
  name: "Engineering Docs",
  description: "Technical documentation, RFCs, and architecture decisions"
)
```

### Notes

- **Empty sources**: Sources can be registered with 0 documents. Files added later will be indexed if `watch: true` is enabled, or on the next scan.
- **Local paths**: Use `create_if_missing: true` to automatically create directories that don't exist. Parent directories are created recursively.
- **S3 sources**: Buckets must exist beforehand. Empty prefixes are supported and will index 0 documents until objects are added.
- **Ontology extraction**: The `extract_ontology` parameter overrides the global `ontology.extraction.extract_on_index` config setting for this source only:
  - `extract_ontology: true` - Force extraction even if globally disabled
  - `extract_ontology: false` - Skip extraction even if globally enabled
  - Not specified - Use global config setting
- **Source naming**: Use `name` and `description` to make sources easier to identify and search. The name can be used to find sources in `list_sources` and filter searches by source.

## create_source

Create and index a source with comprehensive configuration options. This is the advanced version of `index_path` that exposes all source configuration options including file patterns, symlink handling, and S3-specific settings.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `path` | string | Yes | - | Local path or S3 URI (`s3://bucket/prefix`) |
| `name` | string | No | - | Human-readable name for the source |
| `description` | string | No | - | Description of the source contents or purpose |
| `patterns` | string[] | No | `["**/*"]` | Glob patterns to include files (e.g., `["*.md", "**/*.py"]`) |
| `exclude_patterns` | string[] | No | (see notes) | Glob patterns to exclude files (e.g., `["**/node_modules/**"]`) |
| `watch` | boolean | No | `false` | Watch for file changes and auto-reindex (local sources only) |
| `create_if_missing` | boolean | No | `false` | Create the directory if it doesn't exist (local sources only) |
| `follow_symlinks` | boolean | No | `false` | Follow symbolic links when scanning (local sources only) |
| `extract_ontology` | boolean | No | - | Override ontology extraction for this source |
| `region` | string | No | (auto) | AWS region for S3 sources |
| `endpoint_url` | string | No | - | Custom S3 endpoint URL (for MinIO, LocalStack, etc.) |
| `poll_interval_secs` | integer | No | `300` | Polling interval in seconds for S3 change detection |

### Response

```json
{
  "source_id": "/path/to/source",
  "name": "My Documents",
  "source_type": "local",
  "documents_indexed": 42,
  "chunks_created": 156,
  "watching": true,
  "extract_ontology": true,
  "message": "Successfully created source 'My Documents' with 42 documents indexed"
}
```

### Examples

```
# Basic local source with all defaults
create_source(path: "~/documents")

# Local source with custom patterns
create_source(
  path: "~/code/project",
  name: "My Project",
  description: "Main development project with Rust and Python",
  patterns: ["**/*.rs", "**/*.py", "**/*.md"],
  exclude_patterns: ["**/target/**", "**/__pycache__/**", "**/venv/**"],
  watch: true
)

# Create a watched inbox folder for new documents
create_source(
  path: "~/inbox",
  name: "Document Inbox",
  description: "Folder for incoming documents to be processed",
  create_if_missing: true,
  watch: true,
  extract_ontology: true
)

# S3 source with custom settings
create_source(
  path: "s3://my-bucket/documents",
  name: "Company Documents",
  description: "Shared company documentation",
  patterns: ["*.pdf", "*.docx", "*.md"],
  region: "us-west-2",
  poll_interval_secs: 60
)

# MinIO/LocalStack source
create_source(
  path: "s3://local-bucket/data",
  name: "Local Development Data",
  endpoint_url: "http://localhost:9000",
  region: "us-east-1"
)

# Source with symlink following
create_source(
  path: "~/linked-docs",
  name: "Linked Documents",
  follow_symlinks: true,
  patterns: ["**/*.md"]
)
```

### Notes

- **Default exclude patterns** (local sources): `**/node_modules/**`, `**/.git/**`, `**/target/**`, `**/__pycache__/**`, `**/venv/**`, `**/.venv/**`
- **Patterns**: Use glob syntax. Multiple patterns are ORed together. Examples:
  - `*.md` - Markdown files in root
  - `**/*.md` - Markdown files recursively
  - `**/*.{rs,py,js}` - Multiple extensions
  - `docs/**/*` - All files under docs/
- **Symlinks**: By default, symlinks are not followed to prevent infinite loops. Enable `follow_symlinks` if your directory structure requires it.
- **S3 polling**: S3 sources don't have native file watching. Instead, they poll for changes at the configured interval.
- **Ontology extraction**: See `index_path` notes for details on ontology extraction override behavior.

### Comparison with index_path

| Feature | index_path | create_source |
|---------|------------|---------------|
| Basic indexing | ✓ | ✓ |
| Single pattern | ✓ | ✓ |
| Multiple patterns | - | ✓ |
| Exclude patterns | - | ✓ |
| Follow symlinks | - | ✓ |
| S3 region | - | ✓ |
| S3 endpoint URL | - | ✓ |
| S3 poll interval | - | ✓ |

Use `index_path` for quick, simple indexing. Use `create_source` when you need fine-grained control over source configuration.

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

## document_add

Add a document directly to the store from text or base64-encoded content without indexing from a file path. Supports text, markdown, PDF, and other formats. Optionally extracts entities, relationships, and temporal information for the knowledge graph.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `content` | string | Yes | - | Document content (raw text or base64-encoded binary) |
| `content_type` | string | No | `text` | How to interpret content: `text` or `base64` for binary files |
| `mime_type` | string | No | `text/plain` | MIME type for processing (e.g., `text/plain`, `text/markdown`, `application/pdf`) |
| `title` | string | No | - | Document title/name for metadata |
| `source_id` | string | No | `direct-add` | Source ID for grouping documents |
| `extract_ontology` | boolean | No | `false` | Extract entities and relationships for knowledge graph |
| `metadata` | object | No | - | Additional metadata as JSON |

### Response

```json
{
  "success": true,
  "document_id": "doc_abc123",
  "source_id": "direct-add",
  "chunks_created": 3,
  "entities_extracted": 5,
  "relationships_extracted": 2,
  "entity_names": ["Alice Chen", "Project Alpha", "Acme Corp"],
  "processing_ms": 156,
  "message": "Document added successfully"
}
```

### Supported MIME Types

| MIME Type | Description |
|-----------|-------------|
| `text/plain` | Plain text |
| `text/markdown` | Markdown documents |
| `application/pdf` | PDF files (use base64 encoding) |
| `text/html` | HTML documents |
| `application/json` | JSON files |

### Examples

```
# Add plain text document
document_add(
  content: "Meeting notes from Q1 planning session...",
  title: "Q1 Planning Notes",
  source_id: "meetings"
)

# Add markdown with ontology extraction
document_add(
  content: "# Project Alpha\n\nAlice Chen leads the team...",
  mime_type: "text/markdown",
  title: "Project Alpha Overview",
  extract_ontology: true
)

# Add PDF document (base64 encoded)
document_add(
  content: "JVBERi0xLjQKJeLjz9...",
  content_type: "base64",
  mime_type: "application/pdf",
  title: "Contract Draft",
  metadata: { "department": "Legal", "version": "1.0" }
)

# Add to custom source for grouping
document_add(
  content: "API documentation for the payment service...",
  source_id: "api-docs",
  title: "Payment API",
  metadata: { "service": "payments", "version": "2.1" }
)
```

### Use Cases

- **Chat context injection**: Add conversation context or user-provided information directly
- **API integrations**: Ingest documents from external APIs without writing to disk
- **Dynamic content**: Add generated or transformed content to the index
- **Meeting transcripts**: Inject transcripts with ontology extraction to build knowledge graph
- **Quick notes**: Add ad-hoc notes without file system management

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

---

# GTD Tools

## gtd_projects

Manage GTD projects with health scoring and filtering.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `action` | string | Yes | - | Action: `list`, `get`, `create`, `update`, `archive` |
| `project_id` | string | No | - | Project ID (for get/update/archive) |
| `filters` | object | No | - | Filter criteria |

#### Filter Options

| Name | Type | Description |
|------|------|-------------|
| `status` | string | `active`, `on_hold`, `completed`, `archived` |
| `area` | string | Filter by area of focus |
| `has_next_action` | boolean | Projects with/without defined next actions |
| `stalled_days` | integer | Projects without activity for N days |

### Response

```json
{
  "projects": [
    {
      "id": "proj_abc123",
      "name": "Website Redesign",
      "outcome": "Launch new company website with improved UX",
      "status": "active",
      "area": "Marketing",
      "health_score": 85,
      "health": {
        "has_next_action": true,
        "recent_activity": true,
        "clear_outcome": true,
        "reasonable_scope": true,
        "no_blockers": false,
        "linked_to_goal": true,
        "recommendations": ["Resolve blocking task: waiting on design assets"]
      },
      "task_count": 12,
      "next_action": {
        "id": "task_xyz",
        "description": "Review homepage mockups"
      },
      "created_at": "2024-01-10T09:00:00Z",
      "last_activity": "2024-01-14T15:30:00Z"
    }
  ]
}
```

### Examples

```
# List active projects
gtd_projects(action: "list", filters: { status: "active" })

# Get stalled projects (no activity in 7+ days)
gtd_projects(action: "list", filters: { stalled_days: 7 })

# Projects without next actions (need attention)
gtd_projects(action: "list", filters: { has_next_action: false })

# Create a new project
gtd_projects(action: "create", project: {
  name: "Q2 Product Launch",
  outcome: "Successfully launch v2.0 to market",
  area: "Product"
})
```

## gtd_tasks

Manage tasks with context filtering, energy levels, and recommendations.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `action` | string | Yes | - | Action: `list`, `get`, `create`, `update`, `complete`, `recommend` |
| `task_id` | string | No | - | Task ID (for get/update/complete) |
| `filters` | object | No | - | Filter criteria |

#### Filter Options

| Name | Type | Description |
|------|------|-------------|
| `contexts` | array | Context tags: `@home`, `@work`, `@phone`, `@computer`, `@errand` |
| `project` | string | Filter by project ID |
| `status` | string | `next`, `scheduled`, `waiting`, `someday`, `done` |
| `energy_level` | string | `low`, `medium`, `high` |
| `time_available` | integer | Maximum duration in minutes |
| `due_before` | string | ISO date for deadline filter |
| `priority` | string | `low`, `medium`, `high`, `critical` |

### Response

```json
{
  "tasks": [
    {
      "id": "task_xyz789",
      "description": "Review homepage mockups",
      "project": "proj_abc123",
      "contexts": ["@computer"],
      "status": "next",
      "energy_level": "medium",
      "estimated_duration": 30,
      "due_date": "2024-01-20T17:00:00Z",
      "priority": "high",
      "source_document": "doc_meeting_notes",
      "created_at": "2024-01-14T10:00:00Z"
    }
  ]
}
```

### Task Recommendations

When using `action: "recommend"`, the system scores tasks based on:
- Context match (+30 points)
- Energy level match (+20 points)
- Time available fit (+15 points)
- Due date urgency (+20 points)
- Priority (+15 points)

### Examples

```
# Get tasks for @computer context
gtd_tasks(action: "list", filters: { contexts: ["@computer"] })

# Low energy tasks I can do quickly
gtd_tasks(action: "list", filters: { energy_level: "low", time_available: 15 })

# Get smart recommendations for current context
gtd_tasks(action: "recommend", filters: {
  contexts: ["@computer"],
  energy_level: "medium",
  time_available: 60
})

# Tasks due this week
gtd_tasks(action: "list", filters: { due_before: "2024-01-21" })
```

## gtd_waiting

Track delegated items and follow-ups.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `action` | string | Yes | - | Action: `list`, `add`, `resolve`, `remind` |
| `waiting_id` | string | No | - | Waiting item ID |
| `filters` | object | No | - | Filter criteria |

#### Filter Options

| Name | Type | Description |
|------|------|-------------|
| `status` | string | `pending`, `overdue`, `resolved` |
| `delegated_to` | string | Person name or ID |
| `project` | string | Project ID |

### Response

```json
{
  "waiting_items": [
    {
      "id": "wait_abc123",
      "description": "Design mockups for landing page",
      "delegated_to": { "name": "Sarah", "id": "person_sarah" },
      "project": "proj_website",
      "delegated_date": "2024-01-10T09:00:00Z",
      "expected_by": "2024-01-17T17:00:00Z",
      "follow_up_dates": ["2024-01-14T10:00:00Z"],
      "status": "overdue",
      "days_waiting": 7
    }
  ]
}
```

### Examples

```
# List all overdue items
gtd_waiting(action: "list", filters: { status: "overdue" })

# Items waiting on specific person
gtd_waiting(action: "list", filters: { delegated_to: "John" })

# Add new waiting item
gtd_waiting(action: "add", item: {
  description: "Budget approval",
  delegated_to: "Finance Team",
  expected_by: "2024-01-25"
})

# Mark as resolved
gtd_waiting(action: "resolve", waiting_id: "wait_abc123", resolution: "Received mockups")
```

## gtd_someday

Manage someday/maybe items for future consideration.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `action` | string | Yes | - | Action: `list`, `add`, `activate`, `archive` |
| `item_id` | string | No | - | Item ID |
| `category` | string | No | - | Filter by category |

### Response

```json
{
  "items": [
    {
      "id": "someday_xyz",
      "description": "Learn Rust for embedded systems",
      "category": "Learning",
      "trigger": "When current project wraps up",
      "review_date": "2024-03-01",
      "created_at": "2024-01-01T10:00:00Z"
    }
  ]
}
```

### Examples

```
# List all someday items
gtd_someday(action: "list")

# Filter by category
gtd_someday(action: "list", category: "Travel")

# Add new item
gtd_someday(action: "add", item: {
  description: "Write a technical blog",
  category: "Side Projects",
  trigger: "When I have consistent free weekends"
})

# Activate (move to active projects)
gtd_someday(action: "activate", item_id: "someday_xyz")
```

## gtd_weekly_review

Generate comprehensive weekly review reports.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `week_ending` | string | No | today | End date of review week |
| `sections` | array | No | all | Sections to include |

#### Available Sections

- `inbox_status` - Unprocessed items count
- `completed_tasks` - Tasks completed this week
- `projects_review` - Active project health
- `stalled_projects` - Projects needing attention
- `waiting_for` - Overdue delegations
- `upcoming_calendar` - Next week preview
- `someday_maybe` - Deferred items review
- `areas_check` - Area of focus balance

### Response

```json
{
  "period": {
    "start": "2024-01-08",
    "end": "2024-01-14"
  },
  "inbox_count": 5,
  "tasks_completed": 23,
  "projects_active": 8,
  "stalled_projects": [
    {
      "id": "proj_xyz",
      "name": "Mobile App",
      "days_stalled": 12,
      "recommendation": "Define next physical action"
    }
  ],
  "waiting_overdue": [
    {
      "description": "Contract review",
      "delegated_to": "Legal",
      "days_overdue": 3
    }
  ],
  "upcoming_events": 4,
  "suggestions": [
    "Process 5 inbox items to achieve inbox zero",
    "Follow up on 2 overdue waiting items",
    "Review Mobile App project - stalled 12 days"
  ],
  "areas_needing_attention": ["Health", "Learning"]
}
```

### Examples

```
# Full weekly review
gtd_weekly_review()

# Focus on specific sections
gtd_weekly_review(sections: ["projects_review", "waiting_for", "stalled_projects"])

# Review for specific week
gtd_weekly_review(week_ending: "2024-01-14")
```

## gtd_horizons

View commitments across GTD's 6 horizons of focus.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `level` | string | No | - | Specific horizon level |
| `include_entities` | boolean | No | true | Include entity details |

#### Horizon Levels

| Level | Name | Description |
|-------|------|-------------|
| `runway` | Current Actions | Tasks and next actions |
| `h10k` | Projects | Current projects |
| `h20k` | Areas | Areas of focus/responsibility |
| `h30k` | Goals | 1-2 year goals |
| `h40k` | Vision | 3-5 year vision |
| `h50k` | Purpose | Life purpose and principles |

### Response

```json
{
  "horizons": [
    {
      "level": "runway",
      "name": "Current Actions",
      "count": 45,
      "entities": [...]
    },
    {
      "level": "h10k",
      "name": "Projects",
      "count": 12,
      "entities": [...]
    }
  ]
}
```

## gtd_commitments

Track promises made and received.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `action` | string | Yes | - | Action: `list`, `add`, `resolve` |
| `commitment_type` | string | No | - | `made` or `received` |
| `filters` | object | No | - | Filter criteria |

### Response

```json
{
  "commitments": [
    {
      "id": "commit_abc",
      "type": "made",
      "description": "Send quarterly report by Friday",
      "to_person": "Management",
      "due_date": "2024-01-19",
      "status": "pending",
      "source_document": "email_thread_xyz",
      "extracted_text": "I'll have the quarterly report to you by end of day Friday"
    }
  ]
}
```

## gtd_dependencies

Analyze task dependencies and blockers.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | No | - | Analyze specific project |
| `include_critical_path` | boolean | No | true | Calculate critical path |

### Response

```json
{
  "graph": {
    "nodes": [...],
    "edges": [...]
  },
  "critical_path": ["task_a", "task_b", "task_c"],
  "blockers": [
    {
      "blocked_task": "task_xyz",
      "blocked_by": "task_abc",
      "days_blocked": 5,
      "impact": ["task_xyz", "task_def", "task_ghi"]
    }
  ]
}
```

## gtd_attention

Analyze attention distribution across areas and projects.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `period` | string | No | `week` | Analysis period: `day`, `week`, `month` |
| `group_by` | string | No | `area` | Group by: `area`, `project` |

### Response

```json
{
  "period": "2024-01-08 to 2024-01-14",
  "by_area": {
    "Work": { "tasks_completed": 15, "score": 0.45 },
    "Health": { "tasks_completed": 3, "score": 0.08 },
    "Learning": { "tasks_completed": 5, "score": 0.15 }
  },
  "imbalances": [
    {
      "area": "Health",
      "expected": 0.20,
      "actual": 0.08,
      "gap": -0.12,
      "suggestion": "Health area receiving less attention than prioritized"
    }
  ]
}
```

---

# Calendar Tools

## calendar_query

Query calendar events with flexible filtering.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query_type` | string | Yes | - | Query type (see below) |
| `date_range` | object | No | - | Start and end dates |
| `filters` | object | No | - | Additional filters |

#### Query Types

| Type | Description |
|------|-------------|
| `upcoming` | Next N events |
| `today` | Today's events |
| `this_week` | This week's events |
| `date_range` | Events in date range |
| `free_time` | Find available slots |
| `conflicts` | Detect scheduling conflicts |
| `commitments` | Time-specific commitments |

### Response

```json
{
  "events": [
    {
      "id": "event_abc",
      "title": "Team Standup",
      "event_type": "meeting",
      "start": "2024-01-15T09:00:00Z",
      "end": "2024-01-15T09:30:00Z",
      "location": "Zoom",
      "participants": ["Alice", "Bob"],
      "recurrence": "weekly",
      "related_project": "proj_xyz"
    }
  ],
  "conflicts": [],
  "stats": {
    "total_events": 15,
    "meetings": 10,
    "blocked_time": 5
  }
}
```

### Examples

```
# Today's events
calendar_query(query_type: "today")

# This week
calendar_query(query_type: "this_week")

# Find free time slots
calendar_query(query_type: "free_time", date_range: {
  start: "2024-01-15",
  end: "2024-01-19"
})

# Check for conflicts
calendar_query(query_type: "conflicts")

# Specific date range
calendar_query(query_type: "date_range", date_range: {
  start: "2024-02-01",
  end: "2024-02-29"
})
```

---

# Knowledge Tools

## knowledge_query

Query the knowledge graph with semantic search and relationship traversal.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language query |
| `query_type` | string | No | `semantic_search` | Query type (see below) |
| `entity_types` | array | No | - | Filter to specific entity types |
| `limit` | integer | No | 10 | Maximum results |

#### Query Types

| Type | Description |
|------|-------------|
| `semantic_search` | "What do I know about X?" |
| `entity_lookup` | Find specific entity by name |
| `relationship_query` | "Who works with person X?" |
| `topic_summary` | Summarize knowledge on topic |
| `connected_entities` | Graph traversal from entity |
| `expert_finding` | "Who knows about X?" |

### Response

```json
{
  "entities": [
    {
      "id": "entity_abc",
      "type": "Person",
      "name": "Alice Chen",
      "aliases": ["A. Chen"],
      "metadata": {
        "organization": "Engineering",
        "topics": ["Kubernetes", "DevOps"]
      }
    }
  ],
  "relationships": [
    {
      "source": "Alice Chen",
      "type": "works_for",
      "target": "Acme Corp"
    }
  ],
  "source_documents": [
    {
      "id": "doc_xyz",
      "path": "/meetings/2024-01-10.md",
      "relevance": 0.92
    }
  ],
  "summary": "Alice Chen is a DevOps engineer at Acme Corp, frequently mentioned in discussions about Kubernetes deployments...",
  "confidence": 0.85
}
```

### Examples

```
# Semantic search
knowledge_query(query: "machine learning projects")

# Find experts
knowledge_query(query: "Who knows about Kubernetes?", query_type: "expert_finding")

# Topic summary
knowledge_query(query: "authentication", query_type: "topic_summary")

# Entity relationships
knowledge_query(query: "Alice Chen", query_type: "connected_entities")

# Filter by entity type
knowledge_query(query: "database", entity_types: ["Person", "Organization"])
```

---

# Natural Language Query

## query

Unified natural language interface that routes to appropriate subsystem.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language query |
| `query_mode` | string | No | `auto` | Force mode: `auto`, `gtd`, `calendar`, `knowledge` |

### Examples

The query tool automatically routes based on intent:

```
# GTD queries (auto-routes to GTD)
query(query: "What should I work on now?")
query(query: "What's blocking the website project?")
query(query: "Show me stalled projects")
query(query: "What's waiting on John?")

# Calendar queries (auto-routes to calendar)
query(query: "What's on my calendar this week?")
query(query: "When am I free tomorrow?")

# Knowledge queries (auto-routes to knowledge)
query(query: "What do I know about machine learning?")
query(query: "Who can help with AWS migration?")

# Force specific mode
query(query: "review", query_mode: "gtd")
```

### Intent Classification

The query tool uses pattern matching and semantic analysis to classify intent:

| Pattern | Routed To |
|---------|-----------|
| "what should I...", "recommend", "next action" | GTD tasks |
| "blocking", "stalled", "project health" | GTD projects |
| "waiting on", "delegated to" | GTD waiting |
| "calendar", "schedule", "meeting", "free time" | Calendar |
| "who knows", "expert", "what do I know" | Knowledge |

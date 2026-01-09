# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
just build        # Build the project
just test         # Run all tests
just test-unit    # Run unit tests only (cargo test --lib)
just lint         # Run clippy (cargo clippy -- -D warnings)
just fmt          # Format code
just qa           # Run all quality checks (fmt, lint, test)
just release      # Build optimized release binary
```

Run a single test:
```bash
cargo test test_name -- --nocapture
```

Note: Tests marked with `#[ignore]` require embedding model downloads (fastembed models).

## Architecture Overview

Alloy is a hybrid document indexing MCP (Model Context Protocol) server that combines vector similarity search with full-text BM25 matching.

### Core Flow

```
Sources (Local/S3) → Processing → Embedding → Storage → Hybrid Search
```

### Key Components

**IndexCoordinator** (`src/coordinator.rs`): Central orchestrator that manages the indexing pipeline:
- Scans sources for documents
- Processes documents through appropriate processors
- Generates embeddings via provider
- Stores in both vector and full-text backends
- Handles file watching for live updates

**MCP Server** (`src/mcp/server.rs`): Implements the MCP protocol with tools:
- `index_path`: Index local directories or S3 URIs
- `search`: Hybrid search with configurable vector weight
- `get_document`, `list_sources`, `remove_source`, `get_stats`

**CLI** (`src/cli/`): Direct command-line access supporting both local and remote execution modes via `--remote` flag.

### Module Structure

- **embedding/**: Vector embedding providers
  - `local.rs`: fastembed-based local ONNX models (BGE, MiniLM)
  - `api.rs`: OpenAI-compatible API provider
  - `batch.rs`: Batching with rate limiting and retry logic

- **storage/**: Document and vector storage backends
  - `embedded.rs`: Combined Tantivy (full-text) + LanceDB (vectors)
  - `qdrant_backend.rs`: Qdrant vector database + Tantivy hybrid

- **search/**: Hybrid search orchestration
  - `hybrid.rs`: Main orchestrator combining vector and BM25 results
  - `fusion.rs`: RRF (Reciprocal Rank Fusion) and DBSF algorithms
  - `reranker.rs`: Score-based and cross-encoder reranking
  - `expansion.rs`: Query expansion (synonym, embedding-based)
  - `clustering.rs`: K-Means and DBSCAN for result grouping

- **processing/**: Document processing pipeline
  - `text.rs`, `document.rs`, `image.rs`: Format-specific processors
  - `chunker.rs`: Text chunking with configurable size/overlap
  - `registry.rs`: Processor dispatch by MIME type

- **sources/**: Data source abstraction
  - `local.rs`: Local filesystem with glob patterns and file watching
  - `s3.rs`: AWS S3 bucket scanning

### Configuration

Config is loaded from `config.toml` or `~/.config/alloy/config.toml`. Key settings:
- `embedding.provider`: "local" or "api"
- `storage.backend`: "embedded" or "qdrant"
- `processing.chunk_size`: Token count per chunk (default: 512)

### MCP Protocol

Uses the `rmcp` crate from the official Rust MCP SDK. The server supports:
- **stdio** transport: For MCP clients like Claude Desktop
- **http** transport: For network access and remote CLI mode

The `#[tool_router]` and `#[tool]` macros from rmcp define MCP tools with automatic JSON schema generation via schemars.

//! Local execution via IndexCoordinator.
//!
//! This module executes CLI commands directly using the IndexCoordinator.

use alloy::{
    mcp::{
        DocumentDetails, IndexPathResponse, IndexStats, ListSourcesResponse, RemoveSourceResponse,
        SearchResponse, SearchResult, SourceInfo,
    },
    sources::parse_s3_uri,
    Config, IndexCoordinator,
};
use anyhow::Result;

/// Index a local path or S3 URI.
pub async fn index(
    config: Config,
    path: String,
    pattern: Option<String>,
    watch: bool,
) -> Result<IndexPathResponse> {
    let coordinator = IndexCoordinator::new(config).await?;
    let patterns = pattern.map(|p| vec![p]).unwrap_or_default();

    let source = if path.starts_with("s3://") {
        let (bucket, prefix) = parse_s3_uri(&path)?;
        coordinator
            .index_s3(bucket, Some(prefix), patterns, None)
            .await?
    } else {
        let path_buf = std::path::PathBuf::from(&path);
        let path_buf = if path_buf.is_relative() {
            std::env::current_dir()?.join(path_buf)
        } else {
            path_buf
        };
        coordinator
            .index_local(path_buf, patterns, vec![], watch)
            .await?
    };

    Ok(IndexPathResponse {
        source_id: source.id,
        documents_indexed: source.document_count,
        chunks_created: 0, // Not tracked at source level
        watching: source.watching,
        message: format!("Indexed {} documents from {}", source.document_count, path),
    })
}

/// Search indexed documents.
pub async fn search(
    config: Config,
    query: String,
    limit: usize,
    vector_weight: f32,
    source_id: Option<String>,
) -> Result<SearchResponse> {
    use alloy::search::{HybridQuery, SearchFilter};

    let coordinator = IndexCoordinator::new(config).await?;
    let start = std::time::Instant::now();

    let mut hq = HybridQuery::new(&query)
        .limit(limit)
        .vector_weight(vector_weight);

    if let Some(sid) = source_id {
        hq = hq.filter(SearchFilter::new().source(sid));
    }

    let response = coordinator.search(hq).await?;

    Ok(SearchResponse {
        results: response
            .results
            .iter()
            .map(|r| SearchResult {
                document_id: r.document_id.clone(),
                chunk_id: r.chunk_id.clone(),
                source_id: String::new(), // Not available in search result
                path: r.path.clone().unwrap_or_default(),
                content: r.text.clone(),
                score: r.score,
                highlights: vec![],
                metadata: serde_json::json!({}),
            })
            .collect(),
        total_matches: response.results.len(),
        took_ms: start.elapsed().as_millis() as u64,
        query_expanded: if response.stats.query_expanded {
            Some(true)
        } else {
            None
        },
        expanded_query: response.stats.expanded_query.clone(),
        reranked: if response.stats.reranked {
            Some(true)
        } else {
            None
        },
    })
}

/// Get a document by ID.
pub async fn get_document(
    config: Config,
    doc_id: String,
    include_content: bool,
) -> Result<Option<DocumentDetails>> {
    let coordinator = IndexCoordinator::new(config).await?;

    if let Some(doc) = coordinator.get_document(&doc_id).await? {
        Ok(Some(DocumentDetails {
            document_id: doc.id,
            source_id: doc.source_id,
            path: doc.path,
            mime_type: doc.mime_type,
            size_bytes: doc.size,
            chunk_count: 0, // Not tracked at document level
            content: if include_content {
                Some(doc.content)
            } else {
                None
            },
            modified_at: doc.modified_at,
            indexed_at: doc.indexed_at,
            metadata: doc.metadata,
        }))
    } else {
        Ok(None)
    }
}

/// List all indexed sources.
pub async fn list_sources(config: Config) -> Result<ListSourcesResponse> {
    let coordinator = IndexCoordinator::new(config).await?;
    let sources = coordinator.list_sources().await;

    Ok(ListSourcesResponse {
        sources: sources
            .into_iter()
            .map(|s| SourceInfo {
                source_id: s.id,
                source_type: s.source_type,
                path: s.path,
                document_count: s.document_count,
                watching: s.watching,
                last_scan: s.last_scan,
                status: "active".to_string(),
            })
            .collect(),
    })
}

/// Remove an indexed source.
pub async fn remove_source(config: Config, source_id: String) -> Result<RemoveSourceResponse> {
    let coordinator = IndexCoordinator::new(config).await?;
    let removed = coordinator.remove_source(&source_id).await?;

    Ok(RemoveSourceResponse {
        success: removed > 0,
        documents_removed: removed,
        message: if removed > 0 {
            format!("Removed {} documents from source {}", removed, source_id)
        } else {
            format!("Source {} not found", source_id)
        },
    })
}

/// Get index statistics.
pub async fn stats(config: Config) -> Result<IndexStats> {
    let coordinator = IndexCoordinator::new(config.clone()).await?;
    let storage_stats = coordinator.stats().await?;

    Ok(IndexStats {
        source_count: coordinator.list_sources().await.len(),
        document_count: storage_stats.document_count,
        chunk_count: storage_stats.chunk_count,
        storage_bytes: storage_stats.storage_bytes,
        embedding_dimension: coordinator.embedding_dimension(),
        storage_backend: format!("{:?}", config.storage.backend),
        embedding_provider: format!("{:?}", config.embedding.provider),
        uptime_secs: 0, // Not applicable for CLI
    })
}

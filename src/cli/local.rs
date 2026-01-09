//! Local execution via IndexCoordinator.
//!
//! This module executes CLI commands directly using the IndexCoordinator.

use alloy::{
    backup::{BackupManager, ExportFormat, ExportOptions},
    mcp::{
        BackupInfo, CreateBackupResponse, DocumentDetails, ExportDocumentsResponse,
        ImportDocumentsResponse, IndexPathResponse, IndexStats, ListBackupsResponse,
        ListSourcesResponse, RemoveSourceResponse, RestoreBackupResponse, SearchResponse,
        SearchResult, SourceInfo,
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

/// Cluster indexed documents by semantic similarity.
pub async fn cluster(
    config: Config,
    source_id: Option<String>,
    algorithm: Option<String>,
    num_clusters: Option<usize>,
) -> Result<alloy::mcp::ClusterDocumentsResponse> {
    use alloy::mcp::{ClusterDocumentsResponse, ClusterInfo, ClusterMetrics};

    let mut config = config;

    // Parse algorithm if provided
    if let Some(algo) = &algorithm {
        config.search.clustering.algorithm = match algo.as_str() {
            "dbscan" => alloy::config::ClusteringAlgorithm::Dbscan,
            _ => alloy::config::ClusteringAlgorithm::KMeans,
        };
    }

    let coordinator = IndexCoordinator::new(config).await?;

    let clustering_algorithm = algorithm.as_deref().map(|algo| match algo {
        "dbscan" => alloy::config::ClusteringAlgorithm::Dbscan,
        _ => alloy::config::ClusteringAlgorithm::KMeans,
    });

    let result = coordinator
        .cluster_documents(source_id.as_deref(), clustering_algorithm, num_clusters)
        .await?;

    Ok(ClusterDocumentsResponse {
        clusters: result
            .clusters
            .iter()
            .map(|c| ClusterInfo {
                cluster_id: c.cluster_id,
                label: c.label.clone(),
                keywords: c.keywords.clone(),
                size: c.document_ids.len(),
                coherence_score: c.coherence_score,
                representative_docs: c.representative_docs.clone(),
            })
            .collect(),
        outliers: result.outliers,
        metrics: ClusterMetrics {
            silhouette_score: result.metrics.silhouette_score,
            calinski_harabasz_index: if result.metrics.calinski_harabasz_index > 0.0 {
                Some(result.metrics.calinski_harabasz_index)
            } else {
                None
            },
            davies_bouldin_index: if result.metrics.davies_bouldin_index > 0.0 {
                Some(result.metrics.davies_bouldin_index)
            } else {
                None
            },
            num_clusters: result.metrics.num_clusters,
            num_outliers: result.metrics.num_outliers,
        },
        algorithm: result.algorithm_used,
        total_documents: result.total_documents,
    })
}

/// Create a backup of the index.
pub async fn backup(
    config: Config,
    output_path: Option<String>,
    description: Option<String>,
) -> Result<CreateBackupResponse> {
    let coordinator = IndexCoordinator::new(config.clone()).await?;

    let backup_dir = output_path.unwrap_or_else(|| {
        config
            .operations
            .backup
            .backup_dir
            .clone()
            .unwrap_or_else(|| {
                let data_dir = config.storage.data_dir.clone();
                format!("{}/backups", data_dir)
            })
    });

    let manager = BackupManager::new(&backup_dir)?;

    let result = manager
        .create_backup(
            coordinator.storage(),
            &format!("{:?}", config.storage.backend),
            coordinator.embedding_dimension(),
            description,
        )
        .await?;

    Ok(CreateBackupResponse {
        backup_id: result.metadata.backup_id.clone(),
        path: result.path.display().to_string(),
        document_count: result.metadata.document_count,
        size_bytes: result.metadata.size_bytes,
        duration_ms: result.duration_ms,
        success: true,
        message: format!(
            "Backup created successfully: {} documents",
            result.metadata.document_count
        ),
    })
}

/// Restore from a backup.
pub async fn restore(config: Config, input: String) -> Result<RestoreBackupResponse> {
    let coordinator = IndexCoordinator::new(config.clone()).await?;

    let backup_dir = config
        .operations
        .backup
        .backup_dir
        .clone()
        .unwrap_or_else(|| {
            let data_dir = config.storage.data_dir.clone();
            format!("{}/backups", data_dir)
        });

    let manager = BackupManager::new(&backup_dir)?;

    // Try to find backup by ID or use as path
    let backup_path = if std::path::Path::new(&input).exists() {
        std::path::PathBuf::from(&input)
    } else {
        // Find backup file by listing and matching ID
        let backups = manager.list_backups()?;

        backups
            .iter()
            .find(|b| b.backup_id == input)
            .map(|b| {
                let timestamp = b.created_at.format("%Y%m%d_%H%M%S");
                std::path::PathBuf::from(&backup_dir).join(format!("backup_{}.jsonl", timestamp))
            })
            .ok_or_else(|| anyhow::anyhow!("Backup not found: {}", input))?
    };

    let result = manager
        .restore_backup(coordinator.storage(), &backup_path)
        .await?;

    Ok(RestoreBackupResponse {
        backup_id: result.metadata.backup_id.clone(),
        documents_restored: result.documents_restored,
        chunks_restored: result.chunks_restored,
        duration_ms: result.duration_ms,
        success: true,
        message: format!(
            "Restored {} documents from backup",
            result.documents_restored
        ),
    })
}

/// Export documents to a file.
pub async fn export(
    config: Config,
    output_path: String,
    format: String,
    source_id: Option<String>,
    include_embeddings: bool,
) -> Result<ExportDocumentsResponse> {
    let coordinator = IndexCoordinator::new(config.clone()).await?;

    let backup_dir = config
        .operations
        .backup
        .backup_dir
        .clone()
        .unwrap_or_else(|| {
            let data_dir = config.storage.data_dir.clone();
            format!("{}/backups", data_dir)
        });

    let format = match format.as_str() {
        "json" => ExportFormat::Json,
        _ => ExportFormat::Jsonl,
    };

    let options = ExportOptions {
        format,
        include_content: true,
        include_embeddings,
        source_id,
        compress: false,
    };

    let manager = BackupManager::new(&backup_dir)?;
    let start = std::time::Instant::now();
    let result = manager
        .export_documents(coordinator.storage(), &output_path, &options)
        .await?;

    Ok(ExportDocumentsResponse {
        path: result.path.display().to_string(),
        document_count: result.document_count,
        size_bytes: result.size_bytes,
        duration_ms: start.elapsed().as_millis() as u64,
        success: true,
        message: format!(
            "Exported {} documents to {}",
            result.document_count, output_path
        ),
    })
}

/// Import documents from a file.
pub async fn import(config: Config, input_path: String) -> Result<ImportDocumentsResponse> {
    let coordinator = IndexCoordinator::new(config.clone()).await?;

    let backup_dir = config
        .operations
        .backup
        .backup_dir
        .clone()
        .unwrap_or_else(|| {
            let data_dir = config.storage.data_dir.clone();
            format!("{}/backups", data_dir)
        });

    let manager = BackupManager::new(&backup_dir)?;
    let start = std::time::Instant::now();
    let (documents_imported, chunks_imported) = manager
        .import_documents(coordinator.storage(), &input_path)
        .await?;

    Ok(ImportDocumentsResponse {
        documents_imported,
        chunks_imported,
        duration_ms: start.elapsed().as_millis() as u64,
        success: true,
        message: format!(
            "Imported {} documents with {} chunks from {}",
            documents_imported, chunks_imported, input_path
        ),
    })
}

/// List available backups.
pub async fn list_backups(config: Config) -> Result<ListBackupsResponse> {
    let backup_dir = config
        .operations
        .backup
        .backup_dir
        .clone()
        .unwrap_or_else(|| {
            let data_dir = config.storage.data_dir.clone();
            format!("{}/backups", data_dir)
        });

    let manager = BackupManager::new(&backup_dir)?;
    let backups = manager.list_backups()?;

    let backup_infos: Vec<BackupInfo> = backups
        .iter()
        .map(|b| BackupInfo {
            backup_id: b.backup_id.clone(),
            created_at: b.created_at,
            version: b.version.clone(),
            document_count: b.document_count,
            chunk_count: b.chunk_count,
            size_bytes: b.size_bytes,
            description: b.description.clone(),
        })
        .collect();

    Ok(ListBackupsResponse {
        backups: backup_infos,
        message: format!("Found {} backups", backups.len()),
    })
}

// ============================================================================
// GTD, Calendar, Knowledge, Query, Ontology Commands
// ============================================================================

use super::types::{CalendarResult, GtdResult, KnowledgeResult, OntologyResult, QueryResult};
use crate::{CalendarCommand, GtdCommand, KnowledgeCommand, OntologyCommand};

/// Execute GTD commands locally.
pub async fn gtd(_config: Config, command: GtdCommand) -> Result<GtdResult> {
    // Stub implementation - will be fully implemented in Phase 2
    let message = match &command {
        GtdCommand::Projects { action, .. } => {
            format!("GTD projects: {} (not implemented)", action)
        }
        GtdCommand::Tasks { action, .. } => format!("GTD tasks: {} (not implemented)", action),
        GtdCommand::Waiting { action, .. } => format!("GTD waiting: {} (not implemented)", action),
        GtdCommand::Someday { action, .. } => format!("GTD someday: {} (not implemented)", action),
        GtdCommand::Review { .. } => "GTD review (not implemented)".to_string(),
        GtdCommand::Horizons { action, .. } => {
            format!("GTD horizons: {} (not implemented)", action)
        }
        GtdCommand::Commitments { action, .. } => {
            format!("GTD commitments: {} (not implemented)", action)
        }
        GtdCommand::Dependencies { action, .. } => {
            format!("GTD dependencies: {} (not implemented)", action)
        }
        GtdCommand::Attention { .. } => "GTD attention (not implemented)".to_string(),
        GtdCommand::Areas { action, .. } => format!("GTD areas: {} (not implemented)", action),
        GtdCommand::Goals { action, .. } => format!("GTD goals: {} (not implemented)", action),
    };

    Ok(GtdResult {
        success: false,
        message,
        data: serde_json::Value::Null,
    })
}

/// Execute Calendar commands locally.
pub async fn calendar(_config: Config, command: CalendarCommand) -> Result<CalendarResult> {
    // Stub implementation - will be fully implemented in Phase 4
    let message = match &command {
        CalendarCommand::Today => "Calendar today (not implemented)".to_string(),
        CalendarCommand::Week => "Calendar week (not implemented)".to_string(),
        CalendarCommand::Range { start, end } => {
            format!("Calendar range {}-{} (not implemented)", start, end)
        }
        CalendarCommand::Free { start, end, .. } => {
            format!("Calendar free {}-{} (not implemented)", start, end)
        }
        CalendarCommand::Conflicts => "Calendar conflicts (not implemented)".to_string(),
        CalendarCommand::Upcoming { limit } => {
            format!("Calendar upcoming {} (not implemented)", limit)
        }
        CalendarCommand::Events { action, .. } => {
            format!("Calendar events: {} (not implemented)", action)
        }
    };

    Ok(CalendarResult {
        success: false,
        message,
        data: serde_json::Value::Null,
    })
}

/// Execute Knowledge commands locally.
pub async fn knowledge(_config: Config, command: KnowledgeCommand) -> Result<KnowledgeResult> {
    // Stub implementation - will be fully implemented in Phase 5
    let message = match &command {
        KnowledgeCommand::Search { query, .. } => {
            format!("Knowledge search '{}' (not implemented)", query)
        }
        KnowledgeCommand::Entity { name, .. } => {
            format!("Knowledge entity '{}' (not implemented)", name)
        }
        KnowledgeCommand::Expert { topic, .. } => {
            format!("Knowledge expert '{}' (not implemented)", topic)
        }
        KnowledgeCommand::Topic { topic } => {
            format!("Knowledge topic '{}' (not implemented)", topic)
        }
        KnowledgeCommand::Connected { entity, depth } => format!(
            "Knowledge connected '{}' depth {} (not implemented)",
            entity, depth
        ),
    };

    Ok(KnowledgeResult {
        success: false,
        message,
        data: serde_json::Value::Null,
    })
}

/// Execute natural language query locally.
pub async fn query(
    _config: Config,
    query_text: String,
    query_mode: Option<String>,
) -> Result<QueryResult> {
    // Stub implementation - will be fully implemented in Phase 7
    Ok(QueryResult {
        success: false,
        query: query_text,
        mode: query_mode.unwrap_or_else(|| "auto".to_string()),
        message: "Natural language query (not implemented)".to_string(),
        data: serde_json::Value::Null,
    })
}

/// Execute Ontology commands locally.
pub async fn ontology(_config: Config, command: OntologyCommand) -> Result<OntologyResult> {
    // Stub implementation - will be fully implemented in Phase 6
    let message = match &command {
        OntologyCommand::Stats => "Ontology stats (not implemented)".to_string(),
        OntologyCommand::Entities { action, .. } => {
            format!("Ontology entities: {} (not implemented)", action)
        }
        OntologyCommand::Relationships { action, .. } => {
            format!("Ontology relationships: {} (not implemented)", action)
        }
        OntologyCommand::Extract { document, .. } => {
            format!("Ontology extract '{}' (not implemented)", document)
        }
        OntologyCommand::Person { name, .. } => {
            format!("Ontology person '{}' (not implemented)", name)
        }
        OntologyCommand::Organization { name, .. } => {
            format!("Ontology organization '{}' (not implemented)", name)
        }
        OntologyCommand::Topic { name, .. } => {
            format!("Ontology topic '{}' (not implemented)", name)
        }
    };

    Ok(OntologyResult {
        success: false,
        message,
        data: serde_json::Value::Null,
    })
}

//! Local execution via IndexCoordinator.
//!
//! This module executes CLI commands directly using the IndexCoordinator.

use alloy::{
    backup::{BackupManager, ExportFormat, ExportOptions},
    calendar::{
        CalendarEvent, CalendarFilter, CalendarManager, EventType, EventUpdate, FreeTimeParams,
    },
    gtd::{
        AttentionManager, AttentionParams, Commitment, CommitmentDirection, CommitmentFilter,
        CommitmentManager, CommitmentStatus, DependencyManager, DependencyParams, HorizonLevel,
        HorizonManager, HorizonParams, OutputFormat, Project, ProjectFilter, ProjectManager,
        ProjectStatus, ReviewManager, ReviewSection, SomedayFilter, SomedayItem, SomedayManager,
        Task, TaskFilter, TaskManager, TaskStatus, WaitingFilter, WaitingFor, WaitingManager,
        WaitingStatus, WeeklyReviewParams,
    },
    mcp::{
        BackupInfo, CreateBackupResponse, DocumentDetails, ExportDocumentsResponse,
        ImportDocumentsResponse, IndexPathResponse, IndexStats, ListBackupsResponse,
        ListSourcesResponse, RemoveSourceResponse, RestoreBackupResponse, SearchResponse,
        SearchResult, SourceInfo,
    },
    ontology::{
        EmbeddedOntologyStore, Entity, EntityFilter, EntityType, EntityUpdate, OntologyStore,
        RelationType, Relationship, RelationshipFilter,
    },
    sources::parse_s3_uri,
    Config, ExtractionConfig, ExtractionPipeline, IndexCoordinator,
};
use anyhow::Result;
use chrono::{DateTime, NaiveDate, NaiveTime, TimeZone, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;

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
    let result = coordinator.remove_source(&source_id).await?;

    match result {
        Some(docs_removed) => Ok(RemoveSourceResponse {
            success: true,
            documents_removed: docs_removed,
            message: format!(
                "Removed {} documents from source {}",
                docs_removed, source_id
            ),
        }),
        None => Ok(RemoveSourceResponse {
            success: false,
            documents_removed: 0,
            message: format!("Source {} not found", source_id),
        }),
    }
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
// Helper Functions
// ============================================================================

/// Get or create the ontology store.
async fn get_ontology_store(config: &Config) -> Result<Arc<RwLock<EmbeddedOntologyStore>>> {
    let data_dir = std::path::PathBuf::from(&config.storage.data_dir);
    let store = EmbeddedOntologyStore::with_persistence(&data_dir).await?;
    Ok(Arc::new(RwLock::new(store)))
}

/// Parse a date string into DateTime<Utc>.
/// Supports: RFC3339, YYYY-MM-DD, YYYY-MM-DD HH:MM
fn parse_date(s: &str) -> Option<DateTime<Utc>> {
    // Try RFC3339 first
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }

    // Try YYYY-MM-DDTHH:MM
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M") {
        return Some(Utc.from_utc_datetime(&dt));
    }

    // Try YYYY-MM-DD HH:MM
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M") {
        return Some(Utc.from_utc_datetime(&dt));
    }

    // Try YYYY-MM-DD (set to start of day)
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_time(NaiveTime::from_hms_opt(0, 0, 0)?);
        return Some(Utc.from_utc_datetime(&dt));
    }

    None
}

/// Parse project status from string.
fn parse_project_status(s: &str) -> Option<ProjectStatus> {
    match s.to_lowercase().as_str() {
        "active" => Some(ProjectStatus::Active),
        "on_hold" | "onhold" => Some(ProjectStatus::OnHold),
        "completed" => Some(ProjectStatus::Completed),
        "archived" => Some(ProjectStatus::Archived),
        _ => None,
    }
}

/// Parse task status from string.
fn parse_task_status(s: &str) -> Option<TaskStatus> {
    match s.to_lowercase().as_str() {
        "next" => Some(TaskStatus::Next),
        "scheduled" => Some(TaskStatus::Scheduled),
        "waiting" => Some(TaskStatus::Waiting),
        "someday" => Some(TaskStatus::Someday),
        "done" => Some(TaskStatus::Done),
        _ => None,
    }
}

/// Parse waiting status from string.
fn parse_waiting_status(s: &str) -> Option<WaitingStatus> {
    match s.to_lowercase().as_str() {
        "pending" => Some(WaitingStatus::Pending),
        "overdue" => Some(WaitingStatus::Overdue),
        "resolved" => Some(WaitingStatus::Resolved),
        _ => None,
    }
}

/// Parse entity type from string.
fn parse_entity_type(s: &str) -> Option<EntityType> {
    match s.to_lowercase().as_str() {
        "person" => Some(EntityType::Person),
        "organization" | "org" => Some(EntityType::Organization),
        "topic" => Some(EntityType::Topic),
        "concept" => Some(EntityType::Concept),
        "location" => Some(EntityType::Location),
        "project" => Some(EntityType::Project),
        "task" => Some(EntityType::Task),
        "area" => Some(EntityType::Area),
        "goal" => Some(EntityType::Goal),
        "date" => Some(EntityType::Date),
        "commitment" => Some(EntityType::Commitment),
        "context" => Some(EntityType::Context),
        "reference" => Some(EntityType::Reference),
        "vision" => Some(EntityType::Vision),
        "purpose" => Some(EntityType::Purpose),
        "waitingfor" | "waiting_for" => Some(EntityType::WaitingFor),
        "somedaymaybe" | "someday_maybe" => Some(EntityType::SomedayMaybe),
        "calendarevent" | "calendar_event" => Some(EntityType::CalendarEvent),
        "custom" => Some(EntityType::Custom),
        _ => None,
    }
}

/// Parse relation type from string.
fn parse_relation_type(s: &str) -> Option<RelationType> {
    match s.to_lowercase().as_str() {
        "belongstoproject" | "belongs_to_project" => Some(RelationType::BelongsToProject),
        "hascontext" | "has_context" => Some(RelationType::HasContext),
        "inarea" | "in_area" => Some(RelationType::InArea),
        "supportsgoal" | "supports_goal" => Some(RelationType::SupportsGoal),
        "waitingon" | "waiting_on" => Some(RelationType::WaitingOn),
        "delegatedto" | "delegated_to" => Some(RelationType::DelegatedTo),
        "blockedby" | "blocked_by" => Some(RelationType::BlockedBy),
        "dependson" | "depends_on" => Some(RelationType::DependsOn),
        "scheduledfor" | "scheduled_for" => Some(RelationType::ScheduledFor),
        "dueon" | "due_on" => Some(RelationType::DueOn),
        "mentions" => Some(RelationType::Mentions),
        "relatedto" | "related_to" => Some(RelationType::RelatedTo),
        "authoredby" | "authored_by" => Some(RelationType::AuthoredBy),
        "abouttopic" | "about_topic" => Some(RelationType::AboutTopic),
        "locatedat" | "located_at" => Some(RelationType::LocatedAt),
        "worksfor" | "works_for" => Some(RelationType::WorksFor),
        "committedto" | "committed_to" => Some(RelationType::CommittedTo),
        "references" => Some(RelationType::References),
        "partof" | "part_of" => Some(RelationType::PartOf),
        "contains" => Some(RelationType::Contains),
        "sameas" | "same_as" => Some(RelationType::SameAs),
        "custom" => Some(RelationType::Custom),
        _ => None,
    }
}

/// Parse event type from string.
fn parse_event_type(s: &str) -> Option<EventType> {
    match s.to_lowercase().as_str() {
        "event" => Some(EventType::Event),
        "meeting" => Some(EventType::Meeting),
        "deadline" => Some(EventType::Deadline),
        "reminder" => Some(EventType::Reminder),
        "blocked_time" | "blockedtime" => Some(EventType::BlockedTime),
        "milestone" => Some(EventType::Milestone),
        "appointment" => Some(EventType::Appointment),
        "travel" => Some(EventType::Travel),
        "call" => Some(EventType::Call),
        "standup" => Some(EventType::Standup),
        _ => None,
    }
}

// ============================================================================
// GTD, Calendar, Knowledge, Query, Ontology Commands
// ============================================================================

use super::types::{CalendarResult, GtdResult, KnowledgeResult, OntologyResult, QueryResult};
use crate::{CalendarCommand, GtdCommand, KnowledgeCommand, OntologyCommand};

/// Execute GTD commands locally.
pub async fn gtd(config: Config, command: GtdCommand) -> Result<GtdResult> {
    let store = get_ontology_store(&config).await?;

    match command {
        GtdCommand::Projects {
            action,
            id,
            status,
            area,
            stalled,
            no_next_action,
            name,
            outcome,
            set_area,
            goal,
        } => {
            let manager = ProjectManager::new(store);

            match action.as_str() {
                "list" => {
                    let mut filter = ProjectFilter::default();
                    if let Some(s) = status {
                        filter.status = parse_project_status(&s);
                    }
                    if let Some(a) = area {
                        filter.area = Some(a);
                    }
                    if let Some(days) = stalled {
                        filter.stalled_days = Some(days);
                    }
                    if no_next_action {
                        filter.has_next_action = Some(false);
                    }

                    let projects = manager.list(filter).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} projects", projects.len()),
                        data: serde_json::to_value(projects)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "get" => {
                    let project_id = id.ok_or_else(|| anyhow::anyhow!("Project ID required"))?;
                    match manager.get(&project_id).await? {
                        Some(project) => Ok(GtdResult {
                            success: true,
                            message: format!("Project: {}", project.name),
                            data: serde_json::to_value(project)?,
                            extra: serde_json::Map::new(),
                        }),
                        None => Ok(GtdResult {
                            success: false,
                            message: format!("Project not found: {}", project_id),
                            data: serde_json::Value::Null,
                            extra: serde_json::Map::new(),
                        }),
                    }
                }
                "create" => {
                    let project_name =
                        name.ok_or_else(|| anyhow::anyhow!("Project name required"))?;
                    let mut project = Project::new(&project_name);
                    if let Some(o) = outcome {
                        project = project.with_outcome(o);
                    }
                    if let Some(a) = set_area {
                        project = project.with_area(a);
                    }
                    if let Some(g) = goal {
                        project = project.with_goal(g);
                    }

                    let created = manager.create(project).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Created project: {}", created.name),
                        data: serde_json::to_value(created)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "archive" => {
                    let project_id = id.ok_or_else(|| anyhow::anyhow!("Project ID required"))?;
                    let archived = manager.archive(&project_id).await?;
                    Ok(GtdResult {
                        success: archived.is_some(),
                        message: if archived.is_some() {
                            "Project archived".to_string()
                        } else {
                            "Project not found".to_string()
                        },
                        data: serde_json::to_value(archived)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "complete" => {
                    let project_id = id.ok_or_else(|| anyhow::anyhow!("Project ID required"))?;
                    let completed = manager.complete(&project_id).await?;
                    Ok(GtdResult {
                        success: completed.is_some(),
                        message: if completed.is_some() {
                            "Project completed".to_string()
                        } else {
                            "Project not found".to_string()
                        },
                        data: serde_json::to_value(completed)?,
                        extra: serde_json::Map::new(),
                    })
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }

        GtdCommand::Tasks {
            action,
            id,
            context,
            energy,
            time,
            project,
            status,
            due_before,
            description_contains,
            limit,
            description,
            set_contexts,
            set_energy,
            priority,
            duration,
            due,
            scheduled: _scheduled,
            assign_project,
            blocked_by: _blocked_by,
        } => {
            let manager = TaskManager::new(store);

            match action.as_str() {
                "list" => {
                    let mut filter = TaskFilter::default();
                    if let Some(ctx) = context {
                        filter.contexts = ctx;
                    }
                    if let Some(e) = energy {
                        filter.energy_level = match e.to_lowercase().as_str() {
                            "low" => Some(alloy::gtd::EnergyLevel::Low),
                            "medium" => Some(alloy::gtd::EnergyLevel::Medium),
                            "high" => Some(alloy::gtd::EnergyLevel::High),
                            _ => None,
                        };
                    }
                    if let Some(t) = time {
                        filter.time_available = Some(t);
                    }
                    if let Some(p) = project {
                        filter.project_id = Some(p);
                    }
                    if let Some(s) = status {
                        filter.status = parse_task_status(&s);
                    }
                    if let Some(db) = due_before {
                        filter.due_before = parse_date(&db);
                    }
                    if description_contains.is_some() {
                        filter.description_contains = description_contains.clone();
                    }
                    filter.limit = limit;

                    let tasks = manager.list(filter).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} tasks", tasks.len()),
                        data: serde_json::to_value(tasks)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "get" => {
                    let task_id = id.ok_or_else(|| anyhow::anyhow!("Task ID required"))?;
                    match manager.get(&task_id).await? {
                        Some(task) => Ok(GtdResult {
                            success: true,
                            message: format!("Task: {}", task.description),
                            data: serde_json::to_value(task)?,
                            extra: serde_json::Map::new(),
                        }),
                        None => Ok(GtdResult {
                            success: false,
                            message: format!("Task not found: {}", task_id),
                            data: serde_json::Value::Null,
                            extra: serde_json::Map::new(),
                        }),
                    }
                }
                "create" => {
                    let task_desc =
                        description.ok_or_else(|| anyhow::anyhow!("Task description required"))?;
                    let mut task = Task::new(&task_desc);

                    if let Some(ctx_str) = set_contexts {
                        let contexts: Vec<String> =
                            ctx_str.split(',').map(|s| s.trim().to_string()).collect();
                        task = task.with_contexts(contexts);
                    }
                    if let Some(e) = set_energy {
                        let energy_level = match e.to_lowercase().as_str() {
                            "low" => alloy::gtd::EnergyLevel::Low,
                            "high" => alloy::gtd::EnergyLevel::High,
                            _ => alloy::gtd::EnergyLevel::Medium,
                        };
                        task = task.with_energy(energy_level);
                    }
                    if let Some(p) = priority {
                        let prio = match p.to_lowercase().as_str() {
                            "low" => alloy::gtd::Priority::Low,
                            "high" => alloy::gtd::Priority::High,
                            "critical" => alloy::gtd::Priority::Critical,
                            _ => alloy::gtd::Priority::Normal,
                        };
                        task = task.with_priority(prio);
                    }
                    if let Some(d) = duration {
                        task = task.with_duration(d);
                    }
                    if let Some(d) = due {
                        if let Some(dt) = parse_date(&d) {
                            task = task.with_due_date(dt);
                        }
                    }
                    if let Some(p) = assign_project {
                        task = task.with_project(p);
                    }

                    let created = manager.create(task).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Created task: {}", created.description),
                        data: serde_json::to_value(created)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "complete" => {
                    let task_id = id.ok_or_else(|| anyhow::anyhow!("Task ID required"))?;
                    let completed = manager.complete(&task_id).await?;
                    Ok(GtdResult {
                        success: completed.is_some(),
                        message: if completed.is_some() {
                            "Task completed".to_string()
                        } else {
                            "Task not found".to_string()
                        },
                        data: serde_json::to_value(completed)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "recommend" => {
                    let params = alloy::gtd::RecommendParams {
                        current_context: context.and_then(|c| c.first().cloned()),
                        energy_level: energy.and_then(|e| match e.to_lowercase().as_str() {
                            "low" => Some(alloy::gtd::EnergyLevel::Low),
                            "medium" => Some(alloy::gtd::EnergyLevel::Medium),
                            "high" => Some(alloy::gtd::EnergyLevel::High),
                            _ => None,
                        }),
                        time_available: time,
                        focus_area: None,
                        limit,
                    };

                    let recommendations = manager.recommend(params).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} recommendations", recommendations.len()),
                        data: serde_json::to_value(recommendations)?,
                        extra: serde_json::Map::new(),
                    })
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }

        GtdCommand::Waiting {
            action,
            id,
            status,
            person,
            project,
            description,
            delegated_to,
            expected_by,
            for_project,
            resolution,
        } => {
            let manager = WaitingManager::new(store);

            match action.as_str() {
                "list" => {
                    let mut filter = WaitingFilter::default();
                    if let Some(s) = status {
                        filter.status = parse_waiting_status(&s);
                    }
                    if let Some(p) = person {
                        filter.delegated_to = Some(p);
                    }
                    if let Some(p) = project {
                        filter.project_id = Some(p);
                    }

                    let items = manager.list(filter).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} waiting items", items.len()),
                        data: serde_json::to_value(items)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "add" => {
                    let desc =
                        description.ok_or_else(|| anyhow::anyhow!("Description required"))?;
                    let delegate = delegated_to
                        .ok_or_else(|| anyhow::anyhow!("Delegated to person required"))?;

                    let mut item = WaitingFor::new(&desc, &delegate);
                    if let Some(exp) = expected_by {
                        if let Some(dt) = parse_date(&exp) {
                            item = item.with_expected_by(dt);
                        }
                    }
                    if let Some(p) = for_project {
                        item = item.with_project(p);
                    }

                    let created = manager.create(item).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Created waiting item: {}", created.description),
                        data: serde_json::to_value(created)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "resolve" => {
                    let item_id = id.ok_or_else(|| anyhow::anyhow!("Item ID required"))?;
                    let res = resolution.unwrap_or_else(|| "Resolved".to_string());
                    let resolved = manager.resolve(&item_id, &res).await?;
                    Ok(GtdResult {
                        success: resolved.is_some(),
                        message: if resolved.is_some() {
                            "Waiting item resolved".to_string()
                        } else {
                            "Item not found".to_string()
                        },
                        data: serde_json::to_value(resolved)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "remind" => {
                    let item_id = id.ok_or_else(|| anyhow::anyhow!("Item ID required"))?;
                    let reminded = manager.record_follow_up(&item_id).await?;
                    Ok(GtdResult {
                        success: reminded.is_some(),
                        message: if reminded.is_some() {
                            "Follow-up recorded".to_string()
                        } else {
                            "Item not found".to_string()
                        },
                        data: serde_json::to_value(reminded)?,
                        extra: serde_json::Map::new(),
                    })
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }

        GtdCommand::Someday {
            action,
            id,
            category,
            description,
            set_category,
            trigger,
            review_date,
        } => {
            let manager = SomedayManager::new(store);

            match action.as_str() {
                "list" => {
                    let mut filter = SomedayFilter::default();
                    if let Some(c) = category {
                        filter.category = Some(c);
                    }

                    let items = manager.list(filter).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} someday/maybe items", items.len()),
                        data: serde_json::to_value(items)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "add" => {
                    let desc =
                        description.ok_or_else(|| anyhow::anyhow!("Description required"))?;
                    let mut item = SomedayItem::new(&desc);

                    if let Some(c) = set_category {
                        item = item.with_category(c);
                    }
                    if let Some(t) = trigger {
                        item = item.with_trigger(t);
                    }
                    if let Some(rd) = review_date {
                        if let Some(dt) = parse_date(&rd) {
                            item = item.with_review_date(dt);
                        }
                    }

                    let created = manager.create(item).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Created someday item: {}", created.description),
                        data: serde_json::to_value(created)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "activate" => {
                    let item_id = id.ok_or_else(|| anyhow::anyhow!("Item ID required"))?;
                    let activated = manager.activate(&item_id).await?;
                    Ok(GtdResult {
                        success: activated.is_some(),
                        message: if activated.is_some() {
                            "Item activated as project".to_string()
                        } else {
                            "Item not found".to_string()
                        },
                        data: serde_json::to_value(activated)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "archive" => {
                    let item_id = id.ok_or_else(|| anyhow::anyhow!("Item ID required"))?;
                    let archived = manager.delete(&item_id).await?;
                    Ok(GtdResult {
                        success: archived,
                        message: if archived {
                            "Item archived".to_string()
                        } else {
                            "Item not found".to_string()
                        },
                        data: serde_json::Value::Bool(archived),
                        extra: serde_json::Map::new(),
                    })
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }

        GtdCommand::Review {
            week_ending,
            sections,
        } => {
            let manager = ReviewManager::new(store);

            let mut params = WeeklyReviewParams::default();
            if let Some(we) = week_ending {
                if let Some(dt) = parse_date(&we) {
                    params.week_ending = Some(dt);
                }
            }
            if let Some(s) = sections {
                let section_list: Vec<ReviewSection> = s
                    .split(',')
                    .filter_map(|x| match x.trim().to_lowercase().as_str() {
                        "inbox" | "inbox_status" => Some(ReviewSection::InboxStatus),
                        "completed" | "completed_tasks" => Some(ReviewSection::CompletedTasks),
                        "projects" | "projects_review" => Some(ReviewSection::ProjectsReview),
                        "stalled" | "stalled_projects" => Some(ReviewSection::StalledProjects),
                        "waiting" | "waiting_for" => Some(ReviewSection::WaitingFor),
                        "upcoming" | "calendar" | "upcoming_calendar" => {
                            Some(ReviewSection::UpcomingCalendar)
                        }
                        "someday" | "someday_maybe" => Some(ReviewSection::SomedayMaybe),
                        "areas" | "areas_check" => Some(ReviewSection::AreasCheck),
                        "overdue" | "overdue_items" => Some(ReviewSection::OverdueItems),
                        "quick_wins" | "quickwins" => Some(ReviewSection::QuickWins),
                        "all" => Some(ReviewSection::All),
                        _ => None,
                    })
                    .collect();
                if !section_list.is_empty() {
                    params.sections = section_list;
                }
            }

            let report = manager.generate_weekly_review(params).await?;
            Ok(GtdResult {
                success: true,
                message: "Weekly review generated".to_string(),
                data: serde_json::to_value(report)?,
                extra: serde_json::Map::new(),
            })
        }

        GtdCommand::Horizons {
            action,
            level,
            name: _name,
            description: _description,
        } => {
            let manager = HorizonManager::new(store);

            match action.as_str() {
                "list" => {
                    // Parse the horizon level if provided
                    let horizons_filter: Vec<HorizonLevel> = if let Some(ref lvl) = level {
                        match lvl.to_lowercase().as_str() {
                            "runway" | "ground" | "0" => vec![HorizonLevel::Runway],
                            "10k" | "10000" | "projects" => vec![HorizonLevel::H10k],
                            "20k" | "20000" | "areas" => vec![HorizonLevel::H20k],
                            "30k" | "30000" | "goals" => vec![HorizonLevel::H30k],
                            "40k" | "40000" | "vision" => vec![HorizonLevel::H40k],
                            "50k" | "50000" | "purpose" => vec![HorizonLevel::H50k],
                            _ => Vec::new(),
                        }
                    } else {
                        Vec::new()
                    };

                    let params = HorizonParams {
                        horizons: horizons_filter.clone(),
                        include_counts: true,
                        include_health: true,
                        include_alignment: true,
                        items_per_horizon: 10,
                        area: None,
                    };

                    if horizons_filter.is_empty() {
                        // Get full overview of all horizons
                        let overview = manager.overview().await?;
                        Ok(GtdResult {
                            success: true,
                            message: "Horizon overview".to_string(),
                            data: serde_json::to_value(overview)?,
                            extra: serde_json::Map::new(),
                        })
                    } else {
                        // Get specific horizon(s)
                        let map = manager.map(params).await?;
                        Ok(GtdResult {
                            success: true,
                            message: "Horizon map".to_string(),
                            data: serde_json::to_value(map)?,
                            extra: serde_json::Map::new(),
                        })
                    }
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }

        GtdCommand::Commitments {
            action,
            id,
            filter,
            pending,
            description,
            commitment_type,
            person,
            due,
            resolution: _resolution,
        } => {
            let manager = CommitmentManager::new(store);

            match action.as_str() {
                "list" => {
                    let mut f = CommitmentFilter::default();
                    if pending {
                        f.status = Some(CommitmentStatus::Pending);
                    }
                    match filter.as_str() {
                        "made" => f.commitment_type = Some(CommitmentDirection::Made),
                        "received" => f.commitment_type = Some(CommitmentDirection::Received),
                        _ => {}
                    }

                    let commitments = manager.list(f).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} commitments", commitments.len()),
                        data: serde_json::to_value(commitments)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "add" => {
                    let desc =
                        description.ok_or_else(|| anyhow::anyhow!("Description required"))?;
                    let p = person.ok_or_else(|| anyhow::anyhow!("Person required"))?;

                    let direction = match commitment_type.as_deref() {
                        Some("made") => CommitmentDirection::Made,
                        _ => CommitmentDirection::Received,
                    };

                    // Determine from_person and to_person based on direction
                    let (from_person, to_person) = match direction {
                        CommitmentDirection::Made => (Some("self".to_string()), Some(p)),
                        CommitmentDirection::Received => (Some(p), Some("self".to_string())),
                    };

                    let commitment = Commitment {
                        id: uuid::Uuid::new_v4().to_string(),
                        commitment_type: direction,
                        description: desc.clone(),
                        from_person,
                        to_person,
                        due_date: due.and_then(|d| parse_date(&d)),
                        status: CommitmentStatus::Pending,
                        source_document: None,
                        extracted_text: desc,
                        confidence: 1.0,
                        project_id: None,
                        follow_up_date: None,
                        notes: None,
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                    };

                    let created = manager.create(commitment).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Created commitment: {}", created.description),
                        data: serde_json::to_value(created)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "resolve" | "fulfill" => {
                    let commit_id = id.ok_or_else(|| anyhow::anyhow!("Commitment ID required"))?;
                    manager.fulfill(&commit_id).await?;
                    Ok(GtdResult {
                        success: true,
                        message: "Commitment fulfilled".to_string(),
                        data: serde_json::json!({ "id": commit_id, "status": "fulfilled" }),
                        extra: serde_json::Map::new(),
                    })
                }
                "cancel" => {
                    let commit_id = id.ok_or_else(|| anyhow::anyhow!("Commitment ID required"))?;
                    manager.cancel(&commit_id).await?;
                    Ok(GtdResult {
                        success: true,
                        message: "Commitment cancelled".to_string(),
                        data: serde_json::json!({ "id": commit_id, "status": "cancelled" }),
                        extra: serde_json::Map::new(),
                    })
                }
                "summary" => {
                    let summary = manager.summary().await?;
                    Ok(GtdResult {
                        success: true,
                        message: "Commitment summary".to_string(),
                        data: serde_json::to_value(summary)?,
                        extra: serde_json::Map::new(),
                    })
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }

        GtdCommand::Dependencies {
            action,
            project,
            critical_path,
            blocked_task,
            blocking_task,
        } => {
            let manager = DependencyManager::new(store.clone());

            match action.as_str() {
                "list" => {
                    let params = DependencyParams {
                        project_id: project,
                        include_completed: false,
                        max_depth: 5,
                        include_critical_path: critical_path,
                        include_blockers: true,
                        output_format: OutputFormat::Json,
                    };

                    let graph = manager.generate(params).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!(
                            "Dependency analysis: {} nodes, {} edges",
                            graph.nodes.len(),
                            graph.edges.len()
                        ),
                        data: serde_json::to_value(graph)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "add" => {
                    let blocked =
                        blocked_task.ok_or_else(|| anyhow::anyhow!("Blocked task required"))?;
                    let blocking =
                        blocking_task.ok_or_else(|| anyhow::anyhow!("Blocking task required"))?;

                    // Create a BlockedBy relationship in the ontology store
                    let relationship =
                        Relationship::new(&blocked, RelationType::BlockedBy, &blocking);
                    let store_writer = store.write().await;
                    store_writer
                        .create_relationship(relationship.clone())
                        .await?;

                    Ok(GtdResult {
                        success: true,
                        message: format!("Added dependency: {} blocked by {}", blocked, blocking),
                        data: serde_json::to_value(&relationship)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "critical-path" => {
                    let project_id = project
                        .ok_or_else(|| anyhow::anyhow!("Project ID required for critical path"))?;
                    let path = manager.get_critical_path(&project_id).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Critical path: {} items", path.length),
                        data: serde_json::to_value(path)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "blockers" => {
                    let item_id = blocked_task
                        .or(project)
                        .ok_or_else(|| anyhow::anyhow!("Item ID required"))?;
                    let blockers = manager.get_blockers(&item_id).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} blockers", blockers.len()),
                        data: serde_json::to_value(blockers)?,
                        extra: serde_json::Map::new(),
                    })
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }

        GtdCommand::Attention {
            period,
            group_by: _group_by,
        } => {
            let manager = AttentionManager::new(store);

            let period_days = match period.as_str() {
                "day" => 1,
                "month" => 30,
                _ => 7, // default to week
            };

            let params = AttentionParams {
                period_days,
                ..Default::default()
            };

            let metrics = manager.analyze(params).await?;
            Ok(GtdResult {
                success: true,
                message: format!("Attention analysis for {} days", period_days),
                data: serde_json::to_value(metrics)?,
                extra: serde_json::Map::new(),
            })
        }

        GtdCommand::Areas {
            action,
            id: _id,
            name,
            description,
        } => {
            // Areas are stored as entities in the ontology
            let store_guard = store.read().await;

            match action.as_str() {
                "list" => {
                    let filter = EntityFilter::by_types([EntityType::Area]);
                    let areas = store_guard.list_entities(filter).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} areas", areas.len()),
                        data: serde_json::to_value(areas)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "add" => {
                    drop(store_guard);
                    let store_write = store.read().await;
                    let area_name = name.ok_or_else(|| anyhow::anyhow!("Area name required"))?;
                    let mut entity = Entity::new(EntityType::Area, &area_name);
                    if let Some(d) = description {
                        entity = entity.with_metadata("description", serde_json::json!(d));
                    }

                    let created = store_write.create_entity(entity).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Created area: {}", created.name),
                        data: serde_json::to_value(created)?,
                        extra: serde_json::Map::new(),
                    })
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }

        GtdCommand::Goals {
            action,
            id,
            name,
            description,
            target_date,
            area,
        } => {
            let store_guard = store.read().await;

            match action.as_str() {
                "list" => {
                    let filter = EntityFilter::by_types([EntityType::Goal]);
                    let goals = store_guard.list_entities(filter).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Found {} goals", goals.len()),
                        data: serde_json::to_value(goals)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "add" => {
                    drop(store_guard);
                    let store_write = store.read().await;
                    let goal_name = name.ok_or_else(|| anyhow::anyhow!("Goal name required"))?;
                    let mut entity = Entity::new(EntityType::Goal, &goal_name);
                    if let Some(d) = description {
                        entity = entity.with_metadata("description", serde_json::json!(d));
                    }
                    if let Some(td) = target_date {
                        entity = entity.with_metadata("target_date", serde_json::json!(td));
                    }
                    if let Some(a) = area {
                        entity = entity.with_metadata("area_id", serde_json::json!(a));
                    }

                    let created = store_write.create_entity(entity).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Created goal: {}", created.name),
                        data: serde_json::to_value(created)?,
                        extra: serde_json::Map::new(),
                    })
                }
                "complete" => {
                    drop(store_guard);
                    let store_write = store.read().await;
                    let goal_id = id.ok_or_else(|| anyhow::anyhow!("Goal ID required"))?;
                    let update = EntityUpdate::default()
                        .set_meta("status", serde_json::json!("completed"))
                        .set_meta("completed_at", serde_json::json!(Utc::now().to_rfc3339()));
                    let updated = store_write.update_entity(&goal_id, update).await?;
                    Ok(GtdResult {
                        success: true,
                        message: format!("Goal completed: {}", updated.name),
                        data: serde_json::to_value(updated)?,
                        extra: serde_json::Map::new(),
                    })
                }
                _ => Ok(GtdResult {
                    success: false,
                    message: format!("Unknown action: {}", action),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                }),
            }
        }
    }
}

/// Execute Calendar commands locally.
pub async fn calendar(config: Config, command: CalendarCommand) -> Result<CalendarResult> {
    let store = get_ontology_store(&config).await?;
    let manager = CalendarManager::new(store);

    match command {
        CalendarCommand::Today => {
            let events = manager.today().await?;
            Ok(CalendarResult {
                success: true,
                message: format!("Found {} events today", events.len()),
                data: serde_json::to_value(events)?,
                extra: serde_json::Map::new(),
            })
        }

        CalendarCommand::Week => {
            let events = manager.this_week().await?;
            Ok(CalendarResult {
                success: true,
                message: format!("Found {} events this week", events.len()),
                data: serde_json::to_value(events)?,
                extra: serde_json::Map::new(),
            })
        }

        CalendarCommand::Range { start, end } => {
            let start_dt =
                parse_date(&start).ok_or_else(|| anyhow::anyhow!("Invalid start date"))?;
            let end_dt = parse_date(&end).ok_or_else(|| anyhow::anyhow!("Invalid end date"))?;

            let filter = CalendarFilter::date_range(start_dt, end_dt);
            let events = manager.list(&filter).await?;
            Ok(CalendarResult {
                success: true,
                message: format!("Found {} events in range", events.len()),
                data: serde_json::to_value(events)?,
                extra: serde_json::Map::new(),
            })
        }

        CalendarCommand::Free {
            start,
            end,
            min_duration,
        } => {
            let start_dt =
                parse_date(&start).ok_or_else(|| anyhow::anyhow!("Invalid start date"))?;
            let end_dt = parse_date(&end).ok_or_else(|| anyhow::anyhow!("Invalid end date"))?;

            let params = FreeTimeParams {
                range_start: start_dt,
                range_end: end_dt,
                min_duration_minutes: min_duration,
                ..Default::default()
            };

            let slots = manager.find_free_time(&params).await?;
            Ok(CalendarResult {
                success: true,
                message: format!("Found {} free time slots", slots.len()),
                data: serde_json::to_value(slots)?,
                extra: serde_json::Map::new(),
            })
        }

        CalendarCommand::Conflicts => {
            let filter = CalendarFilter::conflicts();
            let conflicts = manager.find_all_conflicts(&filter).await?;
            Ok(CalendarResult {
                success: true,
                message: format!("Found {} conflicts", conflicts.len()),
                data: serde_json::to_value(conflicts)?,
                extra: serde_json::Map::new(),
            })
        }

        CalendarCommand::Upcoming { limit } => {
            let events = manager.upcoming(Some(7)).await?;
            let events: Vec<_> = events.into_iter().take(limit).collect();
            Ok(CalendarResult {
                success: true,
                message: format!("Found {} upcoming events", events.len()),
                data: serde_json::to_value(events)?,
                extra: serde_json::Map::new(),
            })
        }

        CalendarCommand::Events {
            action,
            id,
            title,
            event_type,
            start,
            end,
            location,
            participants,
            project,
            recurrence: _recurrence,
            notes,
            from,
            to,
            filter_type,
        } => match action.as_str() {
            "list" => {
                let mut filter = CalendarFilter::default();
                if let Some(f) = from {
                    filter.start_date = parse_date(&f);
                }
                if let Some(t) = to {
                    filter.end_date = parse_date(&t);
                }
                if let Some(ft) = filter_type {
                    if let Some(et) = parse_event_type(&ft) {
                        filter.event_types.push(et);
                    }
                }

                let events = manager.list(&filter).await?;
                Ok(CalendarResult {
                    success: true,
                    message: format!("Found {} events", events.len()),
                    data: serde_json::to_value(events)?,
                    extra: serde_json::Map::new(),
                })
            }
            "get" => {
                let event_id = id.ok_or_else(|| anyhow::anyhow!("Event ID required"))?;
                match manager.get(&event_id).await? {
                    Some(event) => Ok(CalendarResult {
                        success: true,
                        message: format!("Event: {}", event.title),
                        data: serde_json::to_value(event)?,
                        extra: serde_json::Map::new(),
                    }),
                    None => Ok(CalendarResult {
                        success: false,
                        message: format!("Event not found: {}", event_id),
                        data: serde_json::Value::Null,
                        extra: serde_json::Map::new(),
                    }),
                }
            }
            "add" => {
                let event_title = title.ok_or_else(|| anyhow::anyhow!("Event title required"))?;
                let start_str = start.ok_or_else(|| anyhow::anyhow!("Start time required"))?;
                let start_dt =
                    parse_date(&start_str).ok_or_else(|| anyhow::anyhow!("Invalid start time"))?;

                let mut event = CalendarEvent::new(&event_title, start_dt);

                if let Some(et) = event_type {
                    if let Some(parsed_type) = parse_event_type(&et) {
                        event = event.with_type(parsed_type);
                    }
                }
                if let Some(e) = end {
                    if let Some(end_dt) = parse_date(&e) {
                        event = event.with_end(end_dt);
                    }
                }
                if let Some(l) = location {
                    event = event.with_location(l);
                }
                if let Some(p) = participants {
                    let parts: Vec<String> = p.split(',').map(|s| s.trim().to_string()).collect();
                    event = event.with_participants(parts);
                }
                if let Some(p) = project {
                    event = event.with_project(p);
                }
                if let Some(n) = notes {
                    event.notes = Some(n);
                }

                let created = manager.create(event).await?;
                Ok(CalendarResult {
                    success: true,
                    message: format!("Created event: {}", created.title),
                    data: serde_json::to_value(created)?,
                    extra: serde_json::Map::new(),
                })
            }
            "update" => {
                let event_id = id.ok_or_else(|| anyhow::anyhow!("Event ID required"))?;
                let mut update = EventUpdate::default();

                if let Some(t) = title {
                    update.title = Some(t);
                }
                if let Some(s) = start {
                    update.start = parse_date(&s);
                }
                if let Some(e) = end {
                    update.end = parse_date(&e);
                }
                if let Some(l) = location {
                    update.location = Some(l);
                }
                if let Some(p) = participants {
                    let parts: Vec<String> = p.split(',').map(|s| s.trim().to_string()).collect();
                    update.add_participants = parts;
                }
                if let Some(n) = notes {
                    update.notes = Some(n);
                }

                match manager.update(&event_id, update).await? {
                    Some(event) => Ok(CalendarResult {
                        success: true,
                        message: format!("Updated event: {}", event.title),
                        data: serde_json::to_value(event)?,
                        extra: serde_json::Map::new(),
                    }),
                    None => Ok(CalendarResult {
                        success: false,
                        message: format!("Event not found: {}", event_id),
                        data: serde_json::Value::Null,
                        extra: serde_json::Map::new(),
                    }),
                }
            }
            "delete" => {
                let event_id = id.ok_or_else(|| anyhow::anyhow!("Event ID required"))?;
                let deleted = manager.delete(&event_id).await?;
                Ok(CalendarResult {
                    success: deleted,
                    message: if deleted {
                        "Event deleted".to_string()
                    } else {
                        "Event not found".to_string()
                    },
                    data: serde_json::Value::Bool(deleted),
                    extra: serde_json::Map::new(),
                })
            }
            _ => Ok(CalendarResult {
                success: false,
                message: format!("Unknown action: {}", action),
                data: serde_json::Value::Null,
                extra: serde_json::Map::new(),
            }),
        },
    }
}

/// Execute Knowledge commands locally.
pub async fn knowledge(config: Config, command: KnowledgeCommand) -> Result<KnowledgeResult> {
    let store = get_ontology_store(&config).await?;
    let store_guard = store.read().await;

    match command {
        KnowledgeCommand::Search {
            query,
            types,
            limit,
        } => {
            let type_list: Vec<EntityType> = types
                .map(|t| {
                    t.split(',')
                        .filter_map(|s| parse_entity_type(s.trim()))
                        .collect()
                })
                .unwrap_or_default();

            let filter = EntityFilter {
                name_pattern: Some(query.clone()),
                limit,
                entity_types: type_list,
                ..Default::default()
            };

            let entities = store_guard.list_entities(filter).await?;
            Ok(KnowledgeResult {
                success: true,
                message: format!("Found {} entities matching '{}'", entities.len(), query),
                data: serde_json::to_value(entities)?,
                extra: serde_json::Map::new(),
            })
        }

        KnowledgeCommand::Entity {
            name,
            relationships,
        } => {
            let entities = store_guard.find_entities_by_name(&name, 1).await?;

            if let Some(entity) = entities.first() {
                if relationships {
                    let rels = store_guard.get_relationships_involving(&entity.id).await?;
                    let result = serde_json::json!({
                        "entity": entity,
                        "relationships": rels
                    });
                    Ok(KnowledgeResult {
                        success: true,
                        message: format!(
                            "Entity: {} with {} relationships",
                            entity.name,
                            rels.len()
                        ),
                        data: result,
                        extra: serde_json::Map::new(),
                    })
                } else {
                    Ok(KnowledgeResult {
                        success: true,
                        message: format!("Entity: {}", entity.name),
                        data: serde_json::to_value(entity)?,
                        extra: serde_json::Map::new(),
                    })
                }
            } else {
                Ok(KnowledgeResult {
                    success: false,
                    message: format!("Entity not found: {}", name),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                })
            }
        }

        KnowledgeCommand::Expert { topic, limit } => {
            // Find people who have AboutTopic relationships with the given topic
            let topics = store_guard.find_entities_by_name(&topic, 10).await?;
            let topic_ids: Vec<String> = topics
                .iter()
                .filter(|e| e.entity_type == EntityType::Topic)
                .map(|e| e.id.clone())
                .collect();

            let mut experts = Vec::new();
            for topic_id in topic_ids {
                let rels = store_guard.get_relationships_to(&topic_id).await?;
                for rel in rels {
                    if rel.relationship_type == RelationType::AboutTopic {
                        if let Some(entity) = store_guard.get_entity(&rel.source_entity_id).await? {
                            if entity.entity_type == EntityType::Person
                                && !experts.iter().any(|e: &Entity| e.id == entity.id)
                            {
                                experts.push(entity);
                            }
                        }
                    }
                }
            }

            let experts: Vec<_> = experts.into_iter().take(limit).collect();
            Ok(KnowledgeResult {
                success: true,
                message: format!("Found {} experts on '{}'", experts.len(), topic),
                data: serde_json::to_value(experts)?,
                extra: serde_json::Map::new(),
            })
        }

        KnowledgeCommand::Topic { topic } => {
            let entities = store_guard.find_entities_by_name(&topic, 10).await?;
            let topic_entity = entities.iter().find(|e| e.entity_type == EntityType::Topic);

            if let Some(entity) = topic_entity {
                let rels = store_guard.get_relationships_involving(&entity.id).await?;
                let result = serde_json::json!({
                    "topic": entity,
                    "related_entities": rels.len()
                });
                Ok(KnowledgeResult {
                    success: true,
                    message: format!("Topic: {} with {} relationships", entity.name, rels.len()),
                    data: result,
                    extra: serde_json::Map::new(),
                })
            } else {
                Ok(KnowledgeResult {
                    success: false,
                    message: format!("Topic not found: {}", topic),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                })
            }
        }

        KnowledgeCommand::Connected { entity, depth } => {
            let entities = store_guard.find_entities_by_name(&entity, 1).await?;

            if let Some(start_entity) = entities.first() {
                let connected = store_guard
                    .get_connected_entities(&start_entity.id, None, depth)
                    .await?;

                let result: Vec<serde_json::Value> = connected
                    .into_iter()
                    .map(|(e, rels)| {
                        serde_json::json!({
                            "entity": e,
                            "path_length": rels.len()
                        })
                    })
                    .collect();

                Ok(KnowledgeResult {
                    success: true,
                    message: format!("Found {} connected entities", result.len()),
                    data: serde_json::to_value(result)?,
                    extra: serde_json::Map::new(),
                })
            } else {
                Ok(KnowledgeResult {
                    success: false,
                    message: format!("Entity not found: {}", entity),
                    data: serde_json::Value::Null,
                    extra: serde_json::Map::new(),
                })
            }
        }
    }
}

/// Execute natural language query locally.
pub async fn query(
    config: Config,
    query_text: String,
    query_mode: Option<String>,
) -> Result<QueryResult> {
    let store = get_ontology_store(&config).await?;
    let store_guard = store.read().await;

    // Simple keyword-based query routing
    let mode = query_mode.clone().unwrap_or_else(|| {
        let q = query_text.to_lowercase();
        if q.contains("project")
            || q.contains("task")
            || q.contains("next action")
            || q.contains("waiting")
        {
            "gtd".to_string()
        } else if q.contains("calendar")
            || q.contains("meeting")
            || q.contains("event")
            || q.contains("schedule")
        {
            "calendar".to_string()
        } else {
            "knowledge".to_string()
        }
    });

    // Search entities based on the query
    let filter = EntityFilter {
        name_pattern: Some(query_text.clone()),
        limit: 20,
        ..Default::default()
    };

    let results = store_guard.list_entities(filter).await?;

    Ok(QueryResult {
        success: true,
        query: query_text,
        mode,
        message: format!("Found {} results", results.len()),
        data: serde_json::to_value(results)?,
        extra: serde_json::Map::new(),
    })
}

/// Execute Ontology commands locally.
pub async fn ontology(config: Config, command: OntologyCommand) -> Result<OntologyResult> {
    let store = get_ontology_store(&config).await?;
    let store_guard = store.read().await;

    match command {
        OntologyCommand::Stats => {
            let stats = store_guard.stats().await?;
            Ok(OntologyResult {
                success: true,
                message: format!(
                    "Ontology: {} entities, {} relationships",
                    stats.entity_count, stats.relationship_count
                ),
                data: serde_json::to_value(stats)?,
                extra: serde_json::Map::new(),
            })
        }

        OntologyCommand::Entities {
            action,
            id,
            entity_type,
            name_contains,
            limit,
            set_type,
            name,
            aliases,
            metadata,
            merge_into,
        } => match action.as_str() {
            "list" => {
                let mut filter = EntityFilter::default();
                if let Some(et) = entity_type {
                    if let Some(parsed) = parse_entity_type(&et) {
                        filter.entity_types.push(parsed);
                    }
                }
                if let Some(nc) = name_contains {
                    filter.name_pattern = Some(nc);
                }
                filter.limit = limit;

                let entities = store_guard.list_entities(filter).await?;
                Ok(OntologyResult {
                    success: true,
                    message: format!("Found {} entities", entities.len()),
                    data: serde_json::to_value(entities)?,
                    extra: serde_json::Map::new(),
                })
            }
            "get" => {
                let entity_id = id.ok_or_else(|| anyhow::anyhow!("Entity ID required"))?;
                match store_guard.get_entity(&entity_id).await? {
                    Some(entity) => Ok(OntologyResult {
                        success: true,
                        message: format!("Entity: {}", entity.name),
                        data: serde_json::to_value(entity)?,
                        extra: serde_json::Map::new(),
                    }),
                    None => Ok(OntologyResult {
                        success: false,
                        message: format!("Entity not found: {}", entity_id),
                        data: serde_json::Value::Null,
                        extra: serde_json::Map::new(),
                    }),
                }
            }
            "add" => {
                let entity_name = name.ok_or_else(|| anyhow::anyhow!("Entity name required"))?;
                let type_str = set_type.ok_or_else(|| anyhow::anyhow!("Entity type required"))?;
                let parsed_type = parse_entity_type(&type_str)
                    .ok_or_else(|| anyhow::anyhow!("Invalid entity type: {}", type_str))?;

                let mut entity = Entity::new(parsed_type, &entity_name);
                if let Some(a) = aliases {
                    let alias_list: Vec<String> =
                        a.split(',').map(|s| s.trim().to_string()).collect();
                    entity = entity.with_aliases(alias_list);
                }
                if let Some(m) = metadata {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&m) {
                        if let Some(obj) = parsed.as_object() {
                            for (k, v) in obj {
                                entity = entity.with_metadata(k, v.clone());
                            }
                        }
                    }
                }

                let created = store_guard.create_entity(entity).await?;
                Ok(OntologyResult {
                    success: true,
                    message: format!("Created entity: {}", created.name),
                    data: serde_json::to_value(created)?,
                    extra: serde_json::Map::new(),
                })
            }
            "update" => {
                let entity_id = id.ok_or_else(|| anyhow::anyhow!("Entity ID required"))?;

                // Build update from provided fields
                let mut update = EntityUpdate::default();

                if let Some(n) = name {
                    update.name = Some(n);
                }
                if let Some(t) = set_type {
                    if let Some(parsed) = parse_entity_type(&t) {
                        update.entity_type = Some(parsed);
                    }
                }
                if let Some(a) = aliases {
                    // Treat as setting aliases (add them)
                    update.add_aliases = a.split(',').map(|s| s.trim().to_string()).collect();
                }
                if let Some(m) = metadata {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&m) {
                        if let Some(obj) = parsed.as_object() {
                            for (k, v) in obj {
                                update.set_metadata.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }

                let updated = store_guard.update_entity(&entity_id, update).await?;
                Ok(OntologyResult {
                    success: true,
                    message: format!("Updated entity: {}", updated.name),
                    data: serde_json::to_value(updated)?,
                    extra: serde_json::Map::new(),
                })
            }
            "delete" => {
                let entity_id = id.ok_or_else(|| anyhow::anyhow!("Entity ID required"))?;
                let deleted = store_guard.delete_entity(&entity_id).await?;
                Ok(OntologyResult {
                    success: deleted,
                    message: if deleted {
                        "Entity deleted".to_string()
                    } else {
                        "Entity not found".to_string()
                    },
                    data: serde_json::Value::Bool(deleted),
                    extra: serde_json::Map::new(),
                })
            }
            "merge" => {
                let entity_id = id.ok_or_else(|| anyhow::anyhow!("Entity ID required"))?;
                let target_id =
                    merge_into.ok_or_else(|| anyhow::anyhow!("Target entity ID required"))?;
                let merged = store_guard.merge_entities(&target_id, &entity_id).await?;
                Ok(OntologyResult {
                    success: true,
                    message: format!("Merged into entity: {}", merged.name),
                    data: serde_json::to_value(merged)?,
                    extra: serde_json::Map::new(),
                })
            }
            _ => Ok(OntologyResult {
                success: false,
                message: format!("Unknown action: {}", action),
                data: serde_json::Value::Null,
                extra: serde_json::Map::new(),
            }),
        },

        OntologyCommand::Relationships {
            action,
            id,
            source,
            target,
            rel_type,
            limit,
            from_entity,
            to_entity,
            set_type,
        } => match action.as_str() {
            "list" => {
                let mut filter = RelationshipFilter::default();
                if let Some(s) = source {
                    filter.source_entity_id = Some(s);
                }
                if let Some(t) = target {
                    filter.target_entity_id = Some(t);
                }
                if let Some(rt) = rel_type {
                    if let Some(parsed) = parse_relation_type(&rt) {
                        filter.relationship_types.push(parsed);
                    }
                }
                filter.limit = limit;

                let relationships = store_guard.list_relationships(filter).await?;
                Ok(OntologyResult {
                    success: true,
                    message: format!("Found {} relationships", relationships.len()),
                    data: serde_json::to_value(relationships)?,
                    extra: serde_json::Map::new(),
                })
            }
            "add" => {
                let source_id =
                    from_entity.ok_or_else(|| anyhow::anyhow!("Source entity ID required"))?;
                let target_id =
                    to_entity.ok_or_else(|| anyhow::anyhow!("Target entity ID required"))?;
                let type_str =
                    set_type.ok_or_else(|| anyhow::anyhow!("Relationship type required"))?;
                let parsed_type = parse_relation_type(&type_str)
                    .ok_or_else(|| anyhow::anyhow!("Invalid relationship type: {}", type_str))?;

                let relationship = Relationship::new(&source_id, parsed_type, &target_id);
                let created = store_guard.create_relationship(relationship).await?;
                Ok(OntologyResult {
                    success: true,
                    message: format!("Created relationship: {} -> {}", source_id, target_id),
                    data: serde_json::to_value(created)?,
                    extra: serde_json::Map::new(),
                })
            }
            "delete" => {
                let rel_id = id.ok_or_else(|| anyhow::anyhow!("Relationship ID required"))?;
                let deleted = store_guard.delete_relationship(&rel_id).await?;
                Ok(OntologyResult {
                    success: deleted,
                    message: if deleted {
                        "Relationship deleted".to_string()
                    } else {
                        "Relationship not found".to_string()
                    },
                    data: serde_json::Value::Bool(deleted),
                    extra: serde_json::Map::new(),
                })
            }
            _ => Ok(OntologyResult {
                success: false,
                message: format!("Unknown action: {}", action),
                data: serde_json::Value::Null,
                extra: serde_json::Map::new(),
            }),
        },

        OntologyCommand::Extract {
            document,
            show_confidence,
            auto_add,
        } => {
            // Read the document content - could be a file path or text
            let (text, doc_id) = if std::path::Path::new(&document).exists() {
                // It's a file path - read the file
                let content = std::fs::read_to_string(&document).map_err(|e| {
                    anyhow::anyhow!("Failed to read document '{}': {}", document, e)
                })?;
                let doc_id = std::path::Path::new(&document)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&document)
                    .to_string();
                (content, doc_id)
            } else {
                // Treat the argument as the text itself if it's not a file
                (document.clone(), "cli-input".to_string())
            };

            // Create extraction pipeline with default config
            let extraction_config = ExtractionConfig::default();
            let pipeline = ExtractionPipeline::new(extraction_config);

            // Run extraction
            let result = pipeline.extract(&text, &doc_id).await?;

            // Build extracted entities summary
            let mut entities_summary: Vec<serde_json::Value> = Vec::new();
            for extracted in &result.entities {
                let mut entity_obj = serde_json::json!({
                    "name": extracted.entity.name,
                    "type": format!("{:?}", extracted.entity.entity_type),
                    "source_text": extracted.source_text,
                });

                if show_confidence {
                    entity_obj["confidence"] = serde_json::json!(extracted.entity.confidence);
                    entity_obj["method"] =
                        serde_json::json!(format!("{:?}", extracted.extraction_method));
                }

                entities_summary.push(entity_obj);
            }

            // Build relationships summary
            let mut relationships_summary: Vec<serde_json::Value> = Vec::new();
            for extracted in &result.relationships {
                let mut rel_obj = serde_json::json!({
                    "source": extracted.relationship.source_entity_id,
                    "target": extracted.relationship.target_entity_id,
                    "type": format!("{:?}", extracted.relationship.relationship_type),
                });

                if show_confidence {
                    rel_obj["confidence"] = serde_json::json!(extracted.relationship.confidence);
                }

                relationships_summary.push(rel_obj);
            }

            // Build temporal extractions summary
            let mut temporal_summary: Vec<serde_json::Value> = Vec::new();
            for temporal in &result.temporal {
                let mut temp_obj = serde_json::json!({
                    "text": temporal.original_text,
                    "normalized": temporal.normalized_text,
                    "type": format!("{:?}", temporal.date_type),
                });

                if show_confidence {
                    temp_obj["confidence"] = serde_json::json!(temporal.confidence);
                }

                temporal_summary.push(temp_obj);
            }

            // Build actions summary
            let mut actions_summary: Vec<serde_json::Value> = Vec::new();
            for action in &result.actions {
                let mut action_obj = serde_json::json!({
                    "description": action.description,
                    "type": format!("{:?}", action.action_type),
                    "priority": format!("{:?}", action.priority),
                });

                if show_confidence {
                    action_obj["confidence"] = serde_json::json!(action.confidence);
                }

                actions_summary.push(action_obj);
            }

            // Auto-add entities to ontology if requested
            let mut added_count = 0;
            if auto_add {
                for extracted in &result.entities {
                    // Check if entity already exists by name
                    let existing = store_guard
                        .find_entities_by_name(&extracted.entity.name, 1)
                        .await?;

                    if existing.is_empty() {
                        match store_guard.create_entity(extracted.entity.clone()).await {
                            Ok(_) => added_count += 1,
                            Err(e) => {
                                // Log but continue - entity might already exist
                                eprintln!(
                                    "Warning: Could not add entity '{}': {}",
                                    extracted.entity.name, e
                                );
                            }
                        }
                    }
                }
            }

            // Build the result data
            let mut data = serde_json::json!({
                "document_id": doc_id,
                "entities": entities_summary,
                "relationships": relationships_summary,
                "temporal": temporal_summary,
                "actions": actions_summary,
                "processing_ms": result.metadata.processing_ms,
            });

            if show_confidence {
                data["overall_confidence"] = serde_json::json!(result.confidence);
                data["used_llm"] = serde_json::json!(result.metadata.used_llm);
            }

            // Build summary message
            let message = if auto_add {
                format!(
                    "Extracted {} entities ({} added), {} relationships, {} temporal expressions, {} actions from '{}'",
                    result.entities.len(),
                    added_count,
                    result.relationships.len(),
                    result.temporal.len(),
                    result.actions.len(),
                    doc_id
                )
            } else {
                format!(
                    "Extracted {} entities, {} relationships, {} temporal expressions, {} actions from '{}'",
                    result.entities.len(),
                    result.relationships.len(),
                    result.temporal.len(),
                    result.actions.len(),
                    doc_id
                )
            };

            Ok(OntologyResult {
                success: true,
                message,
                data,
                extra: serde_json::Map::new(),
            })
        }

        OntologyCommand::Person {
            name,
            organization,
            email,
            topics,
            aliases,
        } => {
            let mut entity = Entity::new(EntityType::Person, &name);
            if let Some(a) = aliases {
                let alias_list: Vec<String> = a.split(',').map(|s| s.trim().to_string()).collect();
                entity = entity.with_aliases(alias_list);
            }
            if let Some(e) = email {
                entity = entity.with_metadata("email", serde_json::json!(e));
            }

            let created = store_guard.create_entity(entity).await?;

            // Create relationships if organization or topics specified
            if let Some(org_name) = organization {
                // Find or create organization
                let orgs = store_guard.find_entities_by_name(&org_name, 1).await?;
                let org_id = if let Some(org) = orgs.first() {
                    org.id.clone()
                } else {
                    let org_entity = Entity::new(EntityType::Organization, &org_name);
                    let created_org = store_guard.create_entity(org_entity).await?;
                    created_org.id
                };

                let rel = Relationship::new(&created.id, RelationType::WorksFor, &org_id);
                store_guard.create_relationship(rel).await?;
            }

            if let Some(topic_list) = topics {
                for topic_name in topic_list.split(',').map(|s| s.trim()) {
                    let topics = store_guard.find_entities_by_name(topic_name, 1).await?;
                    let topic_id = if let Some(t) = topics.first() {
                        t.id.clone()
                    } else {
                        let topic_entity = Entity::new(EntityType::Topic, topic_name);
                        let created_topic = store_guard.create_entity(topic_entity).await?;
                        created_topic.id
                    };

                    let rel = Relationship::new(&created.id, RelationType::AboutTopic, &topic_id);
                    store_guard.create_relationship(rel).await?;
                }
            }

            Ok(OntologyResult {
                success: true,
                message: format!("Created person: {}", created.name),
                data: serde_json::to_value(created)?,
                extra: serde_json::Map::new(),
            })
        }

        OntologyCommand::Organization {
            name,
            org_type,
            parent,
            aliases,
        } => {
            let mut entity = Entity::new(EntityType::Organization, &name);
            if let Some(a) = aliases {
                let alias_list: Vec<String> = a.split(',').map(|s| s.trim().to_string()).collect();
                entity = entity.with_aliases(alias_list);
            }
            if let Some(ot) = org_type {
                entity = entity.with_metadata("org_type", serde_json::json!(ot));
            }

            let created = store_guard.create_entity(entity).await?;

            if let Some(parent_name) = parent {
                let parents = store_guard.find_entities_by_name(&parent_name, 1).await?;
                if let Some(parent_entity) = parents.first() {
                    let rel =
                        Relationship::new(&created.id, RelationType::PartOf, &parent_entity.id);
                    store_guard.create_relationship(rel).await?;
                }
            }

            Ok(OntologyResult {
                success: true,
                message: format!("Created organization: {}", created.name),
                data: serde_json::to_value(created)?,
                extra: serde_json::Map::new(),
            })
        }

        OntologyCommand::Topic {
            name,
            description,
            parent,
            aliases,
        } => {
            let mut entity = Entity::new(EntityType::Topic, &name);
            if let Some(a) = aliases {
                let alias_list: Vec<String> = a.split(',').map(|s| s.trim().to_string()).collect();
                entity = entity.with_aliases(alias_list);
            }
            if let Some(d) = description {
                entity = entity.with_metadata("description", serde_json::json!(d));
            }

            let created = store_guard.create_entity(entity).await?;

            if let Some(parent_name) = parent {
                let parents = store_guard.find_entities_by_name(&parent_name, 1).await?;
                if let Some(parent_entity) = parents.first() {
                    let rel =
                        Relationship::new(&created.id, RelationType::PartOf, &parent_entity.id);
                    store_guard.create_relationship(rel).await?;
                }
            }

            Ok(OntologyResult {
                success: true,
                message: format!("Created topic: {}", created.name),
                data: serde_json::to_value(created)?,
                extra: serde_json::Map::new(),
            })
        }
    }
}

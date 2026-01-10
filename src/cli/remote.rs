//! Remote execution via MCP client.
//!
//! This module executes CLI commands by connecting to a remote Alloy MCP server.

use alloy::mcp::{
    CreateBackupResponse, DocumentDetails, ExportDocumentsResponse, ImportDocumentsResponse,
    IndexPathResponse, IndexStats, ListBackupsResponse, ListSourcesResponse, RemoveSourceResponse,
    RestoreBackupResponse, SearchResponse,
};
use anyhow::{anyhow, Result};
use rmcp::{
    model::{CallToolRequestParam, RawContent},
    service::RunningService,
    transport::streamable_http_client::{
        StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
    },
    RoleClient, ServiceExt,
};

/// MCP client for remote Alloy server.
pub struct McpClient {
    client: RunningService<RoleClient, ()>,
}

impl McpClient {
    /// Connect to a remote Alloy MCP server.
    pub async fn connect(url: &str) -> Result<Self> {
        let http_client = reqwest::Client::new();
        let config = StreamableHttpClientTransportConfig::with_uri(url);
        let transport = StreamableHttpClientTransport::with_client(http_client, config);
        let client = ().serve(transport).await?;
        Ok(Self { client })
    }

    /// Call a tool and parse the JSON response.
    async fn call_tool<T: serde::de::DeserializeOwned>(
        &self,
        name: &'static str,
        arguments: serde_json::Value,
    ) -> Result<T> {
        let result = self
            .client
            .call_tool(CallToolRequestParam {
                name: std::borrow::Cow::Borrowed(name),
                arguments: Some(arguments.as_object().cloned().unwrap_or_default()),
                task: None,
            })
            .await?;

        // Extract text content from result
        let text = result
            .content
            .iter()
            .filter_map(|c| {
                // Content is Annotated<RawContent>, we access .raw through Deref
                match &c.raw {
                    RawContent::Text(text_content) => Some(text_content.text.as_str()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            return Err(anyhow!("Empty response from MCP server"));
        }

        serde_json::from_str(&text).map_err(|e| anyhow!("Failed to parse response: {}", e))
    }

    /// Index a path via remote MCP server.
    pub async fn index(
        &self,
        path: String,
        pattern: Option<String>,
        watch: bool,
    ) -> Result<IndexPathResponse> {
        self.call_tool(
            "index_path",
            serde_json::json!({
                "path": path,
                "pattern": pattern,
                "watch": watch,
            }),
        )
        .await
    }

    /// Search via remote MCP server.
    pub async fn search(
        &self,
        query: String,
        limit: usize,
        vector_weight: f32,
        source_id: Option<String>,
    ) -> Result<SearchResponse> {
        self.call_tool(
            "search",
            serde_json::json!({
                "query": query,
                "limit": limit,
                "vector_weight": vector_weight,
                "source_id": source_id,
            }),
        )
        .await
    }

    /// Get document via remote MCP server.
    pub async fn get_document(
        &self,
        document_id: String,
        include_content: bool,
    ) -> Result<Option<DocumentDetails>> {
        self.call_tool(
            "get_document",
            serde_json::json!({
                "document_id": document_id,
                "include_content": include_content,
            }),
        )
        .await
    }

    /// List sources via remote MCP server.
    pub async fn list_sources(&self) -> Result<ListSourcesResponse> {
        self.call_tool("list_sources", serde_json::json!({})).await
    }

    /// Remove source via remote MCP server.
    pub async fn remove_source(&self, source_id: String) -> Result<RemoveSourceResponse> {
        self.call_tool(
            "remove_source",
            serde_json::json!({
                "source_id": source_id,
            }),
        )
        .await
    }

    /// Get stats via remote MCP server.
    pub async fn stats(&self) -> Result<IndexStats> {
        self.call_tool("get_stats", serde_json::json!({})).await
    }
}

// Module-level convenience functions

/// Index a path via remote MCP server.
pub async fn index(
    url: &str,
    path: String,
    pattern: Option<String>,
    watch: bool,
) -> Result<IndexPathResponse> {
    McpClient::connect(url)
        .await?
        .index(path, pattern, watch)
        .await
}

/// Search via remote MCP server.
pub async fn search(
    url: &str,
    query: String,
    limit: usize,
    vector_weight: f32,
    source_id: Option<String>,
) -> Result<SearchResponse> {
    McpClient::connect(url)
        .await?
        .search(query, limit, vector_weight, source_id)
        .await
}

/// Get document via remote MCP server.
pub async fn get_document(
    url: &str,
    document_id: String,
    include_content: bool,
) -> Result<Option<DocumentDetails>> {
    McpClient::connect(url)
        .await?
        .get_document(document_id, include_content)
        .await
}

/// List sources via remote MCP server.
pub async fn list_sources(url: &str) -> Result<ListSourcesResponse> {
    McpClient::connect(url).await?.list_sources().await
}

/// Remove source via remote MCP server.
pub async fn remove_source(url: &str, source_id: String) -> Result<RemoveSourceResponse> {
    McpClient::connect(url)
        .await?
        .remove_source(source_id)
        .await
}

/// Get stats via remote MCP server.
pub async fn stats(url: &str) -> Result<IndexStats> {
    McpClient::connect(url).await?.stats().await
}

/// Cluster documents via remote MCP server.
pub async fn cluster(
    url: &str,
    source_id: Option<String>,
    algorithm: Option<String>,
    num_clusters: Option<usize>,
) -> Result<alloy::mcp::ClusterDocumentsResponse> {
    let client = McpClient::connect(url).await?;
    client
        .call_tool(
            "cluster_documents",
            serde_json::json!({
                "source_id": source_id,
                "algorithm": algorithm,
                "num_clusters": num_clusters,
            }),
        )
        .await
}

/// Create a backup via remote MCP server.
pub async fn backup(
    url: &str,
    output_path: Option<String>,
    description: Option<String>,
) -> Result<CreateBackupResponse> {
    let client = McpClient::connect(url).await?;
    client
        .call_tool(
            "create_backup",
            serde_json::json!({
                "output_path": output_path,
                "description": description,
            }),
        )
        .await
}

/// Restore from a backup via remote MCP server.
pub async fn restore(url: &str, backup_id: String) -> Result<RestoreBackupResponse> {
    let client = McpClient::connect(url).await?;
    client
        .call_tool(
            "restore_backup",
            serde_json::json!({
                "backup_id": backup_id,
            }),
        )
        .await
}

/// Export documents via remote MCP server.
pub async fn export(
    url: &str,
    output_path: String,
    format: String,
    source_id: Option<String>,
    include_embeddings: bool,
) -> Result<ExportDocumentsResponse> {
    let client = McpClient::connect(url).await?;
    client
        .call_tool(
            "export_documents",
            serde_json::json!({
                "output_path": output_path,
                "format": format,
                "source_id": source_id,
                "include_embeddings": include_embeddings,
            }),
        )
        .await
}

/// Import documents via remote MCP server.
pub async fn import(url: &str, input_path: String) -> Result<ImportDocumentsResponse> {
    let client = McpClient::connect(url).await?;
    client
        .call_tool(
            "import_documents",
            serde_json::json!({
                "input_path": input_path,
            }),
        )
        .await
}

/// List backups via remote MCP server.
pub async fn list_backups(url: &str) -> Result<ListBackupsResponse> {
    let client = McpClient::connect(url).await?;
    client
        .call_tool("list_backups", serde_json::json!({}))
        .await
}

// ============================================================================
// GTD, Calendar, Knowledge, Query, Ontology Commands
// ============================================================================

use super::types::{CalendarResult, GtdResult, KnowledgeResult, OntologyResult, QueryResult};
use crate::{CalendarCommand, GtdCommand, KnowledgeCommand, OntologyCommand};

/// Execute GTD commands via remote MCP server.
pub async fn gtd(url: &str, command: GtdCommand) -> Result<GtdResult> {
    let client = McpClient::connect(url).await?;

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
            client
                .call_tool(
                    "gtd_projects",
                    serde_json::json!({
                        "action": action,
                        "project_id": id,
                        "status": status,
                        "area": area,
                        "stalled_days": stalled,
                        "no_next_action": no_next_action,
                        "name": name,
                        "outcome": outcome,
                        "set_area": set_area,
                        "goal": goal,
                    }),
                )
                .await
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
            scheduled,
            assign_project,
            blocked_by,
        } => {
            client
                .call_tool(
                    "gtd_tasks",
                    serde_json::json!({
                        "action": action,
                        "task_id": id,
                        "contexts": context,
                        "energy": energy,
                        "time_available": time,
                        "project_id": project,
                        "status": status,
                        "due_before": due_before,
                        "description_contains": description_contains,
                        "limit": limit,
                        "description": description,
                        "set_contexts": set_contexts,
                        "set_energy": set_energy,
                        "priority": priority,
                        "duration": duration,
                        "due": due,
                        "scheduled": scheduled,
                        "assign_project": assign_project,
                        "blocked_by": blocked_by,
                    }),
                )
                .await
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
            client
                .call_tool(
                    "gtd_waiting",
                    serde_json::json!({
                        "action": action,
                        "item_id": id,
                        "status": status,
                        "person": person,
                        "project_id": project,
                        "description": description,
                        "delegated_to": delegated_to,
                        "expected_by": expected_by,
                        "for_project": for_project,
                        "resolution": resolution,
                    }),
                )
                .await
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
            client
                .call_tool(
                    "gtd_someday",
                    serde_json::json!({
                        "action": action,
                        "item_id": id,
                        "category": category,
                        "description": description,
                        "set_category": set_category,
                        "trigger": trigger,
                        "review_date": review_date,
                    }),
                )
                .await
        }
        GtdCommand::Review {
            week_ending,
            sections,
        } => {
            client
                .call_tool(
                    "gtd_review",
                    serde_json::json!({
                        "week_ending": week_ending,
                        "sections": sections,
                    }),
                )
                .await
        }
        GtdCommand::Horizons {
            action,
            level,
            name,
            description,
        } => {
            client
                .call_tool(
                    "gtd_horizons",
                    serde_json::json!({
                        "action": action,
                        "level": level,
                        "name": name,
                        "description": description,
                    }),
                )
                .await
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
            resolution,
        } => {
            client
                .call_tool(
                    "gtd_commitments",
                    serde_json::json!({
                        "action": action,
                        "commitment_id": id,
                        "filter": filter,
                        "pending": pending,
                        "description": description,
                        "commitment_type": commitment_type,
                        "person": person,
                        "due": due,
                        "resolution": resolution,
                    }),
                )
                .await
        }
        GtdCommand::Dependencies {
            action,
            project,
            critical_path,
            blocked_task,
            blocking_task,
        } => {
            client
                .call_tool(
                    "gtd_dependencies",
                    serde_json::json!({
                        "action": action,
                        "project_id": project,
                        "critical_path": critical_path,
                        "blocked_task": blocked_task,
                        "blocking_task": blocking_task,
                    }),
                )
                .await
        }
        GtdCommand::Attention { period, group_by } => {
            client
                .call_tool(
                    "gtd_attention",
                    serde_json::json!({
                        "period": period,
                        "group_by": group_by,
                    }),
                )
                .await
        }
        GtdCommand::Areas {
            action,
            id,
            name,
            description,
        } => {
            client
                .call_tool(
                    "ontology_entities",
                    serde_json::json!({
                        "action": action,
                        "entity_id": id,
                        "entity_type": "Area",
                        "name": name,
                        "description": description,
                    }),
                )
                .await
        }
        GtdCommand::Goals {
            action,
            id,
            name,
            description,
            target_date,
            area,
        } => {
            client
                .call_tool(
                    "ontology_entities",
                    serde_json::json!({
                        "action": action,
                        "entity_id": id,
                        "entity_type": "Goal",
                        "name": name,
                        "description": description,
                        "target_date": target_date,
                        "area": area,
                    }),
                )
                .await
        }
    }
}

/// Execute Calendar commands via remote MCP server.
pub async fn calendar(url: &str, command: CalendarCommand) -> Result<CalendarResult> {
    let client = McpClient::connect(url).await?;

    match command {
        CalendarCommand::Today => {
            client
                .call_tool(
                    "calendar_query",
                    serde_json::json!({
                        "action": "today",
                    }),
                )
                .await
        }
        CalendarCommand::Week => {
            client
                .call_tool(
                    "calendar_query",
                    serde_json::json!({
                        "action": "week",
                    }),
                )
                .await
        }
        CalendarCommand::Range { start, end } => {
            client
                .call_tool(
                    "calendar_query",
                    serde_json::json!({
                        "action": "range",
                        "start": start,
                        "end": end,
                    }),
                )
                .await
        }
        CalendarCommand::Free {
            start,
            end,
            min_duration,
        } => {
            client
                .call_tool(
                    "calendar_query",
                    serde_json::json!({
                        "action": "free",
                        "start": start,
                        "end": end,
                        "min_duration": min_duration,
                    }),
                )
                .await
        }
        CalendarCommand::Conflicts => {
            client
                .call_tool(
                    "calendar_query",
                    serde_json::json!({
                        "action": "conflicts",
                    }),
                )
                .await
        }
        CalendarCommand::Upcoming { limit } => {
            client
                .call_tool(
                    "calendar_query",
                    serde_json::json!({
                        "action": "upcoming",
                        "limit": limit,
                    }),
                )
                .await
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
            recurrence,
            notes,
            from,
            to,
            filter_type,
        } => {
            client
                .call_tool(
                    "calendar_events",
                    serde_json::json!({
                        "action": action,
                        "event_id": id,
                        "title": title,
                        "event_type": event_type,
                        "start": start,
                        "end": end,
                        "location": location,
                        "participants": participants,
                        "project_id": project,
                        "recurrence": recurrence,
                        "notes": notes,
                        "from": from,
                        "to": to,
                        "filter_type": filter_type,
                    }),
                )
                .await
        }
    }
}

/// Execute Knowledge commands via remote MCP server.
pub async fn knowledge(url: &str, command: KnowledgeCommand) -> Result<KnowledgeResult> {
    let client = McpClient::connect(url).await?;

    match command {
        KnowledgeCommand::Search {
            query,
            types,
            limit,
        } => {
            client
                .call_tool(
                    "knowledge_search",
                    serde_json::json!({
                        "query": query,
                        "entity_types": types,
                        "limit": limit,
                    }),
                )
                .await
        }
        KnowledgeCommand::Entity {
            name,
            relationships,
        } => {
            client
                .call_tool(
                    "knowledge_entity",
                    serde_json::json!({
                        "name": name,
                        "include_relationships": relationships,
                    }),
                )
                .await
        }
        KnowledgeCommand::Expert { topic, limit } => {
            client
                .call_tool(
                    "knowledge_expert",
                    serde_json::json!({
                        "topic": topic,
                        "limit": limit,
                    }),
                )
                .await
        }
        KnowledgeCommand::Topic { topic } => {
            client
                .call_tool(
                    "knowledge_topic",
                    serde_json::json!({
                        "topic": topic,
                    }),
                )
                .await
        }
        KnowledgeCommand::Connected { entity, depth } => {
            client
                .call_tool(
                    "knowledge_connected",
                    serde_json::json!({
                        "entity": entity,
                        "depth": depth,
                    }),
                )
                .await
        }
    }
}

/// Execute natural language query via remote MCP server.
pub async fn query(
    url: &str,
    query_text: String,
    query_mode: Option<String>,
) -> Result<QueryResult> {
    let client = McpClient::connect(url).await?;
    client
        .call_tool(
            "natural_language_query",
            serde_json::json!({
                "query": query_text,
                "mode": query_mode,
            }),
        )
        .await
}

/// Execute Ontology commands via remote MCP server.
pub async fn ontology(url: &str, command: OntologyCommand) -> Result<OntologyResult> {
    let client = McpClient::connect(url).await?;

    match command {
        OntologyCommand::Stats => {
            client
                .call_tool("ontology_stats", serde_json::json!({}))
                .await
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
        } => {
            client
                .call_tool(
                    "ontology_entities",
                    serde_json::json!({
                        "action": action,
                        "entity_id": id,
                        "entity_type": entity_type,
                        "name_contains": name_contains,
                        "limit": limit,
                        "set_type": set_type,
                        "name": name,
                        "aliases": aliases,
                        "metadata": metadata,
                        "merge_into": merge_into,
                    }),
                )
                .await
        }
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
        } => {
            client
                .call_tool(
                    "ontology_relationships",
                    serde_json::json!({
                        "action": action,
                        "relationship_id": id,
                        "source": source,
                        "target": target,
                        "rel_type": rel_type,
                        "limit": limit,
                        "from_entity": from_entity,
                        "to_entity": to_entity,
                        "set_type": set_type,
                    }),
                )
                .await
        }
        OntologyCommand::Extract {
            document,
            show_confidence,
            auto_add,
        } => {
            client
                .call_tool(
                    "ontology_extract",
                    serde_json::json!({
                        "document": document,
                        "show_confidence": show_confidence,
                        "auto_add": auto_add,
                    }),
                )
                .await
        }
        OntologyCommand::Person {
            name,
            organization,
            email,
            topics,
            aliases,
        } => {
            client
                .call_tool(
                    "ontology_person",
                    serde_json::json!({
                        "name": name,
                        "organization": organization,
                        "email": email,
                        "topics": topics,
                        "aliases": aliases,
                    }),
                )
                .await
        }
        OntologyCommand::Organization {
            name,
            org_type,
            parent,
            aliases,
        } => {
            client
                .call_tool(
                    "ontology_organization",
                    serde_json::json!({
                        "name": name,
                        "org_type": org_type,
                        "parent": parent,
                        "aliases": aliases,
                    }),
                )
                .await
        }
        OntologyCommand::Topic {
            name,
            description,
            parent,
            aliases,
        } => {
            client
                .call_tool(
                    "ontology_topic",
                    serde_json::json!({
                        "name": name,
                        "description": description,
                        "parent": parent,
                        "aliases": aliases,
                    }),
                )
                .await
        }
    }
}

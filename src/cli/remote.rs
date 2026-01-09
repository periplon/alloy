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

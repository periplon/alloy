//! Remote execution via MCP client.
//!
//! This module executes CLI commands by connecting to a remote Alloy MCP server.

use alloy::mcp::{
    DocumentDetails, IndexPathResponse, IndexStats, ListSourcesResponse, RemoveSourceResponse,
    SearchResponse,
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

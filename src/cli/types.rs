//! CLI response types for GTD, Calendar, Knowledge, Query, and Ontology commands.
//!
//! These types are used by the CLI to format output for users.
//!
//! The types use `#[serde(flatten)]` to capture any additional fields from MCP responses
//! into a generic `data` field, enabling compatibility with typed MCP response structures
//! (like `GtdProjectsResponse`, `GtdTasksResponse`, etc.) while maintaining flexibility
//! for the CLI output formatters.

use serde::{Deserialize, Serialize};

/// Result type for GTD commands.
///
/// Compatible with MCP responses: `GtdProjectsResponse`, `GtdTasksResponse`,
/// `GtdWaitingResponse`, `GtdSomedayResponse`, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdResult {
    pub success: bool,
    pub message: String,
    /// Additional data from the response (project, projects, task, tasks, item, items, etc.)
    /// Uses flatten to capture extra fields from MCP responses.
    #[serde(flatten)]
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Data field for local execution (not from MCP).
    #[serde(skip_serializing_if = "is_null", default)]
    pub data: serde_json::Value,
}

/// Helper function for serde skip_serializing_if
fn is_null(value: &serde_json::Value) -> bool {
    value.is_null()
}

#[allow(dead_code)]
impl GtdResult {
    /// Get the data as a serde_json::Value for output formatting.
    /// This merges both the `data` field (for local) and `extra` fields (from MCP).
    pub fn data_value(&self) -> serde_json::Value {
        // If we have local data, use it
        if !self.data.is_null() {
            return self.data.clone();
        }
        // Otherwise use flattened extra fields from MCP
        if self.extra.is_empty() {
            serde_json::Value::Null
        } else if let Some(projects) = self.extra.get("projects") {
            projects.clone()
        } else if let Some(project) = self.extra.get("project") {
            project.clone()
        } else if let Some(tasks) = self.extra.get("tasks") {
            tasks.clone()
        } else if let Some(task) = self.extra.get("task") {
            task.clone()
        } else if let Some(items) = self.extra.get("items") {
            items.clone()
        } else if let Some(item) = self.extra.get("item") {
            item.clone()
        } else if let Some(recommendations) = self.extra.get("recommendations") {
            recommendations.clone()
        } else if let Some(health) = self.extra.get("health") {
            health.clone()
        } else if let Some(horizons) = self.extra.get("horizons") {
            serde_json::json!({ "horizons": horizons })
        } else {
            serde_json::Value::Object(self.extra.clone())
        }
    }

    /// Create a new result for local execution.
    pub fn new(success: bool, message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success,
            message: message.into(),
            extra: serde_json::Map::new(),
            data,
        }
    }

    /// Create a successful result with structured data.
    pub fn success(message: impl Into<String>, data: serde_json::Value) -> Self {
        Self::new(true, message, data)
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(false, message, serde_json::Value::Null)
    }
}

/// Result type for Calendar commands.
///
/// Compatible with MCP responses: `CalendarQueryResponse`, `CalendarManageResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarResult {
    pub success: bool,
    pub message: String,
    /// Additional data from MCP responses (events, free_slots, conflicts, event, etc.)
    #[serde(flatten)]
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Data field for local execution.
    #[serde(skip_serializing_if = "is_null", default)]
    pub data: serde_json::Value,
}

#[allow(dead_code)]
impl CalendarResult {
    /// Get the data as a serde_json::Value for output formatting.
    pub fn data_value(&self) -> serde_json::Value {
        if !self.data.is_null() {
            return self.data.clone();
        }
        if self.extra.is_empty() {
            serde_json::Value::Null
        } else if let Some(events) = self.extra.get("events") {
            events.clone()
        } else if let Some(event) = self.extra.get("event") {
            event.clone()
        } else if let Some(free_slots) = self.extra.get("free_slots") {
            free_slots.clone()
        } else if let Some(conflicts) = self.extra.get("conflicts") {
            conflicts.clone()
        } else {
            serde_json::Value::Object(self.extra.clone())
        }
    }

    /// Create a new result for local execution.
    pub fn new(success: bool, message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success,
            message: message.into(),
            extra: serde_json::Map::new(),
            data,
        }
    }

    /// Create a successful result with structured data.
    pub fn success(message: impl Into<String>, data: serde_json::Value) -> Self {
        Self::new(true, message, data)
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(false, message, serde_json::Value::Null)
    }
}

/// Result type for Knowledge commands.
///
/// Compatible with MCP responses: `KnowledgeSearchResponse`, `KnowledgeEntityResponse`,
/// `KnowledgeExpertResponse`, `KnowledgeTopicResponse`, `KnowledgeConnectedResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    pub success: bool,
    pub message: String,
    /// Additional data from MCP responses.
    #[serde(flatten)]
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Data field for local execution.
    #[serde(skip_serializing_if = "is_null", default)]
    pub data: serde_json::Value,
}

#[allow(dead_code)]
impl KnowledgeResult {
    /// Get the data as a serde_json::Value for output formatting.
    pub fn data_value(&self) -> serde_json::Value {
        if !self.data.is_null() {
            return self.data.clone();
        }
        if self.extra.is_empty() {
            serde_json::Value::Null
        } else if let Some(entities) = self.extra.get("entities") {
            entities.clone()
        } else if let Some(entity) = self.extra.get("entity") {
            if let Some(relationships) = self.extra.get("relationships") {
                serde_json::json!({
                    "entity": entity,
                    "relationships": relationships
                })
            } else {
                entity.clone()
            }
        } else if let Some(experts) = self.extra.get("experts") {
            experts.clone()
        } else if let Some(topic) = self.extra.get("topic") {
            serde_json::json!({
                "topic": topic,
                "related_entities": self.extra.get("related_entity_count").cloned()
            })
        } else if let Some(connected) = self.extra.get("connected_entities") {
            connected.clone()
        } else {
            serde_json::Value::Object(self.extra.clone())
        }
    }

    /// Create a new result for local execution.
    pub fn new(success: bool, message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success,
            message: message.into(),
            extra: serde_json::Map::new(),
            data,
        }
    }

    /// Create a successful result with structured data.
    pub fn success(message: impl Into<String>, data: serde_json::Value) -> Self {
        Self::new(true, message, data)
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(false, message, serde_json::Value::Null)
    }
}

/// Result type for natural language Query command.
///
/// Compatible with MCP responses: `QueryResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub success: bool,
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub mode: String,
    pub message: String,
    /// Additional data from MCP responses.
    #[serde(flatten)]
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Data field for local execution.
    #[serde(skip_serializing_if = "is_null", default)]
    pub data: serde_json::Value,
}

#[allow(dead_code)]
impl QueryResult {
    /// Get the data as a serde_json::Value for output formatting.
    pub fn data_value(&self) -> serde_json::Value {
        if !self.data.is_null() {
            return self.data.clone();
        }
        if self.extra.is_empty() {
            serde_json::Value::Null
        } else if let Some(results) = self.extra.get("results") {
            results.clone()
        } else {
            serde_json::Value::Object(self.extra.clone())
        }
    }

    /// Create a new result for local execution.
    pub fn new(
        success: bool,
        query: impl Into<String>,
        mode: impl Into<String>,
        message: impl Into<String>,
        data: serde_json::Value,
    ) -> Self {
        Self {
            success,
            query: query.into(),
            mode: mode.into(),
            message: message.into(),
            extra: serde_json::Map::new(),
            data,
        }
    }

    /// Create a successful result with structured data.
    pub fn success(
        query: impl Into<String>,
        mode: impl Into<String>,
        message: impl Into<String>,
        data: serde_json::Value,
    ) -> Self {
        Self::new(true, query, mode, message, data)
    }

    /// Create an error result.
    pub fn error(query: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(false, query, "", message, serde_json::Value::Null)
    }
}

/// Result type for Ontology commands.
///
/// Compatible with MCP responses: `OntologyStatsResponse`, `OntologyEntitiesResponse`,
/// `OntologyRelationshipsResponse`, `OntologyExtractResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyResult {
    pub success: bool,
    pub message: String,
    /// Additional data from MCP responses.
    #[serde(flatten)]
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Data field for local execution.
    #[serde(skip_serializing_if = "is_null", default)]
    pub data: serde_json::Value,
}

#[allow(dead_code)]
impl OntologyResult {
    /// Get the data as a serde_json::Value for output formatting.
    pub fn data_value(&self) -> serde_json::Value {
        if !self.data.is_null() {
            return self.data.clone();
        }
        if self.extra.is_empty() {
            serde_json::Value::Null
        } else if self.extra.contains_key("entity_count") {
            // Stats response
            serde_json::Value::Object(self.extra.clone())
        } else if let Some(entities) = self.extra.get("entities") {
            entities.clone()
        } else if let Some(entity) = self.extra.get("entity") {
            entity.clone()
        } else if let Some(relationships) = self.extra.get("relationships") {
            relationships.clone()
        } else if let Some(extracted) = self.extra.get("extracted_entities") {
            extracted.clone()
        } else {
            serde_json::Value::Object(self.extra.clone())
        }
    }

    /// Create a new result for local execution.
    pub fn new(success: bool, message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success,
            message: message.into(),
            extra: serde_json::Map::new(),
            data,
        }
    }

    /// Create a successful result with structured data.
    pub fn success(message: impl Into<String>, data: serde_json::Value) -> Self {
        Self::new(true, message, data)
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(false, message, serde_json::Value::Null)
    }
}

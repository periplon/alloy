//! CLI response types for GTD, Calendar, Knowledge, Query, and Ontology commands.
//!
//! These types are used by the CLI to format output for users.

use serde::{Deserialize, Serialize};

/// Result type for GTD commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdResult {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "is_null")]
    pub data: serde_json::Value,
}

/// Result type for Calendar commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarResult {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "is_null")]
    pub data: serde_json::Value,
}

/// Result type for Knowledge commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "is_null")]
    pub data: serde_json::Value,
}

/// Result type for natural language Query command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub success: bool,
    pub query: String,
    pub mode: String,
    pub message: String,
    #[serde(skip_serializing_if = "is_null")]
    pub data: serde_json::Value,
}

/// Result type for Ontology commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyResult {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "is_null")]
    pub data: serde_json::Value,
}

/// Helper function for serde skip_serializing_if
fn is_null(value: &serde_json::Value) -> bool {
    value.is_null()
}

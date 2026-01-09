//! Natural Language Query MCP tool implementations for Alloy.
//!
//! This module provides MCP tools for natural language queries:
//! - `query`: Unified natural language query interface
//! - Automatically routes to GTD, Calendar, Knowledge, or Search

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::query::{QueryMode, QueryStats, UnifiedQueryResult};

// ============================================================================
// Query Tool Types
// ============================================================================

/// Parameters for the unified query tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct QueryParams {
    /// Natural language query.
    ///
    /// Examples:
    /// - "What are my @phone tasks?"
    /// - "What's blocking the website redesign project?"
    /// - "What should I work on now?"
    /// - "What do I know about machine learning?"
    /// - "Who can help with the AWS migration?"
    /// - "What's on my calendar this week?"
    /// - "What commitments did I make last week?"
    /// - "Show me stalled projects"
    /// - "What's waiting on John?"
    pub query: String,

    /// Query mode. If not specified, auto-detection is used.
    ///
    /// - `auto`: Automatically detect the best interpretation (default)
    /// - `gtd`: Force GTD interpretation (tasks, projects, etc.)
    /// - `calendar`: Force calendar interpretation
    /// - `knowledge`: Force knowledge graph interpretation
    /// - `search`: Force document search
    #[serde(default)]
    pub mode: Option<QueryModeParam>,

    /// Maximum number of results (default: 20).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Query mode parameter for MCP.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QueryModeParam {
    /// Auto-detect the best mode
    #[default]
    Auto,
    /// Force GTD interpretation
    Gtd,
    /// Force Calendar interpretation
    Calendar,
    /// Force Knowledge interpretation
    Knowledge,
    /// Force document search
    Search,
}

impl From<QueryModeParam> for QueryMode {
    fn from(mode: QueryModeParam) -> Self {
        match mode {
            QueryModeParam::Auto => QueryMode::Auto,
            QueryModeParam::Gtd => QueryMode::Gtd,
            QueryModeParam::Calendar => QueryMode::Calendar,
            QueryModeParam::Knowledge => QueryMode::Knowledge,
            QueryModeParam::Search => QueryMode::Search,
        }
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Response from the unified query tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Whether the query succeeded.
    pub success: bool,
    /// The interpreted intent.
    pub interpreted_as: String,
    /// Confidence in the interpretation (0.0 to 1.0).
    pub confidence: f32,
    /// Human-readable answer to the query.
    pub answer: String,
    /// Structured result data (format depends on intent).
    pub data: serde_json::Value,
    /// Suggestions for follow-up queries.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
    /// Query execution statistics.
    pub stats: NlQueryStats,
    /// Status message.
    pub message: String,
}

/// Query statistics for MCP response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NlQueryStats {
    /// Classification time in milliseconds.
    pub classification_time_ms: u64,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
    /// Total time in milliseconds.
    pub total_time_ms: u64,
    /// Which subsystem was queried.
    pub subsystem: String,
}

impl From<QueryStats> for NlQueryStats {
    fn from(stats: QueryStats) -> Self {
        Self {
            classification_time_ms: stats.classification_time_ms,
            execution_time_ms: stats.execution_time_ms,
            total_time_ms: stats.total_time_ms,
            subsystem: stats.subsystem,
        }
    }
}

impl QueryResponse {
    /// Create a successful response from a unified query result.
    pub fn from_result(result: UnifiedQueryResult) -> Self {
        Self {
            success: true,
            interpreted_as: result.intent.detailed_name(),
            confidence: result.confidence,
            answer: result.answer,
            data: result.data,
            suggestions: result.suggestions,
            stats: result.stats.into(),
            message: "Query executed successfully".to_string(),
        }
    }

    /// Create an error response.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            interpreted_as: "unknown".to_string(),
            confidence: 0.0,
            answer: String::new(),
            data: serde_json::Value::Null,
            suggestions: Vec::new(),
            stats: NlQueryStats::default(),
            message: message.into(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_mode_conversion() {
        assert!(matches!(
            QueryMode::from(QueryModeParam::Auto),
            QueryMode::Auto
        ));
        assert!(matches!(
            QueryMode::from(QueryModeParam::Gtd),
            QueryMode::Gtd
        ));
        assert!(matches!(
            QueryMode::from(QueryModeParam::Calendar),
            QueryMode::Calendar
        ));
    }

    #[test]
    fn test_error_response() {
        let response = QueryResponse::error("Test error");
        assert!(!response.success);
        assert_eq!(response.message, "Test error");
    }

    #[test]
    fn test_default_params() {
        let params = QueryParams::default();
        assert!(params.query.is_empty());
        assert!(params.mode.is_none());
        assert!(params.limit.is_none());
    }
}

//! Unified response types for MCP tools and CLI commands.
//!
//! These types match the exact specifications from the CLI implementation plan
//! and provide a consistent interface for both MCP tools and CLI commands.
//!
//! ## Type Organization
//!
//! - **GTD Types**: Re-exported from `gtd_tools` module (already match plan spec)
//! - **Calendar Types**: `UnifiedCalendarQueryResponse` (plan-specified structure)
//! - **Knowledge Types**: `UnifiedKnowledgeResult` (plan-specified structure)
//! - **Query Types**: `UnifiedQueryResult` (plan-specified structure)

use serde::{Deserialize, Serialize};

use crate::calendar::{CalendarEvent, CalendarStats, FreeTimeSlot, SchedulingConflict};
use crate::knowledge::DocumentSummary;
use crate::ontology::{Entity, Relationship};

// ============================================================================
// GTD Response Types (Re-exports from gtd_tools)
// ============================================================================

// Note: GtdProjectsResponse and GtdTasksResponse are defined in gtd_tools.rs
// and already match the plan specification exactly. They are re-exported
// via the mcp module's glob exports.

// ============================================================================
// Calendar Response Types
// ============================================================================

/// Unified response for calendar query operations.
///
/// This is the unified calendar response type as specified in the plan,
/// containing events, conflicts, free slots, and statistics.
///
/// Used by:
/// - CLI `alloy calendar` commands
/// - Can be converted from MCP responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedCalendarQueryResponse {
    /// Calendar events matching the query.
    pub events: Vec<CalendarEvent>,
    /// Scheduling conflicts detected.
    #[serde(default)]
    pub conflicts: Vec<SchedulingConflict>,
    /// Free time slots (for free time queries).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_slots: Option<Vec<FreeTimeSlot>>,
    /// Calendar statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<CalendarStats>,
}

impl UnifiedCalendarQueryResponse {
    /// Create a response with events only.
    pub fn events(events: Vec<CalendarEvent>) -> Self {
        Self {
            events,
            conflicts: Vec::new(),
            free_slots: None,
            stats: None,
        }
    }

    /// Create a response with free time slots.
    pub fn free_time(events: Vec<CalendarEvent>, free_slots: Vec<FreeTimeSlot>) -> Self {
        Self {
            events,
            conflicts: Vec::new(),
            free_slots: Some(free_slots),
            stats: None,
        }
    }

    /// Create a response with conflicts.
    pub fn with_conflicts(events: Vec<CalendarEvent>, conflicts: Vec<SchedulingConflict>) -> Self {
        Self {
            events,
            conflicts,
            free_slots: None,
            stats: None,
        }
    }

    /// Create a response with statistics.
    pub fn with_stats(events: Vec<CalendarEvent>, stats: CalendarStats) -> Self {
        Self {
            events,
            conflicts: Vec::new(),
            free_slots: None,
            stats: Some(stats),
        }
    }

    /// Create an empty response.
    pub fn empty() -> Self {
        Self {
            events: Vec::new(),
            conflicts: Vec::new(),
            free_slots: None,
            stats: None,
        }
    }
}

impl Default for UnifiedCalendarQueryResponse {
    fn default() -> Self {
        Self::empty()
    }
}

// ============================================================================
// Knowledge Response Types
// ============================================================================

/// Unified result type for knowledge graph queries.
///
/// This is the unified knowledge result type as specified in the plan,
/// containing entities, relationships, source documents, summary, and confidence.
///
/// Used by:
/// - CLI `alloy knowledge` commands
/// - Can be converted from MCP responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedKnowledgeResult {
    /// Matched entities from the knowledge graph.
    pub entities: Vec<Entity>,
    /// Related relationships found.
    pub relationships: Vec<Relationship>,
    /// Source document summaries with relevance scores.
    pub source_documents: Vec<DocumentSummary>,
    /// Generated summary text (for topic queries).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Overall confidence score (0.0 to 1.0).
    pub confidence: f32,
}

impl UnifiedKnowledgeResult {
    /// Create a new knowledge result with entities.
    pub fn with_entities(entities: Vec<Entity>, confidence: f32) -> Self {
        Self {
            entities,
            relationships: Vec::new(),
            source_documents: Vec::new(),
            summary: None,
            confidence,
        }
    }

    /// Create a knowledge result with summary.
    pub fn with_summary(summary: String, entities: Vec<Entity>, confidence: f32) -> Self {
        Self {
            entities,
            relationships: Vec::new(),
            source_documents: Vec::new(),
            summary: Some(summary),
            confidence,
        }
    }

    /// Create an empty result.
    pub fn empty() -> Self {
        Self {
            entities: Vec::new(),
            relationships: Vec::new(),
            source_documents: Vec::new(),
            summary: None,
            confidence: 0.0,
        }
    }

    /// Add relationships to the result.
    pub fn with_relationships(mut self, relationships: Vec<Relationship>) -> Self {
        self.relationships = relationships;
        self
    }

    /// Add source documents to the result.
    pub fn with_source_documents(mut self, source_documents: Vec<DocumentSummary>) -> Self {
        self.source_documents = source_documents;
        self
    }
}

impl Default for UnifiedKnowledgeResult {
    fn default() -> Self {
        Self::empty()
    }
}

// ============================================================================
// Query Response Types
// ============================================================================

/// GTD-specific query result for natural language queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtdQueryResult {
    /// Query interpretation.
    pub interpretation: String,
    /// Result data (tasks, projects, etc.).
    pub data: serde_json::Value,
    /// Suggestions for follow-up queries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
}

impl GtdQueryResult {
    /// Create a new GTD query result.
    pub fn new(interpretation: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            interpretation: interpretation.into(),
            data,
            suggestions: Vec::new(),
        }
    }

    /// Add suggestions to the result.
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions = suggestions;
        self
    }
}

/// Unified query result for natural language queries.
///
/// This type wraps results from different query subsystems (GTD, Calendar, Knowledge)
/// based on query mode detection.
///
/// Used by:
/// - CLI `alloy query` command
/// - Can be converted from MCP responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedQueryResult {
    /// The query mode that was used (auto, gtd, calendar, knowledge).
    pub query_mode: String,
    /// GTD result (if query was GTD-related).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gtd_result: Option<GtdQueryResult>,
    /// Calendar result (if query was calendar-related).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calendar_result: Option<UnifiedCalendarQueryResponse>,
    /// Knowledge result (if query was knowledge-related).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knowledge_result: Option<UnifiedKnowledgeResult>,
}

impl UnifiedQueryResult {
    /// Create a GTD query result.
    pub fn gtd(result: GtdQueryResult) -> Self {
        Self {
            query_mode: "gtd".to_string(),
            gtd_result: Some(result),
            calendar_result: None,
            knowledge_result: None,
        }
    }

    /// Create a calendar query result.
    pub fn calendar(result: UnifiedCalendarQueryResponse) -> Self {
        Self {
            query_mode: "calendar".to_string(),
            gtd_result: None,
            calendar_result: Some(result),
            knowledge_result: None,
        }
    }

    /// Create a knowledge query result.
    pub fn knowledge(result: UnifiedKnowledgeResult) -> Self {
        Self {
            query_mode: "knowledge".to_string(),
            gtd_result: None,
            calendar_result: None,
            knowledge_result: Some(result),
        }
    }

    /// Create an auto-detected query result.
    pub fn auto(mode: impl Into<String>) -> Self {
        Self {
            query_mode: mode.into(),
            gtd_result: None,
            calendar_result: None,
            knowledge_result: None,
        }
    }

    /// Check if the result is empty.
    pub fn is_empty(&self) -> bool {
        self.gtd_result.is_none()
            && self.calendar_result.is_none()
            && self.knowledge_result.is_none()
    }
}

impl Default for UnifiedQueryResult {
    fn default() -> Self {
        Self {
            query_mode: "auto".to_string(),
            gtd_result: None,
            calendar_result: None,
            knowledge_result: None,
        }
    }
}

// ============================================================================
// Type Aliases for Plan Compatibility
// ============================================================================

/// Alias for plan compatibility: CalendarQueryResponse as specified in the plan.
pub type CalendarQueryResponseSpec = UnifiedCalendarQueryResponse;

/// Alias for plan compatibility: KnowledgeResult as specified in the plan.
pub type KnowledgeResultSpec = UnifiedKnowledgeResult;

/// Alias for plan compatibility: QueryResult as specified in the plan.
pub type QueryResultSpec = UnifiedQueryResult;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calendar_query_response_builders() {
        let response = UnifiedCalendarQueryResponse::empty();
        assert!(response.events.is_empty());
        assert!(response.conflicts.is_empty());
        assert!(response.free_slots.is_none());
        assert!(response.stats.is_none());
    }

    #[test]
    fn test_knowledge_result_builders() {
        let result = UnifiedKnowledgeResult::empty();
        assert!(result.entities.is_empty());
        assert!(result.relationships.is_empty());
        assert!(result.source_documents.is_empty());
        assert!(result.summary.is_none());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_query_result_modes() {
        let gtd_result = UnifiedQueryResult::gtd(GtdQueryResult::new("tasks", serde_json::json!([])));
        assert_eq!(gtd_result.query_mode, "gtd");
        assert!(gtd_result.gtd_result.is_some());
        assert!(gtd_result.calendar_result.is_none());
        assert!(gtd_result.knowledge_result.is_none());

        let cal_result = UnifiedQueryResult::calendar(UnifiedCalendarQueryResponse::empty());
        assert_eq!(cal_result.query_mode, "calendar");
        assert!(cal_result.calendar_result.is_some());

        let know_result = UnifiedQueryResult::knowledge(UnifiedKnowledgeResult::empty());
        assert_eq!(know_result.query_mode, "knowledge");
        assert!(know_result.knowledge_result.is_some());
    }

    #[test]
    fn test_query_result_is_empty() {
        let empty = UnifiedQueryResult::default();
        assert!(empty.is_empty());

        let not_empty = UnifiedQueryResult::gtd(GtdQueryResult::new("test", serde_json::json!({})));
        assert!(!not_empty.is_empty());
    }

    #[test]
    fn test_type_aliases() {
        // Ensure type aliases work correctly
        let _: CalendarQueryResponseSpec = UnifiedCalendarQueryResponse::empty();
        let _: KnowledgeResultSpec = UnifiedKnowledgeResult::empty();
        let _: QueryResultSpec = UnifiedQueryResult::default();
    }
}

//! Calendar MCP tool implementations for Alloy.
//!
//! This module provides MCP tools for calendar functionality:
//! - `calendar_query`: Query calendar events with various filters
//! - `calendar_manage`: Create, update, delete calendar events
//! - `calendar_free_time`: Find free time slots
//! - `calendar_conflicts`: Detect scheduling conflicts

use chrono::{DateTime, Duration, NaiveTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::calendar::{
    CalendarEvent, CalendarFilter, CalendarQueryType, CalendarStats, ConflictSeverity,
    EventRecurrence, EventType, EventUpdate, FreeTimeParams, FreeTimeSlot, SchedulingConflict,
};

// ============================================================================
// Calendar Query Tool Types
// ============================================================================

/// Parameters for the calendar_query tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarQueryParams {
    /// Type of query to execute.
    #[serde(default)]
    pub query_type: CalendarQueryType,
    /// Start date for date range queries (ISO 8601 format).
    #[serde(default)]
    pub start_date: Option<DateTime<Utc>>,
    /// End date for date range queries (ISO 8601 format).
    #[serde(default)]
    pub end_date: Option<DateTime<Utc>>,
    /// Number of days for upcoming queries.
    #[serde(default)]
    pub days: Option<i64>,
    /// Event types to filter.
    #[serde(default)]
    pub event_types: Option<Vec<EventType>>,
    /// Search query text.
    #[serde(default)]
    pub search_query: Option<String>,
    /// Project ID filter.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Participant filter.
    #[serde(default)]
    pub participant: Option<String>,
    /// Maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

impl From<CalendarQueryParams> for CalendarFilter {
    fn from(params: CalendarQueryParams) -> Self {
        Self {
            query_type: params.query_type,
            start_date: params.start_date,
            end_date: params.end_date,
            event_types: params.event_types.unwrap_or_default(),
            project_id: params.project_id,
            participant: params.participant,
            search_query: params.search_query,
            limit: params.limit.unwrap_or(100),
            ..Default::default()
        }
    }
}

/// Response from calendar query operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarQueryResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Events matching the query.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub events: Option<Vec<CalendarEventInfo>>,
    /// Total count of matching events.
    pub count: usize,
    /// Status message.
    pub message: String,
}

/// Simplified calendar event info for responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEventInfo {
    /// Event ID.
    pub id: String,
    /// Event title.
    pub title: String,
    /// Event type.
    pub event_type: EventType,
    /// Start time.
    pub start: DateTime<Utc>,
    /// End time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<DateTime<Utc>>,
    /// Duration in minutes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_minutes: Option<i64>,
    /// Location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// Participants.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub participants: Vec<String>,
    /// Related project ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Is recurring.
    pub is_recurring: bool,
    /// Is extracted from documents.
    pub is_extracted: bool,
    /// Confidence score (for extracted events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl From<CalendarEvent> for CalendarEventInfo {
    fn from(event: CalendarEvent) -> Self {
        let is_recurring = event.recurrence.is_some();
        let is_extracted = event.source_document_id.is_some();
        let confidence = if is_extracted {
            Some(event.confidence)
        } else {
            None
        };
        let duration_minutes = event.duration().map(|d| d.num_minutes());

        Self {
            id: event.id,
            title: event.title,
            event_type: event.event_type,
            start: event.start,
            end: event.end,
            duration_minutes,
            location: event.location,
            participants: event.participants,
            project_id: event.project_id,
            is_recurring,
            is_extracted,
            confidence,
        }
    }
}

// ============================================================================
// Calendar Manage Tool Types
// ============================================================================

/// Action to perform on calendar events.
///
/// Available actions: `create`, `get`, `update`, `delete`, `list`, `stats`
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CalendarAction {
    /// Create a new calendar event. Requires `title`, `start`. Optional: `description`, `event_type`, `end`, `duration_minutes`, `all_day`, `location`, `participants`, `project_id`, `task_id`, `notes`, `recurrence_pattern`, `recurrence_interval`, `recurrence_until`, `recurrence_count`.
    Create,
    /// Get a calendar event by ID. Requires `event_id` parameter.
    Get,
    /// Update an existing calendar event. Requires `event_id`. Optional: any create parameters to modify.
    Update,
    /// Delete a calendar event. Requires `event_id` parameter.
    Delete,
    /// List calendar events with filters. Use with optional `filter` parameter containing query options.
    List,
    /// Get calendar statistics (event counts, busy time, etc.).
    Stats,
}

/// Parameters for the calendar_manage tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarManageParams {
    /// The action to perform.
    pub action: CalendarAction,
    /// Event ID (required for get, update, delete).
    #[serde(default)]
    pub event_id: Option<String>,
    /// Event title (required for create).
    #[serde(default)]
    pub title: Option<String>,
    /// Event description.
    #[serde(default)]
    pub description: Option<String>,
    /// Event type.
    #[serde(default)]
    pub event_type: Option<EventType>,
    /// Start time (ISO 8601 format, required for create).
    #[serde(default)]
    pub start: Option<DateTime<Utc>>,
    /// End time (ISO 8601 format).
    #[serde(default)]
    pub end: Option<DateTime<Utc>>,
    /// Duration in minutes (alternative to end time).
    #[serde(default)]
    pub duration_minutes: Option<i64>,
    /// Is all-day event.
    #[serde(default)]
    pub all_day: Option<bool>,
    /// Location.
    #[serde(default)]
    pub location: Option<String>,
    /// Participants.
    #[serde(default)]
    pub participants: Option<Vec<String>>,
    /// Related project ID.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Related task ID.
    #[serde(default)]
    pub task_id: Option<String>,
    /// Notes.
    #[serde(default)]
    pub notes: Option<String>,
    /// Recurrence pattern.
    #[serde(default)]
    pub recurrence_pattern: Option<String>, // daily, weekly, monthly, yearly
    /// Recurrence interval.
    #[serde(default)]
    pub recurrence_interval: Option<u32>,
    /// Recurrence end date.
    #[serde(default)]
    pub recurrence_until: Option<DateTime<Utc>>,
    /// Recurrence count.
    #[serde(default)]
    pub recurrence_count: Option<u32>,
    /// Filter options for list action.
    #[serde(default)]
    pub filter: Option<CalendarQueryParams>,
}

/// Response from calendar manage operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarManageResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// The event (for create, get, update).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<CalendarEventInfo>,
    /// List of events (for list action).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub events: Option<Vec<CalendarEventInfo>>,
    /// Calendar statistics (for stats action).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<CalendarStats>,
    /// Status message.
    pub message: String,
}

// ============================================================================
// Free Time Tool Types
// ============================================================================

/// Parameters for the calendar_free_time tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarFreeTimeParams {
    /// Start of search range (ISO 8601 format).
    #[serde(default)]
    pub start_date: Option<DateTime<Utc>>,
    /// End of search range (ISO 8601 format).
    #[serde(default)]
    pub end_date: Option<DateTime<Utc>>,
    /// Minimum slot duration in minutes.
    #[serde(default)]
    pub min_duration_minutes: Option<u32>,
    /// Working hours start (HH:MM format).
    #[serde(default)]
    pub working_hours_start: Option<String>,
    /// Working hours end (HH:MM format).
    #[serde(default)]
    pub working_hours_end: Option<String>,
    /// Exclude weekends.
    #[serde(default)]
    pub exclude_weekends: Option<bool>,
    /// Maximum number of slots to return.
    #[serde(default)]
    pub limit: Option<usize>,
}

impl From<CalendarFreeTimeParams> for FreeTimeParams {
    fn from(params: CalendarFreeTimeParams) -> Self {
        let now = Utc::now();
        Self {
            range_start: params.start_date.unwrap_or(now),
            range_end: params.end_date.unwrap_or(now + Duration::days(7)),
            min_duration_minutes: params.min_duration_minutes.unwrap_or(30),
            working_hours_start: params
                .working_hours_start
                .and_then(|s| parse_time(&s))
                .or(NaiveTime::from_hms_opt(9, 0, 0)),
            working_hours_end: params
                .working_hours_end
                .and_then(|s| parse_time(&s))
                .or(NaiveTime::from_hms_opt(17, 0, 0)),
            exclude_weekends: params.exclude_weekends.unwrap_or(true),
        }
    }
}

/// Parse a time string in HH:MM format.
fn parse_time(s: &str) -> Option<NaiveTime> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() >= 2 {
        let hour = parts[0].parse::<u32>().ok()?;
        let minute = parts[1].parse::<u32>().ok()?;
        NaiveTime::from_hms_opt(hour, minute, 0)
    } else {
        None
    }
}

/// Response from free time queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarFreeTimeResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Free time slots.
    pub slots: Vec<FreeTimeSlotInfo>,
    /// Total free time in minutes.
    pub total_free_minutes: i64,
    /// Number of slots found.
    pub count: usize,
    /// Status message.
    pub message: String,
}

/// Simplified free time slot info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeTimeSlotInfo {
    /// Start of free time.
    pub start: DateTime<Utc>,
    /// End of free time.
    pub end: DateTime<Utc>,
    /// Duration in minutes.
    pub duration_minutes: i64,
}

impl From<FreeTimeSlot> for FreeTimeSlotInfo {
    fn from(slot: FreeTimeSlot) -> Self {
        Self {
            start: slot.start,
            end: slot.end,
            duration_minutes: slot.duration_minutes,
        }
    }
}

// ============================================================================
// Conflict Detection Tool Types
// ============================================================================

/// Parameters for the calendar_conflicts tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarConflictsParams {
    /// Start of search range (ISO 8601 format).
    #[serde(default)]
    pub start_date: Option<DateTime<Utc>>,
    /// End of search range (ISO 8601 format).
    #[serde(default)]
    pub end_date: Option<DateTime<Utc>>,
    /// Check conflicts for a specific event ID.
    #[serde(default)]
    pub event_id: Option<String>,
    /// Minimum conflict severity to include.
    #[serde(default)]
    pub min_severity: Option<ConflictSeverity>,
}

/// Response from conflict detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarConflictsResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Scheduling conflicts found.
    pub conflicts: Vec<ConflictInfo>,
    /// Number of conflicts.
    pub count: usize,
    /// Events involved in conflicts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affected_events: Option<Vec<CalendarEventInfo>>,
    /// Status message.
    pub message: String,
}

/// Simplified conflict info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictInfo {
    /// First event ID.
    pub event1_id: String,
    /// Second event ID.
    pub event2_id: String,
    /// Start of overlap.
    pub overlap_start: DateTime<Utc>,
    /// End of overlap.
    pub overlap_end: DateTime<Utc>,
    /// Overlap duration in minutes.
    pub overlap_minutes: i64,
    /// Severity level.
    pub severity: ConflictSeverity,
}

impl From<SchedulingConflict> for ConflictInfo {
    fn from(conflict: SchedulingConflict) -> Self {
        Self {
            event1_id: conflict.event1_id,
            event2_id: conflict.event2_id,
            overlap_start: conflict.overlap_start,
            overlap_end: conflict.overlap_end,
            overlap_minutes: conflict.overlap_minutes,
            severity: conflict.severity,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a CalendarEvent from manage parameters.
pub fn create_event_from_params(params: &CalendarManageParams) -> Option<CalendarEvent> {
    let title = params.title.as_ref()?;
    let start = params.start?;

    let mut event = CalendarEvent::new(title, start);

    if let Some(ref desc) = params.description {
        event = event.with_description(desc);
    }

    if let Some(event_type) = params.event_type {
        event = event.with_type(event_type);
    }

    if let Some(end) = params.end {
        event = event.with_end(end);
    } else if let Some(minutes) = params.duration_minutes {
        event = event.with_duration(Duration::minutes(minutes));
    }

    if params.all_day == Some(true) {
        event = event.all_day_event();
    }

    if let Some(ref location) = params.location {
        event = event.with_location(location);
    }

    if let Some(ref participants) = params.participants {
        event = event.with_participants(participants.clone());
    }

    if let Some(ref project_id) = params.project_id {
        event = event.with_project(project_id);
    }

    if let Some(ref task_id) = params.task_id {
        event = event.with_task(task_id);
    }

    // Handle recurrence
    if let Some(ref pattern) = params.recurrence_pattern {
        let recurrence = create_recurrence_from_params(
            pattern,
            params.recurrence_interval,
            params.recurrence_until,
            params.recurrence_count,
        );
        if let Some(rec) = recurrence {
            event = event.with_recurrence(rec);
        }
    }

    Some(event)
}

/// Create a recurrence from pattern string.
fn create_recurrence_from_params(
    pattern: &str,
    interval: Option<u32>,
    until: Option<DateTime<Utc>>,
    count: Option<u32>,
) -> Option<EventRecurrence> {
    let mut recurrence = match pattern.to_lowercase().as_str() {
        "daily" => EventRecurrence::daily(),
        "weekly" => EventRecurrence::weekly(),
        "monthly" => EventRecurrence::monthly(),
        "yearly" => EventRecurrence::yearly(),
        _ => return None,
    };

    if let Some(i) = interval {
        recurrence = recurrence.every(i);
    }

    if let Some(u) = until {
        recurrence = recurrence.until(u);
    }

    if let Some(c) = count {
        recurrence = recurrence.times(c);
    }

    Some(recurrence)
}

/// Create an EventUpdate from manage parameters.
pub fn create_update_from_params(params: &CalendarManageParams) -> EventUpdate {
    EventUpdate {
        title: params.title.clone(),
        description: params.description.clone(),
        event_type: params.event_type,
        start: params.start,
        end: params.end,
        location: params.location.clone(),
        add_participants: params.participants.clone().unwrap_or_default(),
        remove_participants: Vec::new(),
        project_id: params.project_id.clone(),
        task_id: params.task_id.clone(),
        recurrence: params.recurrence_pattern.as_ref().and_then(|p| {
            create_recurrence_from_params(
                p,
                params.recurrence_interval,
                params.recurrence_until,
                params.recurrence_count,
            )
        }),
        clear_recurrence: false,
        notes: params.notes.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_event_from_params() {
        let params = CalendarManageParams {
            action: CalendarAction::Create,
            event_id: None,
            title: Some("Team Meeting".to_string()),
            description: Some("Weekly sync".to_string()),
            event_type: Some(EventType::Meeting),
            start: Some(Utc::now()),
            end: None,
            duration_minutes: Some(60),
            all_day: None,
            location: Some("Conference Room A".to_string()),
            participants: Some(vec!["John".to_string(), "Jane".to_string()]),
            project_id: None,
            task_id: None,
            notes: None,
            recurrence_pattern: None,
            recurrence_interval: None,
            recurrence_until: None,
            recurrence_count: None,
            filter: None,
        };

        let event = create_event_from_params(&params);
        assert!(event.is_some());

        let event = event.unwrap();
        assert_eq!(event.title, "Team Meeting");
        assert_eq!(event.event_type, EventType::Meeting);
        assert_eq!(event.participants.len(), 2);
    }

    #[test]
    fn test_create_recurring_event() {
        let params = CalendarManageParams {
            action: CalendarAction::Create,
            event_id: None,
            title: Some("Daily Standup".to_string()),
            description: None,
            event_type: Some(EventType::Standup),
            start: Some(Utc::now()),
            end: None,
            duration_minutes: Some(15),
            all_day: None,
            location: None,
            participants: None,
            project_id: None,
            task_id: None,
            notes: None,
            recurrence_pattern: Some("weekly".to_string()),
            recurrence_interval: Some(1),
            recurrence_until: None,
            recurrence_count: Some(10),
            filter: None,
        };

        let event = create_event_from_params(&params);
        assert!(event.is_some());

        let event = event.unwrap();
        assert!(event.recurrence.is_some());

        let recurrence = event.recurrence.unwrap();
        assert_eq!(recurrence.count, Some(10));
    }

    #[test]
    fn test_parse_time() {
        assert_eq!(parse_time("09:00"), NaiveTime::from_hms_opt(9, 0, 0));
        assert_eq!(parse_time("17:30"), NaiveTime::from_hms_opt(17, 30, 0));
        assert_eq!(parse_time("invalid"), None);
    }
}

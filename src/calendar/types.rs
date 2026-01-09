//! Calendar types for event management and temporal intelligence.
//!
//! This module defines the core types for calendar functionality,
//! including events, recurrence rules, and conflict detection.

use chrono::{DateTime, Duration, NaiveTime, Utc, Weekday};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ontology::extraction::{RecurrencePattern, RecurrenceRule};

// ============================================================================
// Calendar Event Types
// ============================================================================

/// A calendar event representing a time-specific commitment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarEvent {
    /// Unique identifier for the event.
    pub id: String,
    /// Event title.
    pub title: String,
    /// Event description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Type of event.
    pub event_type: EventType,
    /// Start time of the event.
    pub start: DateTime<Utc>,
    /// End time of the event (None for all-day or point-in-time events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<DateTime<Utc>>,
    /// Whether this is an all-day event.
    #[serde(default)]
    pub all_day: bool,
    /// Location of the event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// Participants (person entity IDs or names).
    #[serde(default)]
    pub participants: Vec<String>,
    /// Related project ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Related task ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Source document where this event was extracted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document_id: Option<String>,
    /// Original extracted text (for extracted events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extracted_text: Option<String>,
    /// Recurrence rule for recurring events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recurrence: Option<EventRecurrence>,
    /// Event notes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// Reminder settings.
    #[serde(default)]
    pub reminders: Vec<Reminder>,
    /// When the event was created.
    pub created_at: DateTime<Utc>,
    /// When the event was last updated.
    pub updated_at: DateTime<Utc>,
    /// Confidence score for extracted events (0.0-1.0).
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn default_confidence() -> f32 {
    1.0
}

impl CalendarEvent {
    /// Create a new calendar event.
    pub fn new(title: impl Into<String>, start: DateTime<Utc>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: title.into(),
            description: None,
            event_type: EventType::Event,
            start,
            end: None,
            all_day: false,
            location: None,
            participants: Vec::new(),
            project_id: None,
            task_id: None,
            source_document_id: None,
            extracted_text: None,
            recurrence: None,
            notes: None,
            reminders: Vec::new(),
            created_at: now,
            updated_at: now,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Create an event with a specific ID.
    pub fn with_id(id: impl Into<String>, title: impl Into<String>, start: DateTime<Utc>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            title: title.into(),
            description: None,
            event_type: EventType::Event,
            start,
            end: None,
            all_day: false,
            location: None,
            participants: Vec::new(),
            project_id: None,
            task_id: None,
            source_document_id: None,
            extracted_text: None,
            recurrence: None,
            notes: None,
            reminders: Vec::new(),
            created_at: now,
            updated_at: now,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the event type.
    pub fn with_type(mut self, event_type: EventType) -> Self {
        self.event_type = event_type;
        self
    }

    /// Set the end time.
    pub fn with_end(mut self, end: DateTime<Utc>) -> Self {
        self.end = Some(end);
        self
    }

    /// Set the duration (calculates end time).
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.end = Some(self.start + duration);
        self
    }

    /// Set as all-day event.
    pub fn all_day_event(mut self) -> Self {
        self.all_day = true;
        self
    }

    /// Set the location.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Add a participant.
    pub fn with_participant(mut self, participant: impl Into<String>) -> Self {
        self.participants.push(participant.into());
        self
    }

    /// Add multiple participants.
    pub fn with_participants(
        mut self,
        participants: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.participants
            .extend(participants.into_iter().map(|p| p.into()));
        self
    }

    /// Set the related project.
    pub fn with_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Set the related task.
    pub fn with_task(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    /// Set the source document.
    pub fn with_source(mut self, document_id: impl Into<String>) -> Self {
        self.source_document_id = Some(document_id.into());
        self
    }

    /// Set the extracted text.
    pub fn with_extracted_text(mut self, text: impl Into<String>) -> Self {
        self.extracted_text = Some(text.into());
        self
    }

    /// Set recurrence.
    pub fn with_recurrence(mut self, recurrence: EventRecurrence) -> Self {
        self.recurrence = Some(recurrence);
        self
    }

    /// Set the confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Get the duration of the event.
    pub fn duration(&self) -> Option<Duration> {
        self.end.map(|e| e - self.start)
    }

    /// Check if the event is happening at a specific time.
    pub fn is_at(&self, time: DateTime<Utc>) -> bool {
        if let Some(end) = self.end {
            time >= self.start && time < end
        } else {
            time == self.start
        }
    }

    /// Check if the event is within a date range.
    pub fn is_within(&self, range_start: DateTime<Utc>, range_end: DateTime<Utc>) -> bool {
        let event_end = self.end.unwrap_or(self.start);
        // Use >= for range_start to include events that start exactly at the range start
        self.start < range_end && event_end >= range_start
    }

    /// Check if this event overlaps with another.
    pub fn overlaps_with(&self, other: &CalendarEvent) -> bool {
        let self_end = self.end.unwrap_or(self.start);
        let other_end = other.end.unwrap_or(other.start);

        self.start < other_end && self_end > other.start
    }

    /// Check if the event is in the past.
    pub fn is_past(&self) -> bool {
        let end = self.end.unwrap_or(self.start);
        end < Utc::now()
    }

    /// Check if the event is upcoming (within the next N hours).
    pub fn is_upcoming(&self, hours: i64) -> bool {
        let now = Utc::now();
        let cutoff = now + Duration::hours(hours);
        self.start > now && self.start <= cutoff
    }
}

/// Type of calendar event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// A general event.
    #[default]
    Event,
    /// A meeting with participants.
    Meeting,
    /// A deadline for a task or project.
    Deadline,
    /// A reminder.
    Reminder,
    /// Blocked time (focus time, do not disturb).
    BlockedTime,
    /// A milestone or checkpoint.
    Milestone,
    /// An appointment.
    Appointment,
    /// Travel or commute time.
    Travel,
    /// A call or video conference.
    Call,
    /// A recurring stand-up or check-in.
    Standup,
}

impl EventType {
    /// Get a human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            EventType::Event => "Event",
            EventType::Meeting => "Meeting",
            EventType::Deadline => "Deadline",
            EventType::Reminder => "Reminder",
            EventType::BlockedTime => "Blocked Time",
            EventType::Milestone => "Milestone",
            EventType::Appointment => "Appointment",
            EventType::Travel => "Travel",
            EventType::Call => "Call",
            EventType::Standup => "Stand-up",
        }
    }
}

// ============================================================================
// Recurrence Types
// ============================================================================

/// Recurrence configuration for repeating events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EventRecurrence {
    /// The recurrence pattern.
    pub pattern: RecurrencePattern,
    /// Interval (e.g., every 2 weeks).
    #[serde(default = "default_interval")]
    pub interval: u32,
    /// Days of week for weekly patterns.
    #[serde(default)]
    pub days_of_week: Vec<u8>, // 0=Mon, 1=Tue, ..., 6=Sun
    /// Day of month for monthly patterns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub day_of_month: Option<u32>,
    /// Week of month for monthly patterns (e.g., first Monday).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub week_of_month: Option<u8>,
    /// End date for the recurrence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub until: Option<DateTime<Utc>>,
    /// Maximum number of occurrences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    /// Exceptions (dates to skip).
    #[serde(default)]
    pub exceptions: Vec<DateTime<Utc>>,
}

fn default_interval() -> u32 {
    1
}

impl EventRecurrence {
    /// Create a daily recurrence.
    pub fn daily() -> Self {
        Self {
            pattern: RecurrencePattern::Daily,
            interval: 1,
            days_of_week: Vec::new(),
            day_of_month: None,
            week_of_month: None,
            until: None,
            count: None,
            exceptions: Vec::new(),
        }
    }

    /// Create a weekly recurrence.
    pub fn weekly() -> Self {
        Self {
            pattern: RecurrencePattern::Weekly,
            interval: 1,
            days_of_week: Vec::new(),
            day_of_month: None,
            week_of_month: None,
            until: None,
            count: None,
            exceptions: Vec::new(),
        }
    }

    /// Create a weekly recurrence on specific days.
    pub fn weekly_on(days: impl IntoIterator<Item = Weekday>) -> Self {
        Self {
            pattern: RecurrencePattern::Weekly,
            interval: 1,
            days_of_week: days
                .into_iter()
                .map(|d| d.num_days_from_monday() as u8)
                .collect(),
            day_of_month: None,
            week_of_month: None,
            until: None,
            count: None,
            exceptions: Vec::new(),
        }
    }

    /// Create a monthly recurrence.
    pub fn monthly() -> Self {
        Self {
            pattern: RecurrencePattern::Monthly,
            interval: 1,
            days_of_week: Vec::new(),
            day_of_month: None,
            week_of_month: None,
            until: None,
            count: None,
            exceptions: Vec::new(),
        }
    }

    /// Create a yearly recurrence.
    pub fn yearly() -> Self {
        Self {
            pattern: RecurrencePattern::Yearly,
            interval: 1,
            days_of_week: Vec::new(),
            day_of_month: None,
            week_of_month: None,
            until: None,
            count: None,
            exceptions: Vec::new(),
        }
    }

    /// Set the interval.
    pub fn every(mut self, interval: u32) -> Self {
        self.interval = interval;
        self
    }

    /// Set the end date.
    pub fn until(mut self, date: DateTime<Utc>) -> Self {
        self.until = Some(date);
        self
    }

    /// Set the occurrence count.
    pub fn times(mut self, count: u32) -> Self {
        self.count = Some(count);
        self
    }

    /// Create from a RecurrenceRule.
    pub fn from_rule(rule: &RecurrenceRule) -> Self {
        Self {
            pattern: rule.pattern,
            interval: rule.interval,
            days_of_week: rule
                .days_of_week
                .iter()
                .map(|d| d.num_days_from_monday() as u8)
                .collect(),
            day_of_month: rule.day_of_month,
            week_of_month: None,
            until: rule
                .until
                .map(|d| DateTime::from_naive_utc_and_offset(d.and_hms_opt(0, 0, 0).unwrap(), Utc)),
            count: rule.count,
            exceptions: Vec::new(),
        }
    }
}

/// Reminder settings for an event.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Reminder {
    /// Minutes before event to trigger reminder.
    pub minutes_before: u32,
    /// Type of reminder.
    pub reminder_type: ReminderType,
}

/// Type of reminder notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReminderType {
    /// A notification/alert.
    #[default]
    Notification,
    /// An email reminder.
    Email,
    /// An SMS reminder.
    Sms,
}

// ============================================================================
// Calendar Query Types
// ============================================================================

/// Type of calendar query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CalendarQueryType {
    /// Get upcoming events.
    #[default]
    Upcoming,
    /// Get today's events.
    Today,
    /// Get this week's events.
    ThisWeek,
    /// Get next week's events.
    NextWeek,
    /// Get events in a date range.
    DateRange,
    /// Find free time slots.
    FreeTime,
    /// Find scheduling conflicts.
    Conflicts,
    /// Get all commitments (deadlines, meetings).
    Commitments,
    /// Get events by type.
    ByType,
    /// Search events by text.
    Search,
}

/// Filter criteria for calendar queries.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalendarFilter {
    /// Query type.
    #[serde(default)]
    pub query_type: CalendarQueryType,
    /// Start of date range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_date: Option<DateTime<Utc>>,
    /// End of date range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_date: Option<DateTime<Utc>>,
    /// Filter by event types.
    #[serde(default)]
    pub event_types: Vec<EventType>,
    /// Filter by project ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Filter by participant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub participant: Option<String>,
    /// Include recurring event instances.
    #[serde(default = "default_true")]
    pub include_recurring: bool,
    /// Text search query.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_query: Option<String>,
    /// Minimum confidence for extracted events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_confidence: Option<f32>,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

fn default_true() -> bool {
    true
}

fn default_limit() -> usize {
    100
}

impl Default for CalendarFilter {
    fn default() -> Self {
        Self {
            query_type: CalendarQueryType::default(),
            start_date: None,
            end_date: None,
            event_types: Vec::new(),
            project_id: None,
            participant: None,
            include_recurring: true,
            search_query: None,
            min_confidence: None,
            limit: 100,
            offset: 0,
        }
    }
}

impl CalendarFilter {
    /// Create a filter for today's events.
    pub fn today() -> Self {
        Self {
            query_type: CalendarQueryType::Today,
            ..Default::default()
        }
    }

    /// Create a filter for upcoming events.
    pub fn upcoming() -> Self {
        Self {
            query_type: CalendarQueryType::Upcoming,
            ..Default::default()
        }
    }

    /// Create a filter for this week's events.
    pub fn this_week() -> Self {
        Self {
            query_type: CalendarQueryType::ThisWeek,
            ..Default::default()
        }
    }

    /// Create a filter for a date range.
    pub fn date_range(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self {
            query_type: CalendarQueryType::DateRange,
            start_date: Some(start),
            end_date: Some(end),
            ..Default::default()
        }
    }

    /// Create a filter for finding conflicts.
    pub fn conflicts() -> Self {
        Self {
            query_type: CalendarQueryType::Conflicts,
            ..Default::default()
        }
    }

    /// Add an event type filter.
    pub fn with_type(mut self, event_type: EventType) -> Self {
        self.event_types.push(event_type);
        self
    }

    /// Filter by project.
    pub fn for_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Set the limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Check if an event matches this filter.
    pub fn matches(&self, event: &CalendarEvent) -> bool {
        // Check event types
        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }

        // Check project
        if let Some(ref proj_id) = self.project_id {
            if event.project_id.as_ref() != Some(proj_id) {
                return false;
            }
        }

        // Check participant
        if let Some(ref participant) = self.participant {
            if !event.participants.contains(participant) {
                return false;
            }
        }

        // Check confidence
        if let Some(min_conf) = self.min_confidence {
            if event.confidence < min_conf {
                return false;
            }
        }

        // Check text search
        if let Some(ref query) = self.search_query {
            let query_lower = query.to_lowercase();
            let matches_title = event.title.to_lowercase().contains(&query_lower);
            let matches_desc = event
                .description
                .as_ref()
                .is_some_and(|d| d.to_lowercase().contains(&query_lower));
            let matches_notes = event
                .notes
                .as_ref()
                .is_some_and(|n| n.to_lowercase().contains(&query_lower));

            if !matches_title && !matches_desc && !matches_notes {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Conflict Detection Types
// ============================================================================

/// A scheduling conflict between two events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SchedulingConflict {
    /// First event in the conflict.
    pub event1_id: String,
    /// Second event in the conflict.
    pub event2_id: String,
    /// Start of the overlapping period.
    pub overlap_start: DateTime<Utc>,
    /// End of the overlapping period.
    pub overlap_end: DateTime<Utc>,
    /// Duration of the overlap.
    pub overlap_minutes: i64,
    /// Severity of the conflict.
    pub severity: ConflictSeverity,
}

impl SchedulingConflict {
    /// Detect conflict between two events.
    pub fn detect(event1: &CalendarEvent, event2: &CalendarEvent) -> Option<Self> {
        if !event1.overlaps_with(event2) {
            return None;
        }

        let event1_end = event1.end.unwrap_or(event1.start);
        let event2_end = event2.end.unwrap_or(event2.start);

        let overlap_start = event1.start.max(event2.start);
        let overlap_end = event1_end.min(event2_end);
        let overlap_duration = overlap_end - overlap_start;
        let overlap_minutes = overlap_duration.num_minutes();

        // Determine severity based on overlap and event types
        let severity = if overlap_minutes > 60 {
            ConflictSeverity::High
        } else if overlap_minutes > 30 {
            ConflictSeverity::Medium
        } else {
            ConflictSeverity::Low
        };

        Some(Self {
            event1_id: event1.id.clone(),
            event2_id: event2.id.clone(),
            overlap_start,
            overlap_end,
            overlap_minutes,
            severity,
        })
    }
}

/// Severity of a scheduling conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConflictSeverity {
    /// Minor overlap (< 30 minutes).
    #[default]
    Low,
    /// Moderate overlap (30-60 minutes).
    Medium,
    /// Significant overlap (> 60 minutes).
    High,
}

// ============================================================================
// Free Time Types
// ============================================================================

/// A free time slot in the calendar.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FreeTimeSlot {
    /// Start of the free time.
    pub start: DateTime<Utc>,
    /// End of the free time.
    pub end: DateTime<Utc>,
    /// Duration in minutes.
    pub duration_minutes: i64,
}

impl FreeTimeSlot {
    /// Create a new free time slot.
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        let duration = end - start;
        Self {
            start,
            end,
            duration_minutes: duration.num_minutes(),
        }
    }
}

/// Parameters for finding free time.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FreeTimeParams {
    /// Start of the search range.
    pub range_start: DateTime<Utc>,
    /// End of the search range.
    pub range_end: DateTime<Utc>,
    /// Minimum duration in minutes.
    #[serde(default = "default_min_duration")]
    pub min_duration_minutes: u32,
    /// Working hours start (e.g., 09:00).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_hours_start: Option<NaiveTime>,
    /// Working hours end (e.g., 17:00).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_hours_end: Option<NaiveTime>,
    /// Exclude weekends.
    #[serde(default)]
    pub exclude_weekends: bool,
}

fn default_min_duration() -> u32 {
    30
}

impl Default for FreeTimeParams {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            range_start: now,
            range_end: now + Duration::days(7),
            min_duration_minutes: 30,
            working_hours_start: NaiveTime::from_hms_opt(9, 0, 0),
            working_hours_end: NaiveTime::from_hms_opt(17, 0, 0),
            exclude_weekends: true,
        }
    }
}

// ============================================================================
// Calendar Statistics
// ============================================================================

/// Statistics about calendar events.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CalendarStats {
    /// Total number of events.
    pub total_events: usize,
    /// Events by type.
    pub events_by_type: HashMap<String, usize>,
    /// Upcoming events count.
    pub upcoming_events: usize,
    /// Events today.
    pub events_today: usize,
    /// Events this week.
    pub events_this_week: usize,
    /// Recurring events count.
    pub recurring_events: usize,
    /// Extracted events count (from documents).
    pub extracted_events: usize,
    /// Average events per day (for the past 30 days).
    pub avg_events_per_day: f32,
    /// Total scheduled time in hours (this week).
    pub total_scheduled_hours: f32,
    /// Number of conflicts detected.
    pub conflicts: usize,
}

// ============================================================================
// Event Update Types
// ============================================================================

/// Update operations for a calendar event.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct EventUpdate {
    /// New title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// New description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// New event type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_type: Option<EventType>,
    /// New start time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<DateTime<Utc>>,
    /// New end time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<DateTime<Utc>>,
    /// New location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// Participants to add.
    #[serde(default)]
    pub add_participants: Vec<String>,
    /// Participants to remove.
    #[serde(default)]
    pub remove_participants: Vec<String>,
    /// New project ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// New task ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// New recurrence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recurrence: Option<EventRecurrence>,
    /// Clear recurrence.
    #[serde(default)]
    pub clear_recurrence: bool,
    /// New notes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl EventUpdate {
    /// Apply this update to an event.
    pub fn apply_to(&self, event: &mut CalendarEvent) {
        if let Some(ref title) = self.title {
            event.title = title.clone();
        }
        if let Some(ref description) = self.description {
            event.description = Some(description.clone());
        }
        if let Some(event_type) = self.event_type {
            event.event_type = event_type;
        }
        if let Some(start) = self.start {
            event.start = start;
        }
        if let Some(end) = self.end {
            event.end = Some(end);
        }
        if let Some(ref location) = self.location {
            event.location = Some(location.clone());
        }
        for participant in &self.add_participants {
            if !event.participants.contains(participant) {
                event.participants.push(participant.clone());
            }
        }
        for participant in &self.remove_participants {
            event.participants.retain(|p| p != participant);
        }
        if let Some(ref project_id) = self.project_id {
            event.project_id = Some(project_id.clone());
        }
        if let Some(ref task_id) = self.task_id {
            event.task_id = Some(task_id.clone());
        }
        if let Some(ref recurrence) = self.recurrence {
            event.recurrence = Some(recurrence.clone());
        }
        if self.clear_recurrence {
            event.recurrence = None;
        }
        if let Some(ref notes) = self.notes {
            event.notes = Some(notes.clone());
        }
        event.updated_at = Utc::now();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let start = Utc::now();
        let event = CalendarEvent::new("Team Meeting", start)
            .with_description("Weekly sync")
            .with_type(EventType::Meeting)
            .with_duration(Duration::hours(1))
            .with_location("Conference Room A")
            .with_participant("John")
            .with_participant("Jane");

        assert_eq!(event.title, "Team Meeting");
        assert_eq!(event.event_type, EventType::Meeting);
        assert_eq!(event.participants.len(), 2);
        assert!(event.duration().is_some());
        assert_eq!(event.duration().unwrap().num_hours(), 1);
    }

    #[test]
    fn test_event_overlap_detection() {
        let now = Utc::now();

        let event1 = CalendarEvent::new("Meeting 1", now).with_duration(Duration::hours(1));

        let event2 = CalendarEvent::new("Meeting 2", now + Duration::minutes(30))
            .with_duration(Duration::hours(1));

        let event3 = CalendarEvent::new("Meeting 3", now + Duration::hours(2))
            .with_duration(Duration::hours(1));

        assert!(event1.overlaps_with(&event2));
        assert!(event2.overlaps_with(&event1));
        assert!(!event1.overlaps_with(&event3));
    }

    #[test]
    fn test_conflict_detection() {
        let now = Utc::now();

        let event1 = CalendarEvent::new("Meeting 1", now).with_duration(Duration::hours(1));

        let event2 = CalendarEvent::new("Meeting 2", now + Duration::minutes(30))
            .with_duration(Duration::hours(1));

        let conflict = SchedulingConflict::detect(&event1, &event2);
        assert!(conflict.is_some());

        let conflict = conflict.unwrap();
        assert_eq!(conflict.overlap_minutes, 30);
    }

    #[test]
    fn test_recurrence() {
        let recurrence = EventRecurrence::weekly_on([Weekday::Mon, Weekday::Wed, Weekday::Fri])
            .every(1)
            .times(10);

        assert_eq!(recurrence.pattern, RecurrencePattern::Weekly);
        assert_eq!(recurrence.days_of_week.len(), 3);
        assert_eq!(recurrence.count, Some(10));
    }

    #[test]
    fn test_calendar_filter() {
        let now = Utc::now();

        let meeting = CalendarEvent::new("Team Meeting", now)
            .with_type(EventType::Meeting)
            .with_participant("John");

        let deadline = CalendarEvent::new("Project Deadline", now).with_type(EventType::Deadline);

        let filter = CalendarFilter::default().with_type(EventType::Meeting);
        assert!(filter.matches(&meeting));
        assert!(!filter.matches(&deadline));

        let participant_filter = CalendarFilter {
            participant: Some("John".to_string()),
            ..Default::default()
        };
        assert!(participant_filter.matches(&meeting));
        assert!(!participant_filter.matches(&deadline));
    }

    #[test]
    fn test_event_update() {
        let now = Utc::now();
        let mut event = CalendarEvent::new("Original Title", now);

        let update = EventUpdate {
            title: Some("Updated Title".to_string()),
            description: Some("New description".to_string()),
            add_participants: vec!["Alice".to_string(), "Bob".to_string()],
            ..Default::default()
        };

        update.apply_to(&mut event);

        assert_eq!(event.title, "Updated Title");
        assert_eq!(event.description, Some("New description".to_string()));
        assert_eq!(event.participants.len(), 2);
    }

    #[test]
    fn test_event_within_range() {
        let now = Utc::now();
        let event = CalendarEvent::new("Event", now).with_duration(Duration::hours(1));

        let range_start = now - Duration::hours(1);
        let range_end = now + Duration::hours(2);

        assert!(event.is_within(range_start, range_end));

        let past_range_start = now - Duration::hours(5);
        let past_range_end = now - Duration::hours(4);

        assert!(!event.is_within(past_range_start, past_range_end));
    }
}

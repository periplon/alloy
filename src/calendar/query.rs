//! Calendar query engine for natural language and structured queries.
//!
//! This module provides query capabilities for calendar events,
//! supporting both structured filters and natural language queries.

use chrono::{DateTime, Duration, Utc};

use crate::error::Result;
use crate::ontology::OntologyStore;

use super::events::CalendarManager;
use super::types::{
    CalendarEvent, CalendarFilter, CalendarQueryType, CalendarStats, EventType, FreeTimeParams,
    FreeTimeSlot, SchedulingConflict,
};

use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// Calendar Query Engine
// ============================================================================

/// Query engine for calendar operations, wrapping the CalendarManager.
pub struct CalendarQueryEngine<S: OntologyStore> {
    /// The underlying calendar manager.
    pub manager: CalendarManager<S>,
}

impl<S: OntologyStore> CalendarQueryEngine<S> {
    /// Create a new query engine with the given store.
    pub fn new(store: Arc<RwLock<S>>) -> Self {
        Self {
            manager: CalendarManager::new(store),
        }
    }

    /// Get the underlying manager.
    pub fn manager(&self) -> &CalendarManager<S> {
        &self.manager
    }

    // ========================================================================
    // High-Level Query Methods
    // ========================================================================

    /// Execute a calendar query based on query type.
    pub async fn query(&self, params: CalendarQueryParams) -> Result<CalendarQueryResponse> {
        match params.query_type {
            CalendarQueryType::Today => self.query_today().await,
            CalendarQueryType::ThisWeek => self.query_this_week().await,
            CalendarQueryType::NextWeek => self.query_next_week().await,
            CalendarQueryType::Upcoming => self.query_upcoming(params.days).await,
            CalendarQueryType::DateRange => {
                self.query_date_range(params.start_date, params.end_date)
                    .await
            }
            CalendarQueryType::FreeTime => self.query_free_time(params.into()).await,
            CalendarQueryType::Conflicts => {
                self.query_conflicts(params.start_date, params.end_date)
                    .await
            }
            CalendarQueryType::Commitments => {
                self.query_commitments(params.start_date, params.end_date)
                    .await
            }
            CalendarQueryType::ByType => {
                self.query_by_type(params.event_types, params.start_date, params.end_date)
                    .await
            }
            CalendarQueryType::Search => {
                self.query_search(&params.search_query.unwrap_or_default())
                    .await
            }
        }
    }

    /// Get today's events.
    pub async fn query_today(&self) -> Result<CalendarQueryResponse> {
        let events = self.manager.today().await?;
        Ok(CalendarQueryResponse::events(events))
    }

    /// Get this week's events.
    pub async fn query_this_week(&self) -> Result<CalendarQueryResponse> {
        let events = self.manager.this_week().await?;
        Ok(CalendarQueryResponse::events(events))
    }

    /// Get next week's events.
    pub async fn query_next_week(&self) -> Result<CalendarQueryResponse> {
        let filter = CalendarFilter {
            query_type: CalendarQueryType::NextWeek,
            ..Default::default()
        };
        let events = self.manager.list(&filter).await?;
        Ok(CalendarQueryResponse::events(events))
    }

    /// Get upcoming events.
    pub async fn query_upcoming(&self, days: Option<i64>) -> Result<CalendarQueryResponse> {
        let events = self.manager.upcoming(days).await?;
        Ok(CalendarQueryResponse::events(events))
    }

    /// Query events in a date range.
    pub async fn query_date_range(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<CalendarQueryResponse> {
        let now = Utc::now();
        let start = start.unwrap_or(now);
        let end = end.unwrap_or(now + Duration::days(30));

        let filter = CalendarFilter::date_range(start, end);
        let events = self.manager.list(&filter).await?;
        Ok(CalendarQueryResponse::events(events))
    }

    /// Find free time slots.
    pub async fn query_free_time(&self, params: FreeTimeParams) -> Result<CalendarQueryResponse> {
        let slots = self.manager.find_free_time(&params).await?;
        Ok(CalendarQueryResponse::free_time(slots))
    }

    /// Find scheduling conflicts.
    pub async fn query_conflicts(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<CalendarQueryResponse> {
        let now = Utc::now();
        let start = start.unwrap_or(now);
        let end = end.unwrap_or(now + Duration::days(7));

        let filter = CalendarFilter::date_range(start, end);
        let conflicts = self.manager.find_all_conflicts(&filter).await?;

        // Also get the events involved in conflicts
        let events = self
            .manager
            .list(&CalendarFilter {
                query_type: CalendarQueryType::Conflicts,
                start_date: Some(start),
                end_date: Some(end),
                ..Default::default()
            })
            .await?;

        Ok(CalendarQueryResponse::conflicts(conflicts, events))
    }

    /// Get commitments (meetings, deadlines, appointments).
    pub async fn query_commitments(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<CalendarQueryResponse> {
        let now = Utc::now();
        let start = start.unwrap_or(now);
        let end = end.unwrap_or(now + Duration::days(7));

        let filter = CalendarFilter {
            query_type: CalendarQueryType::DateRange,
            start_date: Some(start),
            end_date: Some(end),
            event_types: vec![
                EventType::Meeting,
                EventType::Deadline,
                EventType::Appointment,
                EventType::Call,
                EventType::Milestone,
            ],
            ..Default::default()
        };

        let events = self.manager.list(&filter).await?;
        Ok(CalendarQueryResponse::events(events))
    }

    /// Query events by type.
    pub async fn query_by_type(
        &self,
        event_types: Vec<EventType>,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<CalendarQueryResponse> {
        let now = Utc::now();
        let filter = CalendarFilter {
            query_type: CalendarQueryType::DateRange,
            start_date: Some(start.unwrap_or(now - Duration::days(30))),
            end_date: Some(end.unwrap_or(now + Duration::days(30))),
            event_types,
            ..Default::default()
        };

        let events = self.manager.list(&filter).await?;
        Ok(CalendarQueryResponse::events(events))
    }

    /// Search events by text.
    pub async fn query_search(&self, query: &str) -> Result<CalendarQueryResponse> {
        let events = self.manager.search(query).await?;
        Ok(CalendarQueryResponse::events(events))
    }

    /// Get calendar statistics.
    pub async fn stats(&self) -> Result<CalendarStats> {
        self.manager.stats().await
    }

    // ========================================================================
    // Convenience Methods
    // ========================================================================

    /// Get the next N events.
    pub async fn next_events(&self, count: usize) -> Result<Vec<CalendarEvent>> {
        let filter = CalendarFilter {
            query_type: CalendarQueryType::Upcoming,
            limit: count,
            ..Default::default()
        };
        self.manager.list(&filter).await
    }

    /// Check if a time slot is free.
    pub async fn is_time_free(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<bool> {
        let filter = CalendarFilter::date_range(start, end);
        let events = self.manager.list(&filter).await?;
        Ok(events.is_empty())
    }

    /// Get events for a specific project.
    pub async fn events_for_project(&self, project_id: &str) -> Result<Vec<CalendarEvent>> {
        let filter = CalendarFilter::default().for_project(project_id);
        self.manager.list(&filter).await
    }

    /// Get events with a specific participant.
    pub async fn events_with_participant(&self, participant: &str) -> Result<Vec<CalendarEvent>> {
        let filter = CalendarFilter {
            participant: Some(participant.to_string()),
            ..Default::default()
        };
        self.manager.list(&filter).await
    }

    /// Get overdue deadlines.
    pub async fn overdue_deadlines(&self) -> Result<Vec<CalendarEvent>> {
        let now = Utc::now();
        let filter = CalendarFilter {
            query_type: CalendarQueryType::DateRange,
            start_date: Some(now - Duration::days(365)),
            end_date: Some(now),
            event_types: vec![EventType::Deadline],
            ..Default::default()
        };

        let events = self.manager.list(&filter).await?;
        Ok(events.into_iter().filter(|e| e.is_past()).collect())
    }

    /// Get upcoming deadlines.
    pub async fn upcoming_deadlines(&self, days: Option<i64>) -> Result<Vec<CalendarEvent>> {
        let now = Utc::now();
        let end = now + Duration::days(days.unwrap_or(7));

        let filter = CalendarFilter {
            query_type: CalendarQueryType::DateRange,
            start_date: Some(now),
            end_date: Some(end),
            event_types: vec![EventType::Deadline],
            ..Default::default()
        };

        self.manager.list(&filter).await
    }
}

// ============================================================================
// Query Types
// ============================================================================

/// Parameters for calendar queries.
#[derive(Debug, Clone, Default)]
pub struct CalendarQueryParams {
    /// Type of query to execute.
    pub query_type: CalendarQueryType,
    /// Start date for date range queries.
    pub start_date: Option<DateTime<Utc>>,
    /// End date for date range queries.
    pub end_date: Option<DateTime<Utc>>,
    /// Number of days for upcoming queries.
    pub days: Option<i64>,
    /// Event types to filter.
    pub event_types: Vec<EventType>,
    /// Search query text.
    pub search_query: Option<String>,
    /// Project ID filter.
    pub project_id: Option<String>,
    /// Participant filter.
    pub participant: Option<String>,
    /// Free time parameters.
    pub min_duration_minutes: Option<u32>,
    /// Maximum results.
    pub limit: Option<usize>,
}

impl From<CalendarQueryParams> for FreeTimeParams {
    fn from(params: CalendarQueryParams) -> Self {
        let now = Utc::now();
        FreeTimeParams {
            range_start: params.start_date.unwrap_or(now),
            range_end: params.end_date.unwrap_or(now + Duration::days(7)),
            min_duration_minutes: params.min_duration_minutes.unwrap_or(30),
            ..Default::default()
        }
    }
}

/// Response from a calendar query.
#[derive(Debug, Clone)]
pub struct CalendarQueryResponse {
    /// Events matching the query.
    pub events: Vec<CalendarEvent>,
    /// Free time slots (for free time queries).
    pub free_slots: Vec<FreeTimeSlot>,
    /// Scheduling conflicts (for conflict queries).
    pub conflicts: Vec<SchedulingConflict>,
    /// Summary information.
    pub summary: QuerySummary,
}

impl CalendarQueryResponse {
    /// Create a response with events.
    pub fn events(events: Vec<CalendarEvent>) -> Self {
        let count = events.len();
        Self {
            events,
            free_slots: Vec::new(),
            conflicts: Vec::new(),
            summary: QuerySummary {
                total_results: count,
                ..Default::default()
            },
        }
    }

    /// Create a response with free time slots.
    pub fn free_time(slots: Vec<FreeTimeSlot>) -> Self {
        let count = slots.len();
        let total_minutes: i64 = slots.iter().map(|s| s.duration_minutes).sum();
        Self {
            events: Vec::new(),
            free_slots: slots,
            conflicts: Vec::new(),
            summary: QuerySummary {
                total_results: count,
                total_free_minutes: Some(total_minutes),
                ..Default::default()
            },
        }
    }

    /// Create a response with conflicts.
    pub fn conflicts(conflicts: Vec<SchedulingConflict>, events: Vec<CalendarEvent>) -> Self {
        let conflict_count = conflicts.len();
        Self {
            events,
            free_slots: Vec::new(),
            conflicts,
            summary: QuerySummary {
                total_results: conflict_count,
                ..Default::default()
            },
        }
    }
}

/// Summary of query results.
#[derive(Debug, Clone, Default)]
pub struct QuerySummary {
    /// Total number of results.
    pub total_results: usize,
    /// Total free minutes (for free time queries).
    pub total_free_minutes: Option<i64>,
    /// Whether there are more results.
    pub has_more: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::EmbeddedOntologyStore;

    fn create_test_engine() -> CalendarQueryEngine<EmbeddedOntologyStore> {
        let store = EmbeddedOntologyStore::new();
        CalendarQueryEngine::new(Arc::new(RwLock::new(store)))
    }

    #[tokio::test]
    async fn test_query_today() {
        let engine = create_test_engine();

        // Create an event for today
        let now = Utc::now();
        let event = CalendarEvent::new("Today's Meeting", now);
        engine.manager.create(event).await.unwrap();

        let response = engine.query_today().await.unwrap();
        assert_eq!(response.events.len(), 1);
    }

    #[tokio::test]
    async fn test_query_upcoming() {
        let engine = create_test_engine();

        // Create events - start 1 hour from now to avoid timing issues
        let now = Utc::now();
        for i in 0..5 {
            let event = CalendarEvent::new(
                format!("Event {}", i),
                now + Duration::hours(1) + Duration::days(i),
            );
            engine.manager.create(event).await.unwrap();
        }

        let response = engine.query_upcoming(Some(7)).await.unwrap();
        assert_eq!(response.events.len(), 5);
    }

    #[tokio::test]
    async fn test_next_events() {
        let engine = create_test_engine();

        // Create events
        let now = Utc::now();
        for i in 0..10 {
            let event = CalendarEvent::new(format!("Event {}", i), now + Duration::hours(i));
            engine.manager.create(event).await.unwrap();
        }

        let events = engine.next_events(5).await.unwrap();
        assert_eq!(events.len(), 5);
    }

    #[tokio::test]
    async fn test_is_time_free() {
        let engine = create_test_engine();

        let now = Utc::now();
        let start = now + Duration::hours(1);
        let end = start + Duration::hours(1);

        // Should be free initially
        assert!(engine.is_time_free(start, end).await.unwrap());

        // Add an event
        let event = CalendarEvent::new("Meeting", start).with_duration(Duration::hours(1));
        engine.manager.create(event).await.unwrap();

        // Should not be free now
        assert!(!engine.is_time_free(start, end).await.unwrap());
    }
}

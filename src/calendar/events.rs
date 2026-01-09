//! Calendar event storage and management.
//!
//! This module provides the CalendarManager for storing and querying calendar events,
//! with support for conflict detection, free time finding, and recurring events.

use std::sync::Arc;

use chrono::{DateTime, Datelike, Duration, NaiveTime, Timelike, Utc, Weekday};
use tokio::sync::RwLock;
use tracing::debug;

use crate::error::Result;
use crate::ontology::{Entity, EntityFilter, EntityType, OntologyStore};

use super::types::{
    CalendarEvent, CalendarFilter, CalendarQueryType, CalendarStats,
    EventRecurrence, EventType, EventUpdate, FreeTimeParams, FreeTimeSlot, SchedulingConflict,
};

// ============================================================================
// Calendar Manager
// ============================================================================

/// Manager for calendar events, providing storage and query operations.
pub struct CalendarManager<S: OntologyStore> {
    /// The underlying ontology store.
    pub(crate) store: Arc<RwLock<S>>,
}

impl<S: OntologyStore> CalendarManager<S> {
    /// Create a new CalendarManager with the given store.
    pub fn new(store: Arc<RwLock<S>>) -> Self {
        Self { store }
    }

    // ========================================================================
    // CRUD Operations
    // ========================================================================

    /// Create a new calendar event.
    pub async fn create(&self, event: CalendarEvent) -> Result<CalendarEvent> {
        let entity = self.event_to_entity(&event);
        let store = self.store.read().await;
        store.create_entity(entity).await?;
        debug!("Created calendar event: {} ({})", event.title, event.id);
        Ok(event)
    }

    /// Get an event by ID.
    pub async fn get(&self, id: &str) -> Result<Option<CalendarEvent>> {
        let store = self.store.read().await;
        if let Some(entity) = store.get_entity(id).await? {
            if entity.entity_type == EntityType::CalendarEvent {
                return Ok(Some(self.entity_to_event(&entity)?));
            }
        }
        Ok(None)
    }

    /// Update an existing event.
    pub async fn update(&self, id: &str, update: EventUpdate) -> Result<Option<CalendarEvent>> {
        if let Some(mut event) = self.get(id).await? {
            update.apply_to(&mut event);
            let entity = self.event_to_entity(&event);
            let store = self.store.read().await;
            // Delete old and create new since we don't have a direct update with entity
            store.delete_entity(id).await?;
            store.create_entity(entity).await?;
            debug!("Updated calendar event: {} ({})", event.title, event.id);
            Ok(Some(event))
        } else {
            Ok(None)
        }
    }

    /// Delete an event by ID.
    pub async fn delete(&self, id: &str) -> Result<bool> {
        let store = self.store.read().await;
        store.delete_entity(id).await?;
        debug!("Deleted calendar event: {}", id);
        Ok(true)
    }

    /// List events matching a filter.
    pub async fn list(&self, filter: &CalendarFilter) -> Result<Vec<CalendarEvent>> {
        let store = self.store.read().await;

        // Get date range based on query type
        let (range_start, range_end) = self.get_date_range(filter);

        // Build entity filter
        let entity_filter = EntityFilter::by_types([EntityType::CalendarEvent])
            .with_limit(filter.limit * 2); // Fetch more to account for filtering

        let entities = store.list_entities(entity_filter).await?;

        let mut events: Vec<CalendarEvent> = entities
            .into_iter()
            .filter_map(|e| self.entity_to_event(&e).ok())
            .filter(|e| self.matches_date_range(e, range_start, range_end))
            .filter(|e| filter.matches(e))
            .collect();

        // Sort by start time
        events.sort_by(|a, b| a.start.cmp(&b.start));

        // Apply pagination
        let events: Vec<_> = events
            .into_iter()
            .skip(filter.offset)
            .take(filter.limit)
            .collect();

        Ok(events)
    }

    // ========================================================================
    // Query Operations
    // ========================================================================

    /// Query events based on the filter's query type.
    pub async fn query(&self, filter: &CalendarFilter) -> Result<Vec<CalendarEvent>> {
        match filter.query_type {
            CalendarQueryType::Conflicts => {
                // Get all events in range, then find conflicts
                let events = self.list(filter).await?;
                let conflicts = self.detect_conflicts(&events);
                // Return events that have conflicts
                let conflict_ids: std::collections::HashSet<_> = conflicts
                    .iter()
                    .flat_map(|c| vec![c.event1_id.clone(), c.event2_id.clone()])
                    .collect();
                Ok(events
                    .into_iter()
                    .filter(|e| conflict_ids.contains(&e.id))
                    .collect())
            }
            _ => self.list(filter).await,
        }
    }

    /// Get today's events.
    pub async fn today(&self) -> Result<Vec<CalendarEvent>> {
        self.list(&CalendarFilter::today()).await
    }

    /// Get upcoming events (next 7 days by default).
    pub async fn upcoming(&self, days: Option<i64>) -> Result<Vec<CalendarEvent>> {
        let days = days.unwrap_or(7);
        let now = Utc::now();
        let end = now + Duration::days(days);

        self.list(&CalendarFilter::date_range(now, end)).await
    }

    /// Get events for this week.
    pub async fn this_week(&self) -> Result<Vec<CalendarEvent>> {
        self.list(&CalendarFilter::this_week()).await
    }

    /// Search events by text.
    pub async fn search(&self, query: &str) -> Result<Vec<CalendarEvent>> {
        let filter = CalendarFilter {
            search_query: Some(query.to_string()),
            ..Default::default()
        };
        self.list(&filter).await
    }

    /// Get events for a specific date.
    pub async fn for_date(&self, date: DateTime<Utc>) -> Result<Vec<CalendarEvent>> {
        let start = date.date_naive().and_hms_opt(0, 0, 0).unwrap();
        let end = date.date_naive().and_hms_opt(23, 59, 59).unwrap();
        let start = DateTime::from_naive_utc_and_offset(start, Utc);
        let end = DateTime::from_naive_utc_and_offset(end, Utc);

        self.list(&CalendarFilter::date_range(start, end)).await
    }

    // ========================================================================
    // Conflict Detection
    // ========================================================================

    /// Detect all scheduling conflicts in a list of events.
    pub fn detect_conflicts(&self, events: &[CalendarEvent]) -> Vec<SchedulingConflict> {
        let mut conflicts = Vec::new();

        for i in 0..events.len() {
            for j in (i + 1)..events.len() {
                if let Some(conflict) = SchedulingConflict::detect(&events[i], &events[j]) {
                    conflicts.push(conflict);
                }
            }
        }

        // Sort by overlap duration (highest first)
        conflicts.sort_by(|a, b| b.overlap_minutes.cmp(&a.overlap_minutes));

        conflicts
    }

    /// Check for conflicts with a new event.
    pub async fn check_conflicts(
        &self,
        event: &CalendarEvent,
    ) -> Result<Vec<SchedulingConflict>> {
        // Get events that overlap with the new event's time range
        let buffer = Duration::hours(1);
        let range_start = event.start - buffer;
        let range_end = event.end.unwrap_or(event.start) + buffer;

        let existing = self
            .list(&CalendarFilter::date_range(range_start, range_end))
            .await?;

        let conflicts: Vec<_> = existing
            .iter()
            .filter(|e| e.id != event.id) // Exclude the event itself (for updates)
            .filter_map(|e| SchedulingConflict::detect(event, e))
            .collect();

        Ok(conflicts)
    }

    /// Get all conflicts in a date range.
    pub async fn find_all_conflicts(&self, filter: &CalendarFilter) -> Result<Vec<SchedulingConflict>> {
        let events = self.list(filter).await?;
        Ok(self.detect_conflicts(&events))
    }

    // ========================================================================
    // Free Time Finding
    // ========================================================================

    /// Find free time slots in a date range.
    pub async fn find_free_time(&self, params: &FreeTimeParams) -> Result<Vec<FreeTimeSlot>> {
        // Get all events in the range
        let events = self
            .list(&CalendarFilter::date_range(params.range_start, params.range_end))
            .await?;

        let mut free_slots = Vec::new();
        let min_duration = Duration::minutes(params.min_duration_minutes as i64);

        // Start from range start
        let mut current = params.range_start;

        while current < params.range_end {
            // Check if we should skip this time (outside working hours or weekends)
            if !self.is_working_time(&current, params) {
                current = self.next_working_time(&current, params);
                continue;
            }

            // Find the next event that starts after current time
            let next_event = events
                .iter()
                .filter(|e| e.start > current)
                .min_by_key(|e| e.start);

            let slot_end = match next_event {
                Some(event) => event.start.min(params.range_end),
                None => params.range_end,
            };

            // Adjust for working hours
            let slot_end = self.adjust_for_working_hours(slot_end, params);

            // Check if the slot is long enough
            if slot_end > current {
                let duration = slot_end - current;
                if duration >= min_duration {
                    free_slots.push(FreeTimeSlot::new(current, slot_end));
                }
            }

            // Move to the next potential slot
            current = match next_event {
                Some(event) => event.end.unwrap_or(event.start + Duration::hours(1)),
                None => params.range_end,
            };
        }

        Ok(free_slots)
    }

    /// Check if a time is within working hours.
    fn is_working_time(&self, time: &DateTime<Utc>, params: &FreeTimeParams) -> bool {
        // Check weekends
        if params.exclude_weekends {
            let weekday = time.weekday();
            if weekday == Weekday::Sat || weekday == Weekday::Sun {
                return false;
            }
        }

        // Check working hours
        if let (Some(start), Some(end)) = (params.working_hours_start, params.working_hours_end) {
            let time_only = NaiveTime::from_hms_opt(
                time.hour(),
                time.minute(),
                time.second(),
            ).unwrap_or_default();

            if time_only < start || time_only >= end {
                return false;
            }
        }

        true
    }

    /// Get the next working time from a given time.
    fn next_working_time(&self, time: &DateTime<Utc>, params: &FreeTimeParams) -> DateTime<Utc> {
        let mut next = *time;

        // If it's a weekend and we exclude weekends, move to Monday
        if params.exclude_weekends {
            let weekday = next.weekday();
            if weekday == Weekday::Sat {
                next += Duration::days(2);
            } else if weekday == Weekday::Sun {
                next += Duration::days(1);
            }
        }

        // If before working hours start, move to working hours start
        if let Some(start) = params.working_hours_start {
            let time_only = NaiveTime::from_hms_opt(
                next.hour(),
                next.minute(),
                0,
            ).unwrap_or_default();

            if time_only < start {
                let date = next.date_naive();
                let dt = date.and_time(start);
                return DateTime::from_naive_utc_and_offset(dt, Utc);
            }
        }

        // If after working hours end, move to next day's working hours start
        if let Some(end) = params.working_hours_end {
            let time_only = NaiveTime::from_hms_opt(
                next.hour(),
                next.minute(),
                0,
            ).unwrap_or_default();

            if time_only >= end {
                let mut date = next.date_naive() + Duration::days(1);

                // Skip weekends if needed
                if params.exclude_weekends {
                    let weekday = date.weekday();
                    if weekday == Weekday::Sat {
                        date += chrono::Duration::days(2);
                    } else if weekday == Weekday::Sun {
                        date += chrono::Duration::days(1);
                    }
                }

                let start = params.working_hours_start.unwrap_or(NaiveTime::from_hms_opt(9, 0, 0).unwrap());
                let dt = date.and_time(start);
                return DateTime::from_naive_utc_and_offset(dt, Utc);
            }
        }

        next
    }

    /// Adjust a time to be within working hours.
    fn adjust_for_working_hours(&self, time: DateTime<Utc>, params: &FreeTimeParams) -> DateTime<Utc> {
        if let Some(end) = params.working_hours_end {
            let time_only = NaiveTime::from_hms_opt(
                time.hour(),
                time.minute(),
                0,
            ).unwrap_or_default();

            if time_only > end {
                let date = time.date_naive();
                let dt = date.and_time(end);
                return DateTime::from_naive_utc_and_offset(dt, Utc);
            }
        }
        time
    }

    // ========================================================================
    // Recurring Events
    // ========================================================================

    /// Expand recurring events into instances within a date range.
    pub fn expand_recurring(
        &self,
        event: &CalendarEvent,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
    ) -> Vec<CalendarEvent> {
        let Some(ref recurrence) = event.recurrence else {
            return vec![event.clone()];
        };

        let mut instances = Vec::new();
        let mut current = event.start;
        let duration = event.duration();

        // Cap at 365 days or recurrence end
        let max_end = range_end.min(Utc::now() + Duration::days(365));
        let until = recurrence.until.unwrap_or(max_end);
        let max_count = recurrence.count.unwrap_or(100) as usize;

        while current <= until && current <= range_end && instances.len() < max_count {
            // Check if this occurrence is in our range and not an exception
            if current >= range_start && !recurrence.exceptions.contains(&current) {
                let mut instance = event.clone();
                instance.id = format!("{}-{}", event.id, current.timestamp());
                instance.start = current;
                if let Some(dur) = duration {
                    instance.end = Some(current + dur);
                }
                instances.push(instance);
            }

            // Calculate next occurrence
            current = self.next_occurrence(current, recurrence);
        }

        instances
    }

    /// Calculate the next occurrence based on recurrence rule.
    fn next_occurrence(&self, current: DateTime<Utc>, recurrence: &EventRecurrence) -> DateTime<Utc> {
        use crate::ontology::extraction::RecurrencePattern;

        match recurrence.pattern {
            RecurrencePattern::Daily => current + Duration::days(recurrence.interval as i64),
            RecurrencePattern::Weekly => {
                if recurrence.days_of_week.is_empty() {
                    current + Duration::weeks(recurrence.interval as i64)
                } else {
                    // Find next day in the days_of_week list
                    let current_dow = current.weekday().num_days_from_monday() as u8;
                    let next_dow = recurrence.days_of_week
                        .iter()
                        .find(|&&d| d > current_dow)
                        .or_else(|| recurrence.days_of_week.first());

                    match next_dow {
                        Some(&dow) => {
                            let days_ahead = if dow > current_dow {
                                (dow - current_dow) as i64
                            } else {
                                (7 - current_dow + dow) as i64 + (recurrence.interval as i64 - 1) * 7
                            };
                            current + Duration::days(days_ahead)
                        }
                        None => current + Duration::weeks(recurrence.interval as i64),
                    }
                }
            }
            RecurrencePattern::BiWeekly => current + Duration::weeks(2 * recurrence.interval as i64),
            RecurrencePattern::Monthly => {
                // Simple implementation: add months
                let mut year = current.year();
                let mut month = current.month() as i32 + recurrence.interval as i32;
                while month > 12 {
                    month -= 12;
                    year += 1;
                }
                let day = recurrence.day_of_month.unwrap_or(current.day());

                // Handle month length differences
                let max_day = days_in_month(year, month as u32);
                let day = day.min(max_day);

                current.with_year(year)
                    .and_then(|d| d.with_month(month as u32))
                    .and_then(|d| d.with_day(day))
                    .unwrap_or(current + Duration::days(30))
            }
            RecurrencePattern::Quarterly => {
                let mut year = current.year();
                let mut month = current.month() as i32 + 3 * recurrence.interval as i32;
                while month > 12 {
                    month -= 12;
                    year += 1;
                }
                let day = current.day().min(28); // Safe day for all months

                current.with_year(year)
                    .and_then(|d| d.with_month(month as u32))
                    .and_then(|d| d.with_day(day))
                    .unwrap_or(current + Duration::days(90))
            }
            RecurrencePattern::Yearly => {
                current.with_year(current.year() + recurrence.interval as i32)
                    .unwrap_or(current + Duration::days(365))
            }
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get calendar statistics.
    pub async fn stats(&self) -> Result<CalendarStats> {
        let now = Utc::now();
        let today_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap();
        let today_start = DateTime::from_naive_utc_and_offset(today_start, Utc);
        let today_end = today_start + Duration::days(1);
        let week_end = today_start + Duration::days(7);
        let month_ago = now - Duration::days(30);

        // Get all events
        let all_filter = EntityFilter::by_types([EntityType::CalendarEvent]).with_limit(10000);
        let store = self.store.read().await;
        let entities = store.list_entities(all_filter).await?;
        drop(store);

        let events: Vec<CalendarEvent> = entities
            .into_iter()
            .filter_map(|e| self.entity_to_event(&e).ok())
            .collect();

        let total_events = events.len();

        // Count by type
        let mut events_by_type = std::collections::HashMap::new();
        for event in &events {
            *events_by_type
                .entry(event.event_type.display_name().to_string())
                .or_insert(0) += 1;
        }

        // Upcoming events
        let upcoming_events = events.iter().filter(|e| e.start > now).count();

        // Today's events
        let events_today = events
            .iter()
            .filter(|e| e.is_within(today_start, today_end))
            .count();

        // This week's events
        let events_this_week = events
            .iter()
            .filter(|e| e.is_within(today_start, week_end))
            .count();

        // Recurring events
        let recurring_events = events.iter().filter(|e| e.recurrence.is_some()).count();

        // Extracted events
        let extracted_events = events.iter().filter(|e| e.source_document_id.is_some()).count();

        // Average events per day (last 30 days)
        let recent_events = events
            .iter()
            .filter(|e| e.start >= month_ago && e.start <= now)
            .count();
        let avg_events_per_day = recent_events as f32 / 30.0;

        // Total scheduled hours this week
        let total_scheduled_hours: f32 = events
            .iter()
            .filter(|e| e.is_within(today_start, week_end))
            .filter_map(|e| e.duration().map(|d| d.num_minutes() as f32 / 60.0))
            .sum();

        // Conflicts
        let this_week_events: Vec<_> = events
            .iter()
            .filter(|e| e.is_within(today_start, week_end))
            .cloned()
            .collect();
        let conflicts = self.detect_conflicts(&this_week_events).len();

        Ok(CalendarStats {
            total_events,
            events_by_type,
            upcoming_events,
            events_today,
            events_this_week,
            recurring_events,
            extracted_events,
            avg_events_per_day,
            total_scheduled_hours,
            conflicts,
        })
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Get date range based on query type.
    fn get_date_range(&self, filter: &CalendarFilter) -> (DateTime<Utc>, DateTime<Utc>) {
        let now = Utc::now();
        let today_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap();
        let today_start = DateTime::from_naive_utc_and_offset(today_start, Utc);

        match filter.query_type {
            CalendarQueryType::Today => (today_start, today_start + Duration::days(1)),
            CalendarQueryType::ThisWeek => {
                let days_since_monday = now.weekday().num_days_from_monday() as i64;
                let week_start = today_start - Duration::days(days_since_monday);
                (week_start, week_start + Duration::days(7))
            }
            CalendarQueryType::NextWeek => {
                let days_until_next_monday = 7 - now.weekday().num_days_from_monday() as i64;
                let next_week_start = today_start + Duration::days(days_until_next_monday);
                (next_week_start, next_week_start + Duration::days(7))
            }
            CalendarQueryType::Upcoming => (now, now + Duration::days(14)),
            CalendarQueryType::DateRange => (
                filter.start_date.unwrap_or(now - Duration::days(365)),
                filter.end_date.unwrap_or(now + Duration::days(365)),
            ),
            _ => (
                filter.start_date.unwrap_or(now - Duration::days(365)),
                filter.end_date.unwrap_or(now + Duration::days(365)),
            ),
        }
    }

    /// Check if an event is within a date range.
    fn matches_date_range(
        &self,
        event: &CalendarEvent,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
    ) -> bool {
        event.is_within(range_start, range_end)
    }

    /// Convert a CalendarEvent to an Entity.
    fn event_to_entity(&self, event: &CalendarEvent) -> Entity {
        let mut metadata = event.metadata.clone();

        // Store event fields as metadata
        metadata.insert("title".to_string(), serde_json::json!(event.title));
        if let Some(ref desc) = event.description {
            metadata.insert("description".to_string(), serde_json::json!(desc));
        }
        metadata.insert(
            "event_type".to_string(),
            serde_json::to_value(event.event_type).unwrap(),
        );
        metadata.insert("start".to_string(), serde_json::json!(event.start.to_rfc3339()));
        if let Some(end) = event.end {
            metadata.insert("end".to_string(), serde_json::json!(end.to_rfc3339()));
        }
        metadata.insert("all_day".to_string(), serde_json::json!(event.all_day));
        if let Some(ref location) = event.location {
            metadata.insert("location".to_string(), serde_json::json!(location));
        }
        if !event.participants.is_empty() {
            metadata.insert("participants".to_string(), serde_json::json!(event.participants));
        }
        if let Some(ref project_id) = event.project_id {
            metadata.insert("project_id".to_string(), serde_json::json!(project_id));
        }
        if let Some(ref task_id) = event.task_id {
            metadata.insert("task_id".to_string(), serde_json::json!(task_id));
        }
        if let Some(ref source_id) = event.source_document_id {
            metadata.insert("source_document_id".to_string(), serde_json::json!(source_id));
        }
        if let Some(ref text) = event.extracted_text {
            metadata.insert("extracted_text".to_string(), serde_json::json!(text));
        }
        if let Some(ref recurrence) = event.recurrence {
            metadata.insert(
                "recurrence".to_string(),
                serde_json::to_value(recurrence).unwrap(),
            );
        }
        if let Some(ref notes) = event.notes {
            metadata.insert("notes".to_string(), serde_json::json!(notes));
        }
        if !event.reminders.is_empty() {
            metadata.insert(
                "reminders".to_string(),
                serde_json::to_value(&event.reminders).unwrap(),
            );
        }
        metadata.insert("confidence".to_string(), serde_json::json!(event.confidence));

        Entity::with_id(&event.id, EntityType::CalendarEvent, &event.title)
            .with_confidence(event.confidence)
            .with_metadata("calendar_event", serde_json::json!(metadata))
    }

    /// Convert an Entity to a CalendarEvent.
    fn entity_to_event(&self, entity: &Entity) -> Result<CalendarEvent> {
        let metadata = entity
            .metadata
            .get("calendar_event")
            .and_then(|v| v.as_object())
            .ok_or_else(|| crate::error::AlloyError::Config(
                crate::error::ConfigError::Invalid("Invalid calendar event entity".to_string())
            ))?;

        let title = metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or(&entity.name)
            .to_string();

        let description = metadata
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let event_type: EventType = metadata
            .get("event_type")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let start: DateTime<Utc> = metadata
            .get("start")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let end: Option<DateTime<Utc>> = metadata
            .get("end")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));

        let all_day = metadata
            .get("all_day")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let location = metadata
            .get("location")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let participants: Vec<String> = metadata
            .get("participants")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let project_id = metadata
            .get("project_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let task_id = metadata
            .get("task_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let source_document_id = metadata
            .get("source_document_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let extracted_text = metadata
            .get("extracted_text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let recurrence: Option<EventRecurrence> = metadata
            .get("recurrence")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let notes = metadata
            .get("notes")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let reminders = metadata
            .get("reminders")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let confidence = metadata
            .get("confidence")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0);

        Ok(CalendarEvent {
            id: entity.id.clone(),
            title,
            description,
            event_type,
            start,
            end,
            all_day,
            location,
            participants,
            project_id,
            task_id,
            source_document_id,
            extracted_text,
            recurrence,
            notes,
            reminders,
            created_at: entity.created_at,
            updated_at: entity.updated_at,
            confidence,
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Get the number of days in a month.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::EmbeddedOntologyStore;

    fn create_test_manager() -> CalendarManager<EmbeddedOntologyStore> {
        let store = EmbeddedOntologyStore::new();
        CalendarManager::new(Arc::new(RwLock::new(store)))
    }

    #[tokio::test]
    async fn test_create_and_get_event() {
        let manager = create_test_manager();
        let now = Utc::now();

        let event = CalendarEvent::new("Test Meeting", now)
            .with_description("A test meeting")
            .with_type(EventType::Meeting)
            .with_duration(Duration::hours(1));

        let created = manager.create(event.clone()).await.unwrap();
        assert_eq!(created.title, "Test Meeting");

        let retrieved = manager.get(&created.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().title, "Test Meeting");
    }

    #[tokio::test]
    async fn test_update_event() {
        let manager = create_test_manager();
        let now = Utc::now();

        let event = CalendarEvent::new("Original Title", now);
        let created = manager.create(event).await.unwrap();

        let update = EventUpdate {
            title: Some("Updated Title".to_string()),
            location: Some("Conference Room A".to_string()),
            ..Default::default()
        };

        let updated = manager.update(&created.id, update).await.unwrap();
        assert!(updated.is_some());
        let updated = updated.unwrap();
        assert_eq!(updated.title, "Updated Title");
        assert_eq!(updated.location, Some("Conference Room A".to_string()));
    }

    #[tokio::test]
    async fn test_conflict_detection() {
        let now = Utc::now();

        let event1 = CalendarEvent::new("Meeting 1", now)
            .with_duration(Duration::hours(1));

        let event2 = CalendarEvent::new("Meeting 2", now + Duration::minutes(30))
            .with_duration(Duration::hours(1));

        let event3 = CalendarEvent::new("Meeting 3", now + Duration::hours(2))
            .with_duration(Duration::hours(1));

        let events = vec![event1, event2, event3];
        let manager = create_test_manager();
        let conflicts = manager.detect_conflicts(&events);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].overlap_minutes, 30);
    }

    #[tokio::test]
    async fn test_recurring_expansion() {
        let manager = create_test_manager();
        let now = Utc::now();

        let event = CalendarEvent::new("Weekly Standup", now)
            .with_recurrence(EventRecurrence::weekly().times(5));

        let instances = manager.expand_recurring(
            &event,
            now,
            now + Duration::days(30),
        );

        assert_eq!(instances.len(), 5);
    }
}

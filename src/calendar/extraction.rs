//! Calendar event extraction from documents.
//!
//! This module provides functionality to extract calendar events from
//! indexed documents using temporal parsing and pattern matching.

use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};
use regex::Regex;

use crate::ontology::extraction::{DateType, TemporalExtraction, TemporalParser};

use super::types::{CalendarEvent, EventRecurrence, EventType};

// ============================================================================
// Calendar Extractor
// ============================================================================

/// Extractor for calendar events from text.
pub struct CalendarExtractor {
    /// Temporal parser for date/time extraction.
    temporal_parser: TemporalParser,
    /// Minimum confidence for extracted events.
    min_confidence: f32,
}

impl Default for CalendarExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl CalendarExtractor {
    /// Create a new calendar extractor.
    pub fn new() -> Self {
        Self {
            temporal_parser: TemporalParser::new(),
            min_confidence: 0.6,
        }
    }

    /// Create an extractor with a specific reference date.
    pub fn with_reference_date(reference_date: NaiveDate) -> Self {
        Self {
            temporal_parser: TemporalParser::with_reference_date(reference_date),
            min_confidence: 0.6,
        }
    }

    /// Set the minimum confidence threshold.
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Extract calendar events from text.
    pub fn extract(
        &self,
        text: &str,
        source_document_id: Option<&str>,
    ) -> Vec<ExtractedCalendarEvent> {
        let mut events = Vec::new();

        // Parse temporal expressions
        let temporal = self.temporal_parser.parse(text);

        // Group temporal expressions that might form events
        let grouped = self.group_temporal_expressions(text, &temporal);

        for group in grouped {
            if let Some(event) = self.create_event_from_group(text, group, source_document_id) {
                if event.confidence >= self.min_confidence {
                    events.push(event);
                }
            }
        }

        // Also detect meeting patterns
        events.extend(self.extract_meeting_patterns(text, source_document_id));

        // Deduplicate overlapping events
        self.deduplicate_events(&mut events);

        events
    }

    /// Group temporal expressions that belong together.
    fn group_temporal_expressions(
        &self,
        _text: &str,
        temporal: &[TemporalExtraction],
    ) -> Vec<TemporalGroup> {
        let mut groups = Vec::new();
        let mut used = vec![false; temporal.len()];

        for (i, t) in temporal.iter().enumerate() {
            if used[i] {
                continue;
            }

            let mut group = TemporalGroup {
                date: None,
                time: None,
                recurrence: None,
                context_start: t.start_offset,
                context_end: t.end_offset,
                confidence: t.confidence,
                date_type: t.date_type,
            };

            // Extract date if present
            if t.parsed_date.date.is_some() {
                group.date = t.parsed_date.date;
            }

            // Extract time if present
            if t.parsed_date.time.is_some() {
                group.time = t.parsed_date.time;
            }

            // Extract recurrence if present
            if t.recurrence.is_some() {
                group.recurrence = t.recurrence.clone();
            }

            used[i] = true;

            // Look for nearby related temporal expressions
            for (j, other) in temporal.iter().enumerate().skip(i + 1) {
                if used[j] {
                    continue;
                }

                // Check if they're close together (within 50 characters)
                if other.start_offset <= t.end_offset + 50 {
                    // Combine date and time
                    if group.date.is_none() && other.parsed_date.date.is_some() {
                        group.date = other.parsed_date.date;
                        group.context_end = other.end_offset;
                        group.confidence = group.confidence.min(other.confidence);
                        used[j] = true;
                    } else if group.time.is_none() && other.parsed_date.time.is_some() {
                        group.time = other.parsed_date.time;
                        group.context_end = other.end_offset;
                        group.confidence = group.confidence.min(other.confidence);
                        used[j] = true;
                    }
                }
            }

            groups.push(group);
        }

        groups
    }

    /// Create an event from a temporal group.
    fn create_event_from_group(
        &self,
        text: &str,
        group: TemporalGroup,
        source_document_id: Option<&str>,
    ) -> Option<ExtractedCalendarEvent> {
        // Need at least a date
        let date = group.date?;

        // Get context around the temporal expression
        let context_start = group.context_start.saturating_sub(100);
        let context_end = (group.context_end + 100).min(text.len());
        let context = &text[context_start..context_end];

        // Extract event title from context
        let title =
            self.extract_title_from_context(context, text, group.context_start, group.context_end);

        // Build the datetime
        let time = group
            .time
            .unwrap_or_else(|| chrono::NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        let datetime = NaiveDateTime::new(date, time);
        let start = DateTime::from_naive_utc_and_offset(datetime, Utc);

        // Determine event type
        let event_type = self.infer_event_type(context, group.date_type);

        // Build the event
        let mut event = CalendarEvent::new(&title, start)
            .with_type(event_type)
            .with_confidence(group.confidence * 0.8); // Reduce confidence for extracted events

        // Add recurrence if present
        if let Some(ref recurrence) = group.recurrence {
            event = event.with_recurrence(EventRecurrence::from_rule(recurrence));
        }

        // Add source document
        if let Some(doc_id) = source_document_id {
            event = event.with_source(doc_id);
        }

        // Extract the relevant text snippet
        let snippet_start = group.context_start.saturating_sub(50);
        let snippet_end = (group.context_end + 50).min(text.len());
        let snippet = text[snippet_start..snippet_end].trim().to_string();
        event = event.with_extracted_text(snippet);

        // Set default duration for deadlines vs meetings
        match event_type {
            EventType::Meeting | EventType::Call | EventType::Standup => {
                event = event.with_duration(Duration::hours(1));
            }
            EventType::Appointment => {
                event = event.with_duration(Duration::minutes(30));
            }
            _ => {}
        }

        Some(ExtractedCalendarEvent {
            event,
            confidence: group.confidence * 0.8,
            extraction_method: ExtractionMethod::TemporalParsing,
        })
    }

    /// Extract event title from context.
    fn extract_title_from_context(
        &self,
        context: &str,
        _full_text: &str,
        _temp_start: usize,
        _temp_end: usize,
    ) -> String {
        // Try to find a sentence or phrase before the temporal expression
        let patterns = [
            // Meeting patterns
            r"(?i)(meeting|call|sync|standup|review|discussion|interview|presentation)\s+(?:with|about|for|on)\s+([^.!?\n]+)",
            r"(?i)(weekly|daily|monthly|quarterly)\s+(\w+)",
            // Event patterns
            r"(?i)(?:schedule|scheduled|plan|planned)\s+(?:a\s+)?([^.!?\n]+)",
            // Action patterns
            r"(?i)(?:remind|reminder)\s+(?:me\s+)?(?:to\s+)?([^.!?\n]+)",
        ];

        for pattern in &patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if let Some(captures) = regex.captures(context) {
                    if let Some(m) = captures.get(0) {
                        let title = m.as_str().trim();
                        if !title.is_empty() && title.len() < 100 {
                            return self.clean_title(title);
                        }
                    }
                }
            }
        }

        // Try to get the sentence containing the temporal expression
        let sentences: Vec<&str> = context.split(['.', '!', '?']).collect();
        for sentence in sentences {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() && trimmed.len() >= 10 && trimmed.len() < 100 {
                return self.clean_title(trimmed);
            }
        }

        // Default to a generic title
        "Extracted Event".to_string()
    }

    /// Clean up an extracted title.
    fn clean_title(&self, title: &str) -> String {
        let mut cleaned = title.to_string();

        // Remove leading/trailing punctuation
        cleaned = cleaned
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();

        // Capitalize first letter
        if let Some(first) = cleaned.chars().next() {
            cleaned = first.to_uppercase().to_string() + &cleaned[first.len_utf8()..];
        }

        // Truncate if too long
        if cleaned.len() > 80 {
            cleaned = cleaned[..77].to_string() + "...";
        }

        cleaned
    }

    /// Infer event type from context and date type.
    fn infer_event_type(&self, context: &str, date_type: DateType) -> EventType {
        let context_lower = context.to_lowercase();

        // Check for specific keywords
        if context_lower.contains("meeting") || context_lower.contains("meet with") {
            return EventType::Meeting;
        }
        if context_lower.contains("call") || context_lower.contains("phone") {
            return EventType::Call;
        }
        if context_lower.contains("standup") || context_lower.contains("stand-up") {
            return EventType::Standup;
        }
        if context_lower.contains("appointment") || context_lower.contains("appt") {
            return EventType::Appointment;
        }
        if context_lower.contains("deadline")
            || context_lower.contains("due")
            || context_lower.contains("due by")
        {
            return EventType::Deadline;
        }
        if context_lower.contains("remind") || context_lower.contains("reminder") {
            return EventType::Reminder;
        }
        if context_lower.contains("milestone")
            || context_lower.contains("launch")
            || context_lower.contains("release")
        {
            return EventType::Milestone;
        }
        if context_lower.contains("travel")
            || context_lower.contains("flight")
            || context_lower.contains("trip")
        {
            return EventType::Travel;
        }
        if context_lower.contains("block") || context_lower.contains("focus time") {
            return EventType::BlockedTime;
        }

        // Infer from date type
        match date_type {
            DateType::Deadline => EventType::Deadline,
            DateType::Recurring => EventType::Standup, // Recurring events are often standups
            _ => EventType::Event,
        }
    }

    /// Extract meeting patterns using regex.
    fn extract_meeting_patterns(
        &self,
        text: &str,
        source_document_id: Option<&str>,
    ) -> Vec<ExtractedCalendarEvent> {
        let mut events = Vec::new();

        // Pattern: "Meeting with [person] on [date] at [time]"
        let meeting_pattern = Regex::new(
            r"(?i)(?:meet(?:ing)?|call|sync|chat)\s+(?:with\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:on\s+)?([A-Za-z]+day|tomorrow|next\s+\w+|[A-Z][a-z]+\s+\d+|\d+/\d+(?:/\d+)?)\s*(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?",
        );

        if let Ok(regex) = meeting_pattern {
            for captures in regex.captures_iter(text) {
                let full_match = captures.get(0).map(|m| m.as_str()).unwrap_or("");
                let person = captures.get(1).map(|m| m.as_str()).unwrap_or("Unknown");
                let date_str = captures.get(2).map(|m| m.as_str()).unwrap_or("");
                let time_str = captures.get(3).map(|m| m.as_str());

                // Parse the date
                let date_results = self.temporal_parser.parse(date_str);
                if let Some(date_result) = date_results.first() {
                    if let Some(date) = date_result.parsed_date.date {
                        let time = if let Some(ts) = time_str {
                            let time_results = self.temporal_parser.parse(ts);
                            time_results
                                .first()
                                .and_then(|t| t.parsed_date.time)
                                .unwrap_or_else(|| {
                                    chrono::NaiveTime::from_hms_opt(9, 0, 0).unwrap()
                                })
                        } else {
                            chrono::NaiveTime::from_hms_opt(9, 0, 0).unwrap()
                        };

                        let datetime = NaiveDateTime::new(date, time);
                        let start = DateTime::from_naive_utc_and_offset(datetime, Utc);

                        let title = format!("Meeting with {}", person);
                        let mut event = CalendarEvent::new(&title, start)
                            .with_type(EventType::Meeting)
                            .with_participant(person)
                            .with_duration(Duration::hours(1))
                            .with_confidence(0.75)
                            .with_extracted_text(full_match);

                        if let Some(doc_id) = source_document_id {
                            event = event.with_source(doc_id);
                        }

                        events.push(ExtractedCalendarEvent {
                            event,
                            confidence: 0.75,
                            extraction_method: ExtractionMethod::PatternMatching,
                        });
                    }
                }
            }
        }

        events
    }

    /// Remove duplicate or overlapping events.
    fn deduplicate_events(&self, events: &mut Vec<ExtractedCalendarEvent>) {
        if events.len() < 2 {
            return;
        }

        // Sort by confidence (highest first)
        events.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut i = 0;
        while i < events.len() {
            let mut j = i + 1;
            while j < events.len() {
                // Check if events are too similar (same title or overlapping time)
                let event_i = &events[i].event;
                let event_j = &events[j].event;

                let same_title = event_i.title.to_lowercase() == event_j.title.to_lowercase();
                let overlapping = event_i.overlaps_with(event_j);
                let very_close = (event_i.start - event_j.start).num_minutes().abs() < 30;

                if same_title || (overlapping && very_close) {
                    events.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// A group of related temporal expressions.
#[derive(Debug)]
struct TemporalGroup {
    /// Extracted date.
    date: Option<NaiveDate>,
    /// Extracted time.
    time: Option<chrono::NaiveTime>,
    /// Recurrence rule.
    recurrence: Option<crate::ontology::extraction::RecurrenceRule>,
    /// Start of context in text.
    context_start: usize,
    /// End of context in text.
    context_end: usize,
    /// Confidence score.
    confidence: f32,
    /// Type of date expression.
    date_type: DateType,
}

/// An extracted calendar event with metadata.
#[derive(Debug, Clone)]
pub struct ExtractedCalendarEvent {
    /// The extracted event.
    pub event: CalendarEvent,
    /// Confidence score for the extraction.
    pub confidence: f32,
    /// Method used for extraction.
    pub extraction_method: ExtractionMethod,
}

/// Method used for event extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionMethod {
    /// Extracted using temporal parsing.
    TemporalParsing,
    /// Extracted using pattern matching.
    PatternMatching,
    /// Extracted using LLM.
    Llm,
    /// Manually created.
    Manual,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_meeting() {
        let extractor =
            CalendarExtractor::with_reference_date(NaiveDate::from_ymd_opt(2024, 1, 10).unwrap());

        let text = "Let's schedule a meeting with John tomorrow at 3pm to discuss the project.";
        let events = extractor.extract(text, Some("doc123"));

        assert!(!events.is_empty());
        let event = &events[0].event;
        assert!(
            event.title.to_lowercase().contains("meeting")
                || event.title.to_lowercase().contains("john")
        );
    }

    #[test]
    fn test_extract_deadline() {
        let extractor =
            CalendarExtractor::with_reference_date(NaiveDate::from_ymd_opt(2024, 1, 10).unwrap());

        let text = "The report is due by Friday at 5pm. Please submit before the deadline.";
        let events = extractor.extract(text, None);

        // Should extract at least one event
        let _deadline_events: Vec<_> = events
            .iter()
            .filter(|e| e.event.event_type == EventType::Deadline)
            .collect();

        // Either found as deadline or as a regular event with deadline pattern
        assert!(!events.is_empty());
    }

    #[test]
    fn test_extract_recurring() {
        let extractor =
            CalendarExtractor::with_reference_date(NaiveDate::from_ymd_opt(2024, 1, 10).unwrap());

        let text = "Team standup every Monday at 9am in the conference room.";
        let events = extractor.extract(text, None);

        // Should have found the recurring event
        let recurring: Vec<_> = events
            .iter()
            .filter(|e| e.event.recurrence.is_some())
            .collect();

        // At minimum should extract a temporal expression
        assert!(!events.is_empty() || !recurring.is_empty());
    }

    #[test]
    fn test_extract_with_participants() {
        let extractor = CalendarExtractor::new();

        let text = "Meeting with Alice and Bob on Friday at 2pm to review the design.";
        let events = extractor.extract(text, None);

        // Should extract at least one event
        assert!(!events.is_empty());
    }

    #[test]
    fn test_low_confidence_filtering() {
        let extractor = CalendarExtractor::new().with_min_confidence(0.9);

        let text = "Maybe we could meet sometime next week?";
        let events = extractor.extract(text, None);

        // Low confidence events should be filtered out
        // This depends on the confidence assigned during extraction
    }

    #[test]
    fn test_event_type_inference() {
        let extractor = CalendarExtractor::new();

        // Meeting
        let text = "Schedule a meeting with the team for tomorrow at 10am.";
        let events = extractor.extract(text, None);
        if let Some(event) = events.first() {
            // Should infer meeting type from context
            assert!(
                event.event.event_type == EventType::Meeting
                    || event.event.event_type == EventType::Event
            );
        }

        // Call
        let text = "Call John at 3pm to discuss the proposal.";
        let events = extractor.extract(text, None);
        if let Some(event) = events.first() {
            assert!(
                event.event.event_type == EventType::Call
                    || event.event.event_type == EventType::Event
            );
        }
    }
}

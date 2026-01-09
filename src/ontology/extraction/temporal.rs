//! Temporal parsing for dates, times, deadlines, and recurring patterns.
//!
//! This module provides comprehensive temporal extraction from natural language text,
//! supporting:
//! - Absolute dates: "January 15, 2024", "2024-01-15"
//! - Relative dates: "tomorrow", "next Tuesday", "in 2 weeks"
//! - Times: "3pm", "15:30", "noon"
//! - Date ranges: "from Monday to Friday"
//! - Deadlines: "by end of day", "due Friday"
//! - Recurring patterns: "every Monday", "weekly on Tuesdays"

use chrono::{Datelike, Duration, Local, NaiveDate, NaiveDateTime, NaiveTime, Weekday};
use serde::{Deserialize, Serialize};

// ============================================================================
// Types
// ============================================================================

/// A parsed date/time from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDate {
    /// The parsed date (if available).
    pub date: Option<NaiveDate>,
    /// The parsed time (if available).
    pub time: Option<NaiveTime>,
    /// Whether this is a range end date.
    pub is_end_date: bool,
    /// The original reference point (now) used for relative dates.
    pub reference_date: NaiveDate,
}

impl ParsedDate {
    /// Create a ParsedDate from a NaiveDate.
    pub fn from_date(date: NaiveDate) -> Self {
        Self {
            date: Some(date),
            time: None,
            is_end_date: false,
            reference_date: Local::now().date_naive(),
        }
    }

    /// Create a ParsedDate from a NaiveDateTime.
    pub fn from_datetime(dt: NaiveDateTime) -> Self {
        Self {
            date: Some(dt.date()),
            time: Some(dt.time()),
            is_end_date: false,
            reference_date: Local::now().date_naive(),
        }
    }

    /// Create a ParsedDate with just a time.
    pub fn from_time(time: NaiveTime) -> Self {
        Self {
            date: None,
            time: Some(time),
            is_end_date: false,
            reference_date: Local::now().date_naive(),
        }
    }

    /// Get the full datetime if both date and time are present.
    pub fn as_datetime(&self) -> Option<NaiveDateTime> {
        match (self.date, self.time) {
            (Some(d), Some(t)) => Some(NaiveDateTime::new(d, t)),
            (Some(d), None) => Some(NaiveDateTime::new(d, NaiveTime::from_hms_opt(0, 0, 0)?)),
            _ => None,
        }
    }
}

impl std::fmt::Display for ParsedDate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.date, self.time) {
            (Some(d), Some(t)) => write!(f, "{} {}", d, t),
            (Some(d), None) => write!(f, "{}", d),
            (None, Some(t)) => write!(f, "{}", t),
            (None, None) => write!(f, "(no date/time)"),
        }
    }
}

/// The type of temporal expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DateType {
    /// A specific date (absolute).
    Specific,
    /// A relative date (e.g., "tomorrow").
    Relative,
    /// A deadline (e.g., "by Friday").
    Deadline,
    /// A recurring pattern (e.g., "every Monday").
    Recurring,
    /// A date range.
    Range,
    /// Just a time, no date.
    TimeOnly,
}

/// A complete temporal extraction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalExtraction {
    /// The original text that was parsed.
    pub original_text: String,
    /// Normalized text representation.
    pub normalized_text: String,
    /// The parsed date/time.
    pub parsed_date: ParsedDate,
    /// Type of temporal expression.
    pub date_type: DateType,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// Character offset where the temporal expression starts.
    pub start_offset: usize,
    /// Character offset where the temporal expression ends.
    pub end_offset: usize,
    /// Recurrence rule if this is a recurring pattern.
    pub recurrence: Option<RecurrenceRule>,
    /// End date for ranges.
    pub end_date: Option<ParsedDate>,
}

/// A recurrence pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrenceRule {
    /// The recurrence pattern type.
    pub pattern: RecurrencePattern,
    /// Interval (e.g., every 2 weeks).
    pub interval: u32,
    /// Specific days for weekly patterns.
    #[serde(default)]
    pub days_of_week: Vec<Weekday>,
    /// Day of month for monthly patterns.
    pub day_of_month: Option<u32>,
    /// End date for the recurrence.
    pub until: Option<NaiveDate>,
    /// Maximum occurrences.
    pub count: Option<u32>,
}

/// Type of recurrence pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RecurrencePattern {
    Daily,
    Weekly,
    BiWeekly,
    Monthly,
    Quarterly,
    Yearly,
}

// ============================================================================
// Temporal Parser
// ============================================================================

/// The main temporal parser.
pub struct TemporalParser {
    /// Reference date for relative calculations (defaults to today).
    reference_date: NaiveDate,
}

impl Default for TemporalParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalParser {
    /// Create a new temporal parser with today as the reference date.
    pub fn new() -> Self {
        Self {
            reference_date: Local::now().date_naive(),
        }
    }

    /// Create a parser with a specific reference date.
    pub fn with_reference_date(reference_date: NaiveDate) -> Self {
        Self { reference_date }
    }

    /// Parse all temporal expressions from text.
    pub fn parse(&self, text: &str) -> Vec<TemporalExtraction> {
        let mut results = Vec::new();

        // Parse different types of temporal expressions
        results.extend(self.parse_absolute_dates(text));
        results.extend(self.parse_relative_dates(text));
        results.extend(self.parse_times(text));
        results.extend(self.parse_deadlines(text));
        results.extend(self.parse_recurring(text));

        // Sort by start offset and remove overlaps
        results.sort_by_key(|r| r.start_offset);
        Self::remove_overlaps(&mut results);

        results
    }

    /// Remove overlapping extractions, keeping the higher confidence one.
    fn remove_overlaps(results: &mut Vec<TemporalExtraction>) {
        if results.len() < 2 {
            return;
        }

        let mut i = 0;
        while i < results.len() - 1 {
            let current_end = results[i].end_offset;
            let next_start = results[i + 1].start_offset;

            if next_start < current_end {
                // Overlap detected - keep the higher confidence one
                if results[i].confidence >= results[i + 1].confidence {
                    results.remove(i + 1);
                } else {
                    results.remove(i);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Parse absolute dates like "January 15, 2024" or "2024-01-15".
    fn parse_absolute_dates(&self, text: &str) -> Vec<TemporalExtraction> {
        let mut results = Vec::new();
        let _text_lower = text.to_lowercase(); // Used for case-insensitive matching

        // ISO format: 2024-01-15
        let iso_pattern =
            regex::Regex::new(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b").expect("Invalid regex");
        for cap in iso_pattern.captures_iter(text) {
            if let (Ok(year), Ok(month), Ok(day)) = (
                cap[1].parse::<i32>(),
                cap[2].parse::<u32>(),
                cap[3].parse::<u32>(),
            ) {
                if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                    let full_match = cap.get(0).unwrap();
                    results.push(TemporalExtraction {
                        original_text: full_match.as_str().to_string(),
                        normalized_text: date.format("%Y-%m-%d").to_string(),
                        parsed_date: ParsedDate::from_date(date),
                        date_type: DateType::Specific,
                        confidence: 0.95,
                        start_offset: full_match.start(),
                        end_offset: full_match.end(),
                        recurrence: None,
                        end_date: None,
                    });
                }
            }
        }

        // US format: January 15, 2024 or Jan 15 2024
        let month_names = [
            ("january", 1),
            ("jan", 1),
            ("february", 2),
            ("feb", 2),
            ("march", 3),
            ("mar", 3),
            ("april", 4),
            ("apr", 4),
            ("may", 5),
            ("june", 6),
            ("jun", 6),
            ("july", 7),
            ("jul", 7),
            ("august", 8),
            ("aug", 8),
            ("september", 9),
            ("sept", 9),
            ("sep", 9),
            ("october", 10),
            ("oct", 10),
            ("november", 11),
            ("nov", 11),
            ("december", 12),
            ("dec", 12),
        ];

        for (month_name, month_num) in &month_names {
            let pattern = regex::Regex::new(&format!(
                r"(?i)\b{}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})?\b",
                month_name
            ))
            .expect("Invalid regex");

            for cap in pattern.captures_iter(text) {
                if let Ok(day) = cap[1].parse::<u32>() {
                    let year = cap
                        .get(2)
                        .and_then(|m| m.as_str().parse::<i32>().ok())
                        .unwrap_or(self.reference_date.year());

                    if let Some(date) = NaiveDate::from_ymd_opt(year, *month_num, day) {
                        let full_match = cap.get(0).unwrap();
                        results.push(TemporalExtraction {
                            original_text: full_match.as_str().to_string(),
                            normalized_text: date.format("%B %d, %Y").to_string(),
                            parsed_date: ParsedDate::from_date(date),
                            date_type: DateType::Specific,
                            confidence: 0.9,
                            start_offset: full_match.start(),
                            end_offset: full_match.end(),
                            recurrence: None,
                            end_date: None,
                        });
                    }
                }
            }
        }

        // Numeric format: 1/15/2024 or 15/1/2024 (US vs EU - assume US)
        let numeric_pattern =
            regex::Regex::new(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b").expect("Invalid regex");
        for cap in numeric_pattern.captures_iter(text) {
            if let (Ok(month), Ok(day), Ok(year)) = (
                cap[1].parse::<u32>(),
                cap[2].parse::<u32>(),
                cap[3].parse::<i32>(),
            ) {
                let year = if year < 100 { 2000 + year } else { year };
                if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                    let full_match = cap.get(0).unwrap();
                    results.push(TemporalExtraction {
                        original_text: full_match.as_str().to_string(),
                        normalized_text: date.format("%Y-%m-%d").to_string(),
                        parsed_date: ParsedDate::from_date(date),
                        date_type: DateType::Specific,
                        confidence: 0.8, // Lower confidence due to ambiguity
                        start_offset: full_match.start(),
                        end_offset: full_match.end(),
                        recurrence: None,
                        end_date: None,
                    });
                }
            }
        }

        results
    }

    /// Parse relative dates like "tomorrow", "next Tuesday", "in 2 weeks".
    fn parse_relative_dates(&self, text: &str) -> Vec<TemporalExtraction> {
        let mut results = Vec::new();
        let text_lower = text.to_lowercase();

        // Simple relative terms
        let relative_terms = [
            ("today", 0i64),
            ("tomorrow", 1),
            ("yesterday", -1),
            ("day after tomorrow", 2),
        ];

        for (term, days) in &relative_terms {
            if let Some(pos) = text_lower.find(term) {
                let date = self.reference_date + Duration::days(*days);
                results.push(TemporalExtraction {
                    original_text: term.to_string(),
                    normalized_text: date.format("%Y-%m-%d").to_string(),
                    parsed_date: ParsedDate::from_date(date),
                    date_type: DateType::Relative,
                    confidence: 0.95,
                    start_offset: pos,
                    end_offset: pos + term.len(),
                    recurrence: None,
                    end_date: None,
                });
            }
        }

        // "next/this/last [weekday]"
        let weekday_names = [
            ("monday", Weekday::Mon),
            ("tuesday", Weekday::Tue),
            ("wednesday", Weekday::Wed),
            ("thursday", Weekday::Thu),
            ("friday", Weekday::Fri),
            ("saturday", Weekday::Sat),
            ("sunday", Weekday::Sun),
        ];

        for (name, weekday) in &weekday_names {
            // "next [weekday]"
            let next_pattern =
                regex::Regex::new(&format!(r"(?i)\bnext\s+{}\b", name)).expect("Invalid regex");
            for cap in next_pattern.captures_iter(text) {
                let date = self.next_weekday(*weekday, true);
                let full_match = cap.get(0).unwrap();
                results.push(TemporalExtraction {
                    original_text: full_match.as_str().to_string(),
                    normalized_text: date.format("%Y-%m-%d (%A)").to_string(),
                    parsed_date: ParsedDate::from_date(date),
                    date_type: DateType::Relative,
                    confidence: 0.9,
                    start_offset: full_match.start(),
                    end_offset: full_match.end(),
                    recurrence: None,
                    end_date: None,
                });
            }

            // "this [weekday]"
            let this_pattern =
                regex::Regex::new(&format!(r"(?i)\bthis\s+{}\b", name)).expect("Invalid regex");
            for cap in this_pattern.captures_iter(text) {
                let date = self.next_weekday(*weekday, false);
                let full_match = cap.get(0).unwrap();
                results.push(TemporalExtraction {
                    original_text: full_match.as_str().to_string(),
                    normalized_text: date.format("%Y-%m-%d (%A)").to_string(),
                    parsed_date: ParsedDate::from_date(date),
                    date_type: DateType::Relative,
                    confidence: 0.9,
                    start_offset: full_match.start(),
                    end_offset: full_match.end(),
                    recurrence: None,
                    end_date: None,
                });
            }

            // Just the weekday name (without "next" or "this")
            let bare_pattern =
                regex::Regex::new(&format!(r"(?i)\b(?:on\s+)?{}\b", name)).expect("Invalid regex");
            for cap in bare_pattern.captures_iter(text) {
                // Skip if it's part of "next X" or "this X" or "every X"
                let full_match = cap.get(0).unwrap();
                let prefix_start = full_match.start().saturating_sub(10);
                let prefix = &text_lower[prefix_start..full_match.start()];
                if prefix.contains("next")
                    || prefix.contains("this")
                    || prefix.contains("last")
                    || prefix.contains("every")
                {
                    continue;
                }

                let date = self.next_weekday(*weekday, false);
                results.push(TemporalExtraction {
                    original_text: full_match.as_str().to_string(),
                    normalized_text: date.format("%Y-%m-%d (%A)").to_string(),
                    parsed_date: ParsedDate::from_date(date),
                    date_type: DateType::Relative,
                    confidence: 0.75,
                    start_offset: full_match.start(),
                    end_offset: full_match.end(),
                    recurrence: None,
                    end_date: None,
                });
            }
        }

        // "in N days/weeks/months"
        let in_pattern = regex::Regex::new(
            r"(?i)\bin\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b",
        )
        .expect("Invalid regex");
        for cap in in_pattern.captures_iter(text) {
            if let Ok(num) = cap[1].parse::<i64>() {
                let unit = cap[2].to_lowercase();
                let date = match unit.as_str() {
                    "day" | "days" => self.reference_date + Duration::days(num),
                    "week" | "weeks" => self.reference_date + Duration::weeks(num),
                    "month" | "months" => self
                        .add_months(self.reference_date, num as i32)
                        .unwrap_or(self.reference_date),
                    "year" | "years" => NaiveDate::from_ymd_opt(
                        self.reference_date.year() + num as i32,
                        self.reference_date.month(),
                        self.reference_date.day(),
                    )
                    .unwrap_or(self.reference_date),
                    _ => continue,
                };

                let full_match = cap.get(0).unwrap();
                results.push(TemporalExtraction {
                    original_text: full_match.as_str().to_string(),
                    normalized_text: date.format("%Y-%m-%d").to_string(),
                    parsed_date: ParsedDate::from_date(date),
                    date_type: DateType::Relative,
                    confidence: 0.9,
                    start_offset: full_match.start(),
                    end_offset: full_match.end(),
                    recurrence: None,
                    end_date: None,
                });
            }
        }

        // "end of week/month/year/quarter"
        let end_of_pattern =
            regex::Regex::new(r"(?i)\b(?:end\s+of\s+(?:the\s+)?)(week|month|quarter|year)\b")
                .expect("Invalid regex");
        for cap in end_of_pattern.captures_iter(text) {
            let period = cap[1].to_lowercase();
            let date = match period.as_str() {
                "week" => self.end_of_week(),
                "month" => self.end_of_month(),
                "quarter" => self.end_of_quarter(),
                "year" => self.end_of_year(),
                _ => continue,
            };

            let full_match = cap.get(0).unwrap();
            results.push(TemporalExtraction {
                original_text: full_match.as_str().to_string(),
                normalized_text: date.format("%Y-%m-%d").to_string(),
                parsed_date: ParsedDate::from_date(date),
                date_type: DateType::Relative,
                confidence: 0.85,
                start_offset: full_match.start(),
                end_offset: full_match.end(),
                recurrence: None,
                end_date: None,
            });
        }

        // "Q1/Q2/Q3/Q4 [year]"
        let quarter_pattern =
            regex::Regex::new(r"(?i)\bQ([1-4])(?:\s+(\d{4}))?\b").expect("Invalid regex");
        for cap in quarter_pattern.captures_iter(text) {
            if let Ok(quarter) = cap[1].parse::<u32>() {
                let year = cap
                    .get(2)
                    .and_then(|m| m.as_str().parse::<i32>().ok())
                    .unwrap_or(self.reference_date.year());

                let (start_month, end_month) = match quarter {
                    1 => (1, 3),
                    2 => (4, 6),
                    3 => (7, 9),
                    4 => (10, 12),
                    _ => continue,
                };

                let start_date = NaiveDate::from_ymd_opt(year, start_month, 1);
                let end_date = NaiveDate::from_ymd_opt(year, end_month, 1)
                    .and_then(|d| self.add_months(d, 1))
                    .map(|d| d - Duration::days(1));

                if let (Some(start), Some(end)) = (start_date, end_date) {
                    let full_match = cap.get(0).unwrap();
                    results.push(TemporalExtraction {
                        original_text: full_match.as_str().to_string(),
                        normalized_text: format!("Q{} {} ({} - {})", quarter, year, start, end),
                        parsed_date: ParsedDate::from_date(start),
                        date_type: DateType::Range,
                        confidence: 0.9,
                        start_offset: full_match.start(),
                        end_offset: full_match.end(),
                        recurrence: None,
                        end_date: Some(ParsedDate::from_date(end)),
                    });
                }
            }
        }

        results
    }

    /// Parse times like "3pm", "15:30", "noon".
    fn parse_times(&self, text: &str) -> Vec<TemporalExtraction> {
        let mut results = Vec::new();

        // 12-hour format: 3pm, 3:30pm, 3:30 PM
        let time_12h = regex::Regex::new(r"(?i)\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b")
            .expect("Invalid regex");
        for cap in time_12h.captures_iter(text) {
            if let Ok(mut hour) = cap[1].parse::<u32>() {
                let minute = cap
                    .get(2)
                    .and_then(|m| m.as_str().parse::<u32>().ok())
                    .unwrap_or(0);
                let period = cap[3].to_lowercase();

                // Convert to 24-hour
                if period.starts_with('p') && hour != 12 {
                    hour += 12;
                } else if period.starts_with('a') && hour == 12 {
                    hour = 0;
                }

                if let Some(time) = NaiveTime::from_hms_opt(hour, minute, 0) {
                    let full_match = cap.get(0).unwrap();
                    results.push(TemporalExtraction {
                        original_text: full_match.as_str().to_string(),
                        normalized_text: time.format("%H:%M").to_string(),
                        parsed_date: ParsedDate::from_time(time),
                        date_type: DateType::TimeOnly,
                        confidence: 0.95,
                        start_offset: full_match.start(),
                        end_offset: full_match.end(),
                        recurrence: None,
                        end_date: None,
                    });
                }
            }
        }

        // 24-hour format: 15:30, 09:00
        let time_24h = regex::Regex::new(r"\b([01]?\d|2[0-3]):([0-5]\d)\b").expect("Invalid regex");
        for cap in time_24h.captures_iter(text) {
            if let (Ok(hour), Ok(minute)) = (cap[1].parse::<u32>(), cap[2].parse::<u32>()) {
                if let Some(time) = NaiveTime::from_hms_opt(hour, minute, 0) {
                    let full_match = cap.get(0).unwrap();
                    results.push(TemporalExtraction {
                        original_text: full_match.as_str().to_string(),
                        normalized_text: time.format("%H:%M").to_string(),
                        parsed_date: ParsedDate::from_time(time),
                        date_type: DateType::TimeOnly,
                        confidence: 0.85,
                        start_offset: full_match.start(),
                        end_offset: full_match.end(),
                        recurrence: None,
                        end_date: None,
                    });
                }
            }
        }

        // Named times: noon, midnight, morning, afternoon, evening
        let named_times = [
            ("noon", 12, 0),
            ("midday", 12, 0),
            ("midnight", 0, 0),
            ("morning", 9, 0),
            ("afternoon", 14, 0),
            ("evening", 18, 0),
            ("night", 21, 0),
            ("eod", 17, 0), // End of day
            ("cob", 17, 0), // Close of business
            ("eob", 17, 0), // End of business
            ("end of day", 17, 0),
        ];

        let text_lower = text.to_lowercase();
        for (name, hour, minute) in &named_times {
            if let Some(pos) = text_lower.find(name) {
                if let Some(time) = NaiveTime::from_hms_opt(*hour, *minute, 0) {
                    results.push(TemporalExtraction {
                        original_text: name.to_string(),
                        normalized_text: time.format("%H:%M").to_string(),
                        parsed_date: ParsedDate::from_time(time),
                        date_type: DateType::TimeOnly,
                        confidence: 0.8,
                        start_offset: pos,
                        end_offset: pos + name.len(),
                        recurrence: None,
                        end_date: None,
                    });
                }
            }
        }

        results
    }

    /// Parse deadlines like "by Friday", "due next week", "before Monday".
    fn parse_deadlines(&self, text: &str) -> Vec<TemporalExtraction> {
        let mut results = Vec::new();

        // "by [date expression]" or "due [date expression]" or "before [date expression]"
        let deadline_prefixes = ["by", "due", "before", "until", "deadline"];

        let weekday_names = [
            ("monday", Weekday::Mon),
            ("tuesday", Weekday::Tue),
            ("wednesday", Weekday::Wed),
            ("thursday", Weekday::Thu),
            ("friday", Weekday::Fri),
            ("saturday", Weekday::Sat),
            ("sunday", Weekday::Sun),
        ];

        for prefix in &deadline_prefixes {
            for (weekday_name, weekday) in &weekday_names {
                let pattern = regex::Regex::new(&format!(
                    r"(?i)\b{}\s+(?:next\s+)?{}\b",
                    prefix, weekday_name
                ))
                .expect("Invalid regex");

                for cap in pattern.captures_iter(text) {
                    let full_text = cap.get(0).unwrap().as_str();
                    let is_next = full_text.to_lowercase().contains("next");
                    let date = self.next_weekday(*weekday, is_next);

                    let full_match = cap.get(0).unwrap();
                    results.push(TemporalExtraction {
                        original_text: full_match.as_str().to_string(),
                        normalized_text: format!("Deadline: {}", date.format("%Y-%m-%d (%A)")),
                        parsed_date: ParsedDate::from_date(date),
                        date_type: DateType::Deadline,
                        confidence: 0.9,
                        start_offset: full_match.start(),
                        end_offset: full_match.end(),
                        recurrence: None,
                        end_date: None,
                    });
                }
            }

            // "by end of [period]"
            let eod_pattern = regex::Regex::new(&format!(
                r"(?i)\b{}\s+end\s+of\s+(?:the\s+)?(week|month|quarter|year|day)\b",
                prefix
            ))
            .expect("Invalid regex");

            for cap in eod_pattern.captures_iter(text) {
                let period = cap[1].to_lowercase();
                let date = match period.as_str() {
                    "day" => self.reference_date,
                    "week" => self.end_of_week(),
                    "month" => self.end_of_month(),
                    "quarter" => self.end_of_quarter(),
                    "year" => self.end_of_year(),
                    _ => continue,
                };

                let full_match = cap.get(0).unwrap();
                results.push(TemporalExtraction {
                    original_text: full_match.as_str().to_string(),
                    normalized_text: format!("Deadline: {}", date.format("%Y-%m-%d")),
                    parsed_date: ParsedDate::from_date(date),
                    date_type: DateType::Deadline,
                    confidence: 0.9,
                    start_offset: full_match.start(),
                    end_offset: full_match.end(),
                    recurrence: None,
                    end_date: None,
                });
            }

            // "by tomorrow" / "due today"
            let simple_pattern =
                regex::Regex::new(&format!(r"(?i)\b{}\s+(today|tomorrow|yesterday)\b", prefix))
                    .expect("Invalid regex");

            for cap in simple_pattern.captures_iter(text) {
                let term = cap[1].to_lowercase();
                let days = match term.as_str() {
                    "today" => 0,
                    "tomorrow" => 1,
                    "yesterday" => -1,
                    _ => continue,
                };
                let date = self.reference_date + Duration::days(days);

                let full_match = cap.get(0).unwrap();
                results.push(TemporalExtraction {
                    original_text: full_match.as_str().to_string(),
                    normalized_text: format!("Deadline: {}", date.format("%Y-%m-%d")),
                    parsed_date: ParsedDate::from_date(date),
                    date_type: DateType::Deadline,
                    confidence: 0.95,
                    start_offset: full_match.start(),
                    end_offset: full_match.end(),
                    recurrence: None,
                    end_date: None,
                });
            }
        }

        results
    }

    /// Parse recurring patterns like "every Monday", "weekly", "monthly".
    fn parse_recurring(&self, text: &str) -> Vec<TemporalExtraction> {
        let mut results = Vec::new();

        let weekday_names = [
            ("monday", Weekday::Mon),
            ("tuesday", Weekday::Tue),
            ("wednesday", Weekday::Wed),
            ("thursday", Weekday::Thu),
            ("friday", Weekday::Fri),
            ("saturday", Weekday::Sat),
            ("sunday", Weekday::Sun),
        ];

        // "every [weekday]"
        for (name, weekday) in &weekday_names {
            let pattern =
                regex::Regex::new(&format!(r"(?i)\bevery\s+{}\b", name)).expect("Invalid regex");

            for cap in pattern.captures_iter(text) {
                let next_occurrence = self.next_weekday(*weekday, false);
                let full_match = cap.get(0).unwrap();
                results.push(TemporalExtraction {
                    original_text: full_match.as_str().to_string(),
                    normalized_text: format!("Weekly on {}", name),
                    parsed_date: ParsedDate::from_date(next_occurrence),
                    date_type: DateType::Recurring,
                    confidence: 0.9,
                    start_offset: full_match.start(),
                    end_offset: full_match.end(),
                    recurrence: Some(RecurrenceRule {
                        pattern: RecurrencePattern::Weekly,
                        interval: 1,
                        days_of_week: vec![*weekday],
                        day_of_month: None,
                        until: None,
                        count: None,
                    }),
                    end_date: None,
                });
            }
        }

        // "every [N] [days/weeks/months]"
        let every_n_pattern =
            regex::Regex::new(r"(?i)\bevery\s+(\d+)\s+(day|days|week|weeks|month|months)\b")
                .expect("Invalid regex");

        for cap in every_n_pattern.captures_iter(text) {
            if let Ok(interval) = cap[1].parse::<u32>() {
                let unit = cap[2].to_lowercase();
                let (pattern, next_date) = match unit.as_str() {
                    "day" | "days" => (
                        RecurrencePattern::Daily,
                        self.reference_date + Duration::days(interval as i64),
                    ),
                    "week" | "weeks" => (
                        RecurrencePattern::Weekly,
                        self.reference_date + Duration::weeks(interval as i64),
                    ),
                    "month" | "months" => (
                        RecurrencePattern::Monthly,
                        self.add_months(self.reference_date, interval as i32)
                            .unwrap_or(self.reference_date),
                    ),
                    _ => continue,
                };

                let full_match = cap.get(0).unwrap();
                results.push(TemporalExtraction {
                    original_text: full_match.as_str().to_string(),
                    normalized_text: format!("Every {} {}", interval, unit),
                    parsed_date: ParsedDate::from_date(next_date),
                    date_type: DateType::Recurring,
                    confidence: 0.9,
                    start_offset: full_match.start(),
                    end_offset: full_match.end(),
                    recurrence: Some(RecurrenceRule {
                        pattern,
                        interval,
                        days_of_week: vec![],
                        day_of_month: None,
                        until: None,
                        count: None,
                    }),
                    end_date: None,
                });
            }
        }

        // Simple frequency terms
        let frequency_terms = [
            ("daily", RecurrencePattern::Daily, 1u32),
            ("weekly", RecurrencePattern::Weekly, 1),
            ("biweekly", RecurrencePattern::BiWeekly, 2),
            ("bi-weekly", RecurrencePattern::BiWeekly, 2),
            ("monthly", RecurrencePattern::Monthly, 1),
            ("quarterly", RecurrencePattern::Quarterly, 3),
            ("yearly", RecurrencePattern::Yearly, 1),
            ("annually", RecurrencePattern::Yearly, 1),
        ];

        let text_lower = text.to_lowercase();
        for (term, pattern, interval) in &frequency_terms {
            if let Some(pos) = text_lower.find(term) {
                // Check it's not part of another word
                let before = if pos > 0 {
                    text_lower.chars().nth(pos - 1)
                } else {
                    None
                };
                let after = text_lower.chars().nth(pos + term.len());

                if before.is_none_or(|c| !c.is_alphanumeric())
                    && after.is_none_or(|c| !c.is_alphanumeric())
                {
                    results.push(TemporalExtraction {
                        original_text: term.to_string(),
                        normalized_text: term.to_string(),
                        parsed_date: ParsedDate::from_date(self.reference_date),
                        date_type: DateType::Recurring,
                        confidence: 0.85,
                        start_offset: pos,
                        end_offset: pos + term.len(),
                        recurrence: Some(RecurrenceRule {
                            pattern: *pattern,
                            interval: *interval,
                            days_of_week: vec![],
                            day_of_month: None,
                            until: None,
                            count: None,
                        }),
                        end_date: None,
                    });
                }
            }
        }

        results
    }

    // Helper methods

    /// Get the next occurrence of a weekday.
    fn next_weekday(&self, target: Weekday, skip_this_week: bool) -> NaiveDate {
        let current = self.reference_date.weekday();
        let current_num = current.num_days_from_monday();
        let target_num = target.num_days_from_monday();

        let days_ahead = if target_num > current_num {
            (target_num - current_num) as i64
        } else if target_num < current_num {
            (7 - current_num + target_num) as i64
        } else {
            // Same day
            if skip_this_week {
                7
            } else {
                0
            }
        };

        let days_ahead = if skip_this_week && days_ahead < 7 {
            days_ahead + 7
        } else {
            days_ahead
        };

        self.reference_date + Duration::days(days_ahead)
    }

    /// Get the end of the current week (Sunday).
    fn end_of_week(&self) -> NaiveDate {
        let days_until_sunday = 6 - self.reference_date.weekday().num_days_from_monday() as i64;
        self.reference_date + Duration::days(days_until_sunday)
    }

    /// Get the end of the current month.
    fn end_of_month(&self) -> NaiveDate {
        self.add_months(
            NaiveDate::from_ymd_opt(self.reference_date.year(), self.reference_date.month(), 1)
                .unwrap(),
            1,
        )
        .map(|d| d - Duration::days(1))
        .unwrap_or(self.reference_date)
    }

    /// Get the end of the current quarter.
    fn end_of_quarter(&self) -> NaiveDate {
        let month = self.reference_date.month();
        let quarter_end_month = ((month - 1) / 3 + 1) * 3;
        let year = self.reference_date.year();

        // Get last day of quarter end month
        self.add_months(
            NaiveDate::from_ymd_opt(year, quarter_end_month, 1).unwrap(),
            1,
        )
        .map(|d| d - Duration::days(1))
        .unwrap_or(self.reference_date)
    }

    /// Get the end of the current year.
    fn end_of_year(&self) -> NaiveDate {
        NaiveDate::from_ymd_opt(self.reference_date.year(), 12, 31).unwrap_or(self.reference_date)
    }

    /// Add months to a date.
    fn add_months(&self, date: NaiveDate, months: i32) -> Option<NaiveDate> {
        let total_months = date.year() * 12 + date.month() as i32 - 1 + months;
        let year = total_months / 12;
        let month = (total_months % 12 + 1) as u32;

        // Try the same day, or fall back to last day of month
        NaiveDate::from_ymd_opt(year, month, date.day())
            .or_else(|| NaiveDate::from_ymd_opt(year, month, 28))
            .or_else(|| NaiveDate::from_ymd_opt(year, month, 27))
            .or_else(|| NaiveDate::from_ymd_opt(year, month, 26))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Timelike;

    fn parser_at(year: i32, month: u32, day: u32) -> TemporalParser {
        TemporalParser::with_reference_date(NaiveDate::from_ymd_opt(year, month, day).unwrap())
    }

    #[test]
    fn test_parse_iso_date() {
        let parser = TemporalParser::new();
        let text = "The meeting is on 2024-01-15.";
        let results = parser.parse(text);

        assert!(!results.is_empty());
        let result = &results[0];
        assert_eq!(result.original_text, "2024-01-15");
        assert_eq!(result.date_type, DateType::Specific);
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_parse_month_day_year() {
        let parser = TemporalParser::new();
        let text = "Submit report by January 15, 2024";
        let results = parser.parse(text);

        let date_results: Vec<_> = results
            .iter()
            .filter(|r| r.date_type == DateType::Specific)
            .collect();
        assert!(!date_results.is_empty());
    }

    #[test]
    fn test_parse_tomorrow() {
        let parser = parser_at(2024, 1, 10);
        let text = "I'll do it tomorrow";
        let results = parser.parse(text);

        let tomorrow: Vec<_> = results
            .iter()
            .filter(|r| r.original_text == "tomorrow")
            .collect();
        assert!(!tomorrow.is_empty());

        let date = tomorrow[0].parsed_date.date.unwrap();
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 1, 11).unwrap());
    }

    #[test]
    fn test_parse_next_monday() {
        let parser = parser_at(2024, 1, 10); // Wednesday
        let text = "Let's meet next Monday";
        let results = parser.parse(text);

        let next_monday: Vec<_> = results
            .iter()
            .filter(|r| r.original_text.to_lowercase().contains("monday"))
            .collect();
        assert!(!next_monday.is_empty());

        let date = next_monday[0].parsed_date.date.unwrap();
        assert_eq!(date.weekday(), Weekday::Mon);
    }

    #[test]
    fn test_parse_time_12h() {
        let parser = TemporalParser::new();
        let text = "Meeting at 3:30 PM";
        let results = parser.parse(text);

        let times: Vec<_> = results
            .iter()
            .filter(|r| r.date_type == DateType::TimeOnly)
            .collect();
        assert!(!times.is_empty());

        let time = times[0].parsed_date.time.unwrap();
        assert_eq!(time.hour(), 15);
        assert_eq!(time.minute(), 30);
    }

    #[test]
    fn test_parse_time_24h() {
        let parser = TemporalParser::new();
        let text = "Call scheduled for 15:30";
        let results = parser.parse(text);

        let times: Vec<_> = results
            .iter()
            .filter(|r| r.date_type == DateType::TimeOnly)
            .collect();
        assert!(!times.is_empty());
    }

    #[test]
    fn test_parse_deadline() {
        let parser = parser_at(2024, 1, 10);
        let text = "Report due by Friday";
        let results = parser.parse(text);

        let deadlines: Vec<_> = results
            .iter()
            .filter(|r| r.date_type == DateType::Deadline)
            .collect();
        assert!(!deadlines.is_empty());
    }

    #[test]
    fn test_parse_recurring() {
        let parser = TemporalParser::new();
        let text = "Team standup every Monday";
        let results = parser.parse(text);

        let recurring: Vec<_> = results
            .iter()
            .filter(|r| r.date_type == DateType::Recurring)
            .collect();
        assert!(!recurring.is_empty());
        assert!(recurring[0].recurrence.is_some());
    }

    #[test]
    fn test_parse_end_of_month() {
        let parser = parser_at(2024, 1, 15);
        let text = "Complete by end of month";
        let results = parser.parse(text);

        let eom: Vec<_> = results
            .iter()
            .filter(|r| r.original_text.to_lowercase().contains("end of month"))
            .collect();
        assert!(!eom.is_empty());

        let date = eom[0].parsed_date.date.unwrap();
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 1, 31).unwrap());
    }

    #[test]
    fn test_parse_in_n_weeks() {
        let parser = parser_at(2024, 1, 10);
        let text = "Follow up in 2 weeks";
        let results = parser.parse(text);

        let in_weeks: Vec<_> = results
            .iter()
            .filter(|r| r.original_text.to_lowercase().contains("2 weeks"))
            .collect();
        assert!(!in_weeks.is_empty());

        let date = in_weeks[0].parsed_date.date.unwrap();
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 1, 24).unwrap());
    }

    #[test]
    fn test_parse_quarter() {
        let parser = parser_at(2024, 1, 15);
        let text = "Goals for Q2 2024";
        let results = parser.parse(text);

        let quarters: Vec<_> = results
            .iter()
            .filter(|r| r.original_text.to_uppercase().starts_with('Q'))
            .collect();
        assert!(!quarters.is_empty());
        assert_eq!(quarters[0].date_type, DateType::Range);
    }

    #[test]
    fn test_complex_text() {
        let parser = parser_at(2024, 1, 10);
        let text =
            "Meeting with John tomorrow at 3pm. TODO: Review the quarterly report by Friday. \
                    Weekly standup every Monday at 9am.";
        let results = parser.parse(text);

        // Should find multiple temporal expressions
        assert!(results.len() >= 3);
    }

    #[test]
    fn test_no_duplicates() {
        let parser = TemporalParser::new();
        let text = "Meeting at 3pm today";
        let results = parser.parse(text);

        // Check no overlapping results
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let a = &results[i];
                let b = &results[j];
                // Verify no overlap
                assert!(a.end_offset <= b.start_offset || b.end_offset <= a.start_offset);
            }
        }
    }
}

//! Types for the natural language query system.

use serde::{Deserialize, Serialize};

// ============================================================================
// Query Intent
// ============================================================================

/// High-level intent classification for natural language queries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    /// GTD-related queries (tasks, projects, contexts, etc.)
    Gtd(GtdIntent),
    /// Calendar-related queries
    Calendar(CalendarIntent),
    /// Knowledge graph queries
    Knowledge(KnowledgeIntent),
    /// General search queries
    Search,
    /// Unknown/ambiguous intent
    #[default]
    Unknown,
}

impl QueryIntent {
    /// Get a human-readable name for this intent.
    pub fn display_name(&self) -> &str {
        match self {
            Self::Gtd(_) => "GTD",
            Self::Calendar(_) => "Calendar",
            Self::Knowledge(_) => "Knowledge",
            Self::Search => "Search",
            Self::Unknown => "Unknown",
        }
    }

    /// Get more specific display name.
    pub fn detailed_name(&self) -> String {
        match self {
            Self::Gtd(intent) => format!("GTD: {}", intent.display_name()),
            Self::Calendar(intent) => format!("Calendar: {}", intent.display_name()),
            Self::Knowledge(intent) => format!("Knowledge: {}", intent.display_name()),
            Self::Search => "Document Search".to_string(),
            Self::Unknown => "Unknown".to_string(),
        }
    }
}

// ============================================================================
// GTD Intent
// ============================================================================

/// GTD-specific intents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GtdIntent {
    /// List tasks (with optional context filter)
    ListTasks,
    /// List tasks for a specific context (@phone, @home, etc.)
    TasksByContext { context: String },
    /// Get task recommendations
    RecommendTasks,
    /// Find quick wins (2-minute tasks)
    QuickWins,
    /// List overdue tasks
    OverdueTasks,
    /// List projects
    ListProjects,
    /// Get stalled projects
    StalledProjects,
    /// Project with no next action
    ProjectsWithoutNextAction,
    /// Get project health
    ProjectHealth,
    /// List waiting-for items
    ListWaiting,
    /// Overdue waiting-for items
    OverdueWaiting,
    /// Waiting for a specific person
    WaitingForPerson { person: String },
    /// List someday/maybe items
    ListSomeday,
    /// Weekly review
    WeeklyReview,
    /// Daily review
    DailyReview,
    /// What to work on now
    WhatNow,
    /// General GTD query
    General,
}

impl GtdIntent {
    pub fn display_name(&self) -> &str {
        match self {
            Self::ListTasks => "List Tasks",
            Self::TasksByContext { .. } => "Tasks by Context",
            Self::RecommendTasks => "Task Recommendations",
            Self::QuickWins => "Quick Wins",
            Self::OverdueTasks => "Overdue Tasks",
            Self::ListProjects => "List Projects",
            Self::StalledProjects => "Stalled Projects",
            Self::ProjectsWithoutNextAction => "Projects Without Next Action",
            Self::ProjectHealth => "Project Health",
            Self::ListWaiting => "Waiting For",
            Self::OverdueWaiting => "Overdue Waiting",
            Self::WaitingForPerson { .. } => "Waiting for Person",
            Self::ListSomeday => "Someday/Maybe",
            Self::WeeklyReview => "Weekly Review",
            Self::DailyReview => "Daily Review",
            Self::WhatNow => "What to Work On",
            Self::General => "General GTD",
        }
    }
}

// ============================================================================
// Calendar Intent
// ============================================================================

/// Calendar-specific intents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalendarIntent {
    /// What's on my calendar today
    Today,
    /// What's happening this week
    ThisWeek,
    /// Upcoming events (next N days)
    Upcoming { days: Option<u32> },
    /// Find free time
    FreeTime,
    /// Check for conflicts
    Conflicts,
    /// Events for a specific date
    SpecificDate { date_expr: String },
    /// Events with a specific person
    EventsWithPerson { person: String },
    /// General calendar query
    General,
}

impl CalendarIntent {
    pub fn display_name(&self) -> &str {
        match self {
            Self::Today => "Today's Events",
            Self::ThisWeek => "This Week",
            Self::Upcoming { .. } => "Upcoming Events",
            Self::FreeTime => "Free Time",
            Self::Conflicts => "Scheduling Conflicts",
            Self::SpecificDate { .. } => "Events on Date",
            Self::EventsWithPerson { .. } => "Events with Person",
            Self::General => "General Calendar",
        }
    }
}

// ============================================================================
// Knowledge Intent
// ============================================================================

/// Knowledge graph-specific intents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeIntent {
    /// What do I know about X
    WhatDoIKnowAbout { topic: String },
    /// Who is an expert on X
    FindExpert { topic: String },
    /// Who works with person X
    WhoWorksWith { person: String },
    /// Summarize topic X
    TopicSummary { topic: String },
    /// Find entity by name
    EntityLookup { name: String },
    /// Traverse relationships from entity
    RelationshipQuery { from: String },
    /// General knowledge query
    General { query: String },
}

impl KnowledgeIntent {
    pub fn display_name(&self) -> &str {
        match self {
            Self::WhatDoIKnowAbout { .. } => "Knowledge Search",
            Self::FindExpert { .. } => "Find Expert",
            Self::WhoWorksWith { .. } => "Relationship Query",
            Self::TopicSummary { .. } => "Topic Summary",
            Self::EntityLookup { .. } => "Entity Lookup",
            Self::RelationshipQuery { .. } => "Relationship Traversal",
            Self::General { .. } => "General Knowledge",
        }
    }
}

// ============================================================================
// Classification Result
// ============================================================================

/// Result of query intent classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Classified intent
    pub intent: QueryIntent,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Extracted parameters from the query
    pub extracted_params: ExtractedParams,
    /// Alternative interpretations
    pub alternatives: Vec<(QueryIntent, f32)>,
}

impl ClassificationResult {
    pub fn new(intent: QueryIntent, confidence: f32) -> Self {
        Self {
            intent,
            confidence,
            extracted_params: ExtractedParams::default(),
            alternatives: Vec::new(),
        }
    }

    pub fn with_params(mut self, params: ExtractedParams) -> Self {
        self.extracted_params = params;
        self
    }

    pub fn with_alternatives(mut self, alternatives: Vec<(QueryIntent, f32)>) -> Self {
        self.alternatives = alternatives;
        self
    }
}

// ============================================================================
// Extracted Parameters
// ============================================================================

/// Parameters extracted from natural language queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractedParams {
    /// Context mentions (@phone, @home, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contexts: Vec<String>,
    /// Person names mentioned
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub people: Vec<String>,
    /// Topics/subjects mentioned
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub topics: Vec<String>,
    /// Project names mentioned
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub projects: Vec<String>,
    /// Date expressions (today, tomorrow, next week, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_expression: Option<String>,
    /// Time duration (e.g., "30 minutes", "2 hours")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,
    /// Energy level mentioned
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_level: Option<String>,
    /// Priority mentioned
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    /// Area of focus mentioned
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area: Option<String>,
    /// Limit/count extracted
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// Raw query terms not classified
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub raw_terms: Vec<String>,
}

impl ExtractedParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.contexts.push(context.into());
        self
    }

    pub fn with_person(mut self, person: impl Into<String>) -> Self {
        self.people.push(person.into());
        self
    }

    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.topics.push(topic.into());
        self
    }

    pub fn with_date(mut self, date: impl Into<String>) -> Self {
        self.date_expression = Some(date.into());
        self
    }

    pub fn with_energy(mut self, energy: impl Into<String>) -> Self {
        self.energy_level = Some(energy.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

// ============================================================================
// Query Result
// ============================================================================

/// Unified query result that can hold results from any subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedQueryResult {
    /// The interpreted intent
    pub intent: QueryIntent,
    /// Confidence in the interpretation
    pub confidence: f32,
    /// Human-readable answer
    pub answer: String,
    /// Structured result data
    pub data: serde_json::Value,
    /// Suggestions for follow-up queries
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
    /// Execution statistics
    pub stats: QueryStats,
}

/// Query execution statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Classification time in milliseconds
    pub classification_time_ms: u64,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Total time in milliseconds
    pub total_time_ms: u64,
    /// Which subsystem was queried
    pub subsystem: String,
}

impl UnifiedQueryResult {
    pub fn new(intent: QueryIntent, answer: String, data: serde_json::Value) -> Self {
        Self {
            confidence: 1.0,
            intent,
            answer,
            data,
            suggestions: Vec::new(),
            stats: QueryStats::default(),
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions = suggestions;
        self
    }

    pub fn with_stats(mut self, stats: QueryStats) -> Self {
        self.stats = stats;
        self
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            intent: QueryIntent::Unknown,
            confidence: 0.0,
            answer: message.into(),
            data: serde_json::Value::Null,
            suggestions: Vec::new(),
            stats: QueryStats::default(),
        }
    }
}

// ============================================================================
// Query Mode
// ============================================================================

/// Mode for query processing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
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

impl QueryMode {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Auto => "auto",
            Self::Gtd => "gtd",
            Self::Calendar => "calendar",
            Self::Knowledge => "knowledge",
            Self::Search => "search",
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
    fn test_intent_display_name() {
        assert_eq!(QueryIntent::Search.display_name(), "Search");
        assert_eq!(QueryIntent::Gtd(GtdIntent::QuickWins).display_name(), "GTD");
    }

    #[test]
    fn test_extracted_params_builder() {
        let params = ExtractedParams::new()
            .with_context("@phone")
            .with_person("John")
            .with_topic("Rust")
            .with_date("tomorrow")
            .with_limit(10);

        assert_eq!(params.contexts, vec!["@phone"]);
        assert_eq!(params.people, vec!["John"]);
        assert_eq!(params.topics, vec!["Rust"]);
        assert_eq!(params.date_expression, Some("tomorrow".to_string()));
        assert_eq!(params.limit, Some(10));
    }

    #[test]
    fn test_classification_result() {
        let result = ClassificationResult::new(QueryIntent::Search, 0.95)
            .with_params(ExtractedParams::new().with_topic("testing"))
            .with_alternatives(vec![(QueryIntent::Unknown, 0.05)]);

        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.extracted_params.topics, vec!["testing"]);
        assert_eq!(result.alternatives.len(), 1);
    }
}

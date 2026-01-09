//! Query Intent Classifier.
//!
//! Classifies natural language queries into structured intents for
//! GTD, Calendar, Knowledge, or Search operations.

use std::sync::LazyLock;

use regex::Regex;

use super::types::*;

// ============================================================================
// Intent Classifier
// ============================================================================

/// Classifies natural language queries into structured intents.
pub struct IntentClassifier {
    /// Minimum confidence threshold for classification
    confidence_threshold: f32,
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl IntentClassifier {
    /// Create a new intent classifier with default settings.
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.3,
        }
    }

    /// Create a classifier with a custom confidence threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            confidence_threshold: threshold,
        }
    }

    /// Classify a natural language query into an intent.
    pub fn classify(&self, query: &str) -> ClassificationResult {
        let query_lower = query.to_lowercase();

        // Extract parameters from the query
        let params = self.extract_params(&query_lower);

        // Try each domain classifier
        let gtd_result = self.classify_gtd(&query_lower, &params);
        let calendar_result = self.classify_calendar(&query_lower, &params);
        let knowledge_result = self.classify_knowledge(&query_lower, &params);
        let search_result = self.classify_search(&query_lower, &params);

        // Collect all results and pick the best
        let mut results = vec![
            (gtd_result.0, gtd_result.1),
            (calendar_result.0, calendar_result.1),
            (knowledge_result.0, knowledge_result.1),
            (search_result.0, search_result.1),
        ];

        // Sort by confidence
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_intent, best_confidence) = results.remove(0);

        // Build alternatives list
        let alternatives: Vec<(QueryIntent, f32)> = results
            .into_iter()
            .filter(|(_, conf)| *conf >= self.confidence_threshold)
            .collect();

        ClassificationResult::new(best_intent, best_confidence)
            .with_params(params)
            .with_alternatives(alternatives)
    }

    /// Classify with a forced mode.
    pub fn classify_with_mode(&self, query: &str, mode: QueryMode) -> ClassificationResult {
        let query_lower = query.to_lowercase();
        let params = self.extract_params(&query_lower);

        match mode {
            QueryMode::Auto => self.classify(query),
            QueryMode::Gtd => {
                let (intent, confidence) = self.classify_gtd(&query_lower, &params);
                ClassificationResult::new(intent, confidence).with_params(params)
            }
            QueryMode::Calendar => {
                let (intent, confidence) = self.classify_calendar(&query_lower, &params);
                ClassificationResult::new(intent, confidence).with_params(params)
            }
            QueryMode::Knowledge => {
                let (intent, confidence) = self.classify_knowledge(&query_lower, &params);
                ClassificationResult::new(intent, confidence).with_params(params)
            }
            QueryMode::Search => {
                let (intent, confidence) = self.classify_search(&query_lower, &params);
                ClassificationResult::new(intent, confidence).with_params(params)
            }
        }
    }

    // ========================================================================
    // GTD Classification
    // ========================================================================

    fn classify_gtd(&self, query: &str, params: &ExtractedParams) -> (QueryIntent, f32) {
        // Context-based task queries (@phone, @home, etc.)
        if !params.contexts.is_empty() {
            let context = params.contexts[0].clone();
            return (
                QueryIntent::Gtd(GtdIntent::TasksByContext { context }),
                0.95,
            );
        }

        // Weekly review patterns
        if WEEKLY_REVIEW_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::WeeklyReview), 0.95);
        }

        // Daily review patterns
        if DAILY_REVIEW_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::DailyReview), 0.95);
        }

        // What should I work on / what now patterns
        if WHAT_NOW_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::WhatNow), 0.90);
        }

        // Task recommendation patterns
        if RECOMMEND_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::RecommendTasks), 0.90);
        }

        // Quick wins / 2-minute patterns
        if QUICK_WIN_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::QuickWins), 0.90);
        }

        // Overdue tasks
        if OVERDUE_TASK_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::OverdueTasks), 0.90);
        }

        // List tasks patterns
        if LIST_TASKS_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::ListTasks), 0.85);
        }

        // Stalled projects
        if STALLED_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::StalledProjects), 0.90);
        }

        // Projects without next action
        if NO_NEXT_ACTION_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::ProjectsWithoutNextAction), 0.90);
        }

        // Project health
        if PROJECT_HEALTH_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::ProjectHealth), 0.85);
        }

        // List projects
        if PROJECT_LIST_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::ListProjects), 0.85);
        }

        // Waiting for patterns
        if !params.people.is_empty() && WAITING_PATTERN.is_match(query) {
            let person = params.people[0].clone();
            return (
                QueryIntent::Gtd(GtdIntent::WaitingForPerson { person }),
                0.90,
            );
        }

        // Overdue waiting
        if OVERDUE_WAITING_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::OverdueWaiting), 0.90);
        }

        // List waiting
        if WAITING_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::ListWaiting), 0.85);
        }

        // Someday/maybe
        if SOMEDAY_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::ListSomeday), 0.85);
        }

        // General GTD indicators
        if GTD_GENERAL_PATTERN.is_match(query) {
            return (QueryIntent::Gtd(GtdIntent::General), 0.60);
        }

        (QueryIntent::Gtd(GtdIntent::General), 0.20)
    }

    // ========================================================================
    // Calendar Classification
    // ========================================================================

    fn classify_calendar(&self, query: &str, params: &ExtractedParams) -> (QueryIntent, f32) {
        // Today's calendar
        if TODAY_CALENDAR_PATTERN.is_match(query) {
            return (QueryIntent::Calendar(CalendarIntent::Today), 0.95);
        }

        // Free time (check before this week since "free this week" should match free time)
        if FREE_TIME_PATTERN.is_match(query) {
            return (QueryIntent::Calendar(CalendarIntent::FreeTime), 0.90);
        }

        // This week
        if THIS_WEEK_PATTERN.is_match(query) {
            return (QueryIntent::Calendar(CalendarIntent::ThisWeek), 0.90);
        }

        // Upcoming events
        if UPCOMING_PATTERN.is_match(query) {
            // Try to extract number of days
            let days = extract_days(query);
            return (
                QueryIntent::Calendar(CalendarIntent::Upcoming { days }),
                0.85,
            );
        }

        // Conflicts
        if CONFLICT_PATTERN.is_match(query) {
            return (QueryIntent::Calendar(CalendarIntent::Conflicts), 0.90);
        }

        // Events with a specific person
        if !params.people.is_empty() && MEETING_WITH_PATTERN.is_match(query) {
            let person = params.people[0].clone();
            return (
                QueryIntent::Calendar(CalendarIntent::EventsWithPerson { person }),
                0.85,
            );
        }

        // Specific date expression
        if let Some(ref date_expr) = params.date_expression {
            if CALENDAR_GENERAL_PATTERN.is_match(query) {
                return (
                    QueryIntent::Calendar(CalendarIntent::SpecificDate {
                        date_expr: date_expr.clone(),
                    }),
                    0.80,
                );
            }
        }

        // General calendar indicators
        if CALENDAR_GENERAL_PATTERN.is_match(query) {
            return (QueryIntent::Calendar(CalendarIntent::General), 0.60);
        }

        (QueryIntent::Calendar(CalendarIntent::General), 0.15)
    }

    // ========================================================================
    // Knowledge Classification
    // ========================================================================

    fn classify_knowledge(&self, query: &str, params: &ExtractedParams) -> (QueryIntent, f32) {
        // What do I know about patterns
        if let Some(caps) = WHAT_KNOW_PATTERN.captures(query) {
            let topic = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_else(|| params.topics.first().cloned().unwrap_or_default());
            if !topic.is_empty() {
                return (
                    QueryIntent::Knowledge(KnowledgeIntent::WhatDoIKnowAbout { topic }),
                    0.90,
                );
            }
        }

        // Expert finding patterns
        if let Some(caps) = EXPERT_PATTERN.captures(query) {
            let topic = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_else(|| params.topics.first().cloned().unwrap_or_default());
            if !topic.is_empty() {
                return (
                    QueryIntent::Knowledge(KnowledgeIntent::FindExpert { topic }),
                    0.90,
                );
            }
        }

        // Who works with patterns
        if let Some(caps) = WHO_WORKS_WITH_PATTERN.captures(query) {
            let person = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_else(|| params.people.first().cloned().unwrap_or_default());
            if !person.is_empty() {
                return (
                    QueryIntent::Knowledge(KnowledgeIntent::WhoWorksWith { person }),
                    0.90,
                );
            }
        }

        // Topic summary patterns
        if TOPIC_SUMMARY_PATTERN.is_match(query) {
            if let Some(topic) = params.topics.first().cloned() {
                return (
                    QueryIntent::Knowledge(KnowledgeIntent::TopicSummary { topic }),
                    0.85,
                );
            }
        }

        // Entity lookup (specific person/organization)
        if !params.people.is_empty() && ENTITY_LOOKUP_PATTERN.is_match(query) {
            let name = params.people[0].clone();
            return (
                QueryIntent::Knowledge(KnowledgeIntent::EntityLookup { name }),
                0.80,
            );
        }

        // Relationship query
        if RELATIONSHIP_PATTERN.is_match(query) {
            if let Some(person) = params.people.first().cloned() {
                return (
                    QueryIntent::Knowledge(KnowledgeIntent::RelationshipQuery { from: person }),
                    0.80,
                );
            }
        }

        // General knowledge indicators
        if KNOWLEDGE_GENERAL_PATTERN.is_match(query) {
            let topic = params
                .topics
                .first()
                .cloned()
                .unwrap_or_else(|| query.to_string());
            return (
                QueryIntent::Knowledge(KnowledgeIntent::General { query: topic }),
                0.50,
            );
        }

        let topic = params
            .topics
            .first()
            .cloned()
            .unwrap_or_else(|| query.to_string());
        (
            QueryIntent::Knowledge(KnowledgeIntent::General { query: topic }),
            0.10,
        )
    }

    // ========================================================================
    // Search Classification
    // ========================================================================

    fn classify_search(&self, query: &str, _params: &ExtractedParams) -> (QueryIntent, f32) {
        // Search patterns
        if SEARCH_PATTERN.is_match(query) {
            return (QueryIntent::Search, 0.85);
        }

        // Find document patterns
        if FIND_DOC_PATTERN.is_match(query) {
            return (QueryIntent::Search, 0.80);
        }

        // Default fallback - if nothing else matches well
        (QueryIntent::Search, 0.25)
    }

    // ========================================================================
    // Parameter Extraction
    // ========================================================================

    /// Extract parameters from a query string.
    fn extract_params(&self, query: &str) -> ExtractedParams {
        let mut params = ExtractedParams::default();

        // Extract contexts (@phone, @home, etc.)
        for cap in CONTEXT_PATTERN.find_iter(query) {
            params.contexts.push(cap.as_str().to_string());
        }

        // Extract date expressions
        if let Some(cap) = DATE_EXPR_PATTERN.find(query) {
            params.date_expression = Some(cap.as_str().to_string());
        }

        // Extract energy level
        if let Some(cap) = ENERGY_PATTERN.find(query) {
            params.energy_level = Some(cap.as_str().to_string());
        }

        // Extract duration
        if let Some(cap) = DURATION_PATTERN.find(query) {
            params.duration = Some(cap.as_str().to_string());
        }

        // Extract limit/count
        if let Some(caps) = LIMIT_PATTERN.captures(query) {
            if let Some(num) = caps.get(1).and_then(|m| m.as_str().parse().ok()) {
                params.limit = Some(num);
            }
        }

        // Extract topics (words after "about", "on", "regarding")
        if let Some(caps) = TOPIC_EXTRACT_PATTERN.captures(query) {
            if let Some(topic) = caps.get(1) {
                params.topics.push(topic.as_str().trim().to_string());
            }
        }

        // Extract person names (basic - after "with", "from", "by")
        if let Some(caps) = PERSON_EXTRACT_PATTERN.captures(query) {
            if let Some(person) = caps.get(1) {
                let name = person.as_str().trim();
                // Filter out common non-names
                if !is_common_word(name) {
                    params.people.push(name.to_string());
                }
            }
        }

        params
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn extract_days(query: &str) -> Option<u32> {
    if let Some(caps) = DAYS_PATTERN.captures(query) {
        caps.get(1).and_then(|m| m.as_str().parse().ok())
    } else {
        None
    }
}

fn is_common_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the"
            | "a"
            | "an"
            | "and"
            | "or"
            | "but"
            | "in"
            | "on"
            | "at"
            | "to"
            | "for"
            | "of"
            | "it"
            | "is"
            | "that"
            | "this"
    )
}

// ============================================================================
// Regex Patterns (using LazyLock for static initialization)
// ============================================================================

// Context patterns
static CONTEXT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"@\w+").expect("Invalid regex"));

// GTD patterns
static WEEKLY_REVIEW_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)weekly\s+review").expect("Invalid regex"));
static DAILY_REVIEW_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)daily\s+review").expect("Invalid regex"));
static WHAT_NOW_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(what\s+(should\s+i|can\s+i)\s+(work\s+on|do)|what('s)?\s+next|what\s+now|what\s+to\s+do)")
        .expect("Invalid regex")
});
static RECOMMEND_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(recommend|suggest|propose)\s+(tasks?|actions?|work)").expect("Invalid regex")
});
static QUICK_WIN_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(quick\s+wins?|2[\s-]?minute|two[\s-]?minute|fast\s+tasks?|quick\s+tasks?)")
        .expect("Invalid regex")
});
static OVERDUE_TASK_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(overdue|past\s+due|late)\s+(tasks?|items?)").expect("Invalid regex")
});
static LIST_TASKS_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(list|show|get|my)\s*(all\s+)?(tasks?|actions?|next\s+actions?)")
        .expect("Invalid regex")
});
static STALLED_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(stalled|stuck|inactive|dormant)\s+projects?").expect("Invalid regex")
});
static NO_NEXT_ACTION_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)projects?\s+(with(out)?|no|missing)\s+(next\s+)?actions?")
        .expect("Invalid regex")
});
static PROJECT_HEALTH_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)project\s+(health|status|score)").expect("Invalid regex"));
static PROJECT_LIST_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(list|show|get|my)\s*(all\s+)?projects?").expect("Invalid regex")
});
static WAITING_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)waiting\s+(for|on)|delegated").expect("Invalid regex"));
static OVERDUE_WAITING_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(overdue|past\s+due|late)\s+(waiting|delegated)").expect("Invalid regex")
});
static SOMEDAY_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(someday|maybe|deferred|later)").expect("Invalid regex"));
static GTD_GENERAL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(task|project|action|next\s+action|gtd|todo)").expect("Invalid regex")
});

// Calendar patterns
static TODAY_CALENDAR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(what('s)?|show)\s+(on\s+my\s+)?(calendar\s+)?(today|for\s+today)")
        .expect("Invalid regex")
});
static THIS_WEEK_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(this\s+week|week('s)?\s+(calendar|schedule|events?))").expect("Invalid regex")
});
static UPCOMING_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(upcoming|next|future)\s+(events?|meetings?|appointments?|calendar)")
        .expect("Invalid regex")
});
static FREE_TIME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(free\s+time|available|availability|open\s+slots?|when\s+am\s+i\s+free)")
        .expect("Invalid regex")
});
static CONFLICT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(conflicts?|overlapping|double[\s-]?booked|scheduling\s+conflicts?)")
        .expect("Invalid regex")
});
static MEETING_WITH_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(meeting|event|appointment)\s+with").expect("Invalid regex"));
static CALENDAR_GENERAL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(calendar|schedule|event|meeting|appointment)").expect("Invalid regex")
});

// Knowledge patterns
static WHAT_KNOW_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)what\s+(do\s+i\s+)?know\s+about\s+(.+)").expect("Invalid regex")
});
static EXPERT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(who\s+(knows?|is\s+expert|can\s+help)|find\s+expert|expert\s+on)\s+(.+)")
        .expect("Invalid regex")
});
static WHO_WORKS_WITH_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)who\s+works\s+with\s+(.+)").expect("Invalid regex"));
static TOPIC_SUMMARY_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(summarize|summary|overview)\s+(of\s+)?").expect("Invalid regex")
});
static ENTITY_LOOKUP_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(who\s+is|find|lookup|look\s+up|info\s+(on|about))").expect("Invalid regex")
});
static RELATIONSHIP_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(related\s+to|connected|relationships?|connections?)").expect("Invalid regex")
});
static KNOWLEDGE_GENERAL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(know|knowledge|information|info|about)").expect("Invalid regex")
});

// Search patterns
static SEARCH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(search|find|look\s+for|locate)\s+(in\s+)?(documents?|files?|content)")
        .expect("Invalid regex")
});
static FIND_DOC_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(find|where\s+is|search\s+for)\s+").expect("Invalid regex"));

// Extraction patterns
static DATE_EXPR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(today|tomorrow|yesterday|next\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|this\s+(week|month)|in\s+\d+\s+(days?|weeks?|months?))")
        .expect("Invalid regex")
});
static ENERGY_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(low|medium|high)\s+energy").expect("Invalid regex"));
static DURATION_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(\d+)\s*(minutes?|mins?|hours?|hrs?)").expect("Invalid regex")
});
static LIMIT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(top|first|last|limit)\s+(\d+)").expect("Invalid regex"));
static DAYS_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(\d+)\s*days?").expect("Invalid regex"));
static TOPIC_EXTRACT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:about|on|regarding|concerning)\s+([^?.,]+)").expect("Invalid regex")
});
static PERSON_EXTRACT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:with|from|by|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
        .expect("Invalid regex")
});

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_classification() {
        let classifier = IntentClassifier::new();

        let result = classifier.classify("What are my @phone tasks?");
        assert!(matches!(
            result.intent,
            QueryIntent::Gtd(GtdIntent::TasksByContext { .. })
        ));
        assert!(result.confidence > 0.9);

        let result = classifier.classify("Show @home actions");
        assert!(matches!(
            result.intent,
            QueryIntent::Gtd(GtdIntent::TasksByContext { .. })
        ));
    }

    #[test]
    fn test_weekly_review() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Run my weekly review");
        assert!(matches!(
            result.intent,
            QueryIntent::Gtd(GtdIntent::WeeklyReview)
        ));
    }

    #[test]
    fn test_what_now() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("What should I work on now?");
        assert!(matches!(
            result.intent,
            QueryIntent::Gtd(GtdIntent::WhatNow)
        ));
    }

    #[test]
    fn test_quick_wins() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Show me quick wins");
        assert!(matches!(
            result.intent,
            QueryIntent::Gtd(GtdIntent::QuickWins)
        ));

        let result = classifier.classify("2-minute tasks");
        assert!(matches!(
            result.intent,
            QueryIntent::Gtd(GtdIntent::QuickWins)
        ));
    }

    #[test]
    fn test_calendar_today() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("What's on my calendar today?");
        assert!(matches!(
            result.intent,
            QueryIntent::Calendar(CalendarIntent::Today)
        ));
    }

    #[test]
    fn test_free_time() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("When am I free this week?");
        assert!(matches!(
            result.intent,
            QueryIntent::Calendar(CalendarIntent::FreeTime)
        ));
    }

    #[test]
    fn test_expert_finding() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Who knows about Kubernetes?");
        assert!(matches!(
            result.intent,
            QueryIntent::Knowledge(KnowledgeIntent::FindExpert { .. })
        ));
    }

    #[test]
    fn test_what_do_i_know() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("What do I know about machine learning?");
        assert!(matches!(
            result.intent,
            QueryIntent::Knowledge(KnowledgeIntent::WhatDoIKnowAbout { .. })
        ));
    }

    #[test]
    fn test_stalled_projects() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Show stalled projects");
        assert!(matches!(
            result.intent,
            QueryIntent::Gtd(GtdIntent::StalledProjects)
        ));
    }

    #[test]
    fn test_waiting_for() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("What am I waiting for?");
        assert!(matches!(
            result.intent,
            QueryIntent::Gtd(GtdIntent::ListWaiting)
        ));
    }

    #[test]
    fn test_param_extraction() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Show my @phone tasks about sales");

        assert!(result
            .extracted_params
            .contexts
            .contains(&"@phone".to_string()));
    }
}

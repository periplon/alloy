//! Query Executor.
//!
//! Executes classified queries against the appropriate subsystem
//! (GTD, Calendar, Knowledge, or Search).

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use crate::calendar::{CalendarFilter, CalendarManager};
use crate::coordinator::IndexCoordinator;
use crate::embedding::EmbeddingProvider;
use crate::error::Result;
use crate::gtd::review::{ReviewManager, WeeklyReviewParams};
use crate::gtd::{
    EnergyLevel, ProjectFilter, ProjectManager, ProjectStatus, RecommendParams, SomedayFilter,
    SomedayManager, TaskFilter, TaskManager, TaskStatus, WaitingFilter, WaitingManager,
};
use crate::knowledge::KnowledgeQueryEngine;
use crate::ontology::{EmbeddedOntologyStore, OntologyStore};
use crate::search::HybridQuery;

use super::classifier::IntentClassifier;
use super::types::*;

// ============================================================================
// Query Executor
// ============================================================================

/// Executes natural language queries across all subsystems.
pub struct QueryExecutor {
    /// Intent classifier
    classifier: IntentClassifier,
    /// Ontology store for GTD and Knowledge queries
    ontology_store: Arc<RwLock<EmbeddedOntologyStore>>,
    /// Embedding provider for semantic operations
    embedder: Arc<dyn EmbeddingProvider>,
    /// Index coordinator for search operations
    coordinator: Arc<IndexCoordinator>,
}

impl QueryExecutor {
    /// Create a new query executor.
    pub fn new(
        ontology_store: Arc<RwLock<EmbeddedOntologyStore>>,
        embedder: Arc<dyn EmbeddingProvider>,
        coordinator: Arc<IndexCoordinator>,
    ) -> Self {
        Self {
            classifier: IntentClassifier::new(),
            ontology_store,
            embedder,
            coordinator,
        }
    }

    /// Execute a natural language query.
    pub async fn execute(&self, query: &str, mode: QueryMode) -> Result<UnifiedQueryResult> {
        let total_start = Instant::now();

        // Classify the query
        let class_start = Instant::now();
        let classification = match mode {
            QueryMode::Auto => self.classifier.classify(query),
            _ => self.classifier.classify_with_mode(query, mode),
        };
        let classification_time = class_start.elapsed().as_millis() as u64;

        // Clone before destructuring to avoid partial move
        let intent_clone = classification.intent.clone();
        let params_ref = &classification.extracted_params;

        // Execute based on intent
        let exec_start = Instant::now();
        let mut result = match &intent_clone {
            QueryIntent::Gtd(intent) => {
                self.execute_gtd(intent, params_ref).await?
            }
            QueryIntent::Calendar(intent) => {
                self.execute_calendar(intent, params_ref).await?
            }
            QueryIntent::Knowledge(intent) => {
                self.execute_knowledge(intent, params_ref).await?
            }
            QueryIntent::Search => self.execute_search(query).await?,
            QueryIntent::Unknown => UnifiedQueryResult::error(
                "Could not understand your query. Try being more specific about what you're looking for.",
            ),
        };
        let execution_time = exec_start.elapsed().as_millis() as u64;

        // Add stats
        result.stats = QueryStats {
            classification_time_ms: classification_time,
            execution_time_ms: execution_time,
            total_time_ms: total_start.elapsed().as_millis() as u64,
            subsystem: intent_clone.display_name().to_string(),
        };
        result.confidence = classification.confidence;
        result.intent = intent_clone;

        // Add suggestions based on alternatives
        result.suggestions = self.generate_suggestions(&classification);

        Ok(result)
    }

    // ========================================================================
    // GTD Execution
    // ========================================================================

    async fn execute_gtd(
        &self,
        intent: &GtdIntent,
        params: &ExtractedParams,
    ) -> Result<UnifiedQueryResult> {
        let store = self.ontology_store.clone();
        let limit = params.limit.unwrap_or(20);

        match intent {
            GtdIntent::ListTasks => {
                let manager = TaskManager::new(store);
                let filter = TaskFilter {
                    limit,
                    ..Default::default()
                };
                let tasks = manager.list(filter).await?;
                let count = tasks.len();
                let answer = format!("Found {} tasks.", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&tasks)?,
                ))
            }

            GtdIntent::TasksByContext { context } => {
                let manager = TaskManager::new(store);
                let filter = TaskFilter {
                    contexts: vec![context.clone()],
                    status: Some(TaskStatus::Next),
                    limit,
                    ..Default::default()
                };
                let tasks = manager.list(filter).await?;
                let count = tasks.len();
                let answer = format!("Found {} tasks in context {}.", count, context);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&tasks)?,
                ))
            }

            GtdIntent::RecommendTasks | GtdIntent::WhatNow => {
                let manager = TaskManager::new(store);
                let mut rec_params = RecommendParams::default();

                if let Some(ctx) = params.contexts.first() {
                    rec_params.current_context = Some(ctx.clone());
                }
                if let Some(ref energy) = params.energy_level {
                    rec_params.energy_level = Some(parse_energy_level(energy));
                }
                if let Some(ref duration) = params.duration {
                    rec_params.time_available = parse_duration(duration);
                }
                rec_params.limit = limit;

                let recs = manager.recommend(rec_params).await?;
                let count = recs.len();
                let top_task = recs
                    .first()
                    .map(|r| r.task.description.as_str())
                    .unwrap_or("none");
                let answer = format!(
                    "I recommend {} tasks. Top recommendation: {}",
                    count, top_task
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&recs)?,
                ))
            }

            GtdIntent::QuickWins => {
                let manager = TaskManager::new(store);
                let mut tasks = manager.get_quick_wins().await?;
                tasks.truncate(limit);
                let count = tasks.len();
                let answer = format!("Found {} quick wins (tasks under 2 minutes).", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&tasks)?,
                ))
            }

            GtdIntent::OverdueTasks => {
                let manager = TaskManager::new(store);
                let mut tasks = manager.get_overdue().await?;
                tasks.truncate(limit);
                let count = tasks.len();
                let answer = if count == 0 {
                    "No overdue tasks. Great job staying on top of things!".to_string()
                } else {
                    format!("Found {} overdue tasks that need attention.", count)
                };
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&tasks)?,
                ))
            }

            GtdIntent::ListProjects => {
                let manager = ProjectManager::new(store);
                let filter = ProjectFilter {
                    status: Some(ProjectStatus::Active),
                    limit,
                    ..Default::default()
                };
                let projects = manager.list(filter).await?;
                let count = projects.len();
                let answer = format!("Found {} active projects.", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&projects)?,
                ))
            }

            GtdIntent::StalledProjects => {
                let manager = ProjectManager::new(store);
                let filter = ProjectFilter {
                    stalled_days: Some(7),
                    limit,
                    ..Default::default()
                };
                let projects = manager.list(filter).await?;
                let count = projects.len();
                let answer = if count == 0 {
                    "No stalled projects. All projects have recent activity.".to_string()
                } else {
                    format!("Found {} stalled projects (no activity in 7+ days).", count)
                };
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&projects)?,
                ))
            }

            GtdIntent::ProjectsWithoutNextAction => {
                let manager = ProjectManager::new(store);
                let filter = ProjectFilter {
                    has_next_action: Some(false),
                    limit,
                    ..Default::default()
                };
                let projects = manager.list(filter).await?;
                let count = projects.len();
                let answer = if count == 0 {
                    "All active projects have next actions defined.".to_string()
                } else {
                    format!("Found {} projects without a defined next action.", count)
                };
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&projects)?,
                ))
            }

            GtdIntent::ProjectHealth => {
                let manager = ProjectManager::new(store);
                let filter = ProjectFilter {
                    status: Some(ProjectStatus::Active),
                    limit,
                    ..Default::default()
                };
                let projects = manager.list(filter).await?;

                let mut health_data = Vec::new();
                for project in &projects {
                    let health = manager.get_health(&project.id).await?;
                    health_data.push(serde_json::json!({
                        "project": project.name,
                        "health": health,
                    }));
                }

                let avg_health: f32 = health_data
                    .iter()
                    .filter_map(|h| h["health"]["score"].as_f64())
                    .map(|s| s as f32)
                    .sum::<f32>()
                    / health_data.len().max(1) as f32;

                let answer = format!(
                    "Analyzed {} projects. Average health score: {:.0}%",
                    health_data.len(),
                    avg_health
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&health_data)?,
                ))
            }

            GtdIntent::ListWaiting => {
                let manager = WaitingManager::new(store);
                let filter = WaitingFilter {
                    limit,
                    ..Default::default()
                };
                let items = manager.list(filter).await?;
                let count = items.len();
                let answer = format!("Found {} items you're waiting on.", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&items)?,
                ))
            }

            GtdIntent::OverdueWaiting => {
                let manager = WaitingManager::new(store);
                let mut items = manager.get_overdue().await?;
                items.truncate(limit);
                let count = items.len();
                let answer = if count == 0 {
                    "No overdue waiting-for items.".to_string()
                } else {
                    format!(
                        "Found {} overdue items you're waiting on. Consider following up.",
                        count
                    )
                };
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&items)?,
                ))
            }

            GtdIntent::WaitingForPerson { person } => {
                let manager = WaitingManager::new(store);
                let filter = WaitingFilter {
                    delegated_to: Some(person.clone()),
                    limit,
                    ..Default::default()
                };
                let items = manager.list(filter).await?;
                let count = items.len();
                let answer = format!("Found {} items waiting on {}.", count, person);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&items)?,
                ))
            }

            GtdIntent::ListSomeday => {
                let manager = SomedayManager::new(store);
                let filter = SomedayFilter {
                    limit,
                    ..Default::default()
                };
                let items = manager.list(filter).await?;
                let count = items.len();
                let answer = format!("Found {} someday/maybe items.", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&items)?,
                ))
            }

            GtdIntent::WeeklyReview => {
                let manager = ReviewManager::new(store);
                let review_params = WeeklyReviewParams::default();
                let report = manager.generate_weekly_review(review_params).await?;

                let answer = format!(
                    "Generated weekly review. {} tasks completed, {} projects active, {} stalled.",
                    report.completed_tasks.total_completed,
                    report.active_projects.active_count,
                    report.stalled_projects.len()
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&report)?,
                ))
            }

            GtdIntent::DailyReview => {
                let manager = ReviewManager::new(store);
                let report = manager.generate_daily_review().await?;

                let answer = format!(
                    "Generated daily review. {} tasks completed, {} projects active.",
                    report.completed_tasks.total_completed, report.active_projects.active_count
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&report)?,
                ))
            }

            GtdIntent::General => {
                // Fall back to listing tasks
                let manager = TaskManager::new(store);
                let filter = TaskFilter {
                    status: Some(TaskStatus::Next),
                    limit,
                    ..Default::default()
                };
                let tasks = manager.list(filter).await?;
                let count = tasks.len();
                let answer = format!("Found {} next actions.", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Gtd(intent.clone()),
                    answer,
                    serde_json::to_value(&tasks)?,
                ))
            }
        }
    }

    // ========================================================================
    // Calendar Execution
    // ========================================================================

    async fn execute_calendar(
        &self,
        intent: &CalendarIntent,
        params: &ExtractedParams,
    ) -> Result<UnifiedQueryResult> {
        let store = self.ontology_store.clone();
        let manager = CalendarManager::new(store);
        let _limit = params.limit.unwrap_or(20);

        match intent {
            CalendarIntent::Today => {
                let events = manager.today().await?;
                let count = events.len();
                let answer = if count == 0 {
                    "No events scheduled for today.".to_string()
                } else {
                    format!("You have {} events today.", count)
                };
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Calendar(intent.clone()),
                    answer,
                    serde_json::to_value(&events)?,
                ))
            }

            CalendarIntent::ThisWeek => {
                let events = manager.this_week().await?;
                let count = events.len();
                let answer = format!("You have {} events this week.", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Calendar(intent.clone()),
                    answer,
                    serde_json::to_value(&events)?,
                ))
            }

            CalendarIntent::Upcoming { days } => {
                let num_days = days.unwrap_or(7) as i64;
                let events = manager.upcoming(Some(num_days)).await?;
                let count = events.len();
                let answer = format!(
                    "Found {} upcoming events in the next {} days.",
                    count, num_days
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Calendar(intent.clone()),
                    answer,
                    serde_json::to_value(&events)?,
                ))
            }

            CalendarIntent::FreeTime => {
                let free_params = crate::calendar::FreeTimeParams::default();
                let slots = manager.find_free_time(&free_params).await?;
                let count = slots.len();
                let total_minutes: i64 = slots.iter().map(|s| s.duration_minutes).sum();
                let hours = total_minutes / 60;
                let mins = total_minutes % 60;
                let answer = format!(
                    "Found {} free time slots totaling {}h {}m.",
                    count, hours, mins
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Calendar(intent.clone()),
                    answer,
                    serde_json::to_value(&slots)?,
                ))
            }

            CalendarIntent::Conflicts => {
                // Get upcoming events and detect conflicts
                let events = manager.upcoming(Some(14)).await?;
                let conflicts = manager.detect_conflicts(&events);
                let count = conflicts.len();
                let answer = if count == 0 {
                    "No scheduling conflicts detected.".to_string()
                } else {
                    format!("Found {} scheduling conflicts.", count)
                };
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Calendar(intent.clone()),
                    answer,
                    serde_json::to_value(&conflicts)?,
                ))
            }

            CalendarIntent::SpecificDate { date_expr } => {
                // Parse the date expression (simplified - real implementation would be more robust)
                let answer = format!("Searching for events on: {}", date_expr);
                // For now, just do upcoming events as a fallback
                let events = manager.this_week().await?;
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Calendar(intent.clone()),
                    answer,
                    serde_json::to_value(&events)?,
                ))
            }

            CalendarIntent::EventsWithPerson { person } => {
                let filter = CalendarFilter {
                    participant: Some(person.clone()),
                    ..Default::default()
                };
                let events = manager.list(&filter).await?;
                let count = events.len();
                let answer = format!("Found {} events with {}.", count, person);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Calendar(intent.clone()),
                    answer,
                    serde_json::to_value(&events)?,
                ))
            }

            CalendarIntent::General => {
                let events = manager.upcoming(Some(7)).await?;
                let count = events.len();
                let answer = format!("Found {} upcoming events.", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Calendar(intent.clone()),
                    answer,
                    serde_json::to_value(&events)?,
                ))
            }
        }
    }

    // ========================================================================
    // Knowledge Execution
    // ========================================================================

    async fn execute_knowledge(
        &self,
        intent: &KnowledgeIntent,
        params: &ExtractedParams,
    ) -> Result<UnifiedQueryResult> {
        let engine = KnowledgeQueryEngine::new(self.ontology_store.clone(), self.embedder.clone());
        let limit = params.limit.unwrap_or(20);

        match intent {
            KnowledgeIntent::WhatDoIKnowAbout { topic } => {
                let query_params = crate::knowledge::KnowledgeQueryParams {
                    query: topic.clone(),
                    query_type: crate::knowledge::KnowledgeQueryType::SemanticSearch,
                    limit,
                    include_sources: true,
                    ..Default::default()
                };
                let result = engine.query(query_params).await?;
                let count = result.entities.len();
                let answer = format!(
                    "Found {} entities related to '{}'. Confidence: {:.0}%",
                    count,
                    topic,
                    result.confidence * 100.0
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Knowledge(intent.clone()),
                    answer,
                    serde_json::to_value(&result)?,
                ))
            }

            KnowledgeIntent::FindExpert { topic } => {
                let experts = engine.find_experts(topic, limit).await?;
                let count = experts.len();
                let top_expert = experts
                    .first()
                    .map(|e| e.person.name.as_str())
                    .unwrap_or("none");
                let answer = format!(
                    "Found {} potential experts on '{}'. Top expert: {}",
                    count, topic, top_expert
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Knowledge(intent.clone()),
                    answer,
                    serde_json::to_value(&experts)?,
                ))
            }

            KnowledgeIntent::WhoWorksWith { person } => {
                let query_params = crate::knowledge::KnowledgeQueryParams {
                    query: person.clone(),
                    query_type: crate::knowledge::KnowledgeQueryType::RelationshipQuery,
                    limit,
                    ..Default::default()
                };
                let result = engine.query(query_params).await?;
                let count = result.entities.len().saturating_sub(1); // Exclude the person themselves
                let answer = format!("Found {} people connected to {}.", count, person);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Knowledge(intent.clone()),
                    answer,
                    serde_json::to_value(&result)?,
                ))
            }

            KnowledgeIntent::TopicSummary { topic } => {
                let summary = engine.summarize_topic(topic, limit).await?;
                let answer = format!(
                    "Topic '{}': {} entities, {} documents.",
                    topic, summary.stats.entity_count, summary.stats.document_count
                );
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Knowledge(intent.clone()),
                    answer,
                    serde_json::to_value(&summary)?,
                ))
            }

            KnowledgeIntent::EntityLookup { name } => {
                let query_params = crate::knowledge::KnowledgeQueryParams {
                    query: name.clone(),
                    query_type: crate::knowledge::KnowledgeQueryType::EntityLookup,
                    limit,
                    ..Default::default()
                };
                let result = engine.query(query_params).await?;
                let found = !result.entities.is_empty();
                let answer = if found {
                    format!("Found entity: {}", name)
                } else {
                    format!("No entity found for: {}", name)
                };
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Knowledge(intent.clone()),
                    answer,
                    serde_json::to_value(&result)?,
                ))
            }

            KnowledgeIntent::RelationshipQuery { from } => {
                let store = self.ontology_store.read().await;
                let entities = store.find_entities_by_name(from, 1).await?;
                drop(store);

                if let Some(entity) = entities.first() {
                    let result = engine.traverse_relationships(&entity.id, None, 2).await?;
                    let count = result.discovered_entities.len();
                    let answer = format!(
                        "Found {} entities connected to {} within 2 hops.",
                        count, from
                    );
                    Ok(UnifiedQueryResult::new(
                        QueryIntent::Knowledge(intent.clone()),
                        answer,
                        serde_json::to_value(&result)?,
                    ))
                } else {
                    Ok(UnifiedQueryResult::new(
                        QueryIntent::Knowledge(intent.clone()),
                        format!("Entity not found: {}", from),
                        serde_json::Value::Null,
                    ))
                }
            }

            KnowledgeIntent::General { query } => {
                let query_params = crate::knowledge::KnowledgeQueryParams {
                    query: query.clone(),
                    query_type: crate::knowledge::KnowledgeQueryType::SemanticSearch,
                    limit,
                    include_sources: true,
                    ..Default::default()
                };
                let result = engine.query(query_params).await?;
                let count = result.entities.len();
                let answer = format!("Found {} relevant entities.", count);
                Ok(UnifiedQueryResult::new(
                    QueryIntent::Knowledge(intent.clone()),
                    answer,
                    serde_json::to_value(&result)?,
                ))
            }
        }
    }

    // ========================================================================
    // Search Execution
    // ========================================================================

    async fn execute_search(&self, query: &str) -> Result<UnifiedQueryResult> {
        let search_query = HybridQuery::new(query).limit(20);
        let results = self.coordinator.search(search_query).await?;
        let count = results.results.len();
        let answer = format!("Found {} documents matching '{}'.", count, query);
        Ok(UnifiedQueryResult::new(
            QueryIntent::Search,
            answer,
            serde_json::to_value(&results.results)?,
        ))
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    fn generate_suggestions(&self, classification: &ClassificationResult) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Suggest alternatives if confidence is low
        if classification.confidence < 0.8 {
            for (alt_intent, conf) in &classification.alternatives {
                if *conf > 0.4 {
                    suggestions.push(format!(
                        "Try: {} ({}% match)",
                        alt_intent.detailed_name(),
                        (conf * 100.0) as u32
                    ));
                }
            }
        }

        // Add context-specific suggestions
        match &classification.intent {
            QueryIntent::Gtd(GtdIntent::ListTasks) => {
                suggestions.push("Try: What should I work on now?".to_string());
                suggestions.push("Try: Show me quick wins".to_string());
            }
            QueryIntent::Gtd(GtdIntent::ListProjects) => {
                suggestions.push("Try: Show stalled projects".to_string());
                suggestions.push("Try: Projects without next action".to_string());
            }
            QueryIntent::Calendar(CalendarIntent::Today) => {
                suggestions.push("Try: What's on my calendar this week?".to_string());
                suggestions.push("Try: When am I free?".to_string());
            }
            _ => {}
        }

        suggestions.truncate(3);
        suggestions
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_energy_level(s: &str) -> EnergyLevel {
    match s.to_lowercase().as_str() {
        "low" => EnergyLevel::Low,
        "high" => EnergyLevel::High,
        _ => EnergyLevel::Medium,
    }
}

fn parse_duration(s: &str) -> Option<u32> {
    let s_lower = s.to_lowercase();
    if let Some(caps) = regex::Regex::new(r"(\d+)\s*(minutes?|mins?|hours?|hrs?)")
        .ok()?
        .captures(&s_lower)
    {
        let num: u32 = caps.get(1)?.as_str().parse().ok()?;
        let unit = caps.get(2)?.as_str();
        if unit.starts_with("hour") || unit.starts_with("hr") {
            Some(num * 60)
        } else {
            Some(num)
        }
    } else {
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_energy_level() {
        assert!(matches!(parse_energy_level("low"), EnergyLevel::Low));
        assert!(matches!(parse_energy_level("HIGH"), EnergyLevel::High));
        assert!(matches!(parse_energy_level("medium"), EnergyLevel::Medium));
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration("30 minutes"), Some(30));
        assert_eq!(parse_duration("2 hours"), Some(120));
        assert_eq!(parse_duration("1 hr"), Some(60));
        assert_eq!(parse_duration("invalid"), None);
    }
}

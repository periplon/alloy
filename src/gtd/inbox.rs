//! Inbox processing for GTD.
//!
//! This module provides intelligent inbox processing with auto-classification
//! and suggestions for organizing items into the GTD system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{
    EnergyLevel, Priority, Project, ProjectFilter, ProjectStatus, SomedayItem, Task, WaitingFor,
};
use crate::gtd::{ProjectManager, SomedayManager, TaskManager, WaitingManager};
use crate::ontology::extraction::{ActionDetector, ActionType, DetectedAction, TemporalParser};
use crate::ontology::{EmbeddedOntologyStore, Entity, EntityFilter, EntityType, OntologyStore};
use schemars::JsonSchema;

// ============================================================================
// Inbox Types
// ============================================================================

/// An item in the inbox awaiting processing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InboxItem {
    /// Unique identifier for this inbox item.
    pub id: String,
    /// Raw content of the inbox item.
    pub content: String,
    /// Source document where this item was found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document_id: Option<String>,
    /// Extracted entities from the content.
    #[serde(default)]
    pub extracted_entities: Vec<ExtractedEntitySummary>,
    /// Suggested GTD type classification.
    pub suggested_type: SuggestedGtdType,
    /// Suggested project to associate with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_project: Option<ProjectSuggestion>,
    /// Suggested contexts.
    #[serde(default)]
    pub suggested_contexts: Vec<String>,
    /// Suggested due date (parsed from content).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_due_date: Option<DateTime<Utc>>,
    /// Suggested priority.
    pub suggested_priority: Priority,
    /// Suggested energy level.
    pub suggested_energy: EnergyLevel,
    /// Confidence score for the classification (0.0-1.0).
    pub confidence: f32,
    /// Whether this is a quick task (2-minute rule).
    pub is_quick_task: bool,
    /// Whether this appears to be reference material.
    pub is_reference: bool,
    /// When this inbox item was created.
    pub created_at: DateTime<Utc>,
}

/// Summary of an extracted entity.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExtractedEntitySummary {
    /// Entity type.
    pub entity_type: String,
    /// Entity name.
    pub name: String,
    /// Confidence.
    pub confidence: f32,
}

/// Suggested GTD type for an inbox item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedGtdType {
    /// Single action task.
    Task,
    /// Multi-step project.
    Project,
    /// Waiting on someone else.
    WaitingFor,
    /// Deferred for later.
    SomedayMaybe,
    /// Reference material.
    Reference,
    /// Calendar event.
    CalendarEvent,
    /// Needs more information to classify.
    NeedsClarification,
}

/// A suggested project to associate with.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProjectSuggestion {
    /// Project ID if an existing project matches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub existing_project_id: Option<String>,
    /// Existing project name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub existing_project_name: Option<String>,
    /// Suggested new project name if no match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_project_name: Option<String>,
    /// Similarity score to existing project.
    pub similarity: f32,
}

/// Classification result for an inbox item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// The classified item.
    pub item: InboxItem,
    /// Created task (if classified as task).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_task: Option<Task>,
    /// Created project (if classified as project).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_project: Option<Project>,
    /// Created waiting-for (if classified as waiting).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_waiting: Option<WaitingFor>,
    /// Created someday item (if classified as someday).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_someday: Option<SomedayItem>,
    /// Whether auto-classification was applied.
    pub auto_classified: bool,
    /// Reason for classification.
    pub classification_reason: String,
}

/// Parameters for processing the inbox.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProcessInboxParams {
    /// Process a specific document's items.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,
    /// Process items from a specific source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    /// Enable auto-classification for high-confidence items.
    #[serde(default = "default_true")]
    pub auto_classify: bool,
    /// Confidence threshold for auto-classification.
    #[serde(default = "default_auto_threshold")]
    pub auto_classify_threshold: f32,
    /// Maximum number of items to process.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_true() -> bool {
    true
}

fn default_auto_threshold() -> f32 {
    0.85
}

fn default_batch_size() -> usize {
    50
}

impl Default for ProcessInboxParams {
    fn default() -> Self {
        Self {
            document_id: None,
            source_id: None,
            auto_classify: true,
            auto_classify_threshold: 0.85,
            batch_size: 50,
        }
    }
}

/// Response from processing the inbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInboxResponse {
    /// Items that were processed.
    pub items: Vec<InboxItem>,
    /// Items that were auto-classified.
    pub auto_classified: Vec<ClassificationResult>,
    /// Items needing manual review.
    pub needs_review: Vec<InboxItem>,
    /// Quick wins identified (2-minute tasks).
    pub quick_wins: Vec<InboxItem>,
    /// Total items processed.
    pub total_processed: usize,
    /// Summary statistics.
    pub summary: ProcessingSummary,
}

/// Summary of inbox processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingSummary {
    /// Count by suggested type.
    pub by_type: HashMap<String, usize>,
    /// Tasks created.
    pub tasks_created: usize,
    /// Projects created.
    pub projects_created: usize,
    /// Waiting items created.
    pub waiting_created: usize,
    /// Someday items created.
    pub someday_created: usize,
    /// Average confidence.
    pub average_confidence: f32,
}

// ============================================================================
// Inbox Manager
// ============================================================================

/// Manages GTD inbox processing with intelligent classification.
pub struct InboxManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    task_manager: TaskManager,
    project_manager: ProjectManager,
    waiting_manager: WaitingManager,
    someday_manager: SomedayManager,
    temporal_parser: TemporalParser,
    action_detector: ActionDetector,
    /// Quick task threshold in minutes.
    quick_task_minutes: u32,
    /// Default contexts to suggest.
    default_contexts: Vec<String>,
}

impl InboxManager {
    /// Create a new inbox manager.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self {
            task_manager: TaskManager::new(store.clone()),
            project_manager: ProjectManager::new(store.clone()),
            waiting_manager: WaitingManager::new(store.clone()),
            someday_manager: SomedayManager::new(store.clone()),
            temporal_parser: TemporalParser::new(),
            action_detector: ActionDetector::new(),
            store,
            quick_task_minutes: 2,
            default_contexts: vec![
                "@home".to_string(),
                "@work".to_string(),
                "@phone".to_string(),
                "@computer".to_string(),
                "@errand".to_string(),
                "@anywhere".to_string(),
            ],
        }
    }

    /// Set the quick task threshold.
    pub fn with_quick_threshold(mut self, minutes: u32) -> Self {
        self.quick_task_minutes = minutes;
        self
    }

    /// Set the default contexts.
    pub fn with_default_contexts(mut self, contexts: Vec<String>) -> Self {
        self.default_contexts = contexts;
        self
    }

    /// Process the inbox and classify items.
    pub async fn process(&self, params: ProcessInboxParams) -> Result<ProcessInboxResponse> {
        let store = self.store.read().await;

        // Get unprocessed items from the ontology store
        let mut filter = EntityFilter::default().with_limit(params.batch_size);

        // Filter by document if specified
        if let Some(ref doc_id) = params.document_id {
            filter.source_document_id = Some(doc_id.clone());
        }

        // Get entities that look like action items or unprocessed content
        let task_entities = store
            .find_entities_by_type(EntityType::Task, params.batch_size)
            .await?;
        let commitment_entities = store
            .find_entities_by_type(EntityType::Commitment, params.batch_size)
            .await?;

        drop(store);

        let mut items = Vec::new();
        let mut auto_classified = Vec::new();
        let mut needs_review = Vec::new();
        let mut quick_wins = Vec::new();
        let mut by_type: HashMap<String, usize> = HashMap::new();
        let mut total_confidence = 0.0f32;

        // Get existing projects for matching
        let existing_projects = self
            .project_manager
            .list(ProjectFilter {
                status: Some(ProjectStatus::Active),
                ..Default::default()
            })
            .await?;

        // Process each entity
        for entity in task_entities.into_iter().chain(commitment_entities) {
            // Skip if already processed (has gtd_processed metadata)
            if entity.metadata.contains_key("gtd_processed") {
                continue;
            }

            let inbox_item = self.classify_entity(&entity, &existing_projects).await?;

            total_confidence += inbox_item.confidence;

            // Track by type
            *by_type
                .entry(format!("{:?}", inbox_item.suggested_type))
                .or_default() += 1;

            // Check for quick wins
            if inbox_item.is_quick_task {
                quick_wins.push(inbox_item.clone());
            }

            // Auto-classify if enabled and confidence is high enough
            if params.auto_classify && inbox_item.confidence >= params.auto_classify_threshold {
                if let Some(result) = self.auto_classify_item(&inbox_item).await? {
                    auto_classified.push(result);
                } else {
                    needs_review.push(inbox_item.clone());
                }
            } else {
                needs_review.push(inbox_item.clone());
            }

            items.push(inbox_item);
        }

        let total_processed = items.len();
        let average_confidence = if total_processed > 0 {
            total_confidence / total_processed as f32
        } else {
            0.0
        };

        let summary = ProcessingSummary {
            by_type,
            tasks_created: auto_classified
                .iter()
                .filter(|r| r.created_task.is_some())
                .count(),
            projects_created: auto_classified
                .iter()
                .filter(|r| r.created_project.is_some())
                .count(),
            waiting_created: auto_classified
                .iter()
                .filter(|r| r.created_waiting.is_some())
                .count(),
            someday_created: auto_classified
                .iter()
                .filter(|r| r.created_someday.is_some())
                .count(),
            average_confidence,
        };

        Ok(ProcessInboxResponse {
            items,
            auto_classified,
            needs_review,
            quick_wins,
            total_processed,
            summary,
        })
    }

    /// Classify a single piece of text.
    pub async fn classify_text(
        &self,
        text: &str,
        source_document_id: Option<String>,
    ) -> Result<InboxItem> {
        // Detect actions in the text
        let actions = self.action_detector.detect(text);
        let temporal = self.temporal_parser.parse(text);

        // Get existing projects for matching
        let existing_projects = self
            .project_manager
            .list(ProjectFilter {
                status: Some(ProjectStatus::Active),
                ..Default::default()
            })
            .await?;

        self.build_inbox_item(
            text,
            source_document_id,
            &actions,
            &temporal,
            &existing_projects,
        )
        .await
    }

    /// Manually classify an inbox item.
    pub async fn classify_as(
        &self,
        item: &InboxItem,
        target_type: SuggestedGtdType,
        project_id: Option<String>,
        contexts: Vec<String>,
    ) -> Result<ClassificationResult> {
        match target_type {
            SuggestedGtdType::Task => {
                let mut task = Task::new(&item.content)
                    .with_contexts(contexts)
                    .with_priority(item.suggested_priority)
                    .with_energy(item.suggested_energy);

                if let Some(due) = item.suggested_due_date {
                    task = task.with_due_date(due);
                }

                if let Some(ref proj_id) = project_id {
                    task = task.with_project(proj_id);
                }

                let created = self.task_manager.create(task).await?;

                Ok(ClassificationResult {
                    item: item.clone(),
                    created_task: Some(created),
                    created_project: None,
                    created_waiting: None,
                    created_someday: None,
                    auto_classified: false,
                    classification_reason: "Manually classified as task".to_string(),
                })
            }
            SuggestedGtdType::Project => {
                let project = Project::new(&item.content);
                let created = self.project_manager.create(project).await?;

                Ok(ClassificationResult {
                    item: item.clone(),
                    created_task: None,
                    created_project: Some(created),
                    created_waiting: None,
                    created_someday: None,
                    auto_classified: false,
                    classification_reason: "Manually classified as project".to_string(),
                })
            }
            SuggestedGtdType::WaitingFor => {
                // Extract person from content or use a default
                let person = self
                    .extract_person_from_text(&item.content)
                    .unwrap_or_else(|| "Unknown".to_string());
                let mut waiting = WaitingFor::new(&item.content, person);

                if let Some(due) = item.suggested_due_date {
                    waiting = waiting.with_expected_by(due);
                }

                if let Some(ref proj_id) = project_id {
                    waiting = waiting.with_project(proj_id);
                }

                let created = self.waiting_manager.create(waiting).await?;

                Ok(ClassificationResult {
                    item: item.clone(),
                    created_task: None,
                    created_project: None,
                    created_waiting: Some(created),
                    created_someday: None,
                    auto_classified: false,
                    classification_reason: "Manually classified as waiting-for".to_string(),
                })
            }
            SuggestedGtdType::SomedayMaybe => {
                let someday = SomedayItem::new(&item.content);
                let created = self.someday_manager.create(someday).await?;

                Ok(ClassificationResult {
                    item: item.clone(),
                    created_task: None,
                    created_project: None,
                    created_waiting: None,
                    created_someday: Some(created),
                    auto_classified: false,
                    classification_reason: "Manually classified as someday/maybe".to_string(),
                })
            }
            _ => Ok(ClassificationResult {
                item: item.clone(),
                created_task: None,
                created_project: None,
                created_waiting: None,
                created_someday: None,
                auto_classified: false,
                classification_reason: format!("Type {:?} not automatically created", target_type),
            }),
        }
    }

    /// Process inbox to "zero" - auto-classify high confidence, return rest.
    pub async fn inbox_zero(&self, batch_size: usize) -> Result<ProcessInboxResponse> {
        self.process(ProcessInboxParams {
            auto_classify: true,
            auto_classify_threshold: 0.85,
            batch_size,
            ..Default::default()
        })
        .await
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    async fn classify_entity(
        &self,
        entity: &Entity,
        existing_projects: &[Project],
    ) -> Result<InboxItem> {
        let text = &entity.name;

        // Detect actions and temporal in the text
        let actions = self.action_detector.detect(text);
        let temporal = self.temporal_parser.parse(text);

        self.build_inbox_item(
            text,
            entity.source_refs.first().map(|r| r.document_id.clone()),
            &actions,
            &temporal,
            existing_projects,
        )
        .await
    }

    async fn build_inbox_item(
        &self,
        text: &str,
        source_document_id: Option<String>,
        actions: &[DetectedAction],
        temporal: &[crate::ontology::extraction::TemporalExtraction],
        existing_projects: &[Project],
    ) -> Result<InboxItem> {
        let text_lower = text.to_lowercase();

        // Determine suggested type based on patterns
        let (suggested_type, confidence) = self.determine_type(&text_lower, actions);

        // Extract suggested due date from temporal parsing
        let suggested_due_date = temporal
            .iter()
            .filter(|t| matches!(t.date_type, crate::ontology::extraction::DateType::Deadline))
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .and_then(|t| t.parsed_date.as_datetime())
            .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc));

        // Suggest contexts based on content
        let suggested_contexts = self.suggest_contexts(&text_lower);

        // Suggest project based on semantic similarity
        let suggested_project = self.suggest_project(text, existing_projects);

        // Determine priority and energy
        let (suggested_priority, suggested_energy) =
            self.determine_priority_energy(&text_lower, actions);

        // Check if it's a quick task
        let is_quick_task = self.is_quick_task(&text_lower, actions);

        // Check if it's reference material
        let is_reference = self.is_reference_material(&text_lower);

        // Build extracted entities summary
        let extracted_entities: Vec<ExtractedEntitySummary> = actions
            .iter()
            .map(|a| ExtractedEntitySummary {
                entity_type: format!("{:?}", a.action_type),
                name: a.description.clone(),
                confidence: a.confidence,
            })
            .collect();

        Ok(InboxItem {
            id: uuid::Uuid::new_v4().to_string(),
            content: text.to_string(),
            source_document_id,
            extracted_entities,
            suggested_type,
            suggested_project,
            suggested_contexts,
            suggested_due_date,
            suggested_priority,
            suggested_energy,
            confidence,
            is_quick_task,
            is_reference,
            created_at: Utc::now(),
        })
    }

    fn determine_type(&self, text: &str, actions: &[DetectedAction]) -> (SuggestedGtdType, f32) {
        // Check for explicit patterns
        if text.contains("waiting for")
            || text.contains("waiting on")
            || text.contains("follow up with")
        {
            return (SuggestedGtdType::WaitingFor, 0.85);
        }

        if text.contains("someday")
            || text.contains("maybe")
            || text.contains("would be nice")
            || text.contains("eventually")
        {
            return (SuggestedGtdType::SomedayMaybe, 0.80);
        }

        if text.contains("meeting")
            || text.contains("appointment")
            || text.contains("call at")
            || text.contains("scheduled for")
        {
            return (SuggestedGtdType::CalendarEvent, 0.80);
        }

        if text.contains("reference")
            || text.contains("for info")
            || text.contains("fyi")
            || text.starts_with("note:")
        {
            return (SuggestedGtdType::Reference, 0.75);
        }

        // Check detected actions
        if let Some(action) = actions.first() {
            match action.action_type {
                ActionType::Task => {
                    // Check if it seems like a multi-step project
                    if self.looks_like_project(text) {
                        return (SuggestedGtdType::Project, 0.70);
                    }
                    return (SuggestedGtdType::Task, action.confidence);
                }
                ActionType::Commitment => {
                    // If commitment was made to us (received), it's a waiting-for
                    if action.commitment_to.is_some() {
                        return (SuggestedGtdType::WaitingFor, action.confidence);
                    }
                    return (SuggestedGtdType::Task, action.confidence);
                }
                ActionType::FollowUp => {
                    return (SuggestedGtdType::WaitingFor, action.confidence);
                }
                ActionType::Reminder => {
                    return (SuggestedGtdType::CalendarEvent, action.confidence);
                }
            }
        }

        // Default: likely a task but needs clarification
        if text.len() > 10 {
            (SuggestedGtdType::Task, 0.50)
        } else {
            (SuggestedGtdType::NeedsClarification, 0.30)
        }
    }

    fn looks_like_project(&self, text: &str) -> bool {
        let project_indicators = [
            "project",
            "initiative",
            "launch",
            "implement",
            "develop",
            "build",
            "create",
            "redesign",
            "migration",
            "rollout",
        ];

        project_indicators.iter().any(|ind| text.contains(ind))
    }

    fn suggest_contexts(&self, text: &str) -> Vec<String> {
        let mut contexts = Vec::new();

        // Phone-related
        if text.contains("call")
            || text.contains("phone")
            || text.contains("text")
            || text.contains("message")
        {
            contexts.push("@phone".to_string());
        }

        // Computer-related
        if text.contains("email")
            || text.contains("write")
            || text.contains("code")
            || text.contains("draft")
            || text.contains("research")
            || text.contains("online")
        {
            contexts.push("@computer".to_string());
        }

        // Errand-related
        if text.contains("buy")
            || text.contains("pick up")
            || text.contains("drop off")
            || text.contains("store")
            || text.contains("bank")
        {
            contexts.push("@errand".to_string());
        }

        // Home-related
        if text.contains("home")
            || text.contains("house")
            || text.contains("clean")
            || text.contains("fix")
        {
            contexts.push("@home".to_string());
        }

        // Work-related
        if text.contains("meeting")
            || text.contains("office")
            || text.contains("report")
            || text.contains("present")
        {
            contexts.push("@work".to_string());
        }

        // If no specific context, suggest @anywhere
        if contexts.is_empty() {
            contexts.push("@anywhere".to_string());
        }

        contexts
    }

    fn suggest_project(
        &self,
        text: &str,
        existing_projects: &[Project],
    ) -> Option<ProjectSuggestion> {
        let text_lower = text.to_lowercase();

        // Try to find a matching existing project
        let mut best_match: Option<(&Project, f32)> = None;

        for project in existing_projects {
            let project_name_lower = project.name.to_lowercase();

            // Check for word overlap
            let text_words: std::collections::HashSet<&str> =
                text_lower.split_whitespace().collect();
            let project_words: std::collections::HashSet<&str> =
                project_name_lower.split_whitespace().collect();

            let intersection: usize = text_words.intersection(&project_words).count();
            let union: usize = text_words.union(&project_words).count();

            if union > 0 {
                let jaccard = intersection as f32 / union as f32;
                if jaccard > 0.2 {
                    match best_match {
                        Some((_, score)) if jaccard > score => {
                            best_match = Some((project, jaccard));
                        }
                        None => {
                            best_match = Some((project, jaccard));
                        }
                        _ => {}
                    }
                }
            }
        }

        best_match.map(|(project, similarity)| ProjectSuggestion {
            existing_project_id: Some(project.id.clone()),
            existing_project_name: Some(project.name.clone()),
            new_project_name: None,
            similarity,
        })
    }

    fn determine_priority_energy(
        &self,
        text: &str,
        actions: &[DetectedAction],
    ) -> (Priority, EnergyLevel) {
        // Check actions first
        if let Some(action) = actions.first() {
            return (action.priority, action.energy_level);
        }

        // Pattern-based detection
        let priority = if text.contains("urgent")
            || text.contains("asap")
            || text.contains("critical")
            || text.contains("immediately")
        {
            Priority::Critical
        } else if text.contains("important")
            || text.contains("priority")
            || text.contains("high priority")
        {
            Priority::High
        } else if text.contains("low priority") || text.contains("when possible") {
            Priority::Low
        } else {
            Priority::Normal
        };

        let energy = if text.contains("complex")
            || text.contains("difficult")
            || text.contains("challenging")
            || text.contains("deep work")
        {
            EnergyLevel::High
        } else if text.contains("simple")
            || text.contains("quick")
            || text.contains("easy")
            || text.contains("routine")
        {
            EnergyLevel::Low
        } else {
            EnergyLevel::Medium
        };

        (priority, energy)
    }

    fn is_quick_task(&self, text: &str, _actions: &[DetectedAction]) -> bool {
        // Pattern-based detection
        let quick_patterns = [
            "quick",
            "just",
            "simply",
            "2 minute",
            "two minute",
            "real quick",
            "briefly",
        ];

        quick_patterns.iter().any(|p| text.contains(p))
    }

    fn is_reference_material(&self, text: &str) -> bool {
        let reference_patterns = [
            "for reference",
            "fyi",
            "for your information",
            "note:",
            "reference:",
            "info:",
            "see attached",
            "background:",
        ];

        reference_patterns.iter().any(|p| text.contains(p))
    }

    fn extract_person_from_text(&self, text: &str) -> Option<String> {
        // Simple pattern matching for "waiting for/on [person]"
        let patterns = ["waiting for", "waiting on", "from", "assigned to"];

        for pattern in patterns {
            if let Some(idx) = text.to_lowercase().find(pattern) {
                let after = &text[idx + pattern.len()..];
                let words: Vec<&str> = after.split_whitespace().take(2).collect();
                if !words.is_empty() {
                    return Some(words.join(" "));
                }
            }
        }

        None
    }

    async fn auto_classify_item(&self, item: &InboxItem) -> Result<Option<ClassificationResult>> {
        // Only auto-classify high confidence items
        if item.confidence < 0.85 {
            return Ok(None);
        }

        match item.suggested_type {
            SuggestedGtdType::Task => {
                let mut task = Task::new(&item.content)
                    .with_contexts(item.suggested_contexts.clone())
                    .with_priority(item.suggested_priority)
                    .with_energy(item.suggested_energy);

                if let Some(due) = item.suggested_due_date {
                    task = task.with_due_date(due);
                }

                if let Some(ref suggestion) = item.suggested_project {
                    if let Some(ref proj_id) = suggestion.existing_project_id {
                        task = task.with_project(proj_id);
                    }
                }

                let created = self.task_manager.create(task).await?;

                Ok(Some(ClassificationResult {
                    item: item.clone(),
                    created_task: Some(created),
                    created_project: None,
                    created_waiting: None,
                    created_someday: None,
                    auto_classified: true,
                    classification_reason: format!(
                        "Auto-classified as task with {:.0}% confidence",
                        item.confidence * 100.0
                    ),
                }))
            }
            SuggestedGtdType::WaitingFor => {
                let person = self
                    .extract_person_from_text(&item.content)
                    .unwrap_or_else(|| "Unknown".to_string());
                let mut waiting = WaitingFor::new(&item.content, person);

                if let Some(due) = item.suggested_due_date {
                    waiting = waiting.with_expected_by(due);
                }

                if let Some(ref suggestion) = item.suggested_project {
                    if let Some(ref proj_id) = suggestion.existing_project_id {
                        waiting = waiting.with_project(proj_id);
                    }
                }

                let created = self.waiting_manager.create(waiting).await?;

                Ok(Some(ClassificationResult {
                    item: item.clone(),
                    created_task: None,
                    created_project: None,
                    created_waiting: Some(created),
                    created_someday: None,
                    auto_classified: true,
                    classification_reason: format!(
                        "Auto-classified as waiting-for with {:.0}% confidence",
                        item.confidence * 100.0
                    ),
                }))
            }
            SuggestedGtdType::SomedayMaybe => {
                let someday = SomedayItem::new(&item.content);
                let created = self.someday_manager.create(someday).await?;

                Ok(Some(ClassificationResult {
                    item: item.clone(),
                    created_task: None,
                    created_project: None,
                    created_waiting: None,
                    created_someday: Some(created),
                    auto_classified: true,
                    classification_reason: format!(
                        "Auto-classified as someday/maybe with {:.0}% confidence",
                        item.confidence * 100.0
                    ),
                }))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_manager() -> InboxManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        InboxManager::new(store)
    }

    #[tokio::test]
    async fn test_classify_task() {
        let manager = create_test_manager().await;

        let item = manager
            .classify_text("TODO: Call John about the report", None)
            .await
            .unwrap();

        assert_eq!(item.suggested_type, SuggestedGtdType::Task);
        assert!(item.suggested_contexts.contains(&"@phone".to_string()));
    }

    #[tokio::test]
    async fn test_classify_waiting_for() {
        let manager = create_test_manager().await;

        let item = manager
            .classify_text("Waiting for Sarah to send the report", None)
            .await
            .unwrap();

        assert_eq!(item.suggested_type, SuggestedGtdType::WaitingFor);
    }

    #[tokio::test]
    async fn test_classify_someday() {
        let manager = create_test_manager().await;

        let item = manager
            .classify_text("Someday learn to play guitar", None)
            .await
            .unwrap();

        assert_eq!(item.suggested_type, SuggestedGtdType::SomedayMaybe);
    }

    #[tokio::test]
    async fn test_classify_quick_task() {
        let manager = create_test_manager().await;

        let item = manager
            .classify_text("Quick: Reply to John's email", None)
            .await
            .unwrap();

        assert!(item.is_quick_task);
    }

    #[tokio::test]
    async fn test_suggest_contexts() {
        let manager = create_test_manager().await;

        let item = manager
            .classify_text("Call the bank about account", None)
            .await
            .unwrap();

        assert!(item.suggested_contexts.contains(&"@phone".to_string()));
    }

    #[tokio::test]
    async fn test_extract_person() {
        let manager = create_test_manager().await;

        let person = manager.extract_person_from_text("Waiting for John Smith to respond");
        assert_eq!(person, Some("John Smith".to_string()));
    }
}

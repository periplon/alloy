//! Commitment Tracking for GTD.
//!
//! This module provides tracking and management of commitments - promises made
//! and received. It extracts commitments from documents and tracks their status.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::ontology::extraction::{ActionDetector, ActionType, DetectedAction};
use crate::ontology::{DocumentRef, EmbeddedOntologyStore, Entity, EntityType, OntologyStore};
use schemars::JsonSchema;

// ============================================================================
// Commitment Types
// ============================================================================

/// A commitment (promise made or received).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commitment {
    /// Unique identifier.
    pub id: String,
    /// Type of commitment.
    pub commitment_type: CommitmentDirection,
    /// Description of the commitment.
    pub description: String,
    /// Who made the commitment.
    pub from_person: Option<String>,
    /// Who the commitment is to.
    pub to_person: Option<String>,
    /// Due date (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub due_date: Option<DateTime<Utc>>,
    /// Current status.
    pub status: CommitmentStatus,
    /// Source document where this was extracted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document: Option<String>,
    /// The extracted text that formed the commitment.
    pub extracted_text: String,
    /// Confidence of extraction (0.0-1.0).
    pub confidence: f32,
    /// Related project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Follow-up date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub follow_up_date: Option<DateTime<Utc>>,
    /// Notes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// When this was created.
    pub created_at: DateTime<Utc>,
    /// When this was last updated.
    pub updated_at: DateTime<Utc>,
}

/// Direction of commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CommitmentDirection {
    /// A commitment made by the user.
    Made,
    /// A commitment received from someone else.
    Received,
}

/// Status of a commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CommitmentStatus {
    /// Active commitment, not yet fulfilled.
    #[default]
    Pending,
    /// In progress.
    InProgress,
    /// Completed/fulfilled.
    Fulfilled,
    /// Cancelled or no longer applicable.
    Cancelled,
    /// Broken/not fulfilled.
    Broken,
    /// Overdue.
    Overdue,
}

/// Filter for listing commitments.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CommitmentFilter {
    /// Filter by type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commitment_type: Option<CommitmentDirection>,
    /// Filter by status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<CommitmentStatus>,
    /// Filter by person (from or to).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub person: Option<String>,
    /// Filter by project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Only overdue commitments.
    #[serde(default)]
    pub overdue_only: bool,
    /// Only commitments needing follow-up.
    #[serde(default)]
    pub needs_follow_up: bool,
    /// Maximum results.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    100
}

/// Summary of commitments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentSummary {
    /// Total pending commitments made.
    pub pending_made: usize,
    /// Total pending commitments received.
    pub pending_received: usize,
    /// Overdue commitments.
    pub overdue_count: usize,
    /// Commitments by person.
    pub by_person: HashMap<String, PersonCommitments>,
    /// Commitments due this week.
    pub due_this_week: Vec<Commitment>,
    /// Recently fulfilled.
    pub recently_fulfilled: Vec<Commitment>,
    /// Health score (0-100).
    pub health_score: f32,
    /// Recommendations.
    pub recommendations: Vec<String>,
    /// Generated at.
    pub generated_at: DateTime<Utc>,
}

/// Commitments involving a specific person.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonCommitments {
    /// Person name.
    pub person: String,
    /// Commitments made to this person.
    pub made_to: usize,
    /// Commitments received from this person.
    pub received_from: usize,
    /// Pending count.
    pub pending: usize,
    /// Overdue count.
    pub overdue: usize,
    /// Trust score based on fulfillment history.
    pub trust_score: f32,
}

/// Result of extracting commitments from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentExtractionResult {
    /// Extracted commitments.
    pub commitments: Vec<Commitment>,
    /// Total found.
    pub total_found: usize,
    /// Confidence score.
    pub confidence: f32,
    /// Processing metadata.
    pub metadata: ExtractionMetadata,
}

/// Metadata about extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionMetadata {
    /// Characters processed.
    pub chars_processed: usize,
    /// Processing time in ms.
    pub processing_ms: u64,
    /// Patterns used.
    pub patterns_matched: usize,
}

// ============================================================================
// Commitment Manager
// ============================================================================

/// Manages commitment tracking.
pub struct CommitmentManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    action_detector: ActionDetector,
}

impl CommitmentManager {
    /// Create a new commitment manager.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self {
            store,
            action_detector: ActionDetector::new(),
        }
    }

    /// Extract commitments from text.
    pub fn extract_from_text(
        &self,
        text: &str,
        document_id: Option<&str>,
    ) -> CommitmentExtractionResult {
        let start = std::time::Instant::now();

        let actions = self.action_detector.detect(text);

        // Filter for commitments
        let commitment_actions: Vec<&DetectedAction> = actions
            .iter()
            .filter(|a| a.action_type == ActionType::Commitment)
            .collect();

        let commitments: Vec<Commitment> = commitment_actions
            .iter()
            .map(|action| {
                let direction = if action.commitment_by.as_deref() == Some("self") {
                    CommitmentDirection::Made
                } else {
                    CommitmentDirection::Received
                };

                Commitment {
                    id: uuid::Uuid::new_v4().to_string(),
                    commitment_type: direction,
                    description: action.description.clone(),
                    from_person: action.commitment_by.clone(),
                    to_person: action.commitment_to.clone(),
                    due_date: None,
                    status: CommitmentStatus::Pending,
                    source_document: document_id.map(String::from),
                    extracted_text: action.source_text.clone(),
                    confidence: action.confidence,
                    project_id: None,
                    follow_up_date: None,
                    notes: None,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                }
            })
            .collect();

        let total_found = commitments.len();
        let avg_confidence = if total_found > 0 {
            commitments.iter().map(|c| c.confidence).sum::<f32>() / total_found as f32
        } else {
            1.0
        };

        CommitmentExtractionResult {
            commitments,
            total_found,
            confidence: avg_confidence,
            metadata: ExtractionMetadata {
                chars_processed: text.len(),
                processing_ms: start.elapsed().as_millis() as u64,
                patterns_matched: commitment_actions.len(),
            },
        }
    }

    /// Store a commitment in the ontology.
    pub async fn create(&self, commitment: Commitment) -> Result<Commitment> {
        let store = self.store.write().await;

        // Create entity
        let mut entity = Entity::new(EntityType::Commitment, &commitment.description)
            .with_confidence(commitment.confidence)
            .with_metadata(
                "commitment_type",
                serde_json::json!(format!("{:?}", commitment.commitment_type)),
            )
            .with_metadata(
                "status",
                serde_json::json!(format!("{:?}", commitment.status)),
            )
            .with_metadata(
                "extracted_text",
                serde_json::json!(commitment.extracted_text),
            );

        if let Some(ref from) = commitment.from_person {
            entity = entity.with_metadata("from_person", serde_json::json!(from));
        }

        if let Some(ref to) = commitment.to_person {
            entity = entity.with_metadata("to_person", serde_json::json!(to));
        }

        if let Some(ref doc) = commitment.source_document {
            entity = entity.with_source_ref(DocumentRef::new(doc));
        }

        store.create_entity(entity).await?;

        Ok(commitment)
    }

    /// List commitments with filters.
    pub async fn list(&self, filter: CommitmentFilter) -> Result<Vec<Commitment>> {
        let store = self.store.read().await;
        let entities = store
            .find_entities_by_type(EntityType::Commitment, filter.limit)
            .await?;

        let now = Utc::now();
        let mut commitments = Vec::new();

        for entity in entities {
            let commitment = self.entity_to_commitment(&entity);

            // Apply filters
            if let Some(ref ct) = filter.commitment_type {
                if commitment.commitment_type != *ct {
                    continue;
                }
            }

            if let Some(ref status) = filter.status {
                if commitment.status != *status {
                    continue;
                }
            }

            if let Some(ref person) = filter.person {
                let matches_from = commitment
                    .from_person
                    .as_ref()
                    .map(|f| f.to_lowercase().contains(&person.to_lowercase()))
                    .unwrap_or(false);
                let matches_to = commitment
                    .to_person
                    .as_ref()
                    .map(|t| t.to_lowercase().contains(&person.to_lowercase()))
                    .unwrap_or(false);
                if !matches_from && !matches_to {
                    continue;
                }
            }

            if filter.overdue_only {
                if let Some(due) = commitment.due_date {
                    if due >= now {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            commitments.push(commitment);
        }

        Ok(commitments)
    }

    /// Get a commitment by ID.
    pub async fn get(&self, id: &str) -> Result<Option<Commitment>> {
        let store = self.store.read().await;
        match store.get_entity(id).await? {
            Some(entity) if entity.entity_type == EntityType::Commitment => {
                Ok(Some(self.entity_to_commitment(&entity)))
            }
            _ => Ok(None),
        }
    }

    /// Update a commitment's status.
    pub async fn update_status(&self, id: &str, status: CommitmentStatus) -> Result<()> {
        let store = self.store.read().await;

        if store.get_entity(id).await?.is_some() {
            drop(store);
            let store = self.store.write().await;
            let mut metadata_map = std::collections::HashMap::new();
            metadata_map.insert(
                "status".to_string(),
                serde_json::json!(format!("{:?}", status)),
            );
            let update = crate::ontology::EntityUpdate {
                name: None,
                entity_type: None,
                add_aliases: Vec::new(),
                remove_aliases: Vec::new(),
                embedding: None,
                add_source_refs: Vec::new(),
                remove_source_refs: Vec::new(),
                set_metadata: metadata_map,
                remove_metadata_keys: Vec::new(),
                confidence: None,
            };
            store.update_entity(id, update).await?;
        }

        Ok(())
    }

    /// Mark a commitment as fulfilled.
    pub async fn fulfill(&self, id: &str) -> Result<()> {
        self.update_status(id, CommitmentStatus::Fulfilled).await
    }

    /// Mark a commitment as cancelled.
    pub async fn cancel(&self, id: &str) -> Result<()> {
        self.update_status(id, CommitmentStatus::Cancelled).await
    }

    /// Get overdue commitments.
    pub async fn get_overdue(&self) -> Result<Vec<Commitment>> {
        self.list(CommitmentFilter {
            overdue_only: true,
            ..Default::default()
        })
        .await
    }

    /// Get commitments made to a specific person.
    pub async fn get_made_to(&self, person: &str) -> Result<Vec<Commitment>> {
        let all = self
            .list(CommitmentFilter {
                commitment_type: Some(CommitmentDirection::Made),
                ..Default::default()
            })
            .await?;

        Ok(all
            .into_iter()
            .filter(|c| {
                c.to_person
                    .as_ref()
                    .map(|p| p.to_lowercase().contains(&person.to_lowercase()))
                    .unwrap_or(false)
            })
            .collect())
    }

    /// Get commitments received from a specific person.
    pub async fn get_received_from(&self, person: &str) -> Result<Vec<Commitment>> {
        let all = self
            .list(CommitmentFilter {
                commitment_type: Some(CommitmentDirection::Received),
                ..Default::default()
            })
            .await?;

        Ok(all
            .into_iter()
            .filter(|c| {
                c.from_person
                    .as_ref()
                    .map(|p| p.to_lowercase().contains(&person.to_lowercase()))
                    .unwrap_or(false)
            })
            .collect())
    }

    /// Get a summary of commitments.
    pub async fn summary(&self) -> Result<CommitmentSummary> {
        let all = self.list(CommitmentFilter::default()).await?;
        let now = Utc::now();
        let week_from_now = now + Duration::days(7);

        let pending_made = all
            .iter()
            .filter(|c| {
                c.commitment_type == CommitmentDirection::Made
                    && c.status == CommitmentStatus::Pending
            })
            .count();

        let pending_received = all
            .iter()
            .filter(|c| {
                c.commitment_type == CommitmentDirection::Received
                    && c.status == CommitmentStatus::Pending
            })
            .count();

        let overdue: Vec<&Commitment> = all
            .iter()
            .filter(|c| {
                c.due_date
                    .map(|d| d < now && c.status == CommitmentStatus::Pending)
                    .unwrap_or(false)
            })
            .collect();

        let overdue_count = overdue.len();

        let due_this_week: Vec<Commitment> = all
            .iter()
            .filter(|c| {
                c.due_date
                    .map(|d| d >= now && d <= week_from_now)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        let recently_fulfilled: Vec<Commitment> = all
            .iter()
            .filter(|c| c.status == CommitmentStatus::Fulfilled)
            .take(5)
            .cloned()
            .collect();

        // Calculate by person
        let mut by_person: HashMap<String, PersonCommitments> = HashMap::new();

        for commitment in &all {
            if let Some(ref person) = commitment.to_person {
                let entry = by_person
                    .entry(person.clone())
                    .or_insert_with(|| PersonCommitments {
                        person: person.clone(),
                        made_to: 0,
                        received_from: 0,
                        pending: 0,
                        overdue: 0,
                        trust_score: 1.0,
                    });
                entry.made_to += 1;
                if commitment.status == CommitmentStatus::Pending {
                    entry.pending += 1;
                }
            }

            if let Some(ref person) = commitment.from_person {
                if person != "self" {
                    let entry =
                        by_person
                            .entry(person.clone())
                            .or_insert_with(|| PersonCommitments {
                                person: person.clone(),
                                made_to: 0,
                                received_from: 0,
                                pending: 0,
                                overdue: 0,
                                trust_score: 1.0,
                            });
                    entry.received_from += 1;
                }
            }
        }

        // Calculate health score
        let total_pending = pending_made + pending_received;
        let health_score = if total_pending == 0 {
            100.0
        } else {
            let overdue_ratio = overdue_count as f32 / total_pending as f32;
            (1.0 - overdue_ratio) * 100.0
        };

        // Generate recommendations
        let mut recommendations = Vec::new();

        if overdue_count > 0 {
            recommendations.push(format!(
                "You have {} overdue commitment(s) - address these first",
                overdue_count
            ));
        }

        if pending_made > 10 {
            recommendations.push(
                "You have many outstanding commitments made - consider consolidating".to_string(),
            );
        }

        if due_this_week.len() > 5 {
            recommendations.push(format!(
                "{} commitments due this week - plan accordingly",
                due_this_week.len()
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("Your commitment load looks manageable".to_string());
        }

        Ok(CommitmentSummary {
            pending_made,
            pending_received,
            overdue_count,
            by_person,
            due_this_week,
            recently_fulfilled,
            health_score,
            recommendations,
            generated_at: Utc::now(),
        })
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    fn entity_to_commitment(&self, entity: &Entity) -> Commitment {
        let commitment_type = entity
            .metadata
            .get("commitment_type")
            .and_then(|v| v.as_str())
            .map(|s| {
                if s.contains("Made") {
                    CommitmentDirection::Made
                } else {
                    CommitmentDirection::Received
                }
            })
            .unwrap_or(CommitmentDirection::Made);

        let status = entity
            .metadata
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| {
                if s.contains("Fulfilled") {
                    CommitmentStatus::Fulfilled
                } else if s.contains("Cancelled") {
                    CommitmentStatus::Cancelled
                } else if s.contains("InProgress") {
                    CommitmentStatus::InProgress
                } else if s.contains("Overdue") {
                    CommitmentStatus::Overdue
                } else if s.contains("Broken") {
                    CommitmentStatus::Broken
                } else {
                    CommitmentStatus::Pending
                }
            })
            .unwrap_or(CommitmentStatus::Pending);

        let from_person = entity
            .metadata
            .get("from_person")
            .and_then(|v| v.as_str())
            .map(String::from);

        let to_person = entity
            .metadata
            .get("to_person")
            .and_then(|v| v.as_str())
            .map(String::from);

        let extracted_text = entity
            .metadata
            .get("extracted_text")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_default();

        let source_document = entity.source_refs.first().map(|r| r.document_id.clone());

        Commitment {
            id: entity.id.clone(),
            commitment_type,
            description: entity.name.clone(),
            from_person,
            to_person,
            due_date: None, // Would need to parse from metadata
            status,
            source_document,
            extracted_text,
            confidence: entity.confidence,
            project_id: None,
            follow_up_date: None,
            notes: None,
            created_at: entity.created_at,
            updated_at: entity.updated_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manager() -> CommitmentManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        CommitmentManager::new(store)
    }

    #[test]
    fn test_extract_commitments() {
        let manager = create_test_manager();

        let text = "I'll send you the report by end of day. John will review it tomorrow.";
        let result = manager.extract_from_text(text, Some("test-doc"));

        assert!(result.total_found > 0);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_extract_commitment_made() {
        let manager = create_test_manager();

        let text = "I will complete the project proposal by Friday.";
        let result = manager.extract_from_text(text, None);

        let made_commitments: Vec<_> = result
            .commitments
            .iter()
            .filter(|c| c.commitment_type == CommitmentDirection::Made)
            .collect();

        assert!(!made_commitments.is_empty());
    }

    #[tokio::test]
    async fn test_list_commitments() {
        let manager = create_test_manager();
        let commitments = manager.list(CommitmentFilter::default()).await.unwrap();

        // Empty store should return empty list
        assert!(commitments.is_empty());
    }

    #[tokio::test]
    async fn test_summary() {
        let manager = create_test_manager();
        let summary = manager.summary().await.unwrap();

        assert!(summary.health_score >= 0.0 && summary.health_score <= 100.0);
        assert!(summary.generated_at <= Utc::now());
    }

    #[test]
    fn test_commitment_status() {
        assert_eq!(CommitmentStatus::default(), CommitmentStatus::Pending);
    }

    #[test]
    fn test_commitment_direction() {
        let made = CommitmentDirection::Made;
        let received = CommitmentDirection::Received;
        assert_ne!(made, received);
    }
}

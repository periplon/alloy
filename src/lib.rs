//! Alloy: Hybrid Document Indexing MCP Server
//!
//! A Rust MCP server for indexing local and S3 files with hybrid search
//! combining vector similarity and full-text matching.

pub mod acl;
pub mod api;
pub mod auth;
pub mod backup;
pub mod cache;
pub mod calendar;
pub mod config;
pub mod coordinator;
pub mod embedding;
pub mod error;
pub mod gtd;
pub mod knowledge;
pub mod mcp;
pub mod metrics;
pub mod ontology;
pub mod processing;
pub mod query;
pub mod search;
pub mod sources;
pub mod storage;
pub mod versioning;
pub mod web;
pub mod webhooks;

pub use acl::{
    AclEntry, AclQueryFilter, AclResolver, AclStorage, DocumentAcl, MemoryAclStorage, Permission,
    Principal, SourceAcl,
};
pub use api::{create_rest_router, ApiState, RestApiConfig};
pub use auth::{AuthContext, AuthLayer, AuthMiddleware, Authenticator};
pub use backup::{BackupManager, BackupResult, ExportFormat, ExportOptions, RestoreResult};
pub use cache::{CacheStats, CachedEmbedder, QueryCache};
pub use calendar::{
    CalendarEvent, CalendarExtractor, CalendarFilter, CalendarManager, CalendarQueryEngine,
    CalendarQueryParams, CalendarQueryResponse, CalendarQueryType, CalendarStats, ConflictSeverity,
    EventRecurrence, EventType, EventUpdate, ExtractedCalendarEvent,
    ExtractionMethod as CalendarExtractionMethod, FreeTimeParams, FreeTimeSlot, QuerySummary,
    Reminder, ReminderType, SchedulingConflict,
};
pub use config::Config;
pub use coordinator::{IndexCoordinator, IndexCoordinatorBuilder, IndexProgress, IndexedSource};
pub use error::{AclError, AlloyError, AuthError, Result};
pub use gtd::{
    Project, ProjectFilter, ProjectHealth, ProjectManager, ProjectStatus, RecommendParams,
    SomedayFilter, SomedayItem, SomedayManager, Task, TaskFilter, TaskManager, TaskRecommendation,
    TaskStatus, WaitingFilter, WaitingFor, WaitingManager, WaitingStatus,
};
pub use knowledge::{
    DocumentSummary as KnowledgeDocumentSummary, EvidenceType, Expert, ExpertiseEvidence,
    KnowledgeQueryEngine, KnowledgeQueryParams, KnowledgeQueryResult, KnowledgeQueryType,
    QueryStats as KnowledgeQueryStats, RelatedTopic, RelationshipPath, ScoredEntity,
    TopicExpertise, TopicStats, TopicSummary, TraversalResult, TraversalStats,
};
pub use mcp::{run_server, AlloyServer};
pub use metrics::{get_metrics, HealthCheck, HealthState, HealthStatus, Metrics, MetricsSnapshot};
pub use ontology::extraction::{
    ActionDetector, ActionItem, ActionType, CommitmentType, DateType, DetectedAction,
    DocumentExtractionResult, EnergyLevel, EntityExtractable, EntityExtractionProcessor,
    EntityExtractionProcessorConfig, ExtractedEntity, ExtractedRelationship, ExtractionConfig,
    ExtractionMetadata, ExtractionMethod, ExtractionPipeline, ExtractionResult, LocalNerExtractor,
    NamedEntity, NamedEntityType, ParsedDate, Priority, RecurrencePattern, RecurrenceRule,
    RelationExtractor, RelationshipCandidate, SemanticRelationExtractor, TemporalExtraction,
    TemporalParser,
};
pub use ontology::{
    DocumentRef, EmbeddedOntologyStore, Entity, EntityFilter, EntityType, EntityUpdate,
    OntologyStats, OntologyStore, RelationType, Relationship, RelationshipFilter,
};
pub use query::{
    CalendarIntent, ClassificationResult, ExtractedParams, GtdIntent, IntentClassifier,
    KnowledgeIntent, QueryExecutor, QueryIntent, QueryMode, QueryStats, UnifiedQueryResult,
};
pub use web::{create_web_ui_router, WebUiConfig};
pub use webhooks::{
    create_dispatcher, DocumentDeletedData, DocumentIndexedData, IndexErrorData,
    SharedWebhookDispatcher, SourceAddedData, SourceRemovedData, WebhookConfig,
    WebhookDeliveryResult, WebhookDispatcher, WebhookEvent, WebhookPayload, WebhookStats,
};

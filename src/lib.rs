//! Alloy: Hybrid Document Indexing MCP Server
//!
//! A Rust MCP server for indexing local and S3 files with hybrid search
//! combining vector similarity and full-text matching.

pub mod acl;
pub mod api;
pub mod auth;
pub mod backup;
pub mod cache;
pub mod config;
pub mod coordinator;
pub mod embedding;
pub mod error;
pub mod mcp;
pub mod metrics;
pub mod ontology;
pub mod processing;
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
pub use config::Config;
pub use coordinator::{IndexCoordinator, IndexCoordinatorBuilder, IndexProgress, IndexedSource};
pub use error::{AclError, AlloyError, AuthError, Result};
pub use mcp::{run_server, AlloyServer};
pub use metrics::{get_metrics, HealthCheck, HealthState, HealthStatus, Metrics, MetricsSnapshot};
pub use ontology::{
    DocumentRef, EmbeddedOntologyStore, Entity, EntityFilter, EntityType, EntityUpdate,
    OntologyStats, OntologyStore, RelationType, Relationship, RelationshipFilter,
};
pub use web::{create_web_ui_router, WebUiConfig};
pub use webhooks::{
    create_dispatcher, DocumentDeletedData, DocumentIndexedData, IndexErrorData,
    SharedWebhookDispatcher, SourceAddedData, SourceRemovedData, WebhookConfig,
    WebhookDeliveryResult, WebhookDispatcher, WebhookEvent, WebhookPayload, WebhookStats,
};

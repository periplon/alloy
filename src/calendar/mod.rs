//! Calendar module for event management and temporal intelligence.
//!
//! This module provides comprehensive calendar functionality including:
//!
//! - **Calendar Events**: Time-specific commitments with support for recurrence
//! - **Event Management**: CRUD operations for calendar events
//! - **Query Engine**: Flexible querying with date ranges, filters, and search
//! - **Conflict Detection**: Automatic detection of scheduling conflicts
//! - **Free Time Finding**: Find available time slots in busy schedules
//! - **Event Extraction**: Extract calendar events from indexed documents
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Calendar Layer                              │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │              CalendarQueryEngine                          │  │
//! │  │  - Natural language queries                               │  │
//! │  │  - Free time finding                                      │  │
//! │  │  - Conflict detection                                     │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! │                           │                                      │
//! │                           ▼                                      │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │              CalendarManager                              │  │
//! │  │  - Event CRUD operations                                  │  │
//! │  │  - Date range queries                                     │  │
//! │  │  - Recurring event expansion                              │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! │                           │                                      │
//! │                           ▼                                      │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │              Ontology Store                               │  │
//! │  │  (CalendarEvent entities)                                 │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │              CalendarExtractor                            │  │
//! │  │  - Extract events from documents                          │  │
//! │  │  - Pattern matching                                       │  │
//! │  │  - Temporal parsing                                       │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use alloy::calendar::{CalendarManager, CalendarQueryEngine, CalendarEvent, EventType};
//! use chrono::{Duration, Utc};
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//!
//! // Create a calendar manager with a shared ontology store
//! let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
//! let calendar = CalendarManager::new(store.clone());
//! let query_engine = CalendarQueryEngine::new(store);
//!
//! // Create an event
//! let event = CalendarEvent::new("Team Meeting", Utc::now())
//!     .with_type(EventType::Meeting)
//!     .with_duration(Duration::hours(1))
//!     .with_location("Conference Room A")
//!     .with_participant("John")
//!     .with_participant("Jane");
//!
//! let event = calendar.create(event).await?;
//!
//! // Query today's events
//! let today = query_engine.query_today().await?;
//!
//! // Find free time slots
//! let free_slots = calendar.find_free_time(&FreeTimeParams::default()).await?;
//!
//! // Check for conflicts
//! let conflicts = calendar.check_conflicts(&new_event).await?;
//!
//! // Get calendar statistics
//! let stats = calendar.stats().await?;
//! ```

mod events;
pub mod extraction;
pub mod query;
pub mod types;

pub use events::CalendarManager;
pub use extraction::{CalendarExtractor, ExtractedCalendarEvent, ExtractionMethod};
pub use query::{CalendarQueryEngine, CalendarQueryParams, CalendarQueryResponse, QuerySummary};
pub use types::{
    CalendarEvent, CalendarFilter, CalendarQueryType, CalendarStats, ConflictSeverity,
    EventRecurrence, EventType, EventUpdate, FreeTimeParams, FreeTimeSlot, Reminder, ReminderType,
    SchedulingConflict,
};

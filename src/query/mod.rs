//! Natural Language Query Interface for Alloy.
//!
//! This module provides:
//! - Intent classification for natural language queries
//! - Query translation to structured GTD, Calendar, and Knowledge queries
//! - Unified query execution across all subsystems

pub mod classifier;
pub mod executor;
pub mod types;

pub use classifier::*;
pub use executor::*;
pub use types::*;

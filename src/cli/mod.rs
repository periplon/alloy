//! CLI module for Alloy command-line interface.
//!
//! This module provides command handlers for executing Alloy operations
//! either locally (via IndexCoordinator) or remotely (via MCP client).

mod commands;
mod local;
mod output;
mod remote;
pub mod types;

pub use commands::*;

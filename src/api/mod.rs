//! REST API module for Alloy.
//!
//! Provides a simple HTTP REST API as an alternative to the MCP protocol.
//! This makes it easier to integrate with web applications and services.

mod handlers;
mod rest;

pub use handlers::*;
pub use rest::*;

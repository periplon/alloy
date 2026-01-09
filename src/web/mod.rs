//! Web UI module for Alloy.
//!
//! Provides a simple web dashboard for browsing and searching indexed documents.

mod handlers;
mod static_files;

pub use handlers::*;
pub use static_files::*;

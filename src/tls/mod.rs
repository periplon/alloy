//! TLS certificate management with auto-generation and trust store installation.
//!
//! This module provides automatic local certificate generation similar to Caddy/mkcert,
//! allowing HTTPS to work locally without manual certificate setup.

mod cert_manager;
mod trust_store;

pub use cert_manager::{CertManager, GeneratedCerts};
pub use trust_store::{TrustResult, TrustStore};

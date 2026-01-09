//! Storage module for document and vector storage.
//!
//! This module provides multiple storage backends:
//! - `EmbeddedStorage`: Local storage using Tantivy + LanceDB
//! - `QdrantStorage`: External Qdrant vector database
//! - `QdrantHybridStorage`: Qdrant + Tantivy hybrid

mod embedded;
mod lance_backend;
mod qdrant_backend;
mod tantivy_backend;
mod traits;

pub use embedded::EmbeddedStorage;
pub use lance_backend::{LanceBackend, VectorSearchResult as LanceSearchResult};
pub use qdrant_backend::{QdrantHybridStorage, QdrantStorage};
pub use tantivy_backend::{FullTextResult, TantivyBackend};
pub use traits::*;

use crate::config::{Config, StorageBackendType};
use crate::error::Result;
use std::path::Path;
use std::sync::Arc;

/// Create a storage backend from configuration.
pub async fn create_storage(config: &Config, dimension: usize) -> Result<Arc<dyn StorageBackend>> {
    let data_dir = config.data_dir()?;

    match config.storage.backend {
        StorageBackendType::Embedded => {
            let storage = EmbeddedStorage::new(&data_dir, dimension).await?;
            Ok(Arc::new(storage))
        }
        StorageBackendType::Qdrant => {
            let storage =
                QdrantHybridStorage::new(&config.storage.qdrant, &data_dir, dimension).await?;
            Ok(Arc::new(storage))
        }
    }
}

/// Create an embedded storage backend directly.
pub async fn create_embedded_storage(data_dir: &Path, dimension: usize) -> Result<EmbeddedStorage> {
    EmbeddedStorage::new(data_dir, dimension).await
}

/// Create a Qdrant storage backend directly.
pub async fn create_qdrant_storage(
    config: &crate::config::QdrantConfig,
    dimension: usize,
) -> Result<QdrantStorage> {
    QdrantStorage::new(config, dimension).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_create_embedded_storage() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::default();
        config.storage.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.storage.backend = StorageBackendType::Embedded;

        let storage = create_storage(&config, 384).await;
        assert!(storage.is_ok());
    }
}

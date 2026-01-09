//! Version manager for coordinating versioning operations.
//!
//! Provides high-level API for creating versions, diffing, and restoring.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use super::diff::{compute_diff, compute_unified_diff, DiffResult};
use super::retention::{RetentionEnforcer, RetentionPolicy};
use super::storage::{
    ChangeType, CleanupStats, DocumentVersion, VersionMetadata, VersionStorage, VersionStorageType,
};

/// Versioning configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VersioningConfig {
    /// Enable versioning.
    pub enabled: bool,
    /// Storage type: "memory" or "file".
    pub storage: String,
    /// Store full version every N versions (0 = always full).
    pub delta_threshold: usize,
    /// Retention policy.
    pub retention: RetentionPolicy,
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            storage: "memory".to_string(),
            delta_threshold: 10,
            retention: RetentionPolicy::default(),
        }
    }
}

/// Result of comparing two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Version A info.
    pub version_a: VersionMetadata,
    /// Version B info.
    pub version_b: VersionMetadata,
    /// Unified diff output.
    pub unified_diff: String,
    /// Diff result with changes.
    pub diff: DiffResult,
}

/// Version manager for high-level versioning operations.
pub struct VersionManager {
    /// Version storage backend.
    storage: Arc<dyn VersionStorage>,
    /// Configuration.
    config: VersioningConfig,
    /// Retention enforcer.
    retention: RetentionEnforcer,
}

impl VersionManager {
    /// Create a new version manager.
    pub fn new(storage: Arc<dyn VersionStorage>, config: VersioningConfig) -> Self {
        let retention = RetentionEnforcer::new(config.retention.clone());
        Self {
            storage,
            config,
            retention,
        }
    }

    /// Check if versioning is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Create a new version when document content changes.
    pub async fn create_version(
        &self,
        doc_id: &str,
        content: &str,
        author: Option<String>,
    ) -> crate::error::Result<DocumentVersion> {
        let content_hash = DocumentVersion::compute_hash(content);
        let latest = self.storage.get_latest_version(doc_id).await?;

        // Determine version number and change type
        let (version_number, change_type) = match &latest {
            Some(prev) => {
                // Check if content actually changed
                if prev.content_hash == content_hash {
                    return Ok(prev.clone());
                }
                (prev.version_number + 1, ChangeType::ContentModified)
            }
            None => (1, ChangeType::Created),
        };

        // Decide whether to store full content or delta
        let storage_type = if self.should_store_full(version_number, &latest) {
            VersionStorageType::Full {
                content: content.to_string(),
            }
        } else if let Some(ref prev) = latest {
            // Compute delta from previous version
            let prev_content = self.storage.reconstruct_content(&prev.version_id).await?;
            let operations = super::diff::compute_delta_operations(&prev_content, content);
            VersionStorageType::Delta {
                base_version: prev.version_id.clone(),
                operations,
            }
        } else {
            VersionStorageType::Full {
                content: content.to_string(),
            }
        };

        let version = DocumentVersion {
            version_id: Uuid::new_v4().to_string(),
            document_id: doc_id.to_string(),
            version_number,
            timestamp: Utc::now(),
            author,
            change_type,
            content_hash,
            storage: storage_type,
            size_bytes: content.len(),
        };

        self.storage.store_version(version.clone()).await?;

        // Optionally run cleanup
        if self.config.retention.auto_cleanup {
            let _ = self
                .retention
                .cleanup(&*self.storage, &[doc_id.to_string()])
                .await;
        }

        Ok(version)
    }

    /// Decide whether to store full content or delta.
    fn should_store_full(&self, version_number: u64, latest: &Option<DocumentVersion>) -> bool {
        // Always store full for first version
        if version_number == 1 {
            return true;
        }

        // Store full if no delta threshold configured
        if self.config.delta_threshold == 0 {
            return true;
        }

        // Store full every N versions
        if version_number.is_multiple_of(self.config.delta_threshold as u64) {
            return true;
        }

        // Store full if previous version was also a delta (prevent long chains)
        if let Some(ref prev) = latest {
            if matches!(prev.storage, VersionStorageType::Delta { .. }) {
                // Check chain length - if we'd exceed threshold, store full
                // This is a simplified check; full implementation would traverse the chain
                return version_number.is_multiple_of(self.config.delta_threshold as u64 / 2 + 1);
            }
        }

        false
    }

    /// Mark a document as deleted.
    pub async fn mark_deleted(
        &self,
        doc_id: &str,
        author: Option<String>,
    ) -> crate::error::Result<DocumentVersion> {
        let latest = self.storage.get_latest_version(doc_id).await?;
        let version_number = latest.as_ref().map(|v| v.version_number + 1).unwrap_or(1);

        let version = DocumentVersion {
            version_id: Uuid::new_v4().to_string(),
            document_id: doc_id.to_string(),
            version_number,
            timestamp: Utc::now(),
            author,
            change_type: ChangeType::Deleted,
            content_hash: "deleted".to_string(),
            storage: VersionStorageType::Full {
                content: String::new(),
            },
            size_bytes: 0,
        };

        self.storage.store_version(version.clone()).await?;
        Ok(version)
    }

    /// Get document version history.
    pub async fn get_history(
        &self,
        doc_id: &str,
        limit: Option<usize>,
    ) -> crate::error::Result<Vec<VersionMetadata>> {
        let mut versions = self.storage.list_versions(doc_id).await?;

        if let Some(limit) = limit {
            versions.truncate(limit);
        }

        Ok(versions)
    }

    /// Get a specific version.
    pub async fn get_version(&self, version_id: &str) -> crate::error::Result<Option<DocumentVersion>> {
        self.storage.get_version(version_id).await
    }

    /// Get version content.
    pub async fn get_version_content(&self, version_id: &str) -> crate::error::Result<String> {
        self.storage.reconstruct_content(version_id).await
    }

    /// Compare two versions.
    pub async fn diff_versions(
        &self,
        version_a_id: &str,
        version_b_id: &str,
        context_lines: usize,
    ) -> crate::error::Result<VersionDiff> {
        let version_a = self
            .storage
            .get_version(version_a_id)
            .await?
            .ok_or_else(|| crate::error::StorageError::NotFound(version_a_id.to_string()))?;

        let version_b = self
            .storage
            .get_version(version_b_id)
            .await?
            .ok_or_else(|| crate::error::StorageError::NotFound(version_b_id.to_string()))?;

        let content_a = self.storage.reconstruct_content(version_a_id).await?;
        let content_b = self.storage.reconstruct_content(version_b_id).await?;

        let diff = compute_diff(&content_a, &content_b);
        let unified = compute_unified_diff(
            &content_a,
            &content_b,
            &format!("v{} ({})", version_a.version_number, version_a_id),
            &format!("v{} ({})", version_b.version_number, version_b_id),
            context_lines,
        );

        Ok(VersionDiff {
            version_a: VersionMetadata::from(&version_a),
            version_b: VersionMetadata::from(&version_b),
            unified_diff: unified.to_string(),
            diff,
        })
    }

    /// Restore a document to a previous version.
    pub async fn restore_version(
        &self,
        doc_id: &str,
        version_id: &str,
        author: Option<String>,
    ) -> crate::error::Result<DocumentVersion> {
        // Get the version to restore
        let old_version = self
            .storage
            .get_version(version_id)
            .await?
            .ok_or_else(|| crate::error::StorageError::NotFound(version_id.to_string()))?;

        // Verify it belongs to the right document
        if old_version.document_id != doc_id {
            return Err(crate::error::AlloyError::from(
                crate::error::StorageError::InvalidOperation(
                    "Version does not belong to specified document".to_string(),
                ),
            ));
        }

        // Reconstruct the content
        let content = self.storage.reconstruct_content(version_id).await?;

        // Get latest version number
        let latest_num = self
            .storage
            .get_latest_version_number(doc_id)
            .await?
            .unwrap_or(0);

        // Create new version marked as restoration
        let version = DocumentVersion {
            version_id: Uuid::new_v4().to_string(),
            document_id: doc_id.to_string(),
            version_number: latest_num + 1,
            timestamp: Utc::now(),
            author,
            change_type: ChangeType::Restored {
                from_version: version_id.to_string(),
            },
            content_hash: DocumentVersion::compute_hash(&content),
            storage: VersionStorageType::Full { content },
            size_bytes: old_version.size_bytes,
        };

        self.storage.store_version(version.clone()).await?;

        Ok(version)
    }

    /// Run retention cleanup for specific documents.
    pub async fn cleanup(&self, doc_ids: &[String]) -> crate::error::Result<CleanupStats> {
        self.retention.cleanup(&*self.storage, doc_ids).await
    }

    /// Delete all versions for a document.
    pub async fn delete_document(&self, doc_id: &str) -> crate::error::Result<usize> {
        self.storage.delete_document_versions(doc_id).await
    }

    /// Get total storage size.
    pub async fn total_size(&self) -> crate::error::Result<u64> {
        self.storage.total_size().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::storage::InMemoryVersionStorage;

    fn create_test_manager() -> VersionManager {
        let storage = Arc::new(InMemoryVersionStorage::new());
        let config = VersioningConfig {
            enabled: true,
            delta_threshold: 5,
            ..Default::default()
        };
        VersionManager::new(storage, config)
    }

    #[tokio::test]
    async fn test_create_version() {
        let manager = create_test_manager();

        let v1 = manager
            .create_version("doc1", "Hello, World!", Some("user1".to_string()))
            .await
            .unwrap();

        assert_eq!(v1.version_number, 1);
        assert_eq!(v1.change_type, ChangeType::Created);
        assert!(matches!(v1.storage, VersionStorageType::Full { .. }));
    }

    #[tokio::test]
    async fn test_version_history() {
        let manager = create_test_manager();

        manager
            .create_version("doc1", "Version 1", None)
            .await
            .unwrap();
        manager
            .create_version("doc1", "Version 2", None)
            .await
            .unwrap();
        manager
            .create_version("doc1", "Version 3", None)
            .await
            .unwrap();

        let history = manager.get_history("doc1", None).await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].version_number, 3); // Most recent first
    }

    #[tokio::test]
    async fn test_no_change_no_version() {
        let manager = create_test_manager();

        let v1 = manager
            .create_version("doc1", "Same content", None)
            .await
            .unwrap();

        let v2 = manager
            .create_version("doc1", "Same content", None)
            .await
            .unwrap();

        // Should return the same version
        assert_eq!(v1.version_id, v2.version_id);
    }

    #[tokio::test]
    async fn test_diff_versions() {
        let manager = create_test_manager();

        let v1 = manager
            .create_version("doc1", "Line 1\nLine 2\nLine 3", None)
            .await
            .unwrap();

        let v2 = manager
            .create_version("doc1", "Line 1\nModified Line\nLine 3", None)
            .await
            .unwrap();

        let diff = manager
            .diff_versions(&v1.version_id, &v2.version_id, 3)
            .await
            .unwrap();

        assert!(diff.unified_diff.contains("-Line 2"));
        assert!(diff.unified_diff.contains("+Modified Line"));
    }

    #[tokio::test]
    async fn test_restore_version() {
        let manager = create_test_manager();

        let v1 = manager
            .create_version("doc1", "Original content", None)
            .await
            .unwrap();

        manager
            .create_version("doc1", "Modified content", None)
            .await
            .unwrap();

        let restored = manager
            .restore_version("doc1", &v1.version_id, Some("admin".to_string()))
            .await
            .unwrap();

        assert_eq!(restored.version_number, 3);
        assert!(matches!(
            restored.change_type,
            ChangeType::Restored { from_version } if from_version == v1.version_id
        ));

        let content = manager
            .get_version_content(&restored.version_id)
            .await
            .unwrap();
        assert_eq!(content, "Original content");
    }

    #[tokio::test]
    async fn test_delete_document_versions() {
        let manager = create_test_manager();

        manager.create_version("doc1", "V1", None).await.unwrap();
        manager.create_version("doc1", "V2", None).await.unwrap();
        manager.create_version("doc1", "V3", None).await.unwrap();

        let deleted = manager.delete_document("doc1").await.unwrap();
        assert_eq!(deleted, 3);

        let history = manager.get_history("doc1", None).await.unwrap();
        assert!(history.is_empty());
    }
}

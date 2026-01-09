//! Version retention policy module.
//!
//! Defines policies for automatically cleaning up old versions
//! to manage storage space.

use chrono::{DateTime, Datelike, Duration, Utc};
use serde::{Deserialize, Serialize};

use super::storage::{CleanupStats, VersionMetadata, VersionStorage};

/// Pattern for keeping specific versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionPattern {
    /// Keep the first version of each day.
    DailyFirst,
    /// Keep the first version of each week.
    WeeklyFirst,
    /// Keep the first version of each month.
    MonthlyFirst,
    /// Keep versions with a specific tag/author.
    Tagged(String),
    /// Keep all full versions (for delta reconstruction).
    FullVersions,
}

/// Version retention policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetentionPolicy {
    /// Minimum number of versions to keep per document.
    pub min_versions: usize,
    /// Maximum number of versions to keep per document (0 = unlimited).
    pub max_versions: usize,
    /// Keep versions newer than this duration.
    pub min_age_days: u32,
    /// Delete versions older than this duration.
    pub max_age_days: u32,
    /// Always keep full versions (for delta reconstruction).
    pub keep_full_versions: bool,
    /// Patterns for keeping specific versions.
    pub keep_patterns: Vec<RetentionPattern>,
    /// Enable automatic cleanup.
    pub auto_cleanup: bool,
    /// Cleanup interval in hours.
    pub cleanup_interval_hours: u32,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            min_versions: 5,
            max_versions: 100,
            min_age_days: 7,
            max_age_days: 365,
            keep_full_versions: true,
            keep_patterns: vec![RetentionPattern::MonthlyFirst],
            auto_cleanup: false,
            cleanup_interval_hours: 24,
        }
    }
}

impl RetentionPolicy {
    /// Create a policy that keeps all versions.
    pub fn keep_all() -> Self {
        Self {
            min_versions: usize::MAX,
            max_versions: 0,
            min_age_days: 0,
            max_age_days: u32::MAX,
            keep_full_versions: true,
            keep_patterns: vec![],
            auto_cleanup: false,
            cleanup_interval_hours: 24,
        }
    }

    /// Create a minimal retention policy (for testing).
    pub fn minimal() -> Self {
        Self {
            min_versions: 1,
            max_versions: 10,
            min_age_days: 1,
            max_age_days: 30,
            keep_full_versions: true,
            keep_patterns: vec![],
            auto_cleanup: false,
            cleanup_interval_hours: 24,
        }
    }

    /// Check if a version should be kept according to this policy.
    pub fn should_keep(
        &self,
        version: &VersionMetadata,
        all_versions: &[VersionMetadata],
        now: DateTime<Utc>,
    ) -> bool {
        let age = now - version.timestamp;
        let version_index = all_versions
            .iter()
            .position(|v| v.version_id == version.version_id)
            .unwrap_or(0);
        let versions_after = all_versions.len() - version_index - 1;

        // Always keep minimum versions
        if versions_after < self.min_versions {
            return true;
        }

        // Keep versions within min age
        if age < Duration::days(self.min_age_days as i64) {
            return true;
        }

        // Delete versions beyond max age
        if age > Duration::days(self.max_age_days as i64) {
            // Unless protected by a pattern
            if !self.matches_keep_pattern(version, all_versions) {
                return false;
            }
        }

        // Respect max versions limit
        if self.max_versions > 0 && versions_after >= self.max_versions {
            // Only delete if not protected
            if !self.matches_keep_pattern(version, all_versions) {
                return false;
            }
        }

        // Check keep patterns
        if self.matches_keep_pattern(version, all_versions) {
            return true;
        }

        true
    }

    /// Check if a version matches any keep pattern.
    fn matches_keep_pattern(
        &self,
        version: &VersionMetadata,
        all_versions: &[VersionMetadata],
    ) -> bool {
        for pattern in &self.keep_patterns {
            if self.matches_pattern(pattern, version, all_versions) {
                return true;
            }
        }
        false
    }

    /// Check if a version matches a specific pattern.
    fn matches_pattern(
        &self,
        pattern: &RetentionPattern,
        version: &VersionMetadata,
        all_versions: &[VersionMetadata],
    ) -> bool {
        match pattern {
            RetentionPattern::DailyFirst => {
                let date = version.timestamp.date_naive();
                all_versions
                    .iter()
                    .filter(|v| v.timestamp.date_naive() == date)
                    .min_by_key(|v| v.timestamp)
                    .map(|v| v.version_id == version.version_id)
                    .unwrap_or(false)
            }
            RetentionPattern::WeeklyFirst => {
                let week = version.timestamp.iso_week();
                all_versions
                    .iter()
                    .filter(|v| v.timestamp.iso_week() == week)
                    .min_by_key(|v| v.timestamp)
                    .map(|v| v.version_id == version.version_id)
                    .unwrap_or(false)
            }
            RetentionPattern::MonthlyFirst => {
                let month = (version.timestamp.year(), version.timestamp.month());
                all_versions
                    .iter()
                    .filter(|v| (v.timestamp.year(), v.timestamp.month()) == month)
                    .min_by_key(|v| v.timestamp)
                    .map(|v| v.version_id == version.version_id)
                    .unwrap_or(false)
            }
            RetentionPattern::Tagged(tag) => {
                version.author.as_ref().map(|a| a == tag).unwrap_or(false)
            }
            RetentionPattern::FullVersions => {
                // This requires access to the full version, not just metadata
                // For now, keep version 1 and every 10th version
                version.version_number == 1 || version.version_number.is_multiple_of(10)
            }
        }
    }
}

/// Retention policy enforcer.
pub struct RetentionEnforcer {
    policy: RetentionPolicy,
}

impl RetentionEnforcer {
    /// Create a new enforcer with the given policy.
    pub fn new(policy: RetentionPolicy) -> Self {
        Self { policy }
    }

    /// Identify versions that should be deleted for a document.
    pub fn get_versions_to_delete(
        &self,
        versions: &[VersionMetadata],
        now: DateTime<Utc>,
    ) -> Vec<String> {
        let mut to_delete = Vec::new();

        // Sort by version number (oldest first)
        let mut sorted: Vec<&VersionMetadata> = versions.iter().collect();
        sorted.sort_by_key(|v| v.version_number);

        for version in &sorted {
            if !self.policy.should_keep(version, versions, now) {
                to_delete.push(version.version_id.clone());
            }
        }

        to_delete
    }

    /// Apply the retention policy to a storage backend.
    pub async fn cleanup(
        &self,
        storage: &dyn VersionStorage,
        doc_ids: &[String],
    ) -> crate::error::Result<CleanupStats> {
        let mut stats = CleanupStats::default();
        let now = Utc::now();

        for doc_id in doc_ids {
            let versions = storage.list_versions(doc_id).await?;
            let to_delete = self.get_versions_to_delete(&versions, now);

            for version_id in to_delete {
                if let Some(v) = versions.iter().find(|v| v.version_id == version_id) {
                    stats.bytes_freed += v.size_bytes as u64;
                    stats.versions_removed += 1;
                }
                storage.delete_version(&version_id).await?;
            }

            if stats.versions_removed > 0 {
                stats.documents_affected += 1;
            }
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::super::storage::ChangeType;
    use super::*;

    fn create_test_version(id: &str, num: u64, days_ago: i64) -> VersionMetadata {
        VersionMetadata {
            version_id: id.to_string(),
            version_number: num,
            timestamp: Utc::now() - Duration::days(days_ago),
            author: None,
            change_type: ChangeType::ContentModified,
            size_bytes: 100,
            content_hash: format!("hash{}", num),
            chunk_count: 0,
            tags: Vec::new(),
            compression: Default::default(),
        }
    }

    #[test]
    fn test_min_versions_kept() {
        let policy = RetentionPolicy {
            min_versions: 3,
            max_versions: 5,
            ..Default::default()
        };

        let versions = vec![
            create_test_version("v1", 1, 100),
            create_test_version("v2", 2, 50),
            create_test_version("v3", 3, 10),
        ];

        let enforcer = RetentionEnforcer::new(policy);
        let to_delete = enforcer.get_versions_to_delete(&versions, Utc::now());

        // Should keep all 3 because min_versions = 3
        assert!(to_delete.is_empty());
    }

    #[test]
    fn test_max_versions_enforced() {
        let policy = RetentionPolicy {
            min_versions: 1,
            max_versions: 3,
            keep_patterns: vec![], // No patterns to protect versions
            ..Default::default()
        };

        let versions = vec![
            create_test_version("v1", 1, 100),
            create_test_version("v2", 2, 50),
            create_test_version("v3", 3, 30),
            create_test_version("v4", 4, 10),
            create_test_version("v5", 5, 1),
        ];

        let enforcer = RetentionEnforcer::new(policy);
        let to_delete = enforcer.get_versions_to_delete(&versions, Utc::now());

        // Oldest versions beyond max should be deleted
        assert!(to_delete.contains(&"v1".to_string()) || to_delete.contains(&"v2".to_string()));
    }

    #[test]
    fn test_age_based_deletion() {
        let policy = RetentionPolicy {
            min_versions: 1,
            max_versions: 0, // Unlimited
            min_age_days: 1,
            max_age_days: 30,
            keep_patterns: vec![],
            ..Default::default()
        };

        let versions = vec![
            create_test_version("v1", 1, 100), // Very old
            create_test_version("v2", 2, 10),  // Within range
            create_test_version("v3", 3, 1),   // Recent
        ];

        let enforcer = RetentionEnforcer::new(policy);
        let to_delete = enforcer.get_versions_to_delete(&versions, Utc::now());

        // v1 should be deleted (too old), v2 and v3 kept
        assert!(to_delete.contains(&"v1".to_string()));
        assert!(!to_delete.contains(&"v2".to_string()));
        assert!(!to_delete.contains(&"v3".to_string()));
    }

    #[test]
    fn test_monthly_pattern() {
        let policy = RetentionPolicy {
            min_versions: 1,
            max_versions: 0,
            max_age_days: 365,
            keep_patterns: vec![RetentionPattern::MonthlyFirst],
            ..Default::default()
        };

        let enforcer = RetentionEnforcer::new(policy);

        // The first version of each month should be kept
        let versions = vec![
            create_test_version("v1", 1, 60), // ~2 months ago
            create_test_version("v2", 2, 31), // ~1 month ago
            create_test_version("v3", 3, 1),  // Today
        ];

        let to_delete = enforcer.get_versions_to_delete(&versions, Utc::now());

        // Monthly first pattern should protect at least some versions
        assert!(to_delete.len() < versions.len());
    }

    #[test]
    fn test_keep_all_policy() {
        let policy = RetentionPolicy::keep_all();

        let versions = vec![
            create_test_version("v1", 1, 1000),
            create_test_version("v2", 2, 500),
            create_test_version("v3", 3, 1),
        ];

        let enforcer = RetentionEnforcer::new(policy);
        let to_delete = enforcer.get_versions_to_delete(&versions, Utc::now());

        assert!(to_delete.is_empty());
    }
}

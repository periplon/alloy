//! ACL query filter for rewriting search queries.

use std::sync::Arc;

use super::{AclResolver, Permission};
use crate::auth::AuthContext;
use crate::config::AclConfig;
use crate::error::Result;

/// ACL query filter that rewrites search queries to include access control.
pub struct AclQueryFilter {
    resolver: Arc<AclResolver>,
    config: AclConfig,
}

impl AclQueryFilter {
    /// Create a new ACL query filter.
    pub fn new(resolver: Arc<AclResolver>, config: AclConfig) -> Self {
        Self { resolver, config }
    }

    /// Check if ACL filtering should be applied.
    pub fn should_filter(&self) -> bool {
        self.config.enabled && self.config.enforce_on_search
    }

    /// Get accessible document IDs for a user.
    ///
    /// This is used to filter search results to only include documents
    /// the user has access to.
    pub async fn get_accessible_doc_ids(&self, auth: &AuthContext) -> Result<Option<Vec<String>>> {
        if !self.should_filter() {
            return Ok(None); // No filtering needed
        }

        // If user is anonymous and default is not public, return empty list
        if auth.anonymous && !self.config.default_public {
            return Ok(Some(Vec::new()));
        }

        // For now, we don't have a way to get all accessible documents
        // In a production system, you would query the ACL storage
        // For now, we return None to indicate no filtering
        Ok(None)
    }

    /// Check if a document is accessible to a user.
    pub async fn is_accessible(&self, auth: &AuthContext, doc_id: &str) -> Result<bool> {
        if !self.should_filter() {
            return Ok(true);
        }

        let result = self
            .resolver
            .check_permission(auth, doc_id, Permission::Read)
            .await?;

        Ok(result.allowed)
    }

    /// Filter a list of document IDs to only include accessible ones.
    pub async fn filter_doc_ids(
        &self,
        auth: &AuthContext,
        doc_ids: Vec<String>,
    ) -> Result<Vec<String>> {
        if !self.should_filter() {
            return Ok(doc_ids);
        }

        let mut accessible = Vec::new();
        for doc_id in doc_ids {
            if self.is_accessible(auth, &doc_id).await? {
                accessible.push(doc_id);
            }
        }

        Ok(accessible)
    }

    /// Build a filter expression for the ACL.
    ///
    /// This returns a set of principals that should be matched
    /// in the document's ACL for the user to have access.
    pub fn build_principal_filter(&self, auth: &AuthContext) -> Vec<String> {
        let mut principals = Vec::new();

        // Always include "everyone"
        principals.push("everyone".to_string());

        if !auth.anonymous {
            // Include "authenticated"
            principals.push("authenticated".to_string());

            // Include user ID
            if let Some(ref user_id) = auth.user_id {
                principals.push(format!("user:{}", user_id));
            }

            // Include roles
            for role in &auth.roles {
                principals.push(format!("role:{}", role));
            }

            // Include groups
            for group in &auth.groups {
                principals.push(format!("group:{}", group));
            }
        }

        principals
    }
}

/// Build metadata filter for ACL-aware search.
///
/// This creates a filter that can be applied to search results
/// to only include documents the user has access to.
#[allow(dead_code)]
pub fn build_acl_filter(auth: &AuthContext, config: &AclConfig) -> AclSearchFilter {
    AclSearchFilter {
        principals: build_principals(auth),
        require_public: auth.anonymous && !config.default_authenticated_read,
        allow_authenticated_read: config.default_authenticated_read && !auth.anonymous,
        owner_id: auth.user_id.clone(),
    }
}

/// Build list of principals for a user.
#[allow(dead_code)]
fn build_principals(auth: &AuthContext) -> Vec<String> {
    let mut principals = Vec::new();

    // Everyone always matches
    principals.push("everyone".to_string());

    if !auth.anonymous {
        // Authenticated users
        principals.push("authenticated".to_string());

        if let Some(ref user_id) = auth.user_id {
            principals.push(format!("user:{}", user_id));
        }

        for role in &auth.roles {
            principals.push(format!("role:{}", role));
        }

        for group in &auth.groups {
            principals.push(format!("group:{}", group));
        }
    }

    principals
}

/// ACL search filter for rewriting queries.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AclSearchFilter {
    /// Principals that should match for access.
    pub principals: Vec<String>,
    /// Whether to require public access.
    pub require_public: bool,
    /// Whether authenticated read is allowed.
    pub allow_authenticated_read: bool,
    /// Owner ID (if authenticated).
    pub owner_id: Option<String>,
}

impl AclSearchFilter {
    /// Check if a document with given metadata is accessible.
    #[allow(dead_code)]
    pub fn matches(&self, owner: Option<&str>, acl_principals: &[String], is_public: bool) -> bool {
        // Owner always has access
        if let (Some(ref my_id), Some(doc_owner)) = (&self.owner_id, owner) {
            if my_id == doc_owner {
                return true;
            }
        }

        // Check if public
        if is_public {
            return true;
        }

        // Check if any principal matches
        for principal in &self.principals {
            if acl_principals.contains(principal) {
                return true;
            }
        }

        // Check authenticated read default - only applies when no explicit ACL entries
        // (i.e., document has no ACL and defaults to authenticated read)
        if self.allow_authenticated_read && self.owner_id.is_some() && acl_principals.is_empty() {
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_principal_filter_authenticated() {
        let auth = AuthContext::authenticated(
            "user123".to_string(),
            vec!["editor".to_string(), "viewer".to_string()],
            "test",
        );

        let filter = AclQueryFilter::new(
            Arc::new(AclResolver::new(
                Arc::new(crate::acl::MemoryAclStorage::new()),
                AclConfig::default(),
            )),
            AclConfig::default(),
        );

        let principals = filter.build_principal_filter(&auth);

        assert!(principals.contains(&"everyone".to_string()));
        assert!(principals.contains(&"authenticated".to_string()));
        assert!(principals.contains(&"user:user123".to_string()));
        assert!(principals.contains(&"role:editor".to_string()));
        assert!(principals.contains(&"role:viewer".to_string()));
    }

    #[test]
    fn test_principal_filter_anonymous() {
        let auth = AuthContext::anonymous();

        let filter = AclQueryFilter::new(
            Arc::new(AclResolver::new(
                Arc::new(crate::acl::MemoryAclStorage::new()),
                AclConfig::default(),
            )),
            AclConfig::default(),
        );

        let principals = filter.build_principal_filter(&auth);

        assert!(principals.contains(&"everyone".to_string()));
        assert!(!principals.contains(&"authenticated".to_string()));
        assert_eq!(principals.len(), 1);
    }

    #[test]
    fn test_acl_search_filter() {
        let auth =
            AuthContext::authenticated("user1".to_string(), vec!["editor".to_string()], "test");
        let config = AclConfig {
            enabled: true,
            default_authenticated_read: true,
            ..Default::default()
        };

        let filter = build_acl_filter(&auth, &config);

        // Owner matches
        assert!(filter.matches(Some("user1"), &[], false));

        // Public matches
        assert!(filter.matches(Some("other"), &[], true));

        // Principal matches
        assert!(filter.matches(Some("other"), &["role:editor".to_string()], false));

        // No match
        assert!(!filter.matches(Some("other"), &["role:admin".to_string()], false));
    }

    #[test]
    fn test_anonymous_filter() {
        let auth = AuthContext::anonymous();
        let config = AclConfig {
            enabled: true,
            default_authenticated_read: false,
            default_public: false,
            ..Default::default()
        };

        let filter = build_acl_filter(&auth, &config);

        // Only public matches
        assert!(filter.matches(Some("owner"), &[], true));

        // Everyone principal matches
        assert!(filter.matches(Some("owner"), &["everyone".to_string()], false));

        // Nothing else matches
        assert!(!filter.matches(Some("owner"), &[], false));
    }
}

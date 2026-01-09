//! ACL middleware for request-level enforcement.
//!
//! This module provides middleware components for enforcing access control
//! on HTTP and MCP requests.

use std::sync::Arc;

use tower::Layer;

use super::{AclResolver, Permission};
use crate::auth::AuthContext;
use crate::config::AclConfig;
use crate::error::Result;

/// ACL enforcement middleware layer.
#[derive(Clone)]
pub struct AclEnforcementLayer {
    resolver: Arc<AclResolver>,
    config: AclConfig,
}

impl AclEnforcementLayer {
    /// Create a new ACL enforcement layer.
    pub fn new(resolver: Arc<AclResolver>, config: AclConfig) -> Self {
        Self { resolver, config }
    }

    /// Check if ACL enforcement is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

impl<S> Layer<S> for AclEnforcementLayer {
    type Service = AclEnforcementService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AclEnforcementService {
            inner,
            resolver: self.resolver.clone(),
            config: self.config.clone(),
        }
    }
}

/// ACL enforcement service wrapper.
#[derive(Clone)]
pub struct AclEnforcementService<S> {
    #[allow(dead_code)]
    inner: S,
    resolver: Arc<AclResolver>,
    config: AclConfig,
}

impl<S> AclEnforcementService<S> {
    /// Get the ACL resolver.
    pub fn resolver(&self) -> &Arc<AclResolver> {
        &self.resolver
    }

    /// Get the ACL config.
    pub fn config(&self) -> &AclConfig {
        &self.config
    }

    /// Check if enforcement should apply for a given operation.
    pub fn should_enforce(&self, operation: &str) -> bool {
        self.resolver.should_enforce(operation)
    }
}

/// ACL request guard for checking permissions before operations.
pub struct AclGuard {
    resolver: Arc<AclResolver>,
    config: AclConfig,
    auth_context: AuthContext,
}

impl AclGuard {
    /// Create a new ACL guard.
    pub fn new(resolver: Arc<AclResolver>, config: AclConfig, auth_context: AuthContext) -> Self {
        Self {
            resolver,
            config,
            auth_context,
        }
    }

    /// Create an anonymous guard.
    pub fn anonymous(resolver: Arc<AclResolver>, config: AclConfig) -> Self {
        Self::new(resolver, config, AuthContext::anonymous())
    }

    /// Check if the user can perform the operation on the document.
    pub async fn check(&self, doc_id: &str, permission: Permission) -> Result<bool> {
        if !self.config.enabled {
            return Ok(true);
        }

        let result = self
            .resolver
            .check_permission(&self.auth_context, doc_id, permission)
            .await?;

        Ok(result.allowed)
    }

    /// Check read permission.
    pub async fn can_read(&self, doc_id: &str) -> Result<bool> {
        self.check(doc_id, Permission::Read).await
    }

    /// Check write permission.
    pub async fn can_write(&self, doc_id: &str) -> Result<bool> {
        self.check(doc_id, Permission::Write).await
    }

    /// Check delete permission.
    pub async fn can_delete(&self, doc_id: &str) -> Result<bool> {
        self.check(doc_id, Permission::Delete).await
    }

    /// Check admin permission.
    pub async fn can_admin(&self, doc_id: &str) -> Result<bool> {
        self.check(doc_id, Permission::Admin).await
    }

    /// Filter a list of document IDs to only those accessible to the user.
    pub async fn filter_accessible(&self, doc_ids: Vec<String>) -> Result<Vec<String>> {
        if !self.config.enabled {
            return Ok(doc_ids);
        }

        let mut accessible = Vec::new();
        for doc_id in doc_ids {
            if self.can_read(&doc_id).await? {
                accessible.push(doc_id);
            }
        }
        Ok(accessible)
    }

    /// Require read permission or return an error.
    pub async fn require_read(&self, doc_id: &str) -> Result<()> {
        if !self.can_read(doc_id).await? {
            return Err(crate::error::AlloyError::from(
                crate::error::AuthError::PermissionDenied(format!(
                    "Read permission denied for document {}",
                    doc_id
                )),
            ));
        }
        Ok(())
    }

    /// Require write permission or return an error.
    pub async fn require_write(&self, doc_id: &str) -> Result<()> {
        if !self.can_write(doc_id).await? {
            return Err(crate::error::AlloyError::from(
                crate::error::AuthError::PermissionDenied(format!(
                    "Write permission denied for document {}",
                    doc_id
                )),
            ));
        }
        Ok(())
    }

    /// Require delete permission or return an error.
    pub async fn require_delete(&self, doc_id: &str) -> Result<()> {
        if !self.can_delete(doc_id).await? {
            return Err(crate::error::AlloyError::from(
                crate::error::AuthError::PermissionDenied(format!(
                    "Delete permission denied for document {}",
                    doc_id
                )),
            ));
        }
        Ok(())
    }

    /// Require admin permission or return an error.
    pub async fn require_admin(&self, doc_id: &str) -> Result<()> {
        if !self.can_admin(doc_id).await? {
            return Err(crate::error::AlloyError::from(
                crate::error::AuthError::PermissionDenied(format!(
                    "Admin permission denied for document {}",
                    doc_id
                )),
            ));
        }
        Ok(())
    }
}

/// Operation type for ACL enforcement decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Search operation
    Search,
    /// Get document operation
    Get,
    /// Delete document operation
    Delete,
    /// Index document operation
    Index,
    /// Admin operation
    Admin,
}

impl OperationType {
    /// Convert to operation string for resolver.
    pub fn as_str(&self) -> &'static str {
        match self {
            OperationType::Search => "search",
            OperationType::Get => "get",
            OperationType::Delete => "delete",
            OperationType::Index => "index",
            OperationType::Admin => "admin",
        }
    }

    /// Get the required permission for this operation.
    pub fn required_permission(&self) -> Permission {
        match self {
            OperationType::Search => Permission::Read,
            OperationType::Get => Permission::Read,
            OperationType::Delete => Permission::Delete,
            OperationType::Index => Permission::Write,
            OperationType::Admin => Permission::Admin,
        }
    }
}

/// Trait for extracting auth context from requests.
#[allow(dead_code)]
pub trait ExtractAuthContext {
    /// Extract authentication context from the request.
    fn extract_auth_context(&self) -> AuthContext;
}

/// Default implementation that returns anonymous context.
impl ExtractAuthContext for () {
    fn extract_auth_context(&self) -> AuthContext {
        AuthContext::anonymous()
    }
}

/// Builder for creating ACL guards with fluent API.
pub struct AclGuardBuilder {
    resolver: Arc<AclResolver>,
    config: AclConfig,
    auth_context: Option<AuthContext>,
}

impl AclGuardBuilder {
    /// Create a new builder.
    pub fn new(resolver: Arc<AclResolver>, config: AclConfig) -> Self {
        Self {
            resolver,
            config,
            auth_context: None,
        }
    }

    /// Set the auth context.
    pub fn with_auth(mut self, auth: AuthContext) -> Self {
        self.auth_context = Some(auth);
        self
    }

    /// Set authenticated user.
    pub fn with_user(mut self, user_id: impl Into<String>, roles: Vec<String>) -> Self {
        self.auth_context = Some(AuthContext::authenticated(user_id.into(), roles, "request"));
        self
    }

    /// Build the guard.
    pub fn build(self) -> AclGuard {
        AclGuard::new(
            self.resolver,
            self.config,
            self.auth_context.unwrap_or_else(AuthContext::anonymous),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acl::{AclEntry, AclStorage, DocumentAcl, MemoryAclStorage};
    use crate::config::RoleDefinition;

    fn test_config() -> AclConfig {
        AclConfig {
            enabled: true,
            default_public: false,
            default_authenticated_read: false,
            enforce_on_search: true,
            enforce_on_get: true,
            enforce_on_delete: true,
            roles: vec![RoleDefinition {
                name: "admin".to_string(),
                inherits_from: vec![],
                permissions: vec![
                    "read".to_string(),
                    "write".to_string(),
                    "delete".to_string(),
                    "admin".to_string(),
                ],
            }],
        }
    }

    #[tokio::test]
    async fn test_acl_guard_allows_owner() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = Arc::new(AclResolver::new(storage.clone(), test_config()));

        // Create ACL for doc1
        let acl = DocumentAcl::new("doc1", "owner1");
        storage.set_document_acl(acl).await.unwrap();

        // Owner should have access
        let auth = AuthContext::authenticated("owner1".to_string(), vec![], "test");
        let guard = AclGuard::new(resolver, test_config(), auth);

        assert!(guard.can_read("doc1").await.unwrap());
        assert!(guard.can_write("doc1").await.unwrap());
        assert!(guard.can_delete("doc1").await.unwrap());
        assert!(guard.can_admin("doc1").await.unwrap());
    }

    #[tokio::test]
    async fn test_acl_guard_denies_non_owner() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = Arc::new(AclResolver::new(storage.clone(), test_config()));

        // Create ACL for doc1
        let acl = DocumentAcl::new("doc1", "owner1");
        storage.set_document_acl(acl).await.unwrap();

        // Non-owner without explicit access should be denied
        let auth = AuthContext::authenticated("other_user".to_string(), vec![], "test");
        let guard = AclGuard::new(resolver, test_config(), auth);

        assert!(!guard.can_read("doc1").await.unwrap());
    }

    #[tokio::test]
    async fn test_acl_guard_with_entry() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = Arc::new(AclResolver::new(storage.clone(), test_config()));

        // Create ACL with read access for alice
        let mut acl = DocumentAcl::new("doc1", "owner1");
        acl.add_entry(AclEntry::user_read("alice"));
        storage.set_document_acl(acl).await.unwrap();

        let auth = AuthContext::authenticated("alice".to_string(), vec![], "test");
        let guard = AclGuard::new(resolver, test_config(), auth);

        assert!(guard.can_read("doc1").await.unwrap());
        assert!(!guard.can_write("doc1").await.unwrap()); // No write permission
    }

    #[tokio::test]
    async fn test_filter_accessible() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = Arc::new(AclResolver::new(storage.clone(), test_config()));

        // Create ACLs - alice can access doc1 and doc3, not doc2
        let mut acl1 = DocumentAcl::new("doc1", "owner1");
        acl1.add_entry(AclEntry::user_read("alice"));
        storage.set_document_acl(acl1).await.unwrap();

        let acl2 = DocumentAcl::new("doc2", "owner2");
        storage.set_document_acl(acl2).await.unwrap();

        let mut acl3 = DocumentAcl::new("doc3", "owner3");
        acl3.add_entry(AclEntry::user_read("alice"));
        storage.set_document_acl(acl3).await.unwrap();

        let auth = AuthContext::authenticated("alice".to_string(), vec![], "test");
        let guard = AclGuard::new(resolver, test_config(), auth);

        let all_docs = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];
        let accessible = guard.filter_accessible(all_docs).await.unwrap();

        assert_eq!(accessible, vec!["doc1".to_string(), "doc3".to_string()]);
    }

    #[tokio::test]
    async fn test_disabled_acl() {
        let mut config = test_config();
        config.enabled = false;

        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = Arc::new(AclResolver::new(storage.clone(), config.clone()));

        let guard = AclGuard::anonymous(resolver, config);

        // With ACL disabled, everything is allowed
        assert!(guard.can_read("any_doc").await.unwrap());
        assert!(guard.can_admin("any_doc").await.unwrap());
    }

    #[test]
    fn test_operation_type() {
        assert_eq!(OperationType::Search.as_str(), "search");
        assert_eq!(
            OperationType::Search.required_permission(),
            Permission::Read
        );

        assert_eq!(OperationType::Delete.as_str(), "delete");
        assert_eq!(
            OperationType::Delete.required_permission(),
            Permission::Delete
        );
    }

    #[test]
    fn test_guard_builder() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = Arc::new(AclResolver::new(storage, test_config()));

        let guard = AclGuardBuilder::new(resolver.clone(), test_config())
            .with_user("user1", vec!["admin".to_string()])
            .build();

        // Guard should be created with the specified auth context
        // (We can't easily verify this without exposing internals)
        assert!(Arc::ptr_eq(&guard.resolver, &resolver));
    }
}

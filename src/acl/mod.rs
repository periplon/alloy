//! Access Control List (ACL) module for Alloy.
//!
//! Provides document-level access control for multi-tenant deployments.

mod query_filter;
mod resolver;
mod storage;

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use query_filter::AclQueryFilter;
pub use resolver::AclResolver;
pub use storage::{AclStorage, MemoryAclStorage};

/// Permission levels for documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Permission {
    /// Can read the document.
    Read,
    /// Can update document metadata.
    Write,
    /// Can delete the document.
    Delete,
    /// Can modify ACL settings.
    Admin,
}

impl Permission {
    /// Parse a permission from a string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "read" => Some(Permission::Read),
            "write" => Some(Permission::Write),
            "delete" => Some(Permission::Delete),
            "admin" => Some(Permission::Admin),
            _ => None,
        }
    }

    /// Get all permissions as a set.
    pub fn all() -> HashSet<Permission> {
        let mut set = HashSet::new();
        set.insert(Permission::Read);
        set.insert(Permission::Write);
        set.insert(Permission::Delete);
        set.insert(Permission::Admin);
        set
    }
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Permission::Read => write!(f, "read"),
            Permission::Write => write!(f, "write"),
            Permission::Delete => write!(f, "delete"),
            Permission::Admin => write!(f, "admin"),
        }
    }
}

/// Who a permission applies to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", content = "id")]
#[serde(rename_all = "lowercase")]
pub enum Principal {
    /// Specific user ID.
    User(String),
    /// Role name.
    Role(String),
    /// Group ID.
    Group(String),
    /// Public access (everyone, including anonymous).
    Everyone,
    /// Any authenticated user.
    Authenticated,
}

impl Principal {
    /// Create a user principal.
    pub fn user(id: impl Into<String>) -> Self {
        Principal::User(id.into())
    }

    /// Create a role principal.
    pub fn role(name: impl Into<String>) -> Self {
        Principal::Role(name.into())
    }

    /// Create a group principal.
    pub fn group(id: impl Into<String>) -> Self {
        Principal::Group(id.into())
    }
}

impl std::fmt::Display for Principal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Principal::User(id) => write!(f, "user:{}", id),
            Principal::Role(name) => write!(f, "role:{}", name),
            Principal::Group(id) => write!(f, "group:{}", id),
            Principal::Everyone => write!(f, "everyone"),
            Principal::Authenticated => write!(f, "authenticated"),
        }
    }
}

/// Access control entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclEntry {
    /// Who this entry applies to.
    pub principal: Principal,
    /// Permissions granted.
    pub permissions: Vec<Permission>,
}

impl AclEntry {
    /// Create a new ACL entry.
    pub fn new(principal: Principal, permissions: Vec<Permission>) -> Self {
        Self {
            principal,
            permissions,
        }
    }

    /// Create a read-only entry for a user.
    pub fn user_read(user_id: impl Into<String>) -> Self {
        Self::new(Principal::user(user_id), vec![Permission::Read])
    }

    /// Create a full-access entry for a user.
    pub fn user_admin(user_id: impl Into<String>) -> Self {
        Self::new(
            Principal::user(user_id),
            vec![
                Permission::Read,
                Permission::Write,
                Permission::Delete,
                Permission::Admin,
            ],
        )
    }

    /// Create a read-only entry for a role.
    pub fn role_read(role: impl Into<String>) -> Self {
        Self::new(Principal::role(role), vec![Permission::Read])
    }

    /// Create a public read entry.
    pub fn public_read() -> Self {
        Self::new(Principal::Everyone, vec![Permission::Read])
    }

    /// Create an authenticated users read entry.
    pub fn authenticated_read() -> Self {
        Self::new(Principal::Authenticated, vec![Permission::Read])
    }
}

/// Document access control list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentAcl {
    /// Document ID.
    pub document_id: String,
    /// Owner user ID (always has Admin permission).
    pub owner: String,
    /// Access control entries.
    pub entries: Vec<AclEntry>,
    /// Whether to inherit ACL from source.
    #[serde(default = "default_true")]
    pub inherit_from_source: bool,
    /// When the ACL was created.
    pub created_at: DateTime<Utc>,
    /// When the ACL was last updated.
    pub updated_at: DateTime<Utc>,
}

fn default_true() -> bool {
    true
}

impl DocumentAcl {
    /// Create a new document ACL.
    pub fn new(document_id: impl Into<String>, owner: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            document_id: document_id.into(),
            owner: owner.into(),
            entries: Vec::new(),
            inherit_from_source: true,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a public document ACL.
    pub fn public(document_id: impl Into<String>, owner: impl Into<String>) -> Self {
        let mut acl = Self::new(document_id, owner);
        acl.entries.push(AclEntry::public_read());
        acl
    }

    /// Add an ACL entry.
    pub fn add_entry(&mut self, entry: AclEntry) {
        self.entries.push(entry);
        self.updated_at = Utc::now();
    }

    /// Remove entries for a principal.
    pub fn remove_principal(&mut self, principal: &Principal) {
        self.entries.retain(|e| &e.principal != principal);
        self.updated_at = Utc::now();
    }

    /// Check if the ACL grants public access.
    pub fn is_public(&self) -> bool {
        self.entries
            .iter()
            .any(|e| matches!(e.principal, Principal::Everyone))
    }

    /// Get all principals with any access.
    pub fn principals(&self) -> Vec<&Principal> {
        self.entries.iter().map(|e| &e.principal).collect()
    }
}

/// Source-level access control list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAcl {
    /// Source ID.
    pub source_id: String,
    /// Owner user ID.
    pub owner: String,
    /// Access control entries.
    pub entries: Vec<AclEntry>,
    /// Default ACL entries for new documents.
    pub default_document_acl: Vec<AclEntry>,
    /// When the ACL was created.
    pub created_at: DateTime<Utc>,
    /// When the ACL was last updated.
    pub updated_at: DateTime<Utc>,
}

impl SourceAcl {
    /// Create a new source ACL.
    pub fn new(source_id: impl Into<String>, owner: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            source_id: source_id.into(),
            owner: owner.into(),
            entries: Vec::new(),
            default_document_acl: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// Result of permission check.
#[derive(Debug, Clone)]
pub struct PermissionCheckResult {
    /// Whether permission was granted.
    pub allowed: bool,
    /// Reason for the decision.
    pub reason: String,
    /// Permissions that were checked.
    pub checked_permission: Permission,
    /// Permissions the principal has.
    pub granted_permissions: HashSet<Permission>,
}

impl PermissionCheckResult {
    /// Create an allowed result.
    pub fn allowed(
        permission: Permission,
        granted: HashSet<Permission>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            allowed: true,
            reason: reason.into(),
            checked_permission: permission,
            granted_permissions: granted,
        }
    }

    /// Create a denied result.
    pub fn denied(permission: Permission, reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: reason.into(),
            checked_permission: permission,
            granted_permissions: HashSet::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_parsing() {
        assert_eq!(Permission::from_str("read"), Some(Permission::Read));
        assert_eq!(Permission::from_str("WRITE"), Some(Permission::Write));
        assert_eq!(Permission::from_str("Delete"), Some(Permission::Delete));
        assert_eq!(Permission::from_str("admin"), Some(Permission::Admin));
        assert_eq!(Permission::from_str("unknown"), None);
    }

    #[test]
    fn test_principal_display() {
        assert_eq!(Principal::user("alice").to_string(), "user:alice");
        assert_eq!(Principal::role("admin").to_string(), "role:admin");
        assert_eq!(Principal::Everyone.to_string(), "everyone");
    }

    #[test]
    fn test_document_acl() {
        let mut acl = DocumentAcl::new("doc1", "owner1");
        assert_eq!(acl.owner, "owner1");
        assert!(!acl.is_public());

        acl.add_entry(AclEntry::public_read());
        assert!(acl.is_public());

        acl.remove_principal(&Principal::Everyone);
        assert!(!acl.is_public());
    }

    #[test]
    fn test_acl_entry_constructors() {
        let entry = AclEntry::user_read("alice");
        assert!(matches!(entry.principal, Principal::User(ref id) if id == "alice"));
        assert_eq!(entry.permissions, vec![Permission::Read]);

        let entry = AclEntry::user_admin("bob");
        assert_eq!(entry.permissions.len(), 4);

        let entry = AclEntry::public_read();
        assert!(matches!(entry.principal, Principal::Everyone));
    }
}

//! ACL storage backends.

use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;

use super::{DocumentAcl, Permission, SourceAcl};
use crate::auth::AuthContext;
use crate::error::Result;

/// Trait for ACL storage backends.
#[async_trait]
pub trait AclStorage: Send + Sync {
    /// Get the ACL for a document.
    async fn get_document_acl(&self, doc_id: &str) -> Result<Option<DocumentAcl>>;

    /// Set the ACL for a document.
    async fn set_document_acl(&self, acl: DocumentAcl) -> Result<()>;

    /// Delete the ACL for a document.
    async fn delete_document_acl(&self, doc_id: &str) -> Result<()>;

    /// Get the ACL for a source.
    async fn get_source_acl(&self, source_id: &str) -> Result<Option<SourceAcl>>;

    /// Set the ACL for a source.
    async fn set_source_acl(&self, acl: SourceAcl) -> Result<()>;

    /// Delete the ACL for a source.
    async fn delete_source_acl(&self, source_id: &str) -> Result<()>;

    /// Get all document IDs accessible by a principal with a specific permission.
    async fn get_accessible_documents(
        &self,
        auth: &AuthContext,
        permission: Permission,
    ) -> Result<Vec<String>>;

    /// Check if a principal has a permission on a document.
    async fn check_permission(
        &self,
        auth: &AuthContext,
        doc_id: &str,
        permission: Permission,
    ) -> Result<bool>;

    /// List all document ACLs (for admin purposes).
    async fn list_document_acls(&self, limit: usize, offset: usize) -> Result<Vec<DocumentAcl>>;

    /// List all source ACLs (for admin purposes).
    async fn list_source_acls(&self) -> Result<Vec<SourceAcl>>;
}

/// In-memory ACL storage for testing and simple deployments.
pub struct MemoryAclStorage {
    document_acls: RwLock<HashMap<String, DocumentAcl>>,
    source_acls: RwLock<HashMap<String, SourceAcl>>,
}

impl MemoryAclStorage {
    /// Create a new in-memory ACL storage.
    pub fn new() -> Self {
        Self {
            document_acls: RwLock::new(HashMap::new()),
            source_acls: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryAclStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AclStorage for MemoryAclStorage {
    async fn get_document_acl(&self, doc_id: &str) -> Result<Option<DocumentAcl>> {
        let acls = self.document_acls.read().unwrap();
        Ok(acls.get(doc_id).cloned())
    }

    async fn set_document_acl(&self, acl: DocumentAcl) -> Result<()> {
        let mut acls = self.document_acls.write().unwrap();
        acls.insert(acl.document_id.clone(), acl);
        Ok(())
    }

    async fn delete_document_acl(&self, doc_id: &str) -> Result<()> {
        let mut acls = self.document_acls.write().unwrap();
        acls.remove(doc_id);
        Ok(())
    }

    async fn get_source_acl(&self, source_id: &str) -> Result<Option<SourceAcl>> {
        let acls = self.source_acls.read().unwrap();
        Ok(acls.get(source_id).cloned())
    }

    async fn set_source_acl(&self, acl: SourceAcl) -> Result<()> {
        let mut acls = self.source_acls.write().unwrap();
        acls.insert(acl.source_id.clone(), acl);
        Ok(())
    }

    async fn delete_source_acl(&self, source_id: &str) -> Result<()> {
        let mut acls = self.source_acls.write().unwrap();
        acls.remove(source_id);
        Ok(())
    }

    async fn get_accessible_documents(
        &self,
        auth: &AuthContext,
        permission: Permission,
    ) -> Result<Vec<String>> {
        let acls = self.document_acls.read().unwrap();
        let mut accessible = Vec::new();

        for (doc_id, acl) in acls.iter() {
            if self.principal_has_permission(auth, acl, permission) {
                accessible.push(doc_id.clone());
            }
        }

        Ok(accessible)
    }

    async fn check_permission(
        &self,
        auth: &AuthContext,
        doc_id: &str,
        permission: Permission,
    ) -> Result<bool> {
        let acls = self.document_acls.read().unwrap();

        match acls.get(doc_id) {
            Some(acl) => Ok(self.principal_has_permission(auth, acl, permission)),
            None => {
                // No ACL means use default behavior (allow if ACL is disabled)
                Ok(true)
            }
        }
    }

    async fn list_document_acls(&self, limit: usize, offset: usize) -> Result<Vec<DocumentAcl>> {
        let acls = self.document_acls.read().unwrap();
        let mut list: Vec<_> = acls.values().cloned().collect();
        list.sort_by(|a, b| a.document_id.cmp(&b.document_id));
        Ok(list.into_iter().skip(offset).take(limit).collect())
    }

    async fn list_source_acls(&self) -> Result<Vec<SourceAcl>> {
        let acls = self.source_acls.read().unwrap();
        Ok(acls.values().cloned().collect())
    }
}

impl MemoryAclStorage {
    /// Check if a principal has a permission on a document.
    fn principal_has_permission(
        &self,
        auth: &AuthContext,
        acl: &DocumentAcl,
        permission: Permission,
    ) -> bool {
        // Owner always has all permissions
        if let Some(ref user_id) = auth.user_id {
            if user_id == &acl.owner {
                return true;
            }
        }

        // Check ACL entries
        for entry in &acl.entries {
            if self.principal_matches(auth, &entry.principal)
                && entry.permissions.contains(&permission)
            {
                return true;
            }
        }

        false
    }

    /// Check if an auth context matches a principal.
    fn principal_matches(&self, auth: &AuthContext, principal: &super::Principal) -> bool {
        use super::Principal;

        match principal {
            Principal::Everyone => true,
            Principal::Authenticated => !auth.anonymous,
            Principal::User(id) => auth.user_id.as_ref() == Some(id),
            Principal::Role(role) => auth.roles.contains(role),
            Principal::Group(group) => auth.groups.contains(group),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acl::AclEntry;

    #[tokio::test]
    async fn test_memory_storage() {
        let storage = MemoryAclStorage::new();

        // Create and store ACL
        let mut acl = DocumentAcl::new("doc1", "owner1");
        acl.add_entry(AclEntry::user_read("alice"));
        storage.set_document_acl(acl).await.unwrap();

        // Retrieve ACL
        let retrieved = storage.get_document_acl("doc1").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.owner, "owner1");
        assert_eq!(retrieved.entries.len(), 1);

        // Delete ACL
        storage.delete_document_acl("doc1").await.unwrap();
        let retrieved = storage.get_document_acl("doc1").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_permission_check() {
        let storage = MemoryAclStorage::new();

        // Create ACL with read access for alice
        let mut acl = DocumentAcl::new("doc1", "owner1");
        acl.add_entry(AclEntry::user_read("alice"));
        storage.set_document_acl(acl).await.unwrap();

        // Owner has permission
        let owner_ctx = AuthContext::authenticated("owner1".to_string(), vec![], "test");
        assert!(storage
            .check_permission(&owner_ctx, "doc1", Permission::Read)
            .await
            .unwrap());
        assert!(storage
            .check_permission(&owner_ctx, "doc1", Permission::Admin)
            .await
            .unwrap());

        // Alice has read permission
        let alice_ctx = AuthContext::authenticated("alice".to_string(), vec![], "test");
        assert!(storage
            .check_permission(&alice_ctx, "doc1", Permission::Read)
            .await
            .unwrap());

        // Alice doesn't have write permission
        assert!(!storage
            .check_permission(&alice_ctx, "doc1", Permission::Write)
            .await
            .unwrap());

        // Bob has no access
        let bob_ctx = AuthContext::authenticated("bob".to_string(), vec![], "test");
        assert!(!storage
            .check_permission(&bob_ctx, "doc1", Permission::Read)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_public_access() {
        let storage = MemoryAclStorage::new();

        // Create public ACL
        let acl = DocumentAcl::public("doc1", "owner1");
        storage.set_document_acl(acl).await.unwrap();

        // Anonymous user has read access
        let anon_ctx = AuthContext::anonymous();
        assert!(storage
            .check_permission(&anon_ctx, "doc1", Permission::Read)
            .await
            .unwrap());

        // But not write access
        assert!(!storage
            .check_permission(&anon_ctx, "doc1", Permission::Write)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_role_based_access() {
        let storage = MemoryAclStorage::new();

        // Create ACL with role-based access
        let mut acl = DocumentAcl::new("doc1", "owner1");
        acl.add_entry(AclEntry::role_read("editors"));
        storage.set_document_acl(acl).await.unwrap();

        // User with editor role has access
        let editor_ctx =
            AuthContext::authenticated("user1".to_string(), vec!["editors".to_string()], "test");
        assert!(storage
            .check_permission(&editor_ctx, "doc1", Permission::Read)
            .await
            .unwrap());

        // User without editor role has no access
        let viewer_ctx =
            AuthContext::authenticated("user2".to_string(), vec!["viewers".to_string()], "test");
        assert!(!storage
            .check_permission(&viewer_ctx, "doc1", Permission::Read)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_authenticated_access() {
        let storage = MemoryAclStorage::new();

        // Create ACL with authenticated user access
        let mut acl = DocumentAcl::new("doc1", "owner1");
        acl.add_entry(AclEntry::authenticated_read());
        storage.set_document_acl(acl).await.unwrap();

        // Authenticated user has access
        let auth_ctx = AuthContext::authenticated("user1".to_string(), vec![], "test");
        assert!(storage
            .check_permission(&auth_ctx, "doc1", Permission::Read)
            .await
            .unwrap());

        // Anonymous user has no access
        let anon_ctx = AuthContext::anonymous();
        assert!(!storage
            .check_permission(&anon_ctx, "doc1", Permission::Read)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_get_accessible_documents() {
        let storage = MemoryAclStorage::new();

        // Create some documents with different access
        let acl1 = DocumentAcl::public("doc1", "owner1");
        storage.set_document_acl(acl1).await.unwrap();

        let mut acl2 = DocumentAcl::new("doc2", "owner1");
        acl2.add_entry(AclEntry::user_read("alice"));
        storage.set_document_acl(acl2).await.unwrap();

        let acl3 = DocumentAcl::new("doc3", "owner1"); // No access for anyone but owner
        storage.set_document_acl(acl3).await.unwrap();

        // Alice can access doc1 (public) and doc2 (user access)
        let alice_ctx = AuthContext::authenticated("alice".to_string(), vec![], "test");
        let accessible = storage
            .get_accessible_documents(&alice_ctx, Permission::Read)
            .await
            .unwrap();
        assert!(accessible.contains(&"doc1".to_string()));
        assert!(accessible.contains(&"doc2".to_string()));
        assert!(!accessible.contains(&"doc3".to_string()));

        // Owner can access all
        let owner_ctx = AuthContext::authenticated("owner1".to_string(), vec![], "test");
        let accessible = storage
            .get_accessible_documents(&owner_ctx, Permission::Read)
            .await
            .unwrap();
        assert_eq!(accessible.len(), 3);
    }
}

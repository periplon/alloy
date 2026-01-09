//! ACL permission resolution engine.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use moka::sync::Cache;

use super::{AclStorage, DocumentAcl, Permission, PermissionCheckResult, Principal};
use crate::auth::AuthContext;
use crate::config::{AclConfig, RoleDefinition};
use crate::error::Result;

/// ACL resolver that handles permission resolution with role inheritance.
pub struct AclResolver {
    storage: Arc<dyn AclStorage>,
    config: AclConfig,
    #[allow(dead_code)]
    role_cache: Cache<String, RoleDefinition>,
    role_definitions: HashMap<String, RoleDefinition>,
}

impl AclResolver {
    /// Create a new ACL resolver.
    pub fn new(storage: Arc<dyn AclStorage>, config: AclConfig) -> Self {
        // Build role definitions map
        let role_definitions: HashMap<String, RoleDefinition> = config
            .roles
            .iter()
            .map(|r| (r.name.clone(), r.clone()))
            .collect();

        Self {
            storage,
            config,
            role_cache: Cache::new(1000),
            role_definitions,
        }
    }

    /// Check if ACL is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if a user has a permission on a document.
    pub async fn check_permission(
        &self,
        auth: &AuthContext,
        doc_id: &str,
        permission: Permission,
    ) -> Result<PermissionCheckResult> {
        // If ACL is disabled, allow everything
        if !self.config.enabled {
            return Ok(PermissionCheckResult::allowed(
                permission,
                Permission::all(),
                "ACL disabled",
            ));
        }

        // Get document ACL
        let doc_acl = self.storage.get_document_acl(doc_id).await?;

        match doc_acl {
            Some(acl) => self.resolve_permission(auth, &acl, permission).await,
            None => {
                // No ACL, use default permissions
                if self.config.default_public {
                    Ok(PermissionCheckResult::allowed(
                        permission,
                        HashSet::from([Permission::Read]),
                        "Document has no ACL, default is public",
                    ))
                } else if self.config.default_authenticated_read && !auth.anonymous {
                    Ok(PermissionCheckResult::allowed(
                        permission,
                        HashSet::from([Permission::Read]),
                        "Document has no ACL, authenticated read allowed",
                    ))
                } else {
                    Ok(PermissionCheckResult::denied(
                        permission,
                        "Document has no ACL and defaults deny access",
                    ))
                }
            }
        }
    }

    /// Resolve permissions for a user on a document with ACL.
    async fn resolve_permission(
        &self,
        auth: &AuthContext,
        acl: &DocumentAcl,
        permission: Permission,
    ) -> Result<PermissionCheckResult> {
        // Owner always has all permissions
        if let Some(ref user_id) = auth.user_id {
            if user_id == &acl.owner {
                return Ok(PermissionCheckResult::allowed(
                    permission,
                    Permission::all(),
                    "User is document owner",
                ));
            }
        }

        // Get effective principals for the user
        let effective_principals = self.expand_principals(auth)?;

        // Collect all permissions granted to the user
        let mut granted_permissions = HashSet::new();

        for entry in &acl.entries {
            if self.principal_matches(&entry.principal, &effective_principals) {
                for perm in &entry.permissions {
                    granted_permissions.insert(*perm);
                }
            }
        }

        // Check if the requested permission is granted
        if granted_permissions.contains(&permission) {
            Ok(PermissionCheckResult::allowed(
                permission,
                granted_permissions,
                "Permission granted by ACL entry",
            ))
        } else {
            Ok(PermissionCheckResult::denied(
                permission,
                "Permission not granted by any ACL entry",
            ))
        }
    }

    /// Expand a user's principals including role inheritance.
    fn expand_principals(&self, auth: &AuthContext) -> Result<HashSet<Principal>> {
        let mut principals = HashSet::new();

        // Add user principal if authenticated
        if let Some(ref user_id) = auth.user_id {
            principals.insert(Principal::User(user_id.clone()));
            principals.insert(Principal::Authenticated);
        }

        // Always add Everyone
        principals.insert(Principal::Everyone);

        // Add group principals
        for group in &auth.groups {
            principals.insert(Principal::Group(group.clone()));
        }

        // Expand roles recursively with cycle detection
        let mut visited = HashSet::new();
        let mut to_expand: VecDeque<String> = auth.roles.iter().cloned().collect();

        while let Some(role_name) = to_expand.pop_front() {
            if visited.insert(role_name.clone()) {
                principals.insert(Principal::Role(role_name.clone()));

                // Get parent roles
                if let Some(role_def) = self.role_definitions.get(&role_name) {
                    for parent in &role_def.inherits_from {
                        to_expand.push_back(parent.clone());
                    }
                }
            }
        }

        Ok(principals)
    }

    /// Check if a principal matches any of the effective principals.
    fn principal_matches(&self, principal: &Principal, effective: &HashSet<Principal>) -> bool {
        effective.contains(principal)
    }

    /// Get all permissions for a user on a document.
    pub async fn get_permissions(
        &self,
        auth: &AuthContext,
        doc_id: &str,
    ) -> Result<HashSet<Permission>> {
        // If ACL is disabled, grant all permissions
        if !self.config.enabled {
            return Ok(Permission::all());
        }

        // Get document ACL
        let doc_acl = self.storage.get_document_acl(doc_id).await?;

        match doc_acl {
            Some(acl) => {
                // Owner has all permissions
                if let Some(ref user_id) = auth.user_id {
                    if user_id == &acl.owner {
                        return Ok(Permission::all());
                    }
                }

                // Get effective principals
                let effective_principals = self.expand_principals(auth)?;

                // Collect permissions
                let mut permissions = HashSet::new();
                for entry in &acl.entries {
                    if self.principal_matches(&entry.principal, &effective_principals) {
                        for perm in &entry.permissions {
                            permissions.insert(*perm);
                        }
                    }
                }

                Ok(permissions)
            }
            None => {
                // No ACL, use defaults
                let allow_read = self.config.default_public
                    || (self.config.default_authenticated_read && !auth.anonymous);
                if allow_read {
                    Ok(HashSet::from([Permission::Read]))
                } else {
                    Ok(HashSet::new())
                }
            }
        }
    }

    /// Get the effective roles for a user (including inherited roles).
    pub fn get_effective_roles(&self, auth: &AuthContext) -> HashSet<String> {
        let mut roles = HashSet::new();
        let mut visited = HashSet::new();
        let mut to_expand: VecDeque<String> = auth.roles.iter().cloned().collect();

        while let Some(role_name) = to_expand.pop_front() {
            if visited.insert(role_name.clone()) {
                roles.insert(role_name.clone());

                if let Some(role_def) = self.role_definitions.get(&role_name) {
                    for parent in &role_def.inherits_from {
                        to_expand.push_back(parent.clone());
                    }
                }
            }
        }

        roles
    }

    /// Get the permissions granted by a role.
    pub fn get_role_permissions(&self, role: &str) -> HashSet<Permission> {
        let mut permissions = HashSet::new();
        let mut visited = HashSet::new();
        let mut to_expand = VecDeque::from([role.to_string()]);

        while let Some(role_name) = to_expand.pop_front() {
            if visited.insert(role_name.clone()) {
                if let Some(role_def) = self.role_definitions.get(&role_name) {
                    for perm_str in &role_def.permissions {
                        if let Some(perm) = Permission::from_str(perm_str) {
                            permissions.insert(perm);
                        }
                    }
                    for parent in &role_def.inherits_from {
                        to_expand.push_back(parent.clone());
                    }
                }
            }
        }

        permissions
    }

    /// Check if enforcement is required for a given operation type.
    pub fn should_enforce(&self, operation: &str) -> bool {
        if !self.config.enabled {
            return false;
        }

        match operation {
            "search" => self.config.enforce_on_search,
            "get" => self.config.enforce_on_get,
            "delete" => self.config.enforce_on_delete,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acl::{AclEntry, MemoryAclStorage};

    fn test_config() -> AclConfig {
        AclConfig {
            enabled: true,
            default_public: false,
            default_authenticated_read: true,
            enforce_on_search: true,
            enforce_on_get: true,
            enforce_on_delete: true,
            roles: vec![
                RoleDefinition {
                    name: "admin".to_string(),
                    inherits_from: vec![],
                    permissions: vec![
                        "read".to_string(),
                        "write".to_string(),
                        "delete".to_string(),
                        "admin".to_string(),
                    ],
                },
                RoleDefinition {
                    name: "editor".to_string(),
                    inherits_from: vec!["viewer".to_string()],
                    permissions: vec!["write".to_string()],
                },
                RoleDefinition {
                    name: "viewer".to_string(),
                    inherits_from: vec![],
                    permissions: vec!["read".to_string()],
                },
            ],
        }
    }

    #[tokio::test]
    async fn test_owner_permissions() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = AclResolver::new(storage.clone(), test_config());

        // Create ACL
        let acl = DocumentAcl::new("doc1", "owner1");
        storage.set_document_acl(acl).await.unwrap();

        // Owner has all permissions
        let owner_ctx = AuthContext::authenticated("owner1".to_string(), vec![], "test");
        let result = resolver
            .check_permission(&owner_ctx, "doc1", Permission::Admin)
            .await
            .unwrap();
        assert!(result.allowed);
        assert_eq!(result.granted_permissions, Permission::all());
    }

    #[tokio::test]
    async fn test_acl_entry_permissions() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = AclResolver::new(storage.clone(), test_config());

        // Create ACL with read access for alice
        let mut acl = DocumentAcl::new("doc1", "owner1");
        acl.add_entry(AclEntry::user_read("alice"));
        storage.set_document_acl(acl).await.unwrap();

        // Alice has read permission
        let alice_ctx = AuthContext::authenticated("alice".to_string(), vec![], "test");
        let result = resolver
            .check_permission(&alice_ctx, "doc1", Permission::Read)
            .await
            .unwrap();
        assert!(result.allowed);

        // Alice doesn't have write permission
        let result = resolver
            .check_permission(&alice_ctx, "doc1", Permission::Write)
            .await
            .unwrap();
        assert!(!result.allowed);
    }

    #[tokio::test]
    async fn test_role_inheritance() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = AclResolver::new(storage.clone(), test_config());

        // Create ACL with viewer role access
        let mut acl = DocumentAcl::new("doc1", "owner1");
        acl.add_entry(AclEntry::role_read("viewer"));
        storage.set_document_acl(acl).await.unwrap();

        // User with editor role should have access (inherits from viewer)
        let editor_ctx =
            AuthContext::authenticated("user1".to_string(), vec!["editor".to_string()], "test");
        let result = resolver
            .check_permission(&editor_ctx, "doc1", Permission::Read)
            .await
            .unwrap();
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_default_permissions() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = AclResolver::new(storage.clone(), test_config());

        // No ACL for doc1, authenticated read should be allowed
        let auth_ctx = AuthContext::authenticated("user1".to_string(), vec![], "test");
        let result = resolver
            .check_permission(&auth_ctx, "doc1", Permission::Read)
            .await
            .unwrap();
        assert!(result.allowed);

        // Anonymous user should be denied
        let anon_ctx = AuthContext::anonymous();
        let result = resolver
            .check_permission(&anon_ctx, "doc1", Permission::Read)
            .await
            .unwrap();
        assert!(!result.allowed);
    }

    #[tokio::test]
    async fn test_disabled_acl() {
        let storage = Arc::new(MemoryAclStorage::new());
        let mut config = test_config();
        config.enabled = false;
        let resolver = AclResolver::new(storage, config);

        // Everyone should have access when ACL is disabled
        let anon_ctx = AuthContext::anonymous();
        let result = resolver
            .check_permission(&anon_ctx, "doc1", Permission::Admin)
            .await
            .unwrap();
        assert!(result.allowed);
    }

    #[test]
    fn test_role_permission_expansion() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = AclResolver::new(storage, test_config());

        // Admin role has all permissions
        let admin_perms = resolver.get_role_permissions("admin");
        assert!(admin_perms.contains(&Permission::Read));
        assert!(admin_perms.contains(&Permission::Write));
        assert!(admin_perms.contains(&Permission::Delete));
        assert!(admin_perms.contains(&Permission::Admin));

        // Editor inherits from viewer
        let editor_perms = resolver.get_role_permissions("editor");
        assert!(editor_perms.contains(&Permission::Read)); // Inherited
        assert!(editor_perms.contains(&Permission::Write)); // Direct

        // Viewer only has read
        let viewer_perms = resolver.get_role_permissions("viewer");
        assert!(viewer_perms.contains(&Permission::Read));
        assert!(!viewer_perms.contains(&Permission::Write));
    }

    #[test]
    fn test_effective_roles() {
        let storage = Arc::new(MemoryAclStorage::new());
        let resolver = AclResolver::new(storage, test_config());

        let auth =
            AuthContext::authenticated("user1".to_string(), vec!["editor".to_string()], "test");

        let effective = resolver.get_effective_roles(&auth);
        assert!(effective.contains("editor"));
        assert!(effective.contains("viewer")); // Inherited
        assert!(!effective.contains("admin"));
    }
}

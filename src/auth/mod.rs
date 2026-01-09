//! Authentication module for Alloy.
//!
//! Provides authentication and authorization for the MCP server,
//! supporting API keys, JWT tokens, and basic authentication.

mod middleware;

use std::collections::HashSet;

use base64::Engine;
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

use crate::config::{AuthConfig, AuthMethod};
use crate::error::{AuthError, Result};

pub use middleware::{AuthLayer, AuthMiddleware};

/// Authentication context from a validated request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthContext {
    /// User ID (if authenticated).
    pub user_id: Option<String>,
    /// Roles assigned to the user.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Groups the user belongs to.
    #[serde(default)]
    pub groups: Vec<String>,
    /// Whether this is an anonymous/unauthenticated request.
    pub anonymous: bool,
    /// Authentication method used.
    pub method: Option<String>,
}

impl AuthContext {
    /// Create an anonymous auth context.
    pub fn anonymous() -> Self {
        Self {
            user_id: None,
            roles: Vec::new(),
            groups: Vec::new(),
            anonymous: true,
            method: None,
        }
    }

    /// Create an authenticated context.
    pub fn authenticated(user_id: String, roles: Vec<String>, method: &str) -> Self {
        Self {
            user_id: Some(user_id),
            roles,
            groups: Vec::new(),
            anonymous: false,
            method: Some(method.to_string()),
        }
    }

    /// Check if the user has a specific role.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// Check if the user has any of the specified roles.
    pub fn has_any_role(&self, roles: &[&str]) -> bool {
        roles.iter().any(|r| self.has_role(r))
    }

    /// Check if the user is in a specific group.
    pub fn in_group(&self, group: &str) -> bool {
        self.groups.iter().any(|g| g == group)
    }
}

/// JWT claims structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID).
    pub sub: String,
    /// Issuer.
    pub iss: String,
    /// Audience.
    pub aud: String,
    /// Expiration time (Unix timestamp).
    pub exp: i64,
    /// Issued at (Unix timestamp).
    pub iat: i64,
    /// Roles.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Groups.
    #[serde(default)]
    pub groups: Vec<String>,
}

/// Authenticator that validates credentials.
#[derive(Clone)]
pub struct Authenticator {
    config: AuthConfig,
    api_keys: HashSet<String>,
}

impl Authenticator {
    /// Create a new authenticator from config.
    pub fn new(config: AuthConfig) -> Self {
        // Load API keys from config and environment
        let mut api_keys: HashSet<String> = config.api_keys.iter().cloned().collect();

        // Also load from environment variable
        if let Ok(env_keys) = std::env::var("ALLOY_API_KEYS") {
            for key in env_keys.split(',') {
                let key = key.trim();
                if !key.is_empty() {
                    api_keys.insert(key.to_string());
                }
            }
        }

        Self { config, api_keys }
    }

    /// Check if authentication is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Authenticate a request.
    ///
    /// Returns an AuthContext if authentication succeeds, or an error if it fails.
    pub fn authenticate(
        &self,
        auth_header: Option<&str>,
        api_key_header: Option<&str>,
    ) -> Result<AuthContext> {
        // If authentication is disabled, return anonymous context
        if !self.config.enabled {
            return Ok(AuthContext::anonymous());
        }

        // Try X-API-Key header first
        if let Some(key) = api_key_header {
            return self.authenticate_api_key(key);
        }

        // Try Authorization header
        if let Some(auth) = auth_header {
            return self.authenticate_authorization_header(auth);
        }

        // No credentials provided
        Err(AuthError::MissingCredentials.into())
    }

    /// Authenticate using an API key.
    fn authenticate_api_key(&self, key: &str) -> Result<AuthContext> {
        if self.api_keys.contains(key) {
            // API key authentication doesn't provide user identity
            // We treat it as a service account
            Ok(AuthContext::authenticated(
                format!("api-key:{}", &key[..8.min(key.len())]),
                vec!["authenticated".to_string()],
                "api_key",
            ))
        } else {
            Err(AuthError::InvalidCredentials.into())
        }
    }

    /// Authenticate using the Authorization header.
    fn authenticate_authorization_header(&self, auth: &str) -> Result<AuthContext> {
        if let Some(token) = auth.strip_prefix("Bearer ") {
            match self.config.method {
                AuthMethod::ApiKey => self.authenticate_api_key(token),
                AuthMethod::Jwt => self.authenticate_jwt(token),
                AuthMethod::Basic => Err(AuthError::InvalidCredentials.into()),
            }
        } else if let Some(encoded) = auth.strip_prefix("Basic ") {
            self.authenticate_basic(encoded)
        } else {
            Err(AuthError::InvalidCredentials.into())
        }
    }

    /// Authenticate using a JWT token.
    fn authenticate_jwt(&self, token: &str) -> Result<AuthContext> {
        let secret = self.get_jwt_secret()?;

        let mut validation = Validation::default();
        validation.set_issuer(&[&self.config.jwt.issuer]);
        validation.set_audience(&[&self.config.jwt.audience]);

        if self.config.jwt.expiry_secs == 0 {
            validation.validate_exp = false;
        }

        let token_data = decode::<JwtClaims>(
            token,
            &DecodingKey::from_secret(secret.as_bytes()),
            &validation,
        )
        .map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        let claims = token_data.claims;
        let mut roles = claims.roles;
        roles.push("authenticated".to_string());

        let mut ctx = AuthContext::authenticated(claims.sub, roles, "jwt");
        ctx.groups = claims.groups;

        Ok(ctx)
    }

    /// Authenticate using Basic auth.
    fn authenticate_basic(&self, encoded: &str) -> Result<AuthContext> {
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| AuthError::InvalidCredentials)?;

        let credentials = String::from_utf8(decoded).map_err(|_| AuthError::InvalidCredentials)?;

        let parts: Vec<&str> = credentials.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(AuthError::InvalidCredentials.into());
        }

        let username = parts[0];
        let password = parts[1];

        // Check against configured credentials
        if let Some(stored_hash) = self.config.basic_auth.get(username) {
            // For now, we do a simple comparison
            // In production, you'd want to use proper password hashing
            if stored_hash == password || self.verify_password_hash(password, stored_hash) {
                return Ok(AuthContext::authenticated(
                    username.to_string(),
                    vec!["authenticated".to_string()],
                    "basic",
                ));
            }
        }

        Err(AuthError::InvalidCredentials.into())
    }

    /// Get the JWT secret from config or environment.
    fn get_jwt_secret(&self) -> Result<String> {
        if !self.config.jwt.secret.is_empty() {
            return Ok(self.config.jwt.secret.clone());
        }

        std::env::var("ALLOY_JWT_SECRET").map_err(|_| AuthError::MissingSecret.into())
    }

    /// Verify a password against a hash.
    ///
    /// For now, this is a simple comparison. In production,
    /// you'd want to use bcrypt or argon2.
    fn verify_password_hash(&self, password: &str, hash: &str) -> bool {
        // Simple SHA-256 hash comparison
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        let result = hasher.finalize();
        let hex_hash = hex::encode(result);
        hex_hash == hash
    }

    /// Generate a JWT token for a user.
    pub fn generate_jwt(
        &self,
        user_id: &str,
        roles: Vec<String>,
        groups: Vec<String>,
    ) -> Result<String> {
        let secret = self.get_jwt_secret()?;
        let now = Utc::now();
        let expiry = if self.config.jwt.expiry_secs > 0 {
            now + Duration::seconds(self.config.jwt.expiry_secs as i64)
        } else {
            now + Duration::days(365) // 1 year if no expiry set
        };

        let claims = JwtClaims {
            sub: user_id.to_string(),
            iss: self.config.jwt.issuer.clone(),
            aud: self.config.jwt.audience.clone(),
            exp: expiry.timestamp(),
            iat: now.timestamp(),
            roles,
            groups,
        };

        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .map_err(|e| AuthError::TokenGeneration(e.to_string()).into())
    }
}

/// Simple hex encoding (to avoid adding another dependency).
mod hex {
    pub fn encode(data: impl AsRef<[u8]>) -> String {
        data.as_ref().iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::JwtConfig;

    fn test_config() -> AuthConfig {
        AuthConfig {
            enabled: true,
            method: AuthMethod::ApiKey,
            api_keys: vec!["test-api-key-12345".to_string()],
            jwt: JwtConfig {
                secret: "test-secret-key".to_string(),
                issuer: "alloy".to_string(),
                audience: "alloy-users".to_string(),
                expiry_secs: 3600,
            },
            basic_auth: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_api_key_auth() {
        let auth = Authenticator::new(test_config());

        // Valid key
        let result = auth.authenticate(None, Some("test-api-key-12345"));
        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(!ctx.anonymous);
        assert!(ctx.has_role("authenticated"));

        // Invalid key
        let result = auth.authenticate(None, Some("invalid-key"));
        assert!(result.is_err());
    }

    #[test]
    fn test_bearer_token_as_api_key() {
        let auth = Authenticator::new(test_config());

        let result = auth.authenticate(Some("Bearer test-api-key-12345"), None);
        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(!ctx.anonymous);
    }

    #[test]
    fn test_auth_disabled() {
        let mut config = test_config();
        config.enabled = false;
        let auth = Authenticator::new(config);

        let result = auth.authenticate(None, None);
        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(ctx.anonymous);
    }

    #[test]
    fn test_jwt_generation_and_validation() {
        let mut config = test_config();
        config.method = AuthMethod::Jwt;
        let auth = Authenticator::new(config);

        // Generate token
        let token = auth
            .generate_jwt("user123", vec!["admin".to_string()], vec![])
            .unwrap();

        // Validate token
        let result = auth.authenticate(Some(&format!("Bearer {}", token)), None);
        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert_eq!(ctx.user_id, Some("user123".to_string()));
        assert!(ctx.has_role("admin"));
        assert!(ctx.has_role("authenticated"));
    }

    #[test]
    fn test_missing_credentials() {
        let auth = Authenticator::new(test_config());

        let result = auth.authenticate(None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_context() {
        let ctx = AuthContext::authenticated(
            "user123".to_string(),
            vec!["admin".to_string(), "editor".to_string()],
            "jwt",
        );

        assert!(ctx.has_role("admin"));
        assert!(ctx.has_role("editor"));
        assert!(!ctx.has_role("viewer"));
        assert!(ctx.has_any_role(&["admin", "viewer"]));
        assert!(!ctx.has_any_role(&["viewer", "guest"]));
    }
}

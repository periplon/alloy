//! Authentication middleware for HTTP transport.

use std::sync::Arc;
use std::task::{Context, Poll};

use axum::http::{HeaderMap, Request, Response, StatusCode};
use futures::future::BoxFuture;
use http_body::Body as HttpBody;
use tower::{Layer, Service};

use super::{AuthContext, Authenticator};

/// Auth middleware layer.
#[derive(Clone)]
pub struct AuthLayer {
    authenticator: Arc<Authenticator>,
}

impl AuthLayer {
    /// Create a new auth layer.
    pub fn new(authenticator: Authenticator) -> Self {
        Self {
            authenticator: Arc::new(authenticator),
        }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthMiddleware {
            inner,
            authenticator: self.authenticator.clone(),
        }
    }
}

/// Auth middleware service.
#[derive(Clone)]
pub struct AuthMiddleware<S> {
    inner: S,
    authenticator: Arc<Authenticator>,
}

impl<S> AuthMiddleware<S> {
    /// Get the current auth context from request headers.
    fn extract_auth_context(&self, headers: &HeaderMap) -> AuthContext {
        let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());

        let api_key_header = headers.get("x-api-key").and_then(|v| v.to_str().ok());

        match self.authenticator.authenticate(auth_header, api_key_header) {
            Ok(ctx) => ctx,
            Err(_) if !self.authenticator.is_enabled() => AuthContext::anonymous(),
            Err(_) => {
                // Return anonymous context for now, but we'll check permissions later
                AuthContext::anonymous()
            }
        }
    }
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for AuthMiddleware<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send,
    ReqBody: Send + 'static,
    ResBody: HttpBody + Default + Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<ReqBody>) -> Self::Future {
        let auth_ctx = self.extract_auth_context(req.headers());
        let is_auth_required = self.authenticator.is_enabled();

        // If auth is required and context is anonymous with no valid credentials, reject
        if is_auth_required && auth_ctx.anonymous {
            let has_auth = req.headers().contains_key("authorization")
                || req.headers().contains_key("x-api-key");

            if has_auth {
                // Had credentials but they were invalid
                return Box::pin(async move {
                    let response = Response::builder()
                        .status(StatusCode::UNAUTHORIZED)
                        .header("WWW-Authenticate", "Bearer")
                        .body(ResBody::default())
                        .unwrap();
                    Ok(response)
                });
            }

            // Skip auth check for health/metrics endpoints
            let path = req.uri().path();
            if !matches!(path, "/health" | "/ready" | "/metrics" | "/") {
                return Box::pin(async move {
                    let response = Response::builder()
                        .status(StatusCode::UNAUTHORIZED)
                        .header("WWW-Authenticate", "Bearer")
                        .body(ResBody::default())
                        .unwrap();
                    Ok(response)
                });
            }
        }

        // Store auth context in request extensions
        req.extensions_mut().insert(auth_ctx);

        let inner = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, inner);

        Box::pin(async move { inner.call(req).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AuthConfig, AuthMethod, JwtConfig};

    fn test_authenticator() -> Authenticator {
        Authenticator::new(AuthConfig {
            enabled: true,
            method: AuthMethod::ApiKey,
            api_keys: vec!["test-key".to_string()],
            jwt: JwtConfig::default(),
            basic_auth: std::collections::HashMap::new(),
        })
    }

    #[test]
    fn test_auth_layer_creation() {
        let auth = test_authenticator();
        let _layer = AuthLayer::new(auth);
    }
}

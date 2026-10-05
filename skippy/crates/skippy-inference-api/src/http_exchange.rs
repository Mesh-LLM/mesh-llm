//! Admission and final encoded-byte observation at the typed HTTP boundary.

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::{Body, to_bytes},
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::{
    OpenAiError, OpenAiErrorKind, RequestId,
    router::FrontendState,
    wire_bytes::{WireBytesObserver, observe_response_body},
};

/// The host's admission result and optional observer for the final response.
/// Denial still passes through the observer, so its exact emitted error is visible.
#[derive(Default)]
pub struct HttpExchangeAdmission {
    pub observer: Option<Arc<dyn WireBytesObserver>>,
    pub denial: Option<OpenAiError>,
    pub observation_id: Option<String>,
    /// Bounded host-validated metadata supplied before response headers commit.
    pub response_headers: Vec<(String, String)>,
}

tokio::task_local! { static HTTP_EXCHANGE_OBSERVATION: String; }

/// Exact host-owned observation correlated with this typed backend dispatch.
pub fn current_http_exchange_observation_id() -> Option<String> {
    HTTP_EXCHANGE_OBSERVATION.try_with(Clone::clone).ok()
}

/// Host-owned grant checking and admission over the original HTTP entity.
///
/// Implementations must apply their own bounded deadline and deny/abstain
/// aggregation. An installed plugin is not implicitly granted access here.
#[async_trait]
pub trait HttpExchangePolicy: Send + Sync + 'static {
    /// Avoid body observation overhead when no operator-granted subscriber exists.
    async fn is_enabled(&self) -> bool {
        true
    }
    async fn received(
        &self,
        method: &str,
        path: &str,
        headers: &HeaderMap,
        body: &[u8],
        request_id: RequestId,
    ) -> HttpExchangeAdmission;
}

pub(crate) async fn http_exchange_middleware(
    State(state): State<FrontendState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let Some(policy) = &state.config.http_exchange_policy else {
        return next.run(request).await;
    };
    if !policy.is_enabled().await {
        return next.run(request).await;
    }
    if !matches!(
        request.uri().path(),
        "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
    ) {
        return next.run(request).await;
    }
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .copied()
        .unwrap_or_else(|| crate::request_id_from_headers_or_generate(request.headers()));
    let (parts, body) = request.into_parts();
    let limit = state.config.max_request_body_bytes;
    let body = match to_bytes(body, limit).await {
        Ok(body) => body,
        Err(_) => {
            return OpenAiError::from_kind(
                StatusCode::PAYLOAD_TOO_LARGE,
                OpenAiErrorKind::PayloadTooLarge,
                "request body exceeds lifecycle observation limit",
            )
            .into_response();
        }
    };
    let admission = policy
        .received(
            parts.method.as_str(),
            parts.uri.path(),
            &parts.headers,
            &body,
            request_id,
        )
        .await;
    let mut response = match admission.denial {
        Some(error) => error.into_response(),
        None => {
            let call = next.run(Request::from_parts(parts, Body::from(body)));
            match admission.observation_id {
                Some(id) => HTTP_EXCHANGE_OBSERVATION.scope(id, call).await,
                None => call.await,
            }
        }
    };
    let mut headers = std::collections::BTreeMap::new();
    for (name, value) in admission.response_headers {
        headers.insert(name.to_ascii_lowercase(), value);
    }
    if let Some(observer) = &admission.observer {
        for (name, value) in observer.response_headers() {
            headers.insert(name.to_ascii_lowercase(), value);
        }
    }
    for (name, value) in headers
        .into_iter()
        .filter(|(name, _)| name.starts_with("x-plugin-"))
        .take(16)
    {
        if let (Ok(name), Ok(value)) = (
            axum::http::HeaderName::try_from(name),
            axum::http::HeaderValue::try_from(value),
        ) {
            response.headers_mut().append(name, value);
        }
    }
    match admission.observer {
        Some(observer) => {
            observer.response_status(response.status().as_u16());
            let (parts, body) = response.into_parts();
            Response::from_parts(parts, observe_response_body(body, observer))
        }
        None => response,
    }
}

#[cfg(test)]
#[path = "http_exchange_tests.rs"]
mod tests;

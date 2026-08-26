//! Bounded request-summary metadata captured at trusted lifecycle boundaries.
//!
//! This intentionally retains classifications and selected identifiers only.
//! It never accepts raw paths, query strings, credentials, payloads, or
//! transport targets.

use mesh_llm_events::logging::envelope::{closed_method, closed_source};
use openai_frontend::OpenAiFrontendRoute;

use super::policy::{RedactMode, apply_redaction};

mod caller_identity;

pub(crate) use caller_identity::{CallerPathType, authenticated_endpoint_id};

const MAX_REQUEST_METADATA_CHARS: usize = 64;

/// The small, privacy-safe metadata projection shared by active summaries,
/// durable summaries, and replay filtering.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct RequestSummaryMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    route: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    caller_endpoint_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    caller_addr: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    caller_path_type: Option<CallerPathType>,
}

impl RequestSummaryMetadata {
    /// Build one metadata snapshot from bounded classifications or identifiers.
    pub(crate) fn from_parts(
        route: Option<&str>,
        model: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
    ) -> Self {
        Self {
            route: bounded_metadata(route),
            model: bounded_metadata(model),
            provider: bounded_metadata(provider),
            engine: bounded_metadata(engine),
            source: None,
            method: None,
            caller_endpoint_id: None,
            caller_addr: None,
            caller_path_type: None,
        }
    }

    pub(crate) fn with_source(mut self, source: Option<&str>) -> Self {
        self.source = source.and_then(bounded_source);
        self
    }

    pub(crate) fn with_method(mut self, method: Option<&str>) -> Self {
        self.method = method.and_then(bounded_method);
        self
    }

    /// Capture only the closed frontend route vocabulary. Unknown routes stay
    /// absent rather than retaining an arbitrary path.
    pub(crate) fn from_openai_frontend_route(route: OpenAiFrontendRoute) -> Self {
        Self::from_parts(openai_route_label(route), None, None, None)
    }

    /// Classify a raw host ingress path through the same closed vocabulary as
    /// the embedded frontend. Query text is discarded before comparison and
    /// unknown paths are intentionally omitted.
    pub(crate) fn from_openai_ingress_path(client_path: &str) -> Self {
        let path = client_path
            .split_once('?')
            .map_or(client_path, |(path, _)| path);
        let route = match path {
            "/health" => Some("health"),
            "/healthz" => Some("healthz"),
            "/readyz" => Some("readyz"),
            "/v1/models" => Some("models"),
            "/v1/chat/completions" => Some("chat_completions"),
            "/v1/completions" => Some("completions"),
            "/v1/responses" => Some("responses"),
            _ => None,
        };
        Self::from_parts(route, None, None, None)
    }

    pub(crate) fn route(&self) -> Option<&str> {
        self.route.as_deref()
    }

    pub(crate) fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub(crate) fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    pub(crate) fn engine(&self) -> Option<&str> {
        self.engine.as_deref()
    }

    pub(crate) fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    pub(crate) fn method(&self) -> Option<&str> {
        self.method.as_deref()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.route.is_none()
            && self.model.is_none()
            && self.provider.is_none()
            && self.engine.is_none()
            && self.source.is_none()
            && self.method.is_none()
            && self.caller_endpoint_id.is_none()
            && self.caller_addr.is_none()
            && self.caller_path_type.is_none()
    }

    /// Preserve the first truthful value for each field. A later source can
    /// fill a missing classification but cannot overwrite an earlier one.
    pub(crate) fn merge_missing(&mut self, update: Self) -> bool {
        let Self {
            route,
            model,
            provider,
            engine,
            source,
            method,
            caller_endpoint_id,
            caller_addr,
            caller_path_type,
        } = update;
        let mut changed = false;
        changed |= merge_field(&mut self.route, route);
        changed |= merge_field(&mut self.model, model);
        changed |= merge_field(&mut self.provider, provider);
        changed |= merge_field(&mut self.engine, engine);
        changed |= merge_field(&mut self.source, source);
        changed |= merge_field(&mut self.method, method);
        changed |= self.merge_missing_caller(caller_endpoint_id, caller_addr, caller_path_type);
        changed
    }
}

const fn openai_route_label(route: OpenAiFrontendRoute) -> Option<&'static str> {
    match route {
        OpenAiFrontendRoute::Health => Some("health"),
        OpenAiFrontendRoute::Healthz => Some("healthz"),
        OpenAiFrontendRoute::Readyz => Some("readyz"),
        OpenAiFrontendRoute::Models => Some("models"),
        OpenAiFrontendRoute::ChatCompletions => Some("chat_completions"),
        OpenAiFrontendRoute::Completions => Some("completions"),
        OpenAiFrontendRoute::Responses => Some("responses"),
        OpenAiFrontendRoute::Unknown => None,
    }
}

fn merge_field(current: &mut Option<String>, update: Option<String>) -> bool {
    if current.is_none() && update.is_some() {
        *current = update;
        true
    } else {
        false
    }
}

fn bounded_metadata(value: Option<&str>) -> Option<String> {
    let value = value?.trim();
    let (value, mode) = apply_redaction(value);
    if matches!(mode, RedactMode::FullRedact) || !is_safe_metadata(&value) {
        return None;
    }
    let value = value
        .chars()
        .take(MAX_REQUEST_METADATA_CHARS)
        .collect::<String>();
    (!value.is_empty()).then_some(value)
}

fn bounded_source(value: &str) -> Option<String> {
    closed_source(value)
}

fn bounded_method(value: &str) -> Option<String> {
    // Unrecognized or empty methods classify as OTHER so the method field is
    // never silently missing; recognized methods keep their normalized value.
    closed_method(value).or(Some("OTHER".to_owned()))
}

fn is_safe_metadata(value: &str) -> bool {
    let bytes = value.as_bytes();
    let windows_path = bytes.len() > 1 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic();
    !value.starts_with('/')
        && !value.starts_with("~/")
        && !windows_path
        && !value.contains('\\')
        && !value.contains('?')
        && !value.contains("://")
        && !value.contains("../")
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric()
                || matches!(character, '-' | '_' | '.' | '/' | ':' | '@' | '#' | '+')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_keeps_only_bounded_identifier_vocabulary() {
        let metadata = RequestSummaryMetadata::from_parts(
            Some("/private/operator/path?token=secret"),
            Some("acme/model:Q4_K_M#low-ctx"),
            Some("Bearer credential"),
            Some("raw_ingress"),
        );

        assert!(metadata.route().is_none());
        assert_eq!(metadata.model(), Some("acme/model:Q4_K_M#low-ctx"));
        assert!(metadata.provider().is_none());
        assert_eq!(metadata.engine(), Some("raw_ingress"));
        assert!(metadata.source().is_none());
    }

    #[test]
    fn openai_ingress_classification_never_retains_raw_path_or_query() {
        let metadata =
            RequestSummaryMetadata::from_openai_ingress_path("/v1/responses?token=secret");
        assert_eq!(metadata.route(), Some("responses"));
        assert!(
            RequestSummaryMetadata::from_openai_ingress_path("/private/path?token=secret")
                .is_empty()
        );
    }

    #[test]
    fn source_and_method_keep_only_closed_values() {
        let metadata = RequestSummaryMetadata::from_parts(None, None, None, None)
            .with_source(Some("direct_http"))
            .with_method(Some("post"));
        assert_eq!(metadata.source(), Some("direct_http"));
        assert_eq!(metadata.method(), Some("POST"));
        // Unknown source values are discarded entirely.
        assert!(
            RequestSummaryMetadata::from_parts(None, None, None, None)
                .with_source(Some("10.0.0.1"))
                .is_empty()
        );
        assert_eq!(
            RequestSummaryMetadata::from_parts(None, None, None, None)
                .with_method(Some("PATCH"))
                .method(),
            Some("PATCH")
        );
        // Unrecognized or empty methods classify as OTHER rather than being
        // silently dropped, so the method field is never absent.
        assert_eq!(
            RequestSummaryMetadata::from_parts(None, None, None, None)
                .with_method(Some(""))
                .method(),
            Some("OTHER")
        );
    }
}

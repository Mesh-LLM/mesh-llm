//! Canonical event envelope with versioned schema and identity context.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::events::LifecycleEvent;
use super::identifiers::{EventId, RequestId};
use super::replay::ReplayChannel;

/// Privacy-bounded context used only when a trusted local producer renders a
/// canonical lifecycle envelope. The context is deliberately skipped by
/// serde: durable/replay truth remains the canonical event plus the request
/// summary metadata, while old envelopes continue to deserialize unchanged.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CanonicalPresentationContext {
    route: Option<String>,
    source: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    engine: Option<String>,
    method: Option<String>,
}

impl CanonicalPresentationContext {
    /// Build a local presentation context from already-classified metadata.
    /// Route and source are closed vocabularies; free-form or path-shaped
    /// values are discarded rather than echoed to a console.
    pub fn from_parts(
        route: Option<&str>,
        source: Option<&str>,
        model: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
        method: Option<&str>,
    ) -> Self {
        Self {
            route: route.and_then(closed_route),
            source: source.and_then(closed_source),
            model: model.and_then(bounded_context_value),
            provider: provider.and_then(bounded_context_value),
            engine: engine.and_then(bounded_context_value),
            method: method.and_then(closed_method),
        }
    }

    pub fn route(&self) -> Option<&str> {
        self.route.as_deref()
    }

    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    pub fn engine(&self) -> Option<&str> {
        self.engine.as_deref()
    }

    pub fn method(&self) -> Option<&str> {
        self.method.as_deref()
    }

    pub fn request_kind(&self) -> &'static str {
        match self.route.as_deref() {
            Some("health" | "healthz" | "readyz") => "probe",
            Some("models" | "management_get_models") => "model_listing",
            Some("chat_completions" | "completions" | "responses") => "inference",
            Some(
                "management_get_status"
                | "management_get_events"
                | "management_get_runtime_events"
                | "management_get_other"
                | "management_post"
                | "management_put"
                | "management_patch"
                | "management_delete"
                | "management_other",
            ) => "management",
            _ => "unknown",
        }
    }
}

const MAX_PRESENTATION_CONTEXT_CHARS: usize = 64;

fn bounded_context_value(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > MAX_PRESENTATION_CONTEXT_CHARS
        || value.starts_with('/')
        || value.starts_with("~/")
        || value.contains(['\\', '?'])
        || value.contains("://")
        || value.contains("../")
        || !value.chars().all(|character| {
            character.is_ascii_alphanumeric()
                || matches!(character, '-' | '_' | '.' | '/' | ':' | '@' | '#' | '+')
        })
    {
        return None;
    }
    Some(value.to_owned())
}

fn closed_route(value: &str) -> Option<String> {
    let value = value.trim();
    let accepted = matches!(
        value,
        "health"
            | "healthz"
            | "readyz"
            | "models"
            | "chat_completions"
            | "completions"
            | "responses"
            | "management_get_status"
            | "management_get_models"
            | "management_get_events"
            | "management_get_runtime_events"
            | "management_get_other"
            | "management_post"
            | "management_put"
            | "management_patch"
            | "management_delete"
            | "management_other"
    );
    accepted.then(|| value.to_owned())
}

/// Trim and accept only the closed source vocabulary
/// (`direct_http` | `mesh_forwarded` | `other`). Unknown or empty sources are
/// discarded rather than echoed to a console.
pub fn closed_source(value: &str) -> Option<String> {
    matches!(value.trim(), "direct_http" | "mesh_forwarded" | "other")
        .then(|| value.trim().to_owned())
}

/// Trim, uppercase, and accept only the closed HTTP method vocabulary
/// (`GET` | `POST` | `PUT` | `PATCH` | `DELETE` | `OTHER`). Unknown or empty methods are
/// discarded rather than echoed to a console.
pub fn closed_method(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_uppercase();
    matches!(
        value.as_str(),
        "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "OTHER"
    )
    .then_some(value)
}

/// Current canonical logging schema version. Bump on additive changes to the envelope shape.
pub const SCHEMA_VERSION: u16 = 1;

/// A canonical logging schema version accepted by this build.
///
/// The private representation prevents callers from constructing an envelope with an
/// unsupported version while serde keeps the existing numeric wire representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(into = "u16", try_from = "u16")]
pub struct CanonicalSchemaVersion(u16);

impl CanonicalSchemaVersion {
    /// Return the schema version implemented by this build.
    pub const fn current() -> Self {
        Self(SCHEMA_VERSION)
    }

    /// Return the numeric representation used on the wire.
    pub const fn as_u16(self) -> u16 {
        self.0
    }
}

impl From<CanonicalSchemaVersion> for u16 {
    fn from(version: CanonicalSchemaVersion) -> Self {
        version.0
    }
}

/// Error returned when a canonical logging schema version is not supported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedSchemaVersion {
    pub version: u16,
}

impl fmt::Display for UnsupportedSchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unsupported canonical logging schema version {}; expected {}",
            self.version, SCHEMA_VERSION
        )
    }
}

impl std::error::Error for UnsupportedSchemaVersion {}

/// Errors returned by the explicit canonical-envelope JSON parsing boundary.
#[derive(Debug)]
pub enum CanonicalEnvelopeParseError {
    /// The input was not valid JSON or did not contain a valid envelope.
    Json(serde_json::Error),
    /// The envelope uses a schema version this build does not support.
    UnsupportedSchemaVersion(UnsupportedSchemaVersion),
}

impl fmt::Display for CanonicalEnvelopeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid canonical logging envelope JSON: {error}"),
            Self::UnsupportedSchemaVersion(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CanonicalEnvelopeParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::UnsupportedSchemaVersion(error) => Some(error),
        }
    }
}

impl From<serde_json::Error> for CanonicalEnvelopeParseError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl TryFrom<u16> for CanonicalSchemaVersion {
    type Error = UnsupportedSchemaVersion;

    fn try_from(version: u16) -> Result<Self, Self::Error> {
        if version == SCHEMA_VERSION {
            Ok(Self(version))
        } else {
            Err(UnsupportedSchemaVersion { version })
        }
    }
}

/// Top-level event envelope carrying all metadata required for persistence and replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalEnvelope {
    pub schema_version: CanonicalSchemaVersion,
    pub event_id: EventId,
    pub request_id: RequestId,
    #[serde(rename = "channel")]
    pub channel: ReplayChannel,
    pub sequence: u64,
    /// ISO 8601 timestamp of when the event occurred.
    pub occurred_at: String,

    /// The lifecycle payload for this envelope.
    #[serde(flatten)]
    pub event: LifecycleEvent,

    /// Nullable reserved identity fields (omitted from JSON when None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Process-local projection context. It is reconstructed from trusted
    /// request-summary metadata and intentionally does not alter the wire
    /// schema or persistence payload.
    #[serde(skip)]
    pub(crate) presentation_context: Option<CanonicalPresentationContext>,
}

/// Equality follows the wire representation: the process-local
/// `presentation_context` projection is excluded, so envelopes that serialize
/// identically compare equal.
impl PartialEq for CanonicalEnvelope {
    fn eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.event_id == other.event_id
            && self.request_id == other.request_id
            && self.channel == other.channel
            && self.sequence == other.sequence
            && self.occurred_at == other.occurred_at
            && self.event == other.event
            && self.tenant_id == other.tenant_id
            && self.account_id == other.account_id
            && self.user_id == other.user_id
            && self.role == other.role
    }
}

impl Eq for CanonicalEnvelope {}

impl CanonicalEnvelope {
    /// Create a new envelope with the given fields. Identity fields default to None.
    #[allow(dead_code)]
    pub fn new(
        event_id: EventId,
        request_id: RequestId,
        channel: ReplayChannel,
        sequence: u64,
        occurred_at: String,
        event: LifecycleEvent,
    ) -> Self {
        Self {
            schema_version: CanonicalSchemaVersion::current(),
            event_id,
            request_id,
            channel,
            sequence,
            occurred_at,
            event,
            tenant_id: None,
            account_id: None,
            user_id: None,
            role: None,
            presentation_context: None,
        }
    }

    /// Attach bounded, process-local context for pretty/JSON/TUI output.
    pub fn with_presentation_context(mut self, context: CanonicalPresentationContext) -> Self {
        self.presentation_context = Some(context);
        self
    }

    /// Return the process-local context, if a trusted producer attached it.
    pub fn presentation_context(&self) -> Option<&CanonicalPresentationContext> {
        self.presentation_context.as_ref()
    }

    /// Parse a JSON envelope while exposing unsupported schema versions as a typed error.
    ///
    /// The derived serde implementation remains available for generic serde callers. This
    /// boundary is the intended API when callers need to distinguish an unsupported version
    /// from other JSON or envelope-shape errors.
    pub fn from_json_str(json: &str) -> Result<Self, CanonicalEnvelopeParseError> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        if let Some(raw_version) = value.get("schema_version") {
            let version: u16 = serde_json::from_value(raw_version.clone())?;
            CanonicalSchemaVersion::try_from(version)
                .map_err(CanonicalEnvelopeParseError::UnsupportedSchemaVersion)?;
        }

        Ok(serde_json::from_value(value)?)
    }

    /// Set the tenant ID. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn with_tenant(mut self, tenant_id: String) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    /// Set the account ID. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn with_account(mut self, account_id: String) -> Self {
        self.account_id = Some(account_id);
        self
    }

    /// Set the user ID. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Set the role. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn with_role(mut self, role: String) -> Self {
        self.role = Some(role);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version_constant() {
        assert_eq!(SCHEMA_VERSION, 1);
    }

    #[test]
    fn test_unsupported_schema_version_is_typed_error() {
        let version = SCHEMA_VERSION + 1;
        assert_eq!(
            CanonicalSchemaVersion::try_from(version),
            Err(UnsupportedSchemaVersion { version })
        );
    }

    #[test]
    fn test_envelope_new_minimal() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            0,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        );

        assert_eq!(env.schema_version.as_u16(), SCHEMA_VERSION);
        assert!(env.tenant_id.is_none());
    }

    #[test]
    fn test_envelope_with_identity() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Operations,
            1,
            "2025-06-15T12:30:00Z".into(),
            LifecycleEvent::Completed {
                status_code: Some(200),
                duration_ms: None,
                usage: None,
            },
        )
        .with_tenant("t-abc".into())
        .with_account("a-def".into());

        assert_eq!(env.tenant_id, Some("t-abc".into()));
        assert_eq!(env.account_id, Some("a-def".into()));
    }

    #[test]
    fn test_envelope_serde_roundtrip() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::System,
            42,
            "2025-07-01T10:00:00Z".into(),
            LifecycleEvent::Failed {
                error: "timeout".into(),
                status_code: None,
            },
        );

        let json = serde_json::to_string(&env).unwrap();
        let parsed: CanonicalEnvelope = serde_json::from_str(&json).unwrap();
        let parsed_via_boundary = CanonicalEnvelope::from_json_str(&json).unwrap();
        assert_eq!(parsed, env);
        assert_eq!(parsed_via_boundary, env);
    }

    #[test]
    fn test_envelope_identity_omitted_when_none() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            0,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        );

        let json = serde_json::to_string(&env).unwrap();
        assert!(!json.contains("tenant_id"));
        assert!(!json.contains("account_id"));
        assert!(!json.contains("user_id"));
        assert!(!json.contains("role"));
    }

    #[test]
    fn test_envelope_identity_included_when_set() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            0,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        )
        .with_user("u-xyz".into())
        .with_role("admin".into());

        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains("user_id"));
        assert!(json.contains("role"));
    }

    #[test]
    fn test_envelope_equality_ignores_presentation_context() {
        let base = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            7,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: Some("llama-3".into()),
                method: Some("POST".into()),
            },
        );
        let with_context =
            base.clone()
                .with_presentation_context(CanonicalPresentationContext::from_parts(
                    Some("health"),
                    Some("direct_http"),
                    None,
                    None,
                    None,
                    None,
                ));

        assert_eq!(base, with_context);
        // The process-local context is skipped on the wire, so the two
        // envelopes also serialize identically.
        assert_eq!(
            serde_json::to_value(&base).unwrap(),
            serde_json::to_value(&with_context).unwrap(),
        );
    }

    #[test]
    fn test_envelope_equality_compares_wire_fields() {
        let base = CanonicalEnvelope::new(
            EventId::from(uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()),
            RequestId::from(uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap()),
            ReplayChannel::Requests,
            7,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        );
        let different_request_id = CanonicalEnvelope::new(
            EventId::from(uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()),
            RequestId::from(uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000009").unwrap()),
            ReplayChannel::Requests,
            7,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        );

        assert_ne!(base, different_request_id);
    }
}

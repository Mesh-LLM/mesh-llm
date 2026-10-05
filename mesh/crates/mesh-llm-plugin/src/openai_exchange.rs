//! Generation-3 lifecycle service contract. Transport remains InvokeService.
use crate::{ManifestEntry, proto};
use anyhow::{Result, bail};
use mesh_llm_config::{OpenAiExchangeFailurePolicy, OpenAiExchangeGrant};
use serde::{Deserialize, Serialize};

pub const OPENAI_EXCHANGE_CONTRACT_VERSION: u32 = 1;
pub const OPENAI_EXCHANGE_CAPABILITY: &str = "openai_exchange.v1";

/// Construct a metadata-only observer declaration. Body and admission require opt-in.
pub fn openai_exchange_hook(handler: impl Into<String>) -> proto::OpenAiExchangeHookManifest {
    proto::OpenAiExchangeHookManifest {
        contract_version: OPENAI_EXCHANGE_CONTRACT_VERSION,
        handler: handler.into(),
        endpoints: vec![
            "chat_completions".into(),
            "completions".into(),
            "responses".into(),
        ],
        phases: vec![
            "request_received".into(),
            "backend_selected".into(),
            "exchange_finished".into(),
        ],
        metadata: true,
        deadline_ms: 250,
        max_body_bytes: 1024 * 1024,
        max_queue_bytes: 4 * 1024 * 1024,
        max_in_flight: 32,
        ..Default::default()
    }
}

impl From<proto::OpenAiExchangeHookManifest> for ManifestEntry {
    fn from(value: proto::OpenAiExchangeHookManifest) -> Self {
        Self::OpenAiExchangeHook(value)
    }
}

/// Grants are clipped to declarations; required declarations must be fully authorized.
/// Unsupported required versions fail startup rather than silently losing admission.
pub fn negotiate_openai_exchange(
    declaration: Option<&proto::OpenAiExchangeHookManifest>,
    grant: Option<&OpenAiExchangeGrant>,
) -> Result<Option<OpenAiExchangeGrant>> {
    let Some(request) = declaration else {
        return Ok(None);
    };
    if request.contract_version != OPENAI_EXCHANGE_CONTRACT_VERSION {
        if request.required {
            bail!(
                "unsupported required OpenAI lifecycle contract {}",
                request.contract_version
            );
        }
        return Ok(None);
    }
    validate_openai_exchange_manifest(request)?;
    let Some(grant) = grant else {
        if request.required {
            bail!("required OpenAI lifecycle contract has no operator grant");
        }
        return Ok(None);
    };
    grant.validate().map_err(anyhow::Error::msg)?;
    let mut effective = grant.clone();
    effective
        .endpoints
        .retain(|v| request.endpoints.contains(v));
    effective.phases.retain(|v| request.phases.contains(v));
    effective
        .headers
        .retain(|v| request.headers.iter().any(|h| h.eq_ignore_ascii_case(v)));
    effective
        .signing_scopes
        .retain(|v| request.signing_scopes.contains(v));
    effective.request_body &= request.request_body;
    effective.effective_request_body &= request.effective_request_body;
    effective.response_body &= request.response_body;
    effective.admission &= request.admission;
    effective.metadata &= request.metadata;
    effective.read_identity_bundle &= request.read_identity_bundle;
    effective.delegate_signing_key &= request.delegate_signing_key;
    effective.deadline_ms = grant.deadline_ms.min(request.deadline_ms);
    effective.max_body_bytes = grant.max_body_bytes.min(request.max_body_bytes);
    effective.max_queue_bytes = grant.max_queue_bytes.min(request.max_queue_bytes);
    effective.max_in_flight = grant.max_in_flight.min(request.max_in_flight);
    effective.max_delegation_ttl_secs = grant
        .max_delegation_ttl_secs
        .min(request.max_delegation_ttl_secs);
    if request.required && !fully_granted(request, &effective) {
        bail!("required OpenAI lifecycle permissions are not granted");
    }
    if effective.endpoints.is_empty() || effective.phases.is_empty() {
        return Ok(None);
    }
    Ok(Some(effective))
}

fn fully_granted(r: &proto::OpenAiExchangeHookManifest, g: &OpenAiExchangeGrant) -> bool {
    r.endpoints.iter().all(|v| g.endpoints.contains(v))
        && r.phases.iter().all(|v| g.phases.contains(v))
        && r.headers
            .iter()
            .all(|v| g.headers.iter().any(|h| h.eq_ignore_ascii_case(v)))
        && r.signing_scopes
            .iter()
            .all(|v| g.signing_scopes.contains(v))
        && (!r.request_body || g.request_body)
        && (!r.effective_request_body || g.effective_request_body)
        && (!r.response_body || g.response_body)
        && (!r.admission || g.admission)
        && (!r.metadata || g.metadata)
        && (!r.read_identity_bundle || g.read_identity_bundle)
        && (!r.delegate_signing_key || g.delegate_signing_key)
}

pub fn validate_openai_exchange_manifest(r: &proto::OpenAiExchangeHookManifest) -> Result<()> {
    if r.handler.is_empty() || r.handler.len() > 128 {
        bail!("lifecycle handler must be a nonempty service name of at most 128 bytes");
    }
    let permissions = OpenAiExchangeGrant {
        endpoints: r.endpoints.clone(),
        phases: r.phases.clone(),
        request_body: r.request_body,
        effective_request_body: r.effective_request_body,
        response_body: r.response_body,
        headers: r.headers.clone(),
        admission: r.admission,
        metadata: r.metadata,
        read_identity_bundle: r.read_identity_bundle,
        delegate_signing_key: r.delegate_signing_key,
        signing_scopes: r.signing_scopes.clone(),
        max_delegation_ttl_secs: r.max_delegation_ttl_secs,
        deadline_ms: r.deadline_ms,
        max_body_bytes: r.max_body_bytes,
        max_queue_bytes: r.max_queue_bytes,
        max_in_flight: r.max_in_flight,
        failure_policy: OpenAiExchangeFailurePolicy::BestEffort,
    };
    permissions.validate().map_err(anyhow::Error::msg)
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiAdmissionDecision {
    #[default]
    Abstain,
    Allow,
    Deny,
}

/// A response is read-only. No request or response replacement fields exist.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiExchangeDecision {
    pub decision: OpenAiAdmissionDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default)]
    pub annotations: Vec<(String, String)>,
    #[serde(default)]
    pub response_headers: Vec<(String, String)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grant() -> OpenAiExchangeGrant {
        let r = openai_exchange_hook("observe");
        OpenAiExchangeGrant {
            endpoints: r.endpoints,
            phases: r.phases,
            deadline_ms: 100,
            max_body_bytes: 4096,
            max_queue_bytes: 8192,
            max_in_flight: 2,
            metadata: true,
            ..Default::default()
        }
    }
    #[test]
    fn ordinary_installation_grants_nothing() {
        assert!(
            negotiate_openai_exchange(Some(&openai_exchange_hook("observe")), None)
                .unwrap()
                .is_none()
        );
        assert!(
            negotiate_openai_exchange(None, Some(&grant()))
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn required_admission_requires_operator_permission() {
        let mut r = openai_exchange_hook("observe");
        r.required = true;
        r.admission = true;
        assert!(negotiate_openai_exchange(Some(&r), Some(&grant())).is_err());
        let mut g = grant();
        g.admission = true;
        assert!(
            negotiate_openai_exchange(Some(&r), Some(&g))
                .unwrap()
                .unwrap()
                .admission
        );
    }
    #[test]
    fn optional_permissions_are_intersected_and_bounded() {
        let mut r = openai_exchange_hook("observe");
        r.request_body = true;
        let g = negotiate_openai_exchange(Some(&r), Some(&grant()))
            .unwrap()
            .unwrap();
        assert!(!g.request_body);
        assert_eq!(g.deadline_ms, 100);
        assert_eq!(g.max_body_bytes, 4096);
    }
    #[test]
    fn unsupported_required_contract_fails_clearly() {
        let mut r = openai_exchange_hook("observe");
        r.contract_version = 99;
        assert!(
            negotiate_openai_exchange(Some(&r), Some(&grant()))
                .unwrap()
                .is_none()
        );
        r.required = true;
        assert!(negotiate_openai_exchange(Some(&r), Some(&grant())).is_err());
    }
    #[test]
    fn observer_cannot_return_replacement_bytes() {
        assert!(
            serde_json::from_str::<OpenAiExchangeDecision>(
                r#"{"decision":"allow","body":"changed"}"#
            )
            .is_err()
        );
    }
}

pub type OpenAiExchangeFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = crate::PluginResult<OpenAiExchangeDecision>> + Send + 'a>,
>;
pub type OpenAiExchangeHandler = std::sync::Arc<
    dyn for<'a, 'ctx> Fn(
            String,
            serde_json::Value,
            &'a mut crate::PluginContext<'ctx>,
        ) -> OpenAiExchangeFuture<'a>
        + Send
        + Sync,
>;

/// Evidence publication is independent from inference success. Wire commitments
/// cover HTTP entity bytes, including emitted SSE framing, before final emission.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiExchangeEvent {
    pub exchange_id: String,
    pub phase: String,
    pub endpoint: String,
    pub observation_point: String,
    pub method: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub headers: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<serde_json::Value>,
    /// Bounded exact entity bytes. Larger bodies use correlated side streams.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_hex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_wire_digest: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_request_wire_digest: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_wire_commitment: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_completeness: Option<String>,
}

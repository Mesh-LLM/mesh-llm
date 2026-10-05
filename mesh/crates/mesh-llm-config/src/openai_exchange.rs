//! Operator-owned permissions for OpenAI lifecycle observers.
use serde::{Deserialize, Serialize};

/// A declaration never grants access. Only this host configuration does.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiExchangeGrant {
    pub endpoints: Vec<String>,
    pub phases: Vec<String>,
    pub request_body: bool,
    pub effective_request_body: bool,
    pub response_body: bool,
    pub headers: Vec<String>,
    pub admission: bool,
    pub metadata: bool,
    pub read_identity_bundle: bool,
    pub delegate_signing_key: bool,
    pub signing_scopes: Vec<String>,
    pub max_delegation_ttl_secs: u64,
    pub deadline_ms: u64,
    pub max_body_bytes: u64,
    pub max_queue_bytes: u64,
    pub max_in_flight: u32,
    pub failure_policy: OpenAiExchangeFailurePolicy,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiExchangeFailurePolicy {
    #[default]
    BestEffort,
    Required,
}

pub const OPENAI_EXCHANGE_ENDPOINTS: &[&str] = &["chat_completions", "completions", "responses"];
pub const OPENAI_EXCHANGE_PHASES: &[&str] =
    &["request_received", "backend_selected", "exchange_finished"];

impl OpenAiExchangeGrant {
    pub fn validate(&self) -> Result<(), String> {
        validate_members(&self.endpoints, OPENAI_EXCHANGE_ENDPOINTS, "endpoint")?;
        validate_members(&self.phases, OPENAI_EXCHANGE_PHASES, "phase")?;
        if self.endpoints.is_empty() || self.phases.is_empty() {
            return Err("lifecycle grants require endpoints and phases".into());
        }
        if self.deadline_ms == 0 || self.deadline_ms > 30_000 {
            return Err("deadline_ms must be between 1 and 30000".into());
        }
        if self.max_body_bytes == 0 || self.max_body_bytes > 16 * 1024 * 1024 {
            return Err("max_body_bytes must be between 1 and 16777216".into());
        }
        if self.max_queue_bytes == 0 || self.max_queue_bytes > 64 * 1024 * 1024 {
            return Err("max_queue_bytes must be between 1 and 67108864".into());
        }
        if self.max_in_flight == 0 || self.max_in_flight > 1024 {
            return Err("max_in_flight must be between 1 and 1024".into());
        }
        for (index, header) in self.headers.iter().enumerate() {
            if !safe_exchange_header(header)
                || self.headers[..index]
                    .iter()
                    .any(|previous| previous.eq_ignore_ascii_case(header))
            {
                return Err(format!("header {header:?} is sensitive or invalid"));
            }
        }
        if self.admission
            && !self
                .phases
                .iter()
                .any(|p| p == "request_received" || p == "backend_selected")
        {
            return Err("admission requires a pre-dispatch phase".into());
        }
        if self.delegate_signing_key
            && (!self.read_identity_bundle
                || self.signing_scopes.is_empty()
                || self.max_delegation_ttl_secs == 0
                || self.max_delegation_ttl_secs > 86_400)
        {
            return Err(
                "delegation requires identity permission, scopes, and a TTL of 1..86400".into(),
            );
        }
        if self
            .signing_scopes
            .iter()
            .any(|s| s != "mesh.openai.exchange.evidence.sign.v1")
        {
            return Err("only the registered mesh.openai.exchange.evidence.sign.v1 signing scope is supported".into());
        }
        Ok(())
    }
}

pub fn safe_exchange_header(header: &str) -> bool {
    let lower = header.to_ascii_lowercase();
    !header.is_empty()
        && header
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        && !matches!(
            lower.as_str(),
            "authorization"
                | "proxy-authorization"
                | "cookie"
                | "set-cookie"
                | "x-api-key"
                | "api-key"
        )
        && !lower.contains("token")
        && !lower.contains("secret")
        && !lower.contains("key")
        && !lower.contains("auth")
        && !lower.contains("cookie")
        && !lower.contains("session")
        && !lower.contains("jwt")
        && !lower.starts_with("x-owner-")
        && !lower.starts_with("x-mesh-owner-")
        && !lower.starts_with("x-mesh-control-")
}

fn validate_members(values: &[String], allowed: &[&str], kind: &str) -> Result<(), String> {
    for (index, value) in values.iter().enumerate() {
        if !allowed.contains(&value.as_str()) || values[..index].contains(value) {
            return Err(format!("invalid or duplicate lifecycle {kind} {value:?}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_only_policy_is_valid_but_terminal_only_cannot_admit() {
        let mut grant = OpenAiExchangeGrant {
            endpoints: vec!["chat_completions".into()],
            phases: vec!["backend_selected".into()],
            admission: true,
            deadline_ms: 250,
            max_body_bytes: 1_048_576,
            max_queue_bytes: 4_194_304,
            max_in_flight: 8,
            ..Default::default()
        };
        assert!(grant.validate().is_ok());
        grant.phases = vec!["exchange_finished".into()];
        assert!(grant.validate().unwrap_err().contains("pre-dispatch"));
        grant.admission = false;
        assert!(grant.validate().is_ok());
    }
    #[test]
    fn operator_grants_round_trip_and_reject_invalid_headers() {
        let text = r#"
version = 1
[[plugin]]
name = "observer"
[plugin.openai_exchange_grant]
endpoints = ["chat_completions"]
phases = ["request_received", "exchange_finished"]
metadata = true
deadline_ms = 100
max_body_bytes = 4096
max_queue_bytes = 8192
max_in_flight = 2
"#;
        let config = crate::parse_config_toml(text).unwrap();
        let encoded = crate::config_to_toml(&config).unwrap();
        let decoded = crate::parse_config_toml(&encoded).unwrap();
        assert_eq!(
            config.plugins[0].openai_exchange_grant,
            decoded.plugins[0].openai_exchange_grant
        );
        let invalid = format!("{text}headers = [\"Authorization\"]\n");
        let error = crate::parse_config_toml(&invalid).unwrap_err().to_string();
        assert!(error.contains("openai_exchange_grant"), "{error}");
        assert!(error.contains("sensitive"), "{error}");
    }

    #[test]
    fn headers_cannot_grant_credentials() {
        for header in [
            "Authorization",
            "Cookie",
            "Set-Cookie",
            "X-Api-Key",
            "x-access-token",
            "x-secret",
            "x-owner-signature",
            "x-mesh-owner-control",
            "x-mesh-control-auth",
            "x-custom-auth",
            "x-browser-cookie",
            "x-session-id",
            "x-jwt-assertion",
        ] {
            assert!(!safe_exchange_header(header));
        }
        assert!(safe_exchange_header("content-type"));
        assert!(!safe_exchange_header("x-invalid\n"));
    }
    #[test]
    fn empty_grants_are_not_effective() {
        let grant: OpenAiExchangeGrant = serde_json::from_str("{}").unwrap();
        assert!(!grant.request_body);
        assert!(!grant.admission);
        assert!(grant.validate().is_err());
    }
}

//! Packaged lifecycle access requests, read without launching plugin code.
use serde::{Deserialize, Serialize};

/// Manifest requests are public installation metadata, never operator grants.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct InstalledOpenAiExchangeAccess {
    pub contract_version: u32,
    pub handler: String,
    pub required: bool,
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
    /// Limits the package asks for; the grant made at install uses them.
    pub deadline_ms: u64,
    pub max_body_bytes: u64,
    pub max_queue_bytes: u64,
    pub max_in_flight: u32,
    pub max_delegation_ttl_secs: u64,
}

impl InstalledOpenAiExchangeAccess {
    /// Installation notice: installing the package grants exactly these
    /// requests (a `[[plugin]]` entry can narrow or replace them).
    /// The permission labels this request turns on, in disclosure order.
    fn granted_labels(&self) -> Vec<&'static str> {
        let mut permissions = Vec::new();
        for (enabled, label) in [
            (self.request_body, "original prompts"),
            (self.effective_request_body, "effective prompts"),
            (self.response_body, "generated content"),
            (self.admission, "admission denial"),
            (self.metadata, "annotations/response metadata"),
            (self.read_identity_bundle, "public software identity"),
            (self.delegate_signing_key, "owner-scoped signing delegation"),
        ] {
            if enabled {
                permissions.push(label);
            }
        }
        if !self.headers.is_empty() {
            permissions.push("allowlisted headers");
        }
        permissions
    }

    /// Permissions this request has that `previous` (the access the operator
    /// accepted when the earlier version was installed) did not. An update
    /// is accepted by running it, so anything new is called out rather than
    /// silently widened. Endpoints and phases that were not hooked before
    /// count as new access too.
    pub fn increases_over(&self, previous: &Self) -> Vec<String> {
        let before = previous.granted_labels();
        let mut increases: Vec<String> = self
            .granted_labels()
            .into_iter()
            .filter(|label| !before.contains(label))
            .map(str::to_owned)
            .collect();
        for endpoint in &self.endpoints {
            if !previous.endpoints.contains(endpoint) {
                increases.push(format!("endpoint {endpoint}"));
            }
        }
        for phase in &self.phases {
            if !previous.phases.contains(phase) {
                increases.push(format!("phase {phase}"));
            }
        }
        for header in &self.headers {
            if !previous.headers.contains(header) {
                increases.push(format!("header {header}"));
            }
        }
        for scope in &self.signing_scopes {
            if !previous.signing_scopes.contains(scope) {
                increases.push(format!("signing scope {scope}"));
            }
        }
        increases
    }

    pub fn access_summary(&self) -> String {
        let permissions = self.granted_labels();
        format!(
            "OpenAI lifecycle v{} access: {}. Installing grants these requests; a [[plugin]] entry in config.toml can narrow or replace them. Prompt/content access discloses that data to this plugin process.",
            self.contract_version,
            if permissions.is_empty() {
                "lifecycle metadata".into()
            } else {
                permissions.join(", ")
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn summary_reports_prompt_content_and_identity_requests() {
        let metadata: InstalledOpenAiExchangeAccess = serde_json::from_str(
            r#"{"contract_version":1,"request_body":true,"response_body":true,"delegate_signing_key":true}"#,
        )
        .unwrap();
        let summary = metadata.access_summary();
        for text in [
            "original prompts",
            "generated content",
            "signing delegation",
            "Installing grants these requests",
        ] {
            assert!(summary.contains(text));
        }
    }

    #[test]
    fn manifest_metadata_retains_access_requests_and_accepts_old_packages() {
        let old: super::super::InstalledPluginManifestMetadata =
            serde_json::from_str(r#"{"config_schema":null,"web_ui":null}"#).unwrap();
        assert!(old.openai_exchange_hook.is_none());
        let requested: super::super::InstalledPluginManifestMetadata = serde_json::from_str(
            r#"{"openai_exchange_hook":{"contract_version":1,"request_body":true,"response_body":true,"admission":true}}"#,
        ).unwrap();
        let restored: super::super::InstalledPluginManifestMetadata =
            serde_json::from_slice(&serde_json::to_vec(&requested).unwrap()).unwrap();
        assert_eq!(requested, restored);
        let access = restored.openai_exchange_hook.unwrap();
        assert!(access.request_body && access.response_body && access.admission);
    }

    #[test]
    fn increases_name_only_what_the_previous_version_lacked() {
        let before = InstalledOpenAiExchangeAccess {
            endpoints: vec!["responses".into()],
            phases: vec!["exchange_finished".into()],
            metadata: true,
            ..Default::default()
        };
        let mut now = before.clone();
        assert!(now.increases_over(&before).is_empty());
        now.request_body = true;
        now.delegate_signing_key = true;
        now.endpoints.push("chat_completions".into());
        now.headers.push("x-example".into());
        let listed = now.increases_over(&before);
        for expected in [
            "original prompts",
            "owner-scoped signing delegation",
            "allowlisted headers",
            "endpoint chat_completions",
            "header x-example",
        ] {
            assert!(
                listed.iter().any(|item| item == expected),
                "{expected} missing from {listed:?}"
            );
        }
        assert_eq!(listed.len(), 5);
        // Dropping access is not an increase.
        assert!(before.increases_over(&now).is_empty());
    }
}

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
}

impl InstalledOpenAiExchangeAccess {
    /// Explicit installation warning; requests confer no permissions.
    pub fn access_summary(&self) -> String {
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
        format!(
            "OpenAI lifecycle v{} requests: {}. Explicit operator grants are required; installation grants no access. Prompt/content grants disclose that data to this plugin process.",
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
            "installation grants no access",
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
}

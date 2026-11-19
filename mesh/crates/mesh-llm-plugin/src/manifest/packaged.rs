//! The packaged subset of a plugin manifest.
use super::*;

pub(super) fn render(manifest: &proto::PluginManifest) -> Result<String> {
    Ok(serde_json::to_string_pretty(
        &PackagedPluginManifest::try_from(manifest)?,
    )?)
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(super) struct PackagedPluginManifest {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(super) config_schema: Option<PackagedPluginConfigSchema>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(super) openai_exchange_hook: Option<proto::OpenAiExchangeHookManifest>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(super) peer_evidence_verifier: Option<proto::PeerEvidenceVerifierManifest>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(super) web_ui: Option<PackagedPluginWebUi>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(super) operation_bindings: Vec<proto::OperationBindingManifest>,
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub(super) peer_blocks: bool,
}

impl TryFrom<&proto::PluginManifest> for PackagedPluginManifest {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginManifest) -> Result<Self> {
        let config_schema = value
            .config_schema
            .as_ref()
            .map(PackagedPluginConfigSchema::try_from)
            .transpose()?;
        let web_ui = value
            .web_ui
            .as_ref()
            .map(PackagedPluginWebUi::try_from)
            .transpose()?;

        if let Some(hook) = &value.openai_exchange_hook {
            crate::openai_exchange::validate_openai_exchange_manifest(hook)?;
        }
        crate::peer_evidence::validate_manifest(value)?;
        Ok(Self {
            peer_evidence_verifier: value.peer_evidence_verifier.clone(),
            openai_exchange_hook: value.openai_exchange_hook.as_deref().cloned(),
            operation_bindings: value
                .grant_requests
                .as_deref()
                .map(|requests| requests.operation_bindings.clone())
                .unwrap_or_default(),
            peer_blocks: value
                .grant_requests
                .as_deref()
                .is_some_and(|requests| requests.peer_blocks),
            config_schema,
            web_ui,
        })
    }
}

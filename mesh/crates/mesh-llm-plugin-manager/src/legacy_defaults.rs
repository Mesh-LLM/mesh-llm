//! Grants retained for plugins installed by the retired bundled-default
//! system.
//!
//! MeshLLM releases once bundled default plugins and installed them from a
//! compiled-in pin list. That system is gone: plugins are installed from
//! catalog sets (`crate::sets`) or by the operator. What remains here is the
//! record of what those legacy installs were allowed to use, so migrating a
//! legacy default-managed install to a normal managed one
//! (`PluginStore::migrate_legacy_defaults`) keeps its effective permissions
//! instead of silently expanding them.
//!
//! Nothing here installs anything, and no release pin or checksum lives
//! here: a legacy record only keeps its already-verified identity and this
//! surface allowlist.

/// A host surface a plugin's initialize answer can claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    WebUi,
    Config,
    MeshChannels,
    MeshEvents,
    HttpRoutes,
    /// MCP operations, resources, resource templates, prompts, completions.
    McpOperations,
    InferenceEndpoints,
    /// Any endpoint that is not an inference endpoint.
    OtherEndpoints,
    VirtualModels,
    /// The OpenAI exchange lifecycle hook: it sees every OpenAI exchange the
    /// node handles, prompts and answers included.
    OpenAiExchangeHook,
    /// The plugin's own snapshot, shown in `GET /api/status`.
    StatusContributions,
    /// The local inference observation feed: what this node executes and
    /// its token counts, never prompts or answers.
    InferenceObservation,
}

impl Surface {
    pub fn name(self) -> &'static str {
        match self {
            Self::WebUi => "web UI",
            Self::Config => "config",
            Self::MeshChannels => "mesh channels",
            Self::MeshEvents => "mesh events",
            Self::HttpRoutes => "HTTP routes",
            Self::McpOperations => "MCP operations",
            Self::InferenceEndpoints => "inference endpoints",
            Self::OtherEndpoints => "other endpoints",
            Self::VirtualModels => "virtual models",
            Self::OpenAiExchangeHook => "OpenAI exchange hook",
            Self::StatusContributions => "status contributions",
            Self::InferenceObservation => "local inference observation",
        }
    }

    /// The stable key a stored allowlist names this surface by.
    pub fn key(self) -> &'static str {
        match self {
            Self::WebUi => "web_ui",
            Self::Config => "config",
            Self::MeshChannels => "mesh_channels",
            Self::MeshEvents => "mesh_events",
            Self::HttpRoutes => "http_routes",
            Self::McpOperations => "mcp_operations",
            Self::InferenceEndpoints => "inference_endpoints",
            Self::OtherEndpoints => "other_endpoints",
            Self::VirtualModels => "virtual_models",
            Self::OpenAiExchangeHook => "openai_exchange_hook",
            Self::StatusContributions => "status_contributions",
            Self::InferenceObservation => "inference_observation",
        }
    }

    /// The surface a stored key names; `None` for an unrecognized key, so a
    /// corrupted record narrows the plugin rather than widening it.
    pub fn parse_key(key: &str) -> Option<Self> {
        Some(match key {
            "web_ui" => Self::WebUi,
            "config" => Self::Config,
            "mesh_channels" => Self::MeshChannels,
            "mesh_events" => Self::MeshEvents,
            "http_routes" => Self::HttpRoutes,
            "mcp_operations" => Self::McpOperations,
            "inference_endpoints" => Self::InferenceEndpoints,
            "other_endpoints" => Self::OtherEndpoints,
            "virtual_models" => Self::VirtualModels,
            "openai_exchange_hook" => Self::OpenAiExchangeHook,
            "status_contributions" => Self::StatusContributions,
            "inference_observation" => Self::InferenceObservation,
            _ => return None,
        })
    }
}

/// What a legacy default plugin that is not in the retained map may still
/// use: everything but serving models, observing inference, or seeing
/// OpenAI exchanges, until its operator widens it by installing a release
/// that asks for more.
pub const UNLISTED_DEFAULT_ALLOWS: &[Surface] = &[
    Surface::WebUi,
    Surface::Config,
    Surface::MeshChannels,
    Surface::MeshEvents,
    Surface::HttpRoutes,
    Surface::McpOperations,
    Surface::OtherEndpoints,
    Surface::StatusContributions,
];

/// The exact grants each known legacy default was reviewed with, by its
/// installed name. Only names are retained: releases, pins, and checksums
/// belonged to the retired bundled system and are gone.
const LEGACY_DEFAULT_ALLOWS: &[(&str, &[Surface])] = &[(
    "capsules",
    // An audit plugin: records, its page and its settings; it serves no model.
    &[
        Surface::WebUi,
        Surface::Config,
        Surface::MeshChannels,
        Surface::MeshEvents,
        Surface::HttpRoutes,
        Surface::McpOperations,
        Surface::OtherEndpoints,
    ],
)];

/// The grants a migrated legacy default-managed install keeps. A known
/// legacy default keeps exactly what it was reviewed with; any other legacy
/// default-managed install falls back to [`UNLISTED_DEFAULT_ALLOWS`].
pub fn migrated_allows(name: &str) -> &'static [Surface] {
    LEGACY_DEFAULT_ALLOWS
        .iter()
        .find(|(legacy, _)| *legacy == name)
        .map(|(_, allows)| *allows)
        .unwrap_or(UNLISTED_DEFAULT_ALLOWS)
}

/// The stored keys for [`migrated_allows`], in the allowlist's order.
pub fn migrated_allows_keys(name: &str) -> Vec<String> {
    migrated_allows(name)
        .iter()
        .map(|surface| surface.key().to_string())
        .collect()
}

/// Whether `allows` grants `surface`.
pub fn allows(allows: &[Surface], surface: Surface) -> bool {
    allows.contains(&surface)
}

/// Stored as keys on an installed plugin record, readable as surfaces.
pub fn parse_allows(keys: &[String]) -> Vec<Surface> {
    keys.iter()
        .filter_map(|key| Surface::parse_key(key))
        .collect()
}

pub const NO_DEFAULT_PLUGINS_ENV: &str = "MESH_LLM_NO_DEFAULT_PLUGINS";

/// Whether a set value turns the setting on. Kept for the legacy
/// `MESH_LLM_NO_DEFAULT_PLUGINS` setting, which setup and update honor as an
/// installation-skip default: it skips *optional* plugin installation, never
/// a required repair.
pub fn opts_out(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

pub fn default_plugins_opted_out() -> bool {
    opts_out(std::env::var(NO_DEFAULT_PLUGINS_ENV).ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_keys_round_trip() {
        for surface in [
            Surface::WebUi,
            Surface::Config,
            Surface::MeshChannels,
            Surface::MeshEvents,
            Surface::HttpRoutes,
            Surface::McpOperations,
            Surface::InferenceEndpoints,
            Surface::OtherEndpoints,
            Surface::VirtualModels,
            Surface::OpenAiExchangeHook,
            Surface::StatusContributions,
            Surface::InferenceObservation,
        ] {
            assert_eq!(Surface::parse_key(surface.key()), Some(surface));
        }
        assert_eq!(Surface::parse_key("garbage"), None);
    }

    #[test]
    fn a_known_legacy_default_keeps_its_exact_grants_and_serves_no_model() {
        let grants = migrated_allows("capsules");
        assert!(allows(grants, Surface::WebUi));
        assert!(allows(grants, Surface::McpOperations));
        assert!(!allows(grants, Surface::InferenceEndpoints));
        assert!(!allows(grants, Surface::VirtualModels));
        assert!(!allows(grants, Surface::OpenAiExchangeHook));
        assert!(
            !allows(grants, Surface::StatusContributions),
            "capsules was not reviewed for status contributions; migration must not add it"
        );
    }

    #[test]
    fn an_unlisted_legacy_default_never_serves_a_model_or_sees_exchanges() {
        let grants = migrated_allows("something-else");
        for surface in [
            Surface::InferenceEndpoints,
            Surface::VirtualModels,
            Surface::OpenAiExchangeHook,
            Surface::InferenceObservation,
        ] {
            assert!(!allows(grants, surface), "{surface:?}");
        }
        assert!(allows(grants, Surface::WebUi));
    }

    #[test]
    fn stored_keys_survive_a_round_trip() {
        let keys = migrated_allows_keys("capsules");
        let parsed = parse_allows(&keys);
        assert_eq!(parsed.to_vec(), migrated_allows("capsules").to_vec());
    }

    #[test]
    fn install_time_opt_out_reads_like_a_flag() {
        assert!(!opts_out(None));
        assert!(!opts_out(Some("")));
        assert!(!opts_out(Some("0")));
        assert!(opts_out(Some("1")));
        assert!(opts_out(Some("true")));
    }
}

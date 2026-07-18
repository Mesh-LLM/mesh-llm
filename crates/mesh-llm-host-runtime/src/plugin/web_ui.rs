use super::{PluginSummary, PluginWebUiPreference, proto};
use mesh_llm_plugin_manager::store::{
    InstalledPluginWebUiMetadata, InstalledPluginWebUiValidationStatus,
};
use mesh_llm_plugin_manager::{InstalledPluginMetadata, PluginStore, default_store_root};
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct PluginWebUiState {
    pub state: PluginWebUiStateKind,
    pub declared: bool,
    pub enabled: bool,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub pages: Vec<PluginWebUiPageOverview>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub config_sections: Vec<PluginWebUiConfigSectionOverview>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset_base_url: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PluginWebUiStateKind {
    #[default]
    None,
    Ready,
    Disabled,
    Invalid,
    PluginNotRunning,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginWebUiManifestOverview {
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub pages: Vec<PluginWebUiPageOverview>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub config_sections: Vec<PluginWebUiConfigSectionOverview>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginWebUiPageOverview {
    pub id: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    pub route: String,
    pub bundle_id: String,
    pub entry_script: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginWebUiConfigSectionOverview {
    pub id: String,
    pub title: String,
    pub entry_script: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_tab: Option<String>,
    pub bundle_id: String,
}

pub(crate) struct PluginWebUiStateInput<'a> {
    pub plugin_name: &'a str,
    pub live_manifest: Option<&'a proto::PluginManifest>,
    pub installed_metadata: Option<&'a InstalledPluginMetadata>,
    pub web_ui_enabled: Option<bool>,
    pub runtime_available: bool,
    pub runtime_unavailable_reason: Option<&'a str>,
}

pub(crate) fn derive_plugin_web_ui_state(input: PluginWebUiStateInput<'_>) -> PluginWebUiState {
    let declaration = plugin_web_ui_declaration(&input);
    let Some(declaration) = declaration else {
        return PluginWebUiState::default();
    };
    let preference = plugin_web_ui_preference(input.web_ui_enabled, true);
    match preference {
        PluginWebUiPreference::None => PluginWebUiState::default(),
        PluginWebUiPreference::Disabled => declaration.state(
            PluginWebUiStateKind::Disabled,
            false,
            false,
            Some("web UI disabled by configuration".into()),
        ),
        PluginWebUiPreference::Enabled => enabled_plugin_web_ui_state(input, declaration),
    }
}

fn enabled_plugin_web_ui_state(
    input: PluginWebUiStateInput<'_>,
    declaration: PluginWebUiDeclaration,
) -> PluginWebUiState {
    if let Some(reason) = declaration.invalid_reason.clone() {
        return declaration.state(PluginWebUiStateKind::Invalid, true, false, Some(reason));
    }
    if !input.runtime_available {
        let reason = input
            .runtime_unavailable_reason
            .unwrap_or("plugin process is not running")
            .to_string();
        return declaration.state(
            PluginWebUiStateKind::PluginNotRunning,
            true,
            false,
            Some(reason),
        );
    }
    declaration.state(PluginWebUiStateKind::Ready, true, true, None)
}

pub(super) fn inactive_web_ui_state(
    summary: &PluginSummary,
    web_ui_enabled: Option<bool>,
) -> PluginWebUiState {
    if let Some(metadata) = installed_plugin_metadata(&summary.name) {
        return derive_plugin_web_ui_state(PluginWebUiStateInput {
            plugin_name: &summary.name,
            live_manifest: None,
            installed_metadata: Some(&metadata),
            web_ui_enabled,
            runtime_available: summary.status == "running",
            runtime_unavailable_reason: summary.error.as_deref(),
        });
    }
    projected_existing_web_ui_state(summary, web_ui_enabled)
}

pub(super) fn projected_existing_web_ui_state(
    summary: &PluginSummary,
    web_ui_enabled: Option<bool>,
) -> PluginWebUiState {
    let mut web_ui = summary.web_ui.clone();
    if !web_ui.declared {
        return PluginWebUiState::default();
    }
    match plugin_web_ui_preference(web_ui_enabled, true) {
        PluginWebUiPreference::None => return PluginWebUiState::default(),
        PluginWebUiPreference::Disabled => {
            web_ui.state = PluginWebUiStateKind::Disabled;
            web_ui.enabled = false;
            web_ui.available = false;
            web_ui.unavailable_reason = Some("web UI disabled by configuration".into());
        }
        PluginWebUiPreference::Enabled => {
            web_ui.enabled = true;
            if summary.status == "running" && web_ui.asset_base_url.is_some() {
                web_ui.state = PluginWebUiStateKind::Ready;
                web_ui.available = true;
                web_ui.unavailable_reason = None;
            } else if summary.status != "running" {
                web_ui.state = PluginWebUiStateKind::PluginNotRunning;
                web_ui.available = false;
                web_ui.unavailable_reason = summary
                    .error
                    .clone()
                    .or_else(|| Some("plugin process is not running".to_string()));
            } else if web_ui.state == PluginWebUiStateKind::Disabled {
                web_ui.state = PluginWebUiStateKind::Invalid;
                web_ui.available = false;
                web_ui.unavailable_reason = Some("web UI bundle metadata is unavailable".into());
            }
        }
    }
    web_ui
}

pub(super) fn installed_plugin_metadata(name: &str) -> Option<InstalledPluginMetadata> {
    let root = default_store_root().ok()?;
    PluginStore::new(root).load_optional(name).ok().flatten()
}

fn plugin_web_ui_preference(
    web_ui_enabled: Option<bool>,
    declares_web_ui: bool,
) -> PluginWebUiPreference {
    PluginWebUiPreference::resolve(web_ui_enabled, declares_web_ui)
}

#[derive(Clone, Debug)]
struct PluginWebUiDeclaration {
    pages: Vec<PluginWebUiPageOverview>,
    config_sections: Vec<PluginWebUiConfigSectionOverview>,
    asset_base_url: Option<String>,
    invalid_reason: Option<String>,
}

impl PluginWebUiDeclaration {
    fn state(
        self,
        state: PluginWebUiStateKind,
        enabled: bool,
        available: bool,
        unavailable_reason: Option<String>,
    ) -> PluginWebUiState {
        PluginWebUiState {
            state,
            declared: true,
            enabled,
            available,
            unavailable_reason,
            pages: self.pages,
            config_sections: self.config_sections,
            asset_base_url: self.asset_base_url,
        }
    }
}

fn plugin_web_ui_declaration(input: &PluginWebUiStateInput<'_>) -> Option<PluginWebUiDeclaration> {
    if let Some(metadata) = input.installed_metadata
        && let Some(web_ui) = metadata
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.web_ui.as_ref())
    {
        return Some(plugin_web_ui_declaration_from_installed(
            input.plugin_name,
            metadata,
            web_ui,
        ));
    }
    input
        .live_manifest
        .and_then(|manifest| manifest.web_ui.as_ref())
        .map(plugin_web_ui_declaration_from_proto)
}

fn plugin_web_ui_declaration_from_installed(
    plugin_name: &str,
    metadata: &InstalledPluginMetadata,
    web_ui: &InstalledPluginWebUiMetadata,
) -> PluginWebUiDeclaration {
    let asset_root = metadata
        .web_ui_asset_root_path()
        .filter(|path| path.is_dir());
    let invalid_reason = match web_ui.validation.status {
        InstalledPluginWebUiValidationStatus::Valid if asset_root.is_some() => None,
        InstalledPluginWebUiValidationStatus::Valid => {
            Some("web UI bundle asset root is missing".into())
        }
        InstalledPluginWebUiValidationStatus::Invalid => web_ui
            .validation
            .reason
            .clone()
            .or_else(|| Some("web UI bundle is invalid".into())),
    };
    PluginWebUiDeclaration {
        pages: web_ui
            .pages
            .iter()
            .map(plugin_web_ui_page_from_installed)
            .collect(),
        config_sections: web_ui
            .config_sections
            .iter()
            .map(plugin_web_ui_config_section_from_installed)
            .collect(),
        asset_base_url: asset_root.map(|_| format!("/api/plugins/{plugin_name}/web-ui/assets/")),
        invalid_reason,
    }
}

fn plugin_web_ui_declaration_from_proto(
    web_ui: &proto::PluginWebUiManifest,
) -> PluginWebUiDeclaration {
    PluginWebUiDeclaration {
        pages: web_ui
            .pages
            .iter()
            .map(plugin_web_ui_page_from_proto)
            .collect(),
        config_sections: web_ui
            .config_sections
            .iter()
            .map(plugin_web_ui_config_section_from_proto)
            .collect(),
        asset_base_url: None,
        invalid_reason: Some("web UI bundle metadata is unavailable".into()),
    }
}

pub(super) fn plugin_web_ui_manifest_overview_from_proto(
    web_ui: Option<&proto::PluginWebUiManifest>,
) -> Option<PluginWebUiManifestOverview> {
    web_ui.map(|web_ui| PluginWebUiManifestOverview {
        pages: web_ui
            .pages
            .iter()
            .map(plugin_web_ui_page_from_proto)
            .collect(),
        config_sections: web_ui
            .config_sections
            .iter()
            .map(plugin_web_ui_config_section_from_proto)
            .collect(),
    })
}

fn plugin_web_ui_page_from_proto(page: &proto::PluginWebUiPageManifest) -> PluginWebUiPageOverview {
    PluginWebUiPageOverview {
        id: page.id.clone(),
        label: page.label.clone(),
        icon: page.icon.clone(),
        route: page.route.clone(),
        bundle_id: page.bundle_id.clone(),
        entry_script: page.entry_script.clone(),
    }
}

fn plugin_web_ui_config_section_from_proto(
    section: &proto::PluginWebUiConfigSectionManifest,
) -> PluginWebUiConfigSectionOverview {
    PluginWebUiConfigSectionOverview {
        id: section.id.clone(),
        title: section.title.clone(),
        entry_script: section.entry_script.clone(),
        parent_tab: section.parent_tab.clone(),
        bundle_id: section.bundle_id.clone(),
    }
}

fn plugin_web_ui_page_from_installed(
    page: &mesh_llm_plugin_manager::store::InstalledPluginWebUiPageMetadata,
) -> PluginWebUiPageOverview {
    PluginWebUiPageOverview {
        id: page.id.clone(),
        label: page.label.clone(),
        icon: page.icon.clone(),
        route: page.route.clone(),
        bundle_id: page.bundle_id.clone(),
        entry_script: page.entry_script.clone(),
    }
}

fn plugin_web_ui_config_section_from_installed(
    section: &mesh_llm_plugin_manager::store::InstalledPluginWebUiConfigSectionMetadata,
) -> PluginWebUiConfigSectionOverview {
    PluginWebUiConfigSectionOverview {
        id: section.id.clone(),
        title: section.title.clone(),
        entry_script: section.entry_script.clone(),
        parent_tab: section.parent_tab.clone(),
        bundle_id: section.bundle_id.clone(),
    }
}

#[cfg(test)]
pub(super) fn installed_metadata_with_web_ui(
    validation: InstalledPluginWebUiValidationStatus,
    asset_root: Option<&str>,
) -> InstalledPluginMetadata {
    let install_path = std::env::temp_dir().join("mesh-llm-demo-plugin");
    if let Some(asset_root) = asset_root {
        std::fs::create_dir_all(install_path.join(asset_root)).unwrap();
    }
    InstalledPluginMetadata {
        name: "demo".into(),
        source_repository: "https://github.com/mesh-llm/demo".into(),
        installed_version: "v1.0.0".into(),
        target_triple: "test-target".into(),
        downloaded_asset_name: "demo.tar.gz".into(),
        install_path,
        enabled: true,
        manifest: Some(mesh_llm_plugin_manager::InstalledPluginManifestMetadata {
            config_schema: None,
            web_ui: Some(
                mesh_llm_plugin_manager::store::InstalledPluginWebUiMetadata {
                    pages: vec![
                        mesh_llm_plugin_manager::store::InstalledPluginWebUiPageMetadata {
                            id: "home".into(),
                            label: "Home".into(),
                            icon: Some("icons/home.svg".into()),
                            route: "index.html".into(),
                            bundle_id: "main".into(),
                            entry_script: "assets/app.js".into(),
                        },
                    ],
                    config_sections: vec![
                        mesh_llm_plugin_manager::store::InstalledPluginWebUiConfigSectionMetadata {
                            id: "settings".into(),
                            title: "Settings".into(),
                            entry_script: "assets/settings.js".into(),
                            parent_tab: Some("integrations".into()),
                            bundle_id: "main".into(),
                        },
                    ],
                    bundles: vec![
                        mesh_llm_plugin_manager::store::InstalledPluginWebUiBundleMetadata {
                            id: "main".into(),
                            root_path: "web".into(),
                        },
                    ],
                    asset_root: asset_root.map(std::path::PathBuf::from),
                    validation: mesh_llm_plugin_manager::store::InstalledPluginWebUiValidation {
                        status: validation,
                        reason: Some("bundle failed validation".into()),
                    },
                },
            ),
        }),
        last_protocol_version: Some(1),
        last_status: Some("running".into()),
        last_error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin_web_ui_state_for_test(
        installed_metadata: Option<&InstalledPluginMetadata>,
        web_ui_enabled: Option<bool>,
        runtime_available: bool,
    ) -> PluginWebUiState {
        derive_plugin_web_ui_state(PluginWebUiStateInput {
            plugin_name: "demo",
            live_manifest: None,
            installed_metadata,
            web_ui_enabled,
            runtime_available,
            runtime_unavailable_reason: Some("plugin process is not running"),
        })
    }

    #[test]
    fn plugin_web_ui_state_is_ready_for_enabled_valid_running_plugin() {
        let metadata = installed_metadata_with_web_ui(
            InstalledPluginWebUiValidationStatus::Valid,
            Some("web"),
        );

        let state = plugin_web_ui_state_for_test(Some(&metadata), None, true);

        assert_eq!(state.state, PluginWebUiStateKind::Ready);
        assert!(state.declared);
        assert!(state.enabled);
        assert!(state.available);
        assert_eq!(
            state.asset_base_url.as_deref(),
            Some("/api/plugins/demo/web-ui/assets/")
        );
        assert_eq!(state.pages.len(), 1);
        assert_eq!(state.config_sections.len(), 1);
    }

    #[test]
    fn plugin_web_ui_state_covers_none_disabled_invalid_and_not_running() {
        let valid = installed_metadata_with_web_ui(
            InstalledPluginWebUiValidationStatus::Valid,
            Some("web"),
        );
        let invalid = installed_metadata_with_web_ui(
            InstalledPluginWebUiValidationStatus::Invalid,
            Some("web"),
        );
        let missing_bundle =
            installed_metadata_with_web_ui(InstalledPluginWebUiValidationStatus::Valid, None);

        let none = plugin_web_ui_state_for_test(None, None, true);
        let disabled = plugin_web_ui_state_for_test(Some(&valid), Some(false), true);
        let invalid_state = plugin_web_ui_state_for_test(Some(&invalid), None, true);
        let missing_bundle_state = plugin_web_ui_state_for_test(Some(&missing_bundle), None, true);
        let not_running = plugin_web_ui_state_for_test(Some(&valid), None, false);

        assert_eq!(none.state, PluginWebUiStateKind::None);
        assert_eq!(disabled.state, PluginWebUiStateKind::Disabled);
        assert!(!disabled.enabled);
        assert_eq!(invalid_state.state, PluginWebUiStateKind::Invalid);
        assert_eq!(missing_bundle_state.state, PluginWebUiStateKind::Invalid);
        assert_eq!(not_running.state, PluginWebUiStateKind::PluginNotRunning);
        assert!(not_running.enabled);
        assert!(!not_running.available);
    }
}

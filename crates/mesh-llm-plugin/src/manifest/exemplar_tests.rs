use std::{fs, path::PathBuf};

use super::{PackagedPluginManifest, PackagedPluginValueKind};

fn web_ui_exemplar_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/plugins/exemplars/web-ui")
        .join(file_name)
}

#[test]
fn web_ui_exemplar_package_manifest_matches_packaged_contract() {
    let encoded = fs::read_to_string(web_ui_exemplar_path("plugin.package.json"))
        .expect("read web UI exemplar package manifest");
    let decoded: PackagedPluginManifest =
        serde_json::from_str(&encoded).expect("exemplar manifest should deserialize");

    let schema = decoded.config_schema.expect("config schema");
    assert_eq!(schema.plugin_name, "web-ui-exemplar");
    assert_eq!(schema.settings[0].key, "retention_days");
    assert_eq!(
        schema.settings[0].value_schema.kind,
        PackagedPluginValueKind::Integer
    );

    let web_ui = decoded.web_ui.expect("web_ui");
    assert_eq!(web_ui.bundles.len(), 1);
    assert_eq!(web_ui.bundles[0].root_path, "bundle");
    assert_eq!(web_ui.pages[0].id, "overview");
    assert_eq!(web_ui.pages[0].label, "Exemplar Notes");
    assert_eq!(web_ui.pages[0].entry_script, "register-mesh-plugin-ui.js");
    assert_eq!(web_ui.config_sections[0].id, "page-actions");
    assert_eq!(
        web_ui.config_sections[0].parent_tab.as_deref(),
        Some("integrations")
    );
}

#[test]
fn web_ui_exemplar_lifecycle_states_preserve_non_ui_capability() {
    let encoded = fs::read_to_string(web_ui_exemplar_path("lifecycle-states.json"))
        .expect("read web UI exemplar lifecycle states");
    let decoded: serde_json::Value =
        serde_json::from_str(&encoded).expect("lifecycle states should parse");

    for state in ["none", "ready", "disabled", "invalid", "plugin_not_running"] {
        assert_eq!(decoded["states"][state]["state"], state);
        assert_eq!(
            decoded["states"][state]["non_ui_capabilities"][0],
            "exemplar.notes.v1"
        );
    }
    assert_eq!(
        decoded["states"]["invalid"]["unavailable_reason"],
        "web UI bundle root `bundle` is missing from the installed package"
    );
}

//! Fixtures for tests that drive the real plugin upgrade gate through its
//! production entry points: a plugin store named by `MESH_LLM_PLUGIN_DIR`
//! and a catalog nothing listens on, so the gate decides from what the
//! installed records retained and never reaches the network. Callers are
//! `#[serial_test::serial]`.

use std::path::Path;

use mesh_llm_plugin_manager::{
    HostCompatibility, InstalledPluginMetadata, PluginStore, PluginTarget,
};

pub(crate) fn with_plugin_store<T>(test: impl FnOnce(&Path) -> T) -> T {
    let dir = tempfile::tempdir().expect("temp plugin store");
    let path = dir.path().to_str().expect("UTF-8 temp path").to_string();
    mesh_llm_config::with_env_override_for_test("MESH_LLM_PLUGIN_DIR", &path, || {
        mesh_llm_config::with_env_override_for_test(
            "MESH_LLM_PLUGIN_CATALOG_URL",
            "http://127.0.0.1:1/plugins.jsonl",
            || test(dir.path()),
        )
    })
}

/// Install a `lexe` record declaring the given host bounds.
pub(crate) fn write_plugin(store_root: &Path, min: &str, max: Option<&str>) {
    let store = PluginStore::new(store_root);
    store
        .save(&InstalledPluginMetadata {
            name: "lexe".into(),
            source_repository: "https://github.com/mesh-llm/lexe".into(),
            installed_version: "v1.0.0".into(),
            target_triple: PluginTarget::current()
                .expect("supported test platform")
                .triple()
                .into(),
            downloaded_asset_name: "lexe.tar.gz".into(),
            install_path: store_root.join("lexe"),
            enabled: true,
            default_managed: false,
            host_compatibility: Some(HostCompatibility {
                min: Some(min.into()),
                max: max.map(str::to_string),
            }),
            allowed_surfaces: None,
            manifest: None,
            last_protocol_version: Some(mesh_llm_plugin::PROTOCOL_VERSION),
            last_status: None,
            last_error: None,
            verified_sha256: None,
        })
        .expect("save plugin record");
}

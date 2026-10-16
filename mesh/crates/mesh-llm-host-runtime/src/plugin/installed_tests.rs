use super::super::config::{MeshConfig, PluginConfigEntry, resolve_plugins};
use super::super::{
    BLOBSTORE_PLUGIN_ID, MeshVisibility, PAYMENTS_PLUGIN_ID, PluginHostMode, PluginManager,
};
use super::*;
use mesh_llm_plugin_manager::InstalledPluginManifestMetadata;
use tokio::sync::mpsc;

/// Runs `test` with `MESH_LLM_PLUGIN_DIR` naming `store`, through the
/// registered serial-test helper; callers are `#[serial_test::serial]`.
fn with_plugin_dir<T>(store: &Path, test: impl FnOnce() -> T) -> T {
    let store = store.to_str().expect("plugin store path is UTF-8");
    mesh_llm_config::with_env_override_for_test("MESH_LLM_PLUGIN_DIR", store, test)
}

/// A default-managed `capsules` install whose executable exists, as
/// `mesh-llm plugins install-defaults` leaves it.
fn installed_capsules(store_root: &Path, enabled: bool) -> InstalledPluginMetadata {
    let metadata = InstalledPluginMetadata {
        name: "capsules".to_string(),
        source_repository: "https://github.com/Mesh-LLM/capsules".to_string(),
        installed_version: "v0.1.4".to_string(),
        target_triple: std::env::consts::ARCH.to_string(),
        downloaded_asset_name: "capsules.tar.gz".to_string(),
        install_path: store_root.join("installs").join("capsules"),
        enabled,
        default_managed: true,
        manifest: Some(InstalledPluginManifestMetadata {
            openai_exchange_hook: None,
            config_schema: None,
            web_ui: None,
        }),
        last_protocol_version: Some(1),
        last_status: Some("installed".to_string()),
        last_error: None,
    };
    std::fs::create_dir_all(&metadata.install_path).expect("install dir");
    std::fs::write(metadata.executable_path(), b"").expect("plugin executable");
    PluginStore::new(store_root)
        .save(&metadata)
        .expect("save plugin metadata");
    metadata
}

/// A settings-only `[[plugin]] name = "capsules"` entry: no command, no url.
fn capsules_entry(command: Option<&str>) -> PluginConfigEntry {
    PluginConfigEntry {
        name: "capsules".into(),
        enabled: None,
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        allow_peer_blocks: None,
        openai_exchange_grant: None,
        command: command.map(str::to_string),
        args: Vec::new(),
        url: None,
        settings: Default::default(),
        startup: Default::default(),
    }
}

fn builtin_off(name: &str) -> PluginConfigEntry {
    PluginConfigEntry {
        name: name.into(),
        enabled: Some(false),
        ..capsules_entry(None)
    }
}

#[test]
#[serial_test::serial]
fn settings_only_entry_stays_off_after_plugins_disable() {
    let store = tempfile::TempDir::new().expect("plugin store");
    with_plugin_dir(store.path(), || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(stays_off_after_plugins_disable(store.path()))
    });
}

async fn stays_off_after_plugins_disable(store: &Path) {
    let metadata = installed_capsules(store, false);
    let config = MeshConfig {
        plugins: vec![
            capsules_entry(None),
            builtin_off(BLOBSTORE_PLUGIN_ID),
            builtin_off(PAYMENTS_PLUGIN_ID),
        ],
        ..MeshConfig::default()
    };

    let resolved = resolve_plugins(
        &config,
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        },
    )
    .expect("plugins resolve");

    assert!(
        resolved
            .externals
            .iter()
            .all(|spec| spec.name != "capsules"),
        "a store-disabled plugin must not be launched"
    );
    let summary = resolved
        .inactive
        .iter()
        .find(|summary| summary.name == "capsules")
        .expect("capsules is listed as inactive");
    assert_eq!(summary.status, "disabled");
    assert!(!summary.enabled);
    assert_eq!(summary.version.as_deref(), Some("v0.1.4"));

    let (mesh_tx, _mesh_rx) = mpsc::channel(1);
    let manager = PluginManager::start(
        &resolved,
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        },
        mesh_tx,
    )
    .await
    .expect("plugin manager starts");
    let listed = manager.list().await;
    manager.shutdown().await;
    let summary = listed
        .iter()
        .find(|summary| summary.name == "capsules")
        .expect("startup lists capsules");
    assert_eq!(summary.status, "disabled");
    assert_eq!(summary.pid, None);
    assert_eq!(
        summary.command.as_deref(),
        Some(metadata.executable_path().display().to_string().as_str())
    );
}

#[test]
#[serial_test::serial]
fn explicit_command_still_runs_when_the_store_copy_is_disabled() {
    let store = tempfile::TempDir::new().expect("plugin store");
    installed_capsules(store.path(), false);

    let resolved = with_plugin_dir(store.path(), || {
        configured_external_plugin_spec(&capsules_entry(Some("/opt/capsules/bin")))
    })
    .expect("entry resolves");

    let ConfiguredExternalPlugin::Active(spec) = resolved else {
        panic!("an entry with its own command ignores the store's enabled flag");
    };
    assert_eq!(spec.command, "/opt/capsules/bin");
    assert!(!spec.startup.optional);
    assert!(spec.installed_metadata.is_none());
}

#[test]
#[serial_test::serial]
fn settings_only_entry_runs_the_enabled_store_copy() {
    let store = tempfile::TempDir::new().expect("plugin store");
    let metadata = installed_capsules(store.path(), true);

    let resolved = with_plugin_dir(store.path(), || {
        configured_external_plugin_spec(&capsules_entry(None))
    })
    .expect("entry resolves");

    let ConfiguredExternalPlugin::Active(spec) = resolved else {
        panic!("an enabled store copy must start");
    };
    assert_eq!(
        spec.command,
        metadata.executable_path().display().to_string()
    );
    assert!(spec.startup.optional);
    assert!(
        spec.installed_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.default_managed)
    );
}

#[test]
#[serial_test::serial]
fn explicit_installed_command_keeps_its_packaged_web_ui() {
    use mesh_llm_plugin_manager::store::{
        InstalledPluginWebUiMetadata, InstalledPluginWebUiValidation,
        InstalledPluginWebUiValidationStatus,
    };

    let store = tempfile::TempDir::new().expect("plugin store");
    for default_managed in [false, true] {
        let mut metadata = installed_capsules(store.path(), true);
        metadata.default_managed = default_managed;
        let asset_root = metadata.install_path.join("web");
        std::fs::create_dir_all(&asset_root).unwrap();
        metadata.manifest.as_mut().unwrap().web_ui = Some(InstalledPluginWebUiMetadata {
            pages: Vec::new(),
            config_sections: Vec::new(),
            contributions: Vec::new(),
            bundles: Vec::new(),
            asset_root: Some("web".into()),
            validation: InstalledPluginWebUiValidation {
                status: InstalledPluginWebUiValidationStatus::Valid,
                reason: None,
            },
        });
        PluginStore::new(store.path()).save(&metadata).unwrap();
        let command = metadata.executable_path().display().to_string();
        let resolved = with_plugin_dir(store.path(), || {
            configured_external_plugin_spec(&capsules_entry(Some(&command)))
        })
        .expect("entry resolves");
        let ConfiguredExternalPlugin::Active(spec) = resolved else {
            panic!("the explicit installed executable must start");
        };

        let web_ui = derive_plugin_web_ui_state(super::super::PluginWebUiStateInput {
            plugin_name: &spec.name,
            live_manifest: None,
            installed_metadata: spec.installed_metadata.as_ref(),
            web_ui_enabled: spec.web_ui_enabled,
            web_ui_primary_tab: false,
            runtime_available: true,
            runtime_unavailable_reason: None,
        });
        assert!(
            web_ui.available,
            "the matching package must retain its web UI"
        );
        assert_eq!(
            web_ui.asset_base_url.as_deref(),
            Some("/api/plugins/capsules/web-ui/assets/")
        );
        assert!(!spec.startup.optional, "explicit commands remain required");
        assert_eq!(spec.installed_metadata, Some(metadata));
    }
}

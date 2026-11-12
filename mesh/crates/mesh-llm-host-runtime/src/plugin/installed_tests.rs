use super::super::config::{MeshConfig, PluginConfigEntry, resolve_plugins};
use super::super::{BLOBSTORE_PLUGIN_ID, MeshVisibility, PluginHostMode, PluginManager};
use super::*;
use mesh_llm_plugin_manager::InstalledPluginManifestMetadata;
use tokio::sync::mpsc;

/// Runs `test` with `MESH_LLM_PLUGIN_DIR` naming `store`, through the
/// registered serial-test helper; callers are `#[serial_test::serial]`.
fn with_plugin_dir<T>(store: &Path, test: impl FnOnce() -> T) -> T {
    let store = store.to_str().expect("plugin store path is UTF-8");
    mesh_llm_config::with_env_override_for_test("MESH_LLM_PLUGIN_DIR", store, test)
}

/// A managed `capsules` install whose executable exists, as
/// `mesh-llm plugins install` leaves it.
fn installed_capsules(store_root: &Path, enabled: bool) -> InstalledPluginMetadata {
    let metadata = InstalledPluginMetadata {
        name: "capsules".to_string(),
        source_repository: "https://github.com/Mesh-LLM/capsules".to_string(),
        installed_version: "v0.1.4".to_string(),
        target_triple: std::env::consts::ARCH.to_string(),
        downloaded_asset_name: "capsules.tar.gz".to_string(),
        install_path: store_root.join("installs").join("capsules"),
        enabled,
        default_managed: false,
        host_compatibility: None,
        allowed_surfaces: None,
        manifest: Some(InstalledPluginManifestMetadata {
            openai_exchange_hook: None,
            config_schema: None,
            web_ui: None,
            operation_bindings: Vec::new(),
            peer_blocks: false,
        }),
        last_protocol_version: Some(mesh_llm_plugin::PROTOCOL_VERSION),
        last_status: Some("installed".to_string()),
        last_error: None,
        verified_sha256: None,
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
        operation_bindings: Vec::new(),
        name: "capsules".into(),
        enabled: None,
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        allow_peer_blocks: None,
        allow_inference_observation: None,
        openai_exchange_grant: None,
        command: command.map(str::to_string),
        args: Vec::new(),
        url: None,
        models: None,
        settings: Default::default(),
        startup: Default::default(),
    }
}

fn builtin_off(name: &str) -> PluginConfigEntry {
    PluginConfigEntry {
        operation_bindings: Vec::new(),
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
        plugins: vec![capsules_entry(None), builtin_off(BLOBSTORE_PLUGIN_ID)],
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
    // A bare entry runs the store's copy, which the operator did not choose
    // by path, so it starts optional: a broken plugin install never stops
    // the node.
    assert!(spec.startup.optional);
    assert!(
        spec.installed_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.allowed_surfaces.is_none()),
        "a managed install carries no legacy surface record"
    );
}

#[test]
#[serial_test::serial]
fn configured_store_copies_cannot_bypass_host_compatibility() {
    let root = tempfile::TempDir::new().unwrap();
    for protocol_mismatch in [false, true] {
        let mut metadata = installed_capsules(root.path(), true);
        if protocol_mismatch {
            metadata.last_protocol_version = Some(mesh_llm_plugin::PROTOCOL_VERSION + 1);
        } else {
            metadata.host_compatibility =
                Some(mesh_llm_plugin_manager::compatibility::HostCompatibility {
                    min: None,
                    max: Some("0.0.0".into()),
                });
        }
        PluginStore::new(root.path()).save(&metadata).unwrap();
        let command = metadata.executable_path().display().to_string();
        for configured_command in [None, Some(command.as_str())] {
            let resolved = with_plugin_dir(root.path(), || {
                configured_external_plugin_spec(&capsules_entry(configured_command))
            })
            .unwrap();
            let ConfiguredExternalPlugin::Inactive(summary) = resolved else {
                panic!("configured store copies must receive the same compatibility check");
            };
            assert!(summary.error.unwrap().contains("incompatible with MeshLLM"));
        }
        let unrelated = with_plugin_dir(root.path(), || {
            configured_external_plugin_spec(&capsules_entry(Some("operator-command")))
        })
        .unwrap();
        assert!(matches!(unrelated, ConfiguredExternalPlugin::Active(_)));
    }
}

#[test]
#[serial_test::serial]
fn configured_legacy_grants_apply_before_persistent_migration() {
    let root = tempfile::TempDir::new().unwrap();
    for allowed in [None, Some(Vec::new()), Some(vec!["web_ui".to_string()])] {
        let mut old = installed_capsules(root.path(), true);
        old.default_managed = true;
        old.allowed_surfaces = allowed.clone();
        let store = PluginStore::new(root.path());
        store.save(&old).unwrap();

        let resolved = with_plugin_dir(root.path(), || {
            configured_external_plugin_spec(&capsules_entry(None))
        })
        .unwrap();
        let ConfiguredExternalPlugin::Active(spec) = resolved else {
            panic!("the installed plugin must resolve");
        };
        assert_eq!(
            spec.installed_metadata.unwrap().allowed_surfaces,
            allowed.or_else(|| Some(mesh_llm_plugin_manager::migrated_allows_keys("capsules")))
        );
        assert!(
            store.load("capsules").unwrap().default_managed,
            "the first configured spec must not depend on a persistent migration"
        );
    }
}

#[test]
#[serial_test::serial]
fn explicit_installed_command_keeps_its_packaged_web_ui() {
    use mesh_llm_plugin_manager::store::{
        InstalledPluginWebUiMetadata, InstalledPluginWebUiValidation,
        InstalledPluginWebUiValidationStatus,
    };

    let store = tempfile::TempDir::new().expect("plugin store");
    for allowed_surfaces in [None, Some(vec!["web_ui".to_string()])] {
        let mut metadata = installed_capsules(store.path(), true);
        metadata.allowed_surfaces = allowed_surfaces;
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

#[test]
#[serial_test::serial]
fn explicit_installed_command_still_runs_when_the_store_copy_is_disabled() {
    let store = tempfile::TempDir::new().expect("plugin store");
    let metadata = installed_capsules(store.path(), false);
    let command = metadata.executable_path().display().to_string();

    let resolved = with_plugin_dir(store.path(), || {
        configured_external_plugin_spec(&capsules_entry(Some(&command)))
    })
    .expect("entry resolves");

    let ConfiguredExternalPlugin::Active(spec) = resolved else {
        panic!("an entry that names the store's executable ignores the store's enabled flag");
    };
    assert_eq!(spec.command, command);
    assert!(!spec.startup.optional, "explicit commands remain required");
    assert_eq!(spec.installed_metadata, Some(metadata));
}

/// `[plugin.settings]` are validated and shown in the console; the launch spec
/// is what carries them to the process.
#[test]
#[serial_test::serial]
fn configured_plugin_settings_reach_the_launch_spec() {
    let store = tempfile::TempDir::new().expect("plugin store");
    let mut entry = capsules_entry(Some("/opt/capsules/bin"));
    entry.settings = [(
        "greeting".to_string(),
        toml::Value::String("marker-7f3a".to_string()),
    )]
    .into_iter()
    .collect();

    let resolved = with_plugin_dir(store.path(), || configured_external_plugin_spec(&entry))
        .expect("entry resolves");

    let ConfiguredExternalPlugin::Active(spec) = resolved else {
        panic!("an enabled entry with a command is active");
    };
    assert_eq!(
        spec.settings.get("greeting"),
        Some(&serde_json::json!("marker-7f3a")),
        "[plugin.settings] were dropped from the spec the plugin is started from"
    );
}

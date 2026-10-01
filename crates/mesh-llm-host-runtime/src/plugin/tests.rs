use super::config::{MeshConfig, PluginConfigEntry};
use super::*;

fn private_host_mode() -> PluginHostMode {
    PluginHostMode {
        mesh_visibility: MeshVisibility::Private,
    }
}

fn web_ui_manifest() -> proto::PluginWebUiManifest {
    proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            icon: Some("icons/home.svg".into()),
            route: "index.html".into(),
            bundle_id: "main".into(),
            entry_script: "assets/app.js".into(),
            placement: proto::PluginWebUiPagePlacement::Auxiliary as i32,
        }],
        config_sections: vec![proto::PluginWebUiConfigSectionManifest {
            id: "settings".into(),
            title: "Settings".into(),
            entry_script: "assets/settings.js".into(),
            parent_tab: Some("integrations".into()),
            bundle_id: "main".into(),
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "web".into(),
        }],
    }
}

#[test]
fn plugin_manifest_overview_includes_web_ui_declaration() {
    let manifest = proto::PluginManifest {
        web_ui: Some(web_ui_manifest()),
        ..proto::PluginManifest::default()
    };

    let overview = plugin_manifest_overview(&manifest);

    let web_ui = overview.web_ui.expect("web UI overview should be present");
    assert_eq!(web_ui.pages[0].id, "home");
    assert_eq!(web_ui.config_sections[0].id, "settings");
}

#[test]
fn resolves_default_builtin_plugins() {
    let resolved = resolve_plugins(&MeshConfig::default(), private_host_mode()).unwrap();
    assert_eq!(resolved.externals.len(), 1);
    assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
    assert!(resolved.externals[0].startup.optional);
    assert!(resolved.inactive.is_empty());
}

/// A stanza for the removed built-in Lexe wallet.
fn wallet_entry(enabled: Option<bool>) -> PluginConfigEntry {
    PluginConfigEntry {
        name: RETIRED_WALLET_LEXE_PLUGIN_ID.into(),
        enabled,
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        command: None,
        args: Vec::new(),
        url: None,
        settings: Default::default(),
        startup: Default::default(),
    }
}

#[test]
fn builtin_payments_is_served_in_process_and_can_be_switched_off() {
    super::config::TEST_PAYMENTS_COMPILED_IN.with(|slot| *slot.borrow_mut() = Some(true));
    let resolved = resolve_plugins(&MeshConfig::default(), private_host_mode()).unwrap();
    let payments = resolved
        .externals
        .iter()
        .find(|spec| spec.name == PAYMENTS_PLUGIN_ID)
        .expect("payments builtin registered");
    assert!(
        payments.command.is_empty(),
        "served in-process, not re-exec'd"
    );
    assert!(payments.startup.optional);

    let mut entry = wallet_entry(Some(false));
    entry.name = PAYMENTS_PLUGIN_ID.into();
    let config = MeshConfig {
        plugins: vec![entry],
        ..MeshConfig::default()
    };
    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    super::config::TEST_PAYMENTS_COMPILED_IN.with(|slot| *slot.borrow_mut() = None);
    assert!(
        resolved
            .externals
            .iter()
            .all(|spec| spec.name != PAYMENTS_PLUGIN_ID)
    );
}

#[test]
fn builtin_plugin_arguments_follow_the_plugin_flag() {
    let spec = super::config::builtin_plugin_spec(
        BLOBSTORE_PLUGIN_ID,
        &["--root".into(), "/srv/blobs".into()],
    )
    .unwrap();
    assert_eq!(
        spec.args,
        [
            "--log-format",
            "json",
            "--plugin",
            BLOBSTORE_PLUGIN_ID,
            "--plugin-arg=--root",
            "--plugin-arg=/srv/blobs",
        ]
    );
}

#[tokio::test]
async fn builtin_plugin_without_arguments_refuses_them() {
    let error = run_plugin_process(BLOBSTORE_PLUGIN_ID.into(), vec!["--verbose".into()])
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("takes no arguments"), "{error}");
}

#[test]
fn retired_builtin_wallet_stanza_is_ignored() {
    // The built-in Lexe wallet was removed; an old stanza must not break
    // startup, whatever it sets, and registers nothing.
    let mut with_command = wallet_entry(Some(true));
    with_command.command = Some("/opt/wallets/my-wallet".into());
    for entry in [
        wallet_entry(Some(true)),
        wallet_entry(Some(false)),
        wallet_entry(None),
        with_command,
    ] {
        let config = MeshConfig {
            plugins: vec![entry],
            defaults: None,
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode())
            .expect("a retired stanza must not break startup");
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }
}

#[test]
fn external_wallet_plugin_is_an_ordinary_plugin() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "lexe-wallet".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: Some("/opt/wallets/lexe-wallet".into()),
            args: Vec::new(),
            url: None,
            settings: Default::default(),
            startup: Default::default(),
        }],
        defaults: None,
        ..MeshConfig::default()
    };
    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    let names: Vec<_> = resolved.externals.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["lexe-wallet", BLOBSTORE_PLUGIN_ID]);
    assert!(!resolved.externals[0].startup.optional);
}

#[test]
fn external_plugin_can_be_configured() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "demo".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: Some("mesh-llm-plugin-demo".into()),
            args: vec!["--stdio".into()],
            url: None,
            settings: Default::default(),
            startup: Default::default(),
        }],
        defaults: None,
        ..MeshConfig::default()
    };
    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    assert_eq!(resolved.externals.len(), 2);
    assert_eq!(resolved.externals[0].name, "demo");
    assert_eq!(resolved.externals[0].command, "mesh-llm-plugin-demo");
    assert_eq!(resolved.externals[0].args, ["--stdio"]);
    assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
    assert!(resolved.inactive.is_empty());
}

#[test]
fn failed_plugin_summary_redacts_urls_before_serialization() {
    let spec = ExternalPluginSpec {
        name: "remote".into(),
        command: "mesh-llm-plugin-remote".into(),
        args: Vec::new(),
        url: Some("https://plugin.example.test/v1".into()),
        env: BTreeMap::new(),
        startup: PluginStartupOptions {
            optional: true,
            ..PluginStartupOptions::default()
        },
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        installed_metadata: None,
    };
    let error = anyhow::anyhow!(
        "connection to tcp://user:secret@127.0.0.1:19091/control?token=private failed"
    );

    let summary = PluginManager::plugin_load_failure_summary(&spec, &error);
    let serialized = serde_json::to_string(&summary).expect("summary serializes");

    assert!(!serialized.contains("user"));
    assert!(!serialized.contains("secret"));
    assert!(!serialized.contains("private"));
}

#[test]
fn blobstore_can_be_disabled() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: BLOBSTORE_PLUGIN_ID.into(),
            enabled: Some(false),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: None,
            args: Vec::new(),
            url: None,
            settings: Default::default(),
            startup: Default::default(),
        }],
        defaults: None,
        ..MeshConfig::default()
    };
    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    assert!(resolved.externals.is_empty());
    assert!(resolved.inactive.is_empty());
}

#[test]
fn external_plugin_can_be_enabled_with_url() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "endpoint-plugin".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: Some("endpoint-plugin".into()),
            args: Vec::new(),
            url: Some("http://gpu-box:8000/v1".into()),
            settings: Default::default(),
            startup: Default::default(),
        }],
        defaults: None,
        ..MeshConfig::default()
    };
    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    assert_eq!(resolved.externals.len(), 2);
    assert_eq!(resolved.externals[0].name, "endpoint-plugin");
    assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
    let spec = &resolved.externals[0];
    assert_eq!(spec.command, "endpoint-plugin");
    assert!(spec.args.is_empty());
    assert_eq!(spec.url.as_deref(), Some("http://gpu-box:8000/v1"));
}

#[test]
fn external_plugin_rejects_url_that_is_empty_after_normalization() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "endpoint-plugin".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: Some("endpoint-plugin".into()),
            args: Vec::new(),
            url: Some("\u{2003}\t\n".into()),
            settings: Default::default(),
            startup: Default::default(),
        }],
        ..MeshConfig::default()
    };

    let error = resolve_plugins(&config, private_host_mode())
        .expect_err("normalized plugin URL must not be empty");
    assert!(error.to_string().contains("plugin URL must not be empty"));
}

#[test]
fn remote_plugin_control_url_is_rejected_without_authentication() {
    let raw_url = "\u{2003}tcp://user:secret@127.0.0.1:19091/control?token=private\u{2003}";
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "remote-plugin".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: None,
            args: Vec::new(),
            url: Some(raw_url.into()),
            settings: Default::default(),
            startup: Default::default(),
        }],
        defaults: None,
        ..MeshConfig::default()
    };

    let error = resolve_plugins(&config, private_host_mode())
        .expect_err("unauthenticated remote plugin control must be rejected");
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("authenticated capability handshake"));
    assert!(!diagnostic.contains("user"));
    assert!(!diagnostic.contains("secret"));
    assert!(!diagnostic.contains("private"));
    assert!(!diagnostic.contains(raw_url));
}

#[test]
fn external_plugin_can_be_enabled_with_command_args() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "endpoint-plugin".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: Some("/opt/plugins/endpoint-plugin".into()),
            args: vec!["--verbose".into()],
            url: None,
            settings: Default::default(),
            startup: Default::default(),
        }],
        defaults: None,
        ..MeshConfig::default()
    };
    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    assert_eq!(resolved.externals.len(), 2);
    assert_eq!(resolved.externals[0].name, "endpoint-plugin");
    assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
    let spec = &resolved.externals[0];
    assert_eq!(spec.command, "/opt/plugins/endpoint-plugin");
    assert_eq!(spec.args, vec!["--verbose"]);
}

#[test]
fn external_plugin_ignores_disabled_entry_without_install() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "endpoint-plugin".into(),
            enabled: Some(false),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: None,
            args: Vec::new(),
            url: Some("http://gpu-box:8000/v1".into()),
            settings: Default::default(),
            startup: Default::default(),
        }],
        defaults: None,
        ..MeshConfig::default()
    };
    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    assert_eq!(resolved.externals.len(), 1);
    assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
}

#[test]
fn default_builtins_are_resolved_on_public_meshes() {
    let resolved = resolve_plugins(
        &MeshConfig::default(),
        PluginHostMode {
            mesh_visibility: MeshVisibility::Public,
        },
    )
    .unwrap();
    assert_eq!(resolved.externals.len(), 1);
    assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
    assert!(resolved.inactive.is_empty());
}

#[test]
fn resolves_external_plugin() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "demo".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            command: Some("/tmp/demo".into()),
            args: vec!["--flag".into()],
            url: None,
            settings: Default::default(),
            startup: Default::default(),
        }],
        defaults: None,
        ..MeshConfig::default()
    };
    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    assert_eq!(resolved.externals.len(), 2);
    assert_eq!(resolved.externals[0].name, "demo");
    assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
    assert!(resolved.inactive.is_empty());
}

#[tokio::test]
async fn plugin_load_failure_keeps_declared_web_ui_metadata() {
    let specs = ResolvedPlugins {
        externals: vec![ExternalPluginSpec {
            name: "demo".into(),
            command: "mesh-llm-definitely-missing-plugin-binary".into(),
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
            startup: PluginStartupOptions {
                optional: true,
                ..PluginStartupOptions::default()
            },
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            installed_metadata: Some(installed_metadata_with_web_ui(
                InstalledPluginWebUiValidationStatus::Valid,
                Some("web"),
            )),
        }],
        inactive: Vec::new(),
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);

    let manager = PluginManager::start(&specs, private_host_mode(), mesh_tx)
        .await
        .expect("broken plugin should not stop manager startup");
    let summaries = manager.list().await;
    manager.shutdown().await;

    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].name, "demo");
    assert_eq!(summaries[0].status, "error");
    assert_eq!(
        summaries[0].web_ui.state,
        PluginWebUiStateKind::PluginNotRunning
    );
    assert_eq!(summaries[0].web_ui.pages.len(), 1);
    assert_eq!(summaries[0].web_ui.config_sections.len(), 1);
}

#[test]
fn instance_ids_include_pid_and_random_suffix() {
    let instance_id = make_instance_id();
    let prefix = format!("p{}-", std::process::id());
    assert!(instance_id.starts_with(&prefix));
    assert_eq!(instance_id.len(), prefix.len() + 8);
    assert!(
        instance_id[prefix.len()..]
            .chars()
            .all(|ch| ch.is_ascii_hexdigit())
    );
}

#[cfg(unix)]
#[test]
fn unix_socket_path_is_namespaced_by_instance_id() {
    let path = unix_socket_path("p1234-deadbeef", "Pipes").unwrap();
    assert_eq!(
        path.file_name().and_then(|value| value.to_str()),
        Some("p1234-deadbeef-Pipes.sock")
    );
}

#[cfg(windows)]
#[test]
fn windows_pipe_name_is_namespaced_by_instance_id() {
    assert_eq!(
        windows_pipe_name("p1234-deadbeef", "Pipes"),
        r"\\.\pipe\mesh-llm-p1234-deadbeef-Pipes"
    );
}

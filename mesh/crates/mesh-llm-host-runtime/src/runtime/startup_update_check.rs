//! The release check that finishes before startup continues.

use std::future::Future;

use mesh_llm_events::{OutputEvent, emit_event};

use super::RuntimeOptions;
use crate::plugin::MeshConfig;
use crate::system::autoupdate::{self, UpdateNotice};

/// Shows a notice when a newer release is out.
pub(super) async fn announce_newer_release(
    options: &RuntimeOptions,
    checked_updates: bool,
    config: &MeshConfig,
) {
    let skip = skip_release_check(options, checked_updates, config);
    let notice =
        newer_release_notice(skip, || autoupdate::check_for_update(crate::BUILD_VERSION)).await;
    if let Some(notice) = notice {
        let _ = emit_event(OutputEvent::Info {
            message: notice.message,
            context: None,
        });
    }
}

/// Skips when `--auto-update` already checked, the command is `update` or
/// prints machine output, or `updates.check` turns the check off.
fn skip_release_check(
    options: &RuntimeOptions,
    checked_updates: bool,
    config: &MeshConfig,
) -> bool {
    checked_updates
        || options.command_is_update
        || options.command_uses_machine_output
        || !config.updates.check_enabled()
}

async fn newer_release_notice<F, Fut>(skip: bool, check: F) -> Option<UpdateNotice>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Option<UpdateNotice>>,
{
    if skip { None } else { check().await }
}

/// The automatic update at node start. It owes installed plugins the same
/// check and repairs `mesh-llm update` runs before it replaces the host, so
/// the only way to reach `maybe_auto_update` here is with that gate.
pub(super) async fn auto_update_with_plugin_gate(
    options: autoupdate::AutoUpdateOptions,
) -> anyhow::Result<bool> {
    autoupdate::maybe_auto_update(options, auto_update_plugin_gate).await
}

/// The plugin compatibility check and repairs `mesh-llm update` runs before
/// it replaces the host, run here before an automatic update does. A refusal
/// keeps the running host, so the node starts on the version it has.
async fn auto_update_plugin_gate(target_version: &str) -> anyhow::Result<()> {
    mesh_llm_plugin_manager::upgrade_gate::check_and_repair_installed_plugins(
        target_version,
        mesh_llm_plugin::PROTOCOL_VERSION,
        &mut |line| {
            let _ = mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::AutoUpdate {
                message: line.to_string(),
                version: None,
            });
        },
        &mut |event: mesh_llm_plugin_manager::PluginProgressEvent| {
            if let mesh_llm_plugin_manager::PluginProgressEvent::Warning { message } = event {
                let _ = mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::AutoUpdate {
                    message: format!("⚠️  {message}"),
                    version: None,
                });
            }
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_check_false_skips_the_release_check() {
        let options = RuntimeOptions::default();
        let mut config = MeshConfig::default();
        assert!(!skip_release_check(&options, false, &config));

        config.updates.check = Some(false);
        assert!(skip_release_check(&options, false, &config));

        config.updates.check_override = Some(true);
        assert!(!skip_release_check(&options, false, &config));
    }

    #[test]
    fn update_commands_and_machine_output_skip_the_release_check() {
        let config = MeshConfig::default();
        assert!(skip_release_check(
            &RuntimeOptions::default(),
            true,
            &config
        ));
        let options = RuntimeOptions {
            command_is_update: true,
            ..RuntimeOptions::default()
        };
        assert!(skip_release_check(&options, false, &config));
        let options = RuntimeOptions {
            command_uses_machine_output: true,
            ..RuntimeOptions::default()
        };
        assert!(skip_release_check(&options, false, &config));
    }

    #[tokio::test]
    async fn a_skipped_check_sends_no_request() {
        let notice = newer_release_notice(true, || async {
            panic!("the release check must not run");
        })
        .await;
        assert!(notice.is_none());
    }

    #[tokio::test]
    async fn a_check_that_runs_returns_its_notice() {
        let notice = newer_release_notice(false, || async {
            Some(UpdateNotice {
                message: "new".to_string(),
            })
        })
        .await;
        assert_eq!(notice.map(|notice| notice.message).as_deref(), Some("new"));
    }

    #[test]
    #[serial_test::serial]
    fn the_auto_update_gate_refuses_a_plugin_the_target_host_cannot_run() {
        let store = tempfile::tempdir().unwrap();
        let path = store.path().to_str().unwrap().to_string();
        let gate = |plugin_max: Option<&str>| {
            let plugins = mesh_llm_plugin_manager::PluginStore::new(store.path());
            for existing in plugins.list().unwrap() {
                std::fs::remove_dir_all(plugins.root().join(&existing.name)).ok();
            }
            plugins
                .save(&mesh_llm_plugin_manager::InstalledPluginMetadata {
                    name: "lexe".into(),
                    source_repository: "https://github.com/mesh-llm/lexe".into(),
                    installed_version: "v1.0.0".into(),
                    target_triple: mesh_llm_plugin_manager::PluginTarget::current()
                        .unwrap()
                        .triple()
                        .into(),
                    downloaded_asset_name: "lexe.tar.gz".into(),
                    install_path: store.path().join("lexe"),
                    enabled: true,
                    default_managed: false,
                    host_compatibility: Some(mesh_llm_plugin_manager::HostCompatibility {
                        min: Some("0.1.0".into()),
                        max: plugin_max.map(str::to_string),
                    }),
                    allowed_surfaces: None,
                    manifest: None,
                    last_protocol_version: Some(mesh_llm_plugin::PROTOCOL_VERSION),
                    last_status: None,
                    last_error: None,
                    verified_sha256: None,
                })
                .unwrap();
            // Nothing listens on the catalog, so the gate decides from what
            // the installed record retained and never reaches the network.
            mesh_llm_config::with_env_override_for_test("MESH_LLM_PLUGIN_DIR", &path, || {
                mesh_llm_config::with_env_override_for_test(
                    "MESH_LLM_PLUGIN_CATALOG_URL",
                    "http://127.0.0.1:1/plugins.jsonl",
                    || {
                        tokio::runtime::Runtime::new()
                            .unwrap()
                            .block_on(auto_update_plugin_gate("0.81.0"))
                    },
                )
            })
        };
        assert!(
            gate(Some("0.5.0")).is_err(),
            "incompatible plugin not refused"
        );
        assert!(gate(None).is_ok(), "compatible plugin was refused");
    }
}

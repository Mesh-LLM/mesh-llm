//! Offer the default plugin list once on a fresh node, before plugins are
//! resolved, so a new install starts with them. See
//! `mesh_llm_plugin_manager::defaults`. Best effort and bounded: a node always
//! starts, with or without its defaults. `--no-default-plugins` or
//! `MESH_LLM_NO_DEFAULT_PLUGINS=1` skips it.

use std::collections::BTreeSet;
use std::time::Duration;

use anyhow::Result;
use mesh_llm_plugin_manager::defaults::{
    DEFAULT_PLUGINS, DefaultPlugin, DefaultPluginOutcome, DefaultPluginsRun,
    default_plugins_opted_out, install_default_plugins,
};
use mesh_llm_plugin_manager::{PluginInstallOptions, PluginProgressEvent};

use super::RuntimeOptions;
use crate::plugin;

/// Bounds how long a node's start can wait on a first-run install (an
/// offline node waits this long on each start until the defaults are offered).
const FIRST_RUN_INSTALL_TIMEOUT: Duration = Duration::from_secs(30);

/// Offer the defaults, then resolve plugins from `config` as before.
pub(super) async fn resolve_after_defaults(
    config: &plugin::MeshConfig,
    options: &RuntimeOptions,
) -> Result<plugin::ResolvedPlugins> {
    if offers_defaults(
        DEFAULT_PLUGINS,
        options.no_default_plugins,
        default_plugins_opted_out(),
    ) {
        offer_default_plugins(config).await;
    }
    super::run_auto::resolve_plugins_from_config(config, options)
}

/// Whether this start offers the default list: not when the list is empty,
/// and not when the operator opted out by flag or environment.
fn offers_defaults(defaults: &[DefaultPlugin], flag_opt_out: bool, env_opt_out: bool) -> bool {
    !defaults.is_empty() && !flag_opt_out && !env_opt_out
}

async fn offer_default_plugins(config: &plugin::MeshConfig) {
    let Some(options) = install_options() else {
        return;
    };
    let configured: BTreeSet<String> = config
        .plugins
        .iter()
        .map(|entry| entry.name.clone())
        .collect();
    let mut progress = |_event: PluginProgressEvent| {};
    let install = install_default_plugins(DEFAULT_PLUGINS, &configured, &options, &mut progress);
    match tokio::time::timeout(FIRST_RUN_INSTALL_TIMEOUT, install).await {
        Ok(run) => log_run(run),
        Err(_) => {
            warn("Default plugins: first-run install timed out; tried again next start".into())
        }
    }
}

fn install_options() -> Option<PluginInstallOptions> {
    PluginInstallOptions::from_env()
        .inspect_err(|error| {
            warn(format!(
                "Default plugins skipped: no plugin store ({error:#})"
            ))
        })
        .ok()
}

fn log_run(run: DefaultPluginsRun) {
    for (name, outcome) in run.outcomes {
        log_outcome(name, outcome);
    }
    if let Some(error) = run.record_error {
        warn(format!("Default plugins: {error}"));
    }
}

fn log_outcome(name: &str, outcome: DefaultPluginOutcome) {
    // Output events, not `tracing`: the runtime's default log filter drops
    // host-runtime info and warnings, and an operator should see what a
    // first run installed and how to undo it.
    let event = match outcome {
        DefaultPluginOutcome::Installed(installed) => mesh_llm_events::OutputEvent::Info {
            message: format!(
                "Installed default plugin {name} {}; remove it with `mesh-llm plugins delete \
                 {name}`, or start with --no-default-plugins to skip defaults",
                installed.metadata.installed_version
            ),
            context: Some("default_plugins".to_string()),
        },
        DefaultPluginOutcome::NotInstalled(reason) => {
            return warn(format!("Default plugin {name} not installed: {reason}"));
        }
        // Nothing happened, so nothing is logged: a node that has offered its
        // defaults is silent about them on every later start.
        DefaultPluginOutcome::AlreadyOffered | DefaultPluginOutcome::OperatorChose => return,
    };
    let _ = mesh_llm_events::emit_event(event);
}

fn warn(message: String) {
    let _ = mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::Warning {
        message,
        context: Some("default_plugins".to_string()),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_offered_unless_the_operator_opts_out() {
        let one = [DefaultPlugin {
            name: "notes",
            version: "1.0.0",
            sha256: &[],
        }];
        assert!(offers_defaults(&one, false, false), "on by default");
        assert!(!offers_defaults(&one, true, false), "--no-default-plugins");
        assert!(
            !offers_defaults(&one, false, true),
            "MESH_LLM_NO_DEFAULT_PLUGINS=1"
        );
        assert!(
            !offers_defaults(&[], false, false),
            "an empty list does nothing, not even a catalog lookup"
        );
    }

    #[test]
    fn the_default_list_carries_the_plugin() {
        assert!(
            DEFAULT_PLUGINS
                .iter()
                .any(|default| default.name == "capsule-emit-mesh")
        );
    }
}

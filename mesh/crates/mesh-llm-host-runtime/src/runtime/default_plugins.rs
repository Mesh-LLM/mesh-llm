//! Install the default plugins from this release's bundled copy when the node
//! starts, before plugins are resolved, so a node installed by a package, a
//! formula or `install.sh` starts with them, and an updated node with the new
//! release's copy. See `mesh_llm_plugin_manager::defaults`.
//!
//! Local only: the archive comes from the release (`plugins/`), is checked
//! against the reviewed pin, and is never downloaded; a missing copy is
//! reported, not fetched. A node always starts, with or without its defaults.
//! `MESH_LLM_NO_DEFAULT_PLUGINS=1` skips it; `mesh-llm plugins install-defaults
//! --off` (and the installers' and `mesh-llm update`'s `--no-default-plugins`)
//! turns the defaults off for good.

use std::collections::BTreeSet;

use mesh_llm_plugin_manager::defaults::{
    DEFAULT_PLUGINS, DefaultPluginOutcome, default_plugins_opted_out, provision_bundled_defaults,
};
use mesh_llm_plugin_manager::{PluginInstallOptions, PluginProgressEvent};

use crate::plugin;

/// Install the defaults from the bundled copy, then let plugins be resolved
/// from `config` as before.
pub(super) fn provision_bundled_defaults_at_start(config: &plugin::MeshConfig) {
    if DEFAULT_PLUGINS.is_empty() || default_plugins_opted_out() {
        return;
    }
    let options = match PluginInstallOptions::from_env() {
        Ok(options) => options,
        Err(error) => {
            return warn(format!(
                "Default plugins skipped: no plugin store ({error:#})"
            ));
        }
    };
    let mut progress = |_event: PluginProgressEvent| {};
    let outcomes = provision_bundled_defaults(
        DEFAULT_PLUGINS,
        &operator_run_plugins(config),
        &options,
        &mut progress,
    );
    for (name, outcome) in outcomes {
        log_outcome(name, outcome);
    }
}

/// The plugins the operator runs from config: an entry that says how to start
/// the plugin (`command` or `url`). A bare `[[plugin]]` entry holding only
/// settings, as the console writes, leaves a default default-managed.
fn operator_run_plugins(config: &plugin::MeshConfig) -> BTreeSet<String> {
    config
        .plugins
        .iter()
        .filter(|entry| entry.command.is_some() || entry.url.is_some())
        .map(|entry| entry.name.clone())
        .collect()
}

fn log_outcome(name: &str, outcome: DefaultPluginOutcome) {
    // Output events, not `tracing`: the runtime's default log filter drops
    // host-runtime info and warnings, and an operator should see what a
    // start installed and how to undo it.
    let event = match outcome {
        DefaultPluginOutcome::Installed(installed) => mesh_llm_events::OutputEvent::Info {
            message: format!(
                "Installed default plugin {name} {} from this release; turn it off with \
                 `mesh-llm plugins disable {name}`",
                installed.metadata.installed_version
            ),
            context: Some("default_plugins".to_string()),
        },
        DefaultPluginOutcome::NotInstalled(reason) => {
            return warn(format!("Default plugin {name} not installed: {reason}"));
        }
        DefaultPluginOutcome::NotBundled => {
            return warn(format!(
                "Default plugin {name} not installed: this install carries no bundled copy, \
                 and a default plugin is never downloaded"
            ));
        }
        // Nothing to do, or the operator's choice: a node is silent about it
        // on every start.
        DefaultPluginOutcome::AlreadyCurrent
        | DefaultPluginOutcome::OperatorManaged
        | DefaultPluginOutcome::Disabled
        | DefaultPluginOutcome::TurnedOff
        | DefaultPluginOutcome::UnsupportedPlatform => return,
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
    fn only_entries_that_start_a_plugin_take_it_out_of_default_management() {
        let config: plugin::MeshConfig = toml::from_str(
            r#"
[[plugin]]
name = "capsules"
[plugin.settings]
share_history_segments = "off"

[[plugin]]
name = "operator-run"
command = "/opt/plugins/operator-run"

[[plugin]]
name = "remote"
url = "unix:///run/remote.sock"
"#,
        )
        .unwrap();
        let configured = operator_run_plugins(&config);
        assert!(!configured.contains("capsules"));
        assert!(configured.contains("operator-run"));
        assert!(configured.contains("remote"));
    }
}

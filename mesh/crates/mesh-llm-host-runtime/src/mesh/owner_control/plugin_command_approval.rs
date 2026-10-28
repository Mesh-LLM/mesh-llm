//! Separate approval for executable plugin command changes.
//!
//! An apply-config can add or replace a plugin's `command` executable, which
//! the host then spawns as the service account. The apply request must
//! therefore name every such change in `plugin_command_approvals`; the target
//! rejects applies whose declared approvals do not cover the actual command
//! diff.

use std::collections::HashSet;

use crate::plugin::{MeshConfig, PluginConfigEntry};
use crate::proto::node::OwnerControlPluginCommandApproval;

/// Plugin commands that the incoming config adds or changes relative to the
/// current config. Removals and plugins without a command are not execution
/// risks and are not reported.
pub(crate) fn changed_plugin_commands(
    current: &[PluginConfigEntry],
    incoming: &MeshConfig,
) -> Vec<(String, String)> {
    let mut changed = Vec::new();
    for plugin in &incoming.plugins {
        let Some(command) = plugin.command.as_deref() else {
            continue;
        };
        let previous = current
            .iter()
            .find(|entry| entry.name == plugin.name)
            .and_then(|entry| entry.command.as_deref());
        if previous != Some(command) {
            changed.push((plugin.name.clone(), command.to_owned()));
        }
    }
    changed
}

/// Changed commands the request does not declare an approval for. Empty means
/// the request is fully approved.
pub(crate) fn missing_plugin_command_approvals(
    changed: &[(String, String)],
    declared: &[OwnerControlPluginCommandApproval],
) -> Vec<String> {
    let declared: HashSet<(String, String)> = declared
        .iter()
        .map(|entry| (entry.plugin.clone(), entry.command.clone()))
        .collect();
    changed
        .iter()
        .filter(|(plugin, command)| !declared.contains(&(plugin.clone(), command.clone())))
        .map(|(plugin, _)| plugin.clone())
        .collect()
}

/// The rejection message for an apply whose executable plugin command changes
/// are not fully declared in `declared`, or `None` when the apply is approved.
pub(crate) fn plugin_command_approval_error(
    current_plugins: &[PluginConfigEntry],
    incoming: &MeshConfig,
    declared: &[OwnerControlPluginCommandApproval],
) -> Option<String> {
    let changed = changed_plugin_commands(current_plugins, incoming);
    let missing = missing_plugin_command_approvals(&changed, declared);
    (!missing.is_empty()).then(|| {
        format!(
            "executable plugin command changes require separate approval: {}",
            missing.join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::PluginStartupConfig;

    fn entry(name: &str, command: Option<&str>) -> PluginConfigEntry {
        PluginConfigEntry {
            name: name.to_owned(),
            enabled: None,
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            allow_peer_blocks: None,
            allow_inference_observation: None,
            openai_exchange_grant: None,
            command: command.map(str::to_owned),
            args: Vec::new(),
            url: None,
            settings: std::collections::BTreeMap::new(),
            startup: PluginStartupConfig::default(),
        }
    }

    fn incoming_with_command(plugin: &str, command: &str) -> MeshConfig {
        let mut config = MeshConfig::default();
        config.plugins.push(entry(plugin, Some(command)));
        config
    }

    #[test]
    fn added_and_changed_commands_are_reported() {
        let current = vec![entry("metrics", Some("old-binary"))];
        let mut incoming = incoming_with_command("metrics", "new-binary");
        incoming
            .plugins
            .push(entry("second", Some("second-binary")));
        let changed = changed_plugin_commands(&current, &incoming);
        assert_eq!(
            changed,
            vec![
                ("metrics".to_string(), "new-binary".to_string()),
                ("second".to_string(), "second-binary".to_string())
            ]
        );
    }

    #[test]
    fn unchanged_and_commandless_plugins_are_not_reported() {
        let current = vec![entry("metrics", Some("same-binary")), entry("ui", None)];
        let mut incoming = incoming_with_command("metrics", "same-binary");
        incoming.plugins.push(entry("ui", None));
        assert!(changed_plugin_commands(&current, &incoming).is_empty());
    }

    #[test]
    fn removed_commands_are_not_reported() {
        let current = vec![entry("metrics", Some("old-binary"))];
        let mut incoming = MeshConfig::default();
        incoming.plugins.push(entry("metrics", None));
        assert!(changed_plugin_commands(&current, &incoming).is_empty());
    }

    #[test]
    fn declared_approvals_cover_the_changed_commands() {
        let changed = vec![("metrics".to_string(), "new-binary".to_string())];
        let declared = vec![OwnerControlPluginCommandApproval {
            plugin: "metrics".to_string(),
            command: "new-binary".to_string(),
        }];
        assert!(missing_plugin_command_approvals(&changed, &declared).is_empty());
    }

    #[test]
    fn undeclared_changed_commands_are_listed() {
        let changed = vec![
            ("metrics".to_string(), "new-binary".to_string()),
            ("second".to_string(), "second-binary".to_string()),
        ];
        let declared = vec![OwnerControlPluginCommandApproval {
            plugin: "metrics".to_string(),
            command: "new-binary".to_string(),
        }];
        assert_eq!(
            missing_plugin_command_approvals(&changed, &declared),
            vec!["second".to_string()]
        );
    }

    #[test]
    fn approvals_for_unchanged_commands_are_ignored() {
        let changed = vec![];
        let declared = vec![OwnerControlPluginCommandApproval {
            plugin: "metrics".to_string(),
            command: "whatever".to_string(),
        }];
        assert!(missing_plugin_command_approvals(&changed, &declared).is_empty());
    }

    #[test]
    fn fully_approved_apply_has_no_rejection_message() {
        let current = vec![entry("metrics", Some("old-binary"))];
        let incoming = incoming_with_command("metrics", "new-binary");
        let declared = vec![OwnerControlPluginCommandApproval {
            plugin: "metrics".to_string(),
            command: "new-binary".to_string(),
        }];
        assert!(plugin_command_approval_error(&current, &incoming, &declared).is_none());
    }

    #[test]
    fn unapproved_command_change_produces_the_rejection_message() {
        let current = vec![entry("metrics", Some("old-binary"))];
        let incoming = incoming_with_command("metrics", "new-binary");
        let message = plugin_command_approval_error(&current, &incoming, &[])
            .expect("an undeclared command change must be rejected");
        assert!(message.contains("separate approval"), "message: {message}");
        assert!(message.contains("metrics"), "message: {message}");
    }
}

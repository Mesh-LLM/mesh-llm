//! Which `wallet.v1` plugin backs the payment ledger.
//!
//! The ledger's pin wins over everything: outstanding reservations and
//! receivables are only meaningful against the wallet that created them, so a
//! pinned ledger never silently moves to another plugin. Without a pin the
//! operator's `[payments] wallet` choice wins, and without that the host uses
//! the only wallet plugin. The built-in Lexe wallet runs by default, so it
//! yields to any other wallet plugin the operator configured.
//!
//! Choosing automatically pins the choice, so it is refused while any plugin
//! that could be a wallet is down. A plugin that crashes loses its capability
//! list, so a stopped wallet plugin looks like any other stopped plugin; if
//! the host chose among the running ones, a crash would hand the ledger to
//! Lexe for good.

use anyhow::{Result, bail};
use mesh_llm_config::PaymentsConfig;
use mesh_llm_wallet::contract::CAPABILITY;
use mesh_llm_wallet::provisioning::WalletPin;

use crate::plugin::{PluginSummary, WALLET_LEXE_PLUGIN_ID};

/// The wallet plugin the operator chose in `[payments] wallet`, if any. A
/// blank value means no choice.
pub fn configured_wallet(config: &PaymentsConfig) -> Option<String> {
    config
        .wallet
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
}

/// Enabled plugins that are not running and might serve `wallet.v1`: either
/// they declared it, or their capabilities are unknown because they failed to
/// start or have crashed. A lazily started plugin that has not been started
/// yet is left out: the host never starts one to find a wallet, so it could
/// not be chosen anyway.
pub fn possible_wallets_not_running(plugins: &[PluginSummary]) -> Vec<String> {
    plugins
        .iter()
        .filter(|plugin| {
            plugin.enabled && !matches!(plugin.status.as_str(), "running" | "deferred")
        })
        .filter(|plugin| {
            plugin.capabilities.is_empty() || plugin.capabilities.iter().any(|c| c == CAPABILITY)
        })
        .map(|plugin| plugin.name.clone())
        .collect()
}

/// Choose the wallet plugin to open among the `running` `wallet.v1`
/// providers. `down` lists the plugins that might be wallets but are not
/// running; see [`possible_wallets_not_running`].
pub fn select_wallet_plugin(
    running: &[String],
    down: &[String],
    pin: Option<&WalletPin>,
    configured: Option<&str>,
) -> Result<String> {
    let is_running = |name: &str| running.iter().any(|running| running == name);
    if let Some(pin) = pin {
        if let Some(chosen) = configured
            && chosen != pin.plugin
        {
            bail!(
                "[payments] wallet = '{chosen}', but the payment ledger is pinned to wallet \
                 plugin '{}'. Stop mesh-llm and run `mesh-llm wallet unpin` to switch wallets.",
                pin.plugin
            );
        }
        if !is_running(&pin.plugin) {
            bail!(
                "the payment ledger is pinned to wallet plugin '{}', which is not running{}",
                pin.plugin,
                running_suffix(running)
            );
        }
        return Ok(pin.plugin.clone());
    }
    if let Some(chosen) = configured {
        if !is_running(chosen) {
            bail!(
                "[payments] wallet = '{chosen}', but that wallet plugin is not running{}",
                running_suffix(running)
            );
        }
        return Ok(chosen.to_owned());
    }
    if !down.is_empty() {
        bail!(
            "cannot choose a wallet plugin while {} not running: it may be the wallet you \
             meant. Start it, or name the wallet with `[payments] wallet`.",
            down.join(", ")
        );
    }
    let preferred: Vec<&String> = running
        .iter()
        .filter(|name| name.as_str() != WALLET_LEXE_PLUGIN_ID)
        .collect();
    match (preferred.as_slice(), running) {
        ([only], _) => Ok((*only).clone()),
        ([], [only]) => Ok(only.clone()),
        ([], []) => bail!("no wallet plugin is running (capability '{CAPABILITY}' unavailable)"),
        _ => bail!(
            "several wallet plugins are running ({}); choose one with `[payments] wallet`",
            running.join(", ")
        ),
    }
}

fn running_suffix(running: &[String]) -> String {
    if running.is_empty() {
        String::new()
    } else {
        format!(" (running: {})", running.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn pin(plugin: &str) -> WalletPin {
        WalletPin {
            plugin: plugin.into(),
            wallet_id: "w".into(),
            provider: "p".into(),
            network: "mainnet".into(),
        }
    }

    fn summary(name: &str, status: &str, capabilities: &[&str]) -> PluginSummary {
        PluginSummary {
            name: name.into(),
            kind: "external".into(),
            enabled: true,
            status: status.into(),
            pid: None,
            version: None,
            capabilities: names(capabilities),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            web_ui: Default::default(),
            startup: None,
            error: None,
        }
    }

    #[test]
    fn the_only_running_wallet_is_used() {
        let running = names(&[WALLET_LEXE_PLUGIN_ID]);
        let selected = select_wallet_plugin(&running, &[], None, None).unwrap();
        assert_eq!(selected, WALLET_LEXE_PLUGIN_ID);
    }

    #[test]
    fn another_wallet_plugin_is_preferred_over_the_default_lexe_wallet() {
        let running = names(&["wallet-nwc", WALLET_LEXE_PLUGIN_ID]);
        let selected = select_wallet_plugin(&running, &[], None, None).unwrap();
        assert_eq!(selected, "wallet-nwc");
    }

    #[test]
    fn a_plugin_that_might_be_a_wallet_blocks_automatic_choice() {
        let running = names(&[WALLET_LEXE_PLUGIN_ID]);
        let down = names(&["wallet-ldk-server"]);
        let error = select_wallet_plugin(&running, &down, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("wallet-ldk-server"), "{error}");
        assert!(error.contains("[payments] wallet"), "{error}");

        // An explicit choice or a pin says what the operator meant.
        let chosen = select_wallet_plugin(&running, &down, None, Some(WALLET_LEXE_PLUGIN_ID));
        assert_eq!(chosen.unwrap(), WALLET_LEXE_PLUGIN_ID);
        let pinned = pin(WALLET_LEXE_PLUGIN_ID);
        let reopened = select_wallet_plugin(&running, &down, Some(&pinned), None);
        assert_eq!(reopened.unwrap(), WALLET_LEXE_PLUGIN_ID);
    }

    #[test]
    fn a_crashed_plugin_counts_as_a_possible_wallet() {
        // What `handle_runtime_failure` leaves behind: restarting, with its
        // capabilities cleared.
        let crashed = summary("wallet-ldk-server", "restarting", &[]);
        let failed_wallet = summary("my-wallet", "error", &[CAPABILITY]);
        let other = summary("search", "stopped", &["search.v1"]);
        let running = summary(WALLET_LEXE_PLUGIN_ID, "running", &[]);
        let mut disabled = summary("old-wallet", "disabled", &[]);
        disabled.enabled = false;
        let deferred = summary("lazy-tools", "deferred", &[]);
        let plugins = [crashed, failed_wallet, other, running, disabled, deferred];
        assert_eq!(
            possible_wallets_not_running(&plugins),
            ["wallet-ldk-server", "my-wallet"]
        );
    }

    #[test]
    fn two_non_default_wallets_need_an_explicit_choice() {
        let running = names(&["wallet-a", "wallet-b", WALLET_LEXE_PLUGIN_ID]);
        let error = select_wallet_plugin(&running, &[], None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("[payments] wallet"), "{error}");
        let selected = select_wallet_plugin(&running, &[], None, Some("wallet-b")).unwrap();
        assert_eq!(selected, "wallet-b");
    }

    #[test]
    fn an_explicit_choice_must_be_running() {
        let running = names(&[WALLET_LEXE_PLUGIN_ID]);
        let error = select_wallet_plugin(&running, &[], None, Some("wallet-nwc"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("not running"), "{error}");
        assert!(error.contains(WALLET_LEXE_PLUGIN_ID), "{error}");
    }

    #[test]
    fn a_pin_overrides_preference_but_not_an_explicit_conflicting_choice() {
        let running = names(&["wallet-nwc", WALLET_LEXE_PLUGIN_ID]);
        let pinned = pin(WALLET_LEXE_PLUGIN_ID);
        let selected = select_wallet_plugin(&running, &[], Some(&pinned), None).unwrap();
        assert_eq!(selected, WALLET_LEXE_PLUGIN_ID);

        let error = select_wallet_plugin(&running, &[], Some(&pinned), Some("wallet-nwc"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("mesh-llm wallet unpin"), "{error}");
    }

    #[test]
    fn a_pinned_wallet_that_is_not_running_is_not_replaced() {
        let running = names(&["wallet-nwc"]);
        let error = select_wallet_plugin(&running, &[], Some(&pin(WALLET_LEXE_PLUGIN_ID)), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not running"), "{error}");
    }

    #[test]
    fn nothing_running_is_a_clear_error() {
        let error = select_wallet_plugin(&[], &[], None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("no wallet plugin is running"), "{error}");
    }

    #[test]
    fn blank_configured_wallet_means_unset() {
        let blank = PaymentsConfig {
            wallet: Some("  ".into()),
        };
        assert_eq!(configured_wallet(&blank), None);
        let set = PaymentsConfig {
            wallet: Some(" wallet-nwc ".into()),
        };
        assert_eq!(configured_wallet(&set).as_deref(), Some("wallet-nwc"));
    }
}

//! Reviewed default plugin pins, provisioned by installers and `mesh-llm update`.
//! Node startup never downloads plugins. The installed plugin metadata records
//! default ownership, so a later upgrade can move only default-managed plugins
//! to the next reviewed pin.

use crate::catalog::PinnedRelease;
use crate::install::{
    InstallOutcome, PluginInstallOptions, PluginProgressReporter, install_default_plugin_at,
};
use crate::store::{InstalledPluginMetadata, PluginStore};
use std::collections::BTreeSet;
use std::time::Duration;

const INSTALL_TIMEOUT: Duration = Duration::from_secs(60);
pub const NO_DEFAULT_PLUGINS_ENV: &str = "MESH_LLM_NO_DEFAULT_PLUGINS";

pub fn opts_out(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

pub fn default_plugins_opted_out() -> bool {
    opts_out(std::env::var(NO_DEFAULT_PLUGINS_ENV).ok().as_deref())
}

/// One default plugin, pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefaultPlugin {
    /// The catalog name, which is also the installed plugin name.
    pub name: &'static str,
    /// The exact release to install.
    pub version: &'static str,
    /// `(target triple, lowercase hex SHA-256 of that release's archive)`.
    pub sha256: &'static [(&'static str, &'static str)],
    /// The host surfaces this default may use. The host keeps only these from
    /// the plugin's initialize answer and removes the rest.
    pub allows: &'static [Surface],
    /// The capability strings this default may declare; none by default.
    pub capabilities: &'static [&'static str],
}

/// A host surface a plugin's initialize answer can claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    WebUi,
    Config,
    MeshChannels,
    MeshEvents,
    HttpRoutes,
    /// MCP operations, resources, resource templates, prompts, completions.
    McpOperations,
    InferenceEndpoints,
    /// Any endpoint that is not an inference endpoint.
    OtherEndpoints,
    VirtualModels,
}

impl Surface {
    pub fn name(self) -> &'static str {
        match self {
            Self::WebUi => "web UI",
            Self::Config => "config",
            Self::MeshChannels => "mesh channels",
            Self::MeshEvents => "mesh events",
            Self::HttpRoutes => "HTTP routes",
            Self::McpOperations => "MCP operations",
            Self::InferenceEndpoints => "inference endpoints",
            Self::OtherEndpoints => "other endpoints",
            Self::VirtualModels => "virtual models",
        }
    }
}

/// What a default-managed plugin that is no longer on this build's list may
/// still use: everything but serving models, until it is reviewed again.
pub const UNLISTED_DEFAULT_ALLOWS: &[Surface] = &[
    Surface::WebUi,
    Surface::Config,
    Surface::MeshChannels,
    Surface::MeshEvents,
    Surface::HttpRoutes,
    Surface::McpOperations,
    Surface::OtherEndpoints,
];

/// This build's entry for a default plugin, by its installed name.
pub fn default_plugin(name: &str) -> Option<&'static DefaultPlugin> {
    DEFAULT_PLUGINS.iter().find(|default| default.name == name)
}

impl DefaultPlugin {
    pub(crate) fn pin_for(&self, target: &str) -> Option<PinnedRelease<'static>> {
        self.sha256
            .iter()
            .find(|(triple, _)| *triple == target)
            .map(|&(_, sha256)| PinnedRelease {
                version: self.version,
                sha256,
            })
    }
}

/// The plugins a fresh node installs. Each entry is added by its own PR, and
/// bumping one is a one-entry change per release. Payment and wallet
/// plugins are never on this list: a node pays or gets paid only through a
/// plugin its operator chose.
pub const DEFAULT_PLUGINS: &[DefaultPlugin] = &[DefaultPlugin {
    name: "capsule-emit-mesh",
    version: "0.1.2",
    sha256: &[
        (
            "aarch64-apple-darwin",
            "593566c4f0bcb9edc804fb2face0b07e7508a0d5891f7962c14b6c06578b09a6",
        ),
        (
            "x86_64-unknown-linux-gnu",
            "696c60023f4d0868f94e1fdee4799616c6a6ad75a3a067337a0e098f698414fc",
        ),
        (
            "aarch64-unknown-linux-gnu",
            "578c89497a61591907bd065d1b751a2c2cc2c545b62c105ce1aab96e3db647ac",
        ),
    ],
    // An audit plugin: records, its page and its settings; it serves no model.
    // Other (non-inference) endpoints stay allowed so a later release can
    // declare an MCP endpoint without the host stripping it.
    allows: &[
        Surface::WebUi,
        Surface::Config,
        Surface::MeshChannels,
        Surface::MeshEvents,
        Surface::HttpRoutes,
        Surface::McpOperations,
        Surface::OtherEndpoints,
    ],
    capabilities: &[],
}];

/// What provisioning did with one default.
#[derive(Debug, Clone, PartialEq)]
pub enum DefaultPluginOutcome {
    Installed(Box<InstallOutcome>),
    AlreadyCurrent,
    OperatorManaged,
    Disabled,
    /// Turned off with `plugins disable` while it was not installed.
    TurnedOff,
    UnsupportedPlatform,
    NotInstalled(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    SkipOperator,
    SkipDisabled,
    SkipCurrent,
    Install(PinnedRelease<'static>),
    NoPinForTarget,
}

fn plan(default: &DefaultPlugin, current: Option<&InstalledPluginMetadata>, target: &str) -> Plan {
    let Some(pin) = default.pin_for(target) else {
        return Plan::NoPinForTarget;
    };
    match current {
        Some(metadata) if !metadata.default_managed => Plan::SkipOperator,
        Some(metadata) if !metadata.enabled => Plan::SkipDisabled,
        Some(metadata)
            if metadata.installed_version.trim_start_matches('v') == default.version
                && metadata.target_triple == target =>
        {
            Plan::SkipCurrent
        }
        _ => Plan::Install(pin),
    }
}

/// Provision defaults from the reviewed pins. A deleted default is installed
/// again on the next install/update invocation; disabling it persists in its
/// installed metadata and prevents an automatic pin upgrade.
pub async fn install_default_plugins(
    defaults: &[DefaultPlugin],
    configured: &BTreeSet<String>,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Vec<(&'static str, DefaultPluginOutcome)> {
    let store = PluginStore::new(&options.store_root);
    let target = options.target.triple();
    let mut outcomes = Vec::new();
    for default in defaults {
        if configured.contains(default.name) {
            outcomes.push((default.name, DefaultPluginOutcome::OperatorManaged));
            continue;
        }
        if store.default_turned_off(default.name) {
            outcomes.push((default.name, DefaultPluginOutcome::TurnedOff));
            continue;
        }
        let outcome = match store.load_optional(default.name) {
            Err(error) => {
                DefaultPluginOutcome::NotInstalled(format!("read installed metadata: {error:#}"))
            }
            Ok(current) => match plan(default, current.as_ref(), target) {
                Plan::SkipOperator => DefaultPluginOutcome::OperatorManaged,
                Plan::SkipDisabled => DefaultPluginOutcome::Disabled,
                Plan::SkipCurrent => DefaultPluginOutcome::AlreadyCurrent,
                Plan::NoPinForTarget => DefaultPluginOutcome::UnsupportedPlatform,
                Plan::Install(pin) => {
                    match tokio::time::timeout(
                        INSTALL_TIMEOUT,
                        install_default_plugin_at(default.name, pin, options, progress),
                    )
                    .await
                    {
                        Ok(Ok(installed)) => DefaultPluginOutcome::Installed(Box::new(installed)),
                        Ok(Err(error)) => DefaultPluginOutcome::NotInstalled(format!("{error:#}")),
                        Err(_) => DefaultPluginOutcome::NotInstalled(
                            "install timed out after 60 seconds".into(),
                        ),
                    }
                }
            },
        };
        outcomes.push((default.name, outcome));
    }
    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINUX: &str = "x86_64-unknown-linux-gnu";
    const NOTES: DefaultPlugin = DefaultPlugin {
        name: "notes",
        version: "1.0.0",
        sha256: &[(
            LINUX,
            "abababababababababababababababababababababababababababababababab",
        )],
        allows: &[Surface::WebUi],
        capabilities: &[],
    };

    fn installed(version: &str, managed: bool, enabled: bool) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: "notes".into(),
            source_repository: "https://github.com/example/notes".into(),
            installed_version: version.into(),
            target_triple: LINUX.into(),
            downloaded_asset_name: "notes.tar.gz".into(),
            install_path: "/tmp/notes".into(),
            enabled,
            default_managed: managed,
            manifest: None,
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        }
    }

    #[test]
    fn pin_bump_updates_only_enabled_default_managed_installs() {
        assert!(matches!(
            plan(&NOTES, Some(&installed("0.9.0", true, true)), LINUX),
            Plan::Install(_)
        ));
        assert_eq!(
            plan(&NOTES, Some(&installed("v1.0.0", true, true)), LINUX),
            Plan::SkipCurrent
        );
        assert_eq!(
            plan(&NOTES, Some(&installed("0.9.0", false, true)), LINUX),
            Plan::SkipOperator
        );
        assert_eq!(
            plan(&NOTES, Some(&installed("0.9.0", true, false)), LINUX),
            Plan::SkipDisabled
        );
    }

    #[test]
    fn missing_plugin_uses_reviewed_pin() {
        assert_eq!(
            plan(&NOTES, None, LINUX),
            Plan::Install(PinnedRelease {
                version: "1.0.0",
                sha256: NOTES.sha256[0].1,
            })
        );
    }

    #[test]
    fn unsupported_platform_has_no_install() {
        assert_eq!(
            plan(&NOTES, None, "aarch64-apple-darwin"),
            Plan::NoPinForTarget
        );
    }

    #[test]
    fn install_time_opt_out_reads_like_a_flag() {
        assert!(!opts_out(None));
        assert!(!opts_out(Some("")));
        assert!(!opts_out(Some("0")));
        assert!(opts_out(Some("1")));
        assert!(opts_out(Some("true")));
    }

    #[test]
    fn the_shipped_list_names_valid_unique_plugins() {
        let unique: BTreeSet<_> = DEFAULT_PLUGINS.iter().map(|default| default.name).collect();
        assert_eq!(unique.len(), DEFAULT_PLUGINS.len());
        for default in DEFAULT_PLUGINS {
            assert!(
                crate::source_ref::is_valid_name(default.name),
                "{}",
                default.name
            );
            assert!(
                crate::source_ref::PluginVersion::new(default.version).is_ok(),
                "{} pins an invalid version {:?}",
                default.name,
                default.version
            );
            for (triple, _) in default.sha256 {
                assert!(
                    crate::target::PluginTarget::is_supported_triple(triple),
                    "{} pins an unsupported target triple {triple}",
                    default.name
                );
            }
        }
    }

    #[test]
    fn no_payment_or_wallet_plugin_is_a_default() {
        for default in DEFAULT_PLUGINS {
            for word in ["wallet", "payment", "lightning", "lexe"] {
                assert!(
                    !default.name.contains(word),
                    "{} looks like a payment or wallet plugin; those are never defaults",
                    default.name
                );
            }
        }
    }

    /// The placeholder a pin carries before its release exists. The installer
    /// refuses it as not a SHA-256; this test refuses to pass while one remains.
    const TODO_AFTER_TAG: &str = "TODO-AFTER-TAG";

    /// Fails while any pin is still a placeholder, so the list cannot ship
    /// unfilled: each digest must be the release archive's real SHA-256.
    #[test]
    fn the_shipped_pins_are_filled_in() {
        for default in DEFAULT_PLUGINS {
            for (triple, digest) in default.sha256 {
                assert_ne!(
                    *digest, TODO_AFTER_TAG,
                    "{} {} pin for {triple} is still {TODO_AFTER_TAG}: fill it from the \
                     release's SHA256SUMS before merging",
                    default.name, default.version
                );
                assert!(
                    crate::catalog::is_sha256_hex(digest),
                    "{} pins a malformed digest for {triple}",
                    default.name
                );
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_default_disabled_after_a_delete_stays_off() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut record = installed("v1.0.0", true, true);
        record.install_path = temp.path().join("installed").join("notes");
        std::fs::create_dir_all(&record.install_path).unwrap();
        store.save(&record).unwrap();

        // plugins delete, then plugins disable on the now-uninstalled default.
        store.delete("notes").unwrap();
        store.set_default_turned_off("notes", true).unwrap();
        assert!(store.load_optional("notes").unwrap().is_none());
        assert!(store.list().unwrap().is_empty(), "a turned-off default is not listed");

        // The next installer or update run leaves it off and fetches nothing.
        let options = PluginInstallOptions {
            store_root: temp.path().to_path_buf(),
            install_root: temp.path().join("installed"),
            catalog_url: "http://127.0.0.1:9/unreachable".into(),
            target: crate::target::PluginTarget::from_os_arch("linux", "x86_64").unwrap(),
        };
        let mut events = 0;
        let outcomes = install_default_plugins(&[NOTES], &BTreeSet::new(), &options, &mut |_: crate::install::PluginProgressEvent| {
            events += 1
        })
        .await;
        assert!(matches!(
            outcomes.as_slice(),
            [("notes", DefaultPluginOutcome::TurnedOff)]
        ));
        assert_eq!(events, 0, "nothing was fetched");
        assert!(store.load_optional("notes").unwrap().is_none());

        // plugins enable undoes it.
        store.set_default_turned_off("notes", false).unwrap();
        assert!(!store.default_turned_off("notes"));
    }

    #[test]
    fn no_shipped_default_may_serve_a_model() {
        for default in DEFAULT_PLUGINS {
            assert!(
                !default.allows.contains(&Surface::InferenceEndpoints),
                "{}",
                default.name
            );
            assert!(
                !default.allows.contains(&Surface::VirtualModels),
                "{}",
                default.name
            );
        }
        assert!(!UNLISTED_DEFAULT_ALLOWS.contains(&Surface::InferenceEndpoints));
        assert!(!UNLISTED_DEFAULT_ALLOWS.contains(&Surface::VirtualModels));
    }
}

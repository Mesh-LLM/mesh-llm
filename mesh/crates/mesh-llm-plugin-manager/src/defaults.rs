//! Default plugins: a short list a fresh node installs once, on first run.
//!
//! Each entry names a catalog plugin and pins it here, in reviewed code: an
//! exact release version and the SHA-256 of that release's archive for each
//! platform. The catalog only says where the plugin lives; if its entry pins
//! the same platform too, the two pins must agree. The download must match the
//! pin as well as GitHub's digest, or nothing is installed. Bumping a default
//! is one entry here per release.
//!
//! A default is offered once. After a successful install, or when the plugin
//! is already installed or named in the operator's config, its name is
//! recorded in `defaults-offered.json` in the plugin store, so an operator who
//! disables or removes it never gets it back behind their back. A record that
//! exists but can't be read offers nothing: it is never treated as empty. A
//! failed attempt (no network, no pin for this platform, a digest mismatch) is
//! not recorded and is tried again on the next start. `--no-default-plugins`
//! or `MESH_LLM_NO_DEFAULT_PLUGINS=1` skips the list.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::catalog::PinnedRelease;
use crate::install::{
    InstallOutcome, PluginInstallOptions, PluginProgressReporter, install_default_plugin_at,
};
use crate::store::PluginStore;

/// Set to any non-empty value other than `0` to skip the default list.
pub const NO_DEFAULT_PLUGINS_ENV: &str = "MESH_LLM_NO_DEFAULT_PLUGINS";
const OFFERED_FILE: &str = "defaults-offered.json";

/// One default plugin, pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefaultPlugin {
    /// The catalog name, which is also the installed plugin name.
    pub name: &'static str,
    /// The exact release to install.
    pub version: &'static str,
    /// `(target triple, lowercase hex SHA-256 of that release's archive)`.
    pub sha256: &'static [(&'static str, &'static str)],
}

impl DefaultPlugin {
    fn pin_for(&self, target: &str) -> Option<PinnedRelease<'static>> {
        self.sha256
            .iter()
            .find(|(triple, _)| *triple == target)
            .map(|&(_, sha256)| PinnedRelease {
                version: self.version,
                sha256,
            })
    }
}

/// The plugins a fresh node installs. Empty: an entry is added by its own PR,
/// and bumping one is a one-entry change per release. Payment and wallet
/// plugins are never on this list: a node pays or gets paid only through a
/// plugin its operator chose.
pub const DEFAULT_PLUGINS: &[DefaultPlugin] = &[];

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Offered {
    #[serde(default)]
    names: BTreeSet<String>,
}

/// What first run did with one default.
#[derive(Debug, Clone, PartialEq)]
pub enum DefaultPluginOutcome {
    Installed(Box<InstallOutcome>),
    /// Offered on an earlier run; never offered again.
    AlreadyOffered,
    /// Installed or configured by the operator; recorded as offered.
    OperatorChose,
    /// Not installed this time; tried again on the next start.
    NotInstalled(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    Skip,
    RecordOnly,
    Install(PinnedRelease<'static>),
    NoPinForTarget,
}

fn plan(
    default: &DefaultPlugin,
    offered: &Offered,
    installed: bool,
    configured: bool,
    target: &str,
) -> Plan {
    if offered.names.contains(default.name) {
        return Plan::Skip;
    }
    // An operator who already names the plugin in their config, or installed
    // it themselves, has decided; don't second-guess that.
    if installed || configured {
        return Plan::RecordOnly;
    }
    match default.pin_for(target) {
        Some(pin) => Plan::Install(pin),
        None => Plan::NoPinForTarget,
    }
}

fn offered_path(store_root: &Path) -> PathBuf {
    store_root.join(OFFERED_FILE)
}

/// A missing record is a fresh node. A record that exists but can't be read
/// is an error: treating it as empty would reinstall a default the operator
/// removed.
fn load_offered(store_root: &Path) -> Result<Offered> {
    let path = offered_path(store_root);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Offered::default());
        }
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// The record to plan against, or why this run offers nothing.
fn offered_or_refusal(store_root: &Path) -> std::result::Result<Offered, String> {
    load_offered(store_root).map_err(|error| format!("{error:#}; offering no defaults"))
}

fn save_offered(store_root: &Path, offered: &Offered) -> Result<()> {
    std::fs::create_dir_all(store_root)
        .with_context(|| format!("create plugin store {}", store_root.display()))?;
    let path = offered_path(store_root);
    let temp = path.with_extension("json.tmp");
    std::fs::write(&temp, serde_json::to_vec_pretty(offered)?)
        .with_context(|| format!("write {}", temp.display()))?;
    std::fs::rename(&temp, &path).with_context(|| format!("write {}", path.display()))
}

/// Whether `value`, the value of [`NO_DEFAULT_PLUGINS_ENV`], opts out.
pub fn opts_out(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

/// Whether the environment opts this node out of the default list.
pub fn default_plugins_opted_out() -> bool {
    opts_out(std::env::var(NO_DEFAULT_PLUGINS_ENV).ok().as_deref())
}

/// What one first-run pass did.
#[derive(Debug, Default)]
pub struct DefaultPluginsRun {
    pub outcomes: Vec<(&'static str, DefaultPluginOutcome)>,
    /// Set when the offered-defaults record could not be read (nothing was
    /// offered) or could not be saved.
    pub record_error: Option<String>,
}

/// Offer each of `defaults` once. `configured` names plugins the operator's
/// config already lists. Never fails the node's start: every problem is an
/// outcome or a `record_error`, and the caller logs it.
pub async fn install_default_plugins(
    defaults: &[DefaultPlugin],
    configured: &BTreeSet<String>,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> DefaultPluginsRun {
    let store = PluginStore::new(&options.store_root);
    let mut offered = match offered_or_refusal(&options.store_root) {
        Ok(offered) => offered,
        Err(record_error) => {
            return DefaultPluginsRun {
                outcomes: Vec::new(),
                record_error: Some(record_error),
            };
        }
    };
    let before = offered.clone();
    let mut outcomes = Vec::new();
    let target = options.target.triple();
    for default in defaults {
        let name = default.name;
        let installed = matches!(store.load_optional(name), Ok(Some(_)));
        let outcome = match plan(
            default,
            &offered,
            installed,
            configured.contains(name),
            target,
        ) {
            Plan::Skip => DefaultPluginOutcome::AlreadyOffered,
            Plan::RecordOnly => {
                offered.names.insert(name.to_string());
                DefaultPluginOutcome::OperatorChose
            }
            Plan::NoPinForTarget => {
                DefaultPluginOutcome::NotInstalled(format!("no pinned archive for {target}"))
            }
            Plan::Install(pin) => {
                match install_default_plugin_at(name, pin, options, progress).await {
                    Ok(outcome) => {
                        offered.names.insert(name.to_string());
                        DefaultPluginOutcome::Installed(Box::new(outcome))
                    }
                    Err(error) => DefaultPluginOutcome::NotInstalled(format!("{error:#}")),
                }
            }
        };
        outcomes.push((name, outcome));
    }
    // Nothing new to record on a node that has already offered everything:
    // a later start leaves the store untouched.
    let record_error = (offered != before)
        .then(|| save_offered(&options.store_root, &offered).err())
        .flatten()
        .map(|error| format!("could not record offered defaults: {error:#}"));
    DefaultPluginsRun {
        outcomes,
        record_error,
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::*;
    use crate::install::PluginProgressEvent;

    fn offered(names: &[&str]) -> Offered {
        Offered {
            names: names.iter().map(|name| name.to_string()).collect(),
        }
    }

    const LINUX: &str = "x86_64-unknown-linux-gnu";
    const NOTES: DefaultPlugin = DefaultPlugin {
        name: "notes",
        version: "1.0.0",
        sha256: &[(
            LINUX,
            "abababababababababababababababababababababababababababababababab",
        )],
    };

    #[test]
    fn a_fresh_node_installs_each_default_once_at_its_pin() {
        assert_eq!(
            plan(&NOTES, &offered(&[]), false, false, LINUX),
            Plan::Install(PinnedRelease {
                version: "1.0.0",
                sha256: NOTES.sha256[0].1,
            })
        );
        assert_eq!(
            plan(&NOTES, &offered(&["notes"]), false, false, LINUX),
            Plan::Skip,
            "a removed default is never offered again"
        );
    }

    #[test]
    fn a_platform_without_a_pin_is_not_installed() {
        assert_eq!(
            plan(&NOTES, &offered(&[]), false, false, "aarch64-apple-darwin"),
            Plan::NoPinForTarget
        );
    }

    #[test]
    fn an_operator_choice_is_recorded_not_overridden() {
        for (installed, configured) in [(true, false), (false, true)] {
            assert_eq!(
                plan(&NOTES, &offered(&[]), installed, configured, LINUX),
                Plan::RecordOnly
            );
        }
    }

    #[test]
    fn the_environment_opt_out_reads_like_a_flag() {
        assert!(!opts_out(None));
        assert!(!opts_out(Some("")));
        assert!(!opts_out(Some("0")));
        assert!(opts_out(Some("1")));
        assert!(opts_out(Some("true")));
    }

    #[test]
    fn a_node_that_has_offered_everything_does_nothing_on_later_starts() {
        let temp = tempfile::tempdir().unwrap();
        save_offered(temp.path(), &offered(&["notes"])).unwrap();
        let record = offered_path(temp.path());
        let written = std::fs::metadata(&record).unwrap().modified().unwrap();
        let options = PluginInstallOptions {
            store_root: temp.path().to_path_buf(),
            install_root: temp.path().join("installed"),
            catalog_url: "http://127.0.0.1:9/unreachable".to_string(),
            target: crate::target::PluginTarget::current().unwrap(),
        };
        let mut events: Vec<PluginProgressEvent> = Vec::new();

        let configured = BTreeSet::new();
        let run = {
            let mut progress = |event: PluginProgressEvent| events.push(event);
            let mut run = std::pin::pin!(install_default_plugins(
                &[NOTES],
                &configured,
                &options,
                &mut progress,
            ));
            // Polled once with a waker that never fires: an idle start
            // finishes without waiting on anything, so it never reaches the
            // network.
            match run
                .as_mut()
                .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
            {
                std::task::Poll::Ready(run) => run,
                std::task::Poll::Pending => panic!("an idle start must not wait on I/O"),
            }
        };

        assert_eq!(
            run.outcomes,
            [("notes", DefaultPluginOutcome::AlreadyOffered)]
        );
        assert!(run.record_error.is_none());
        assert!(
            events.is_empty(),
            "no catalog lookup, no download, no output"
        );
        assert_eq!(
            std::fs::metadata(&record).unwrap().modified().unwrap(),
            written,
            "the record is not rewritten"
        );
    }

    #[test]
    fn offered_names_survive_a_reload_and_a_missing_record_is_a_fresh_node() {
        let temp = tempfile::tempdir().unwrap();
        assert_eq!(load_offered(temp.path()).unwrap(), Offered::default());
        save_offered(temp.path(), &offered(&["notes"])).unwrap();
        assert_eq!(load_offered(temp.path()).unwrap(), offered(&["notes"]));
    }

    #[test]
    fn an_unreadable_record_offers_nothing_and_is_kept() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(offered_path(temp.path()), b"not json").unwrap();

        let refusal =
            offered_or_refusal(temp.path()).expect_err("a corrupt record must not read as empty");
        assert!(refusal.contains("offering no defaults"), "{refusal}");
        assert_eq!(
            std::fs::read(offered_path(temp.path())).unwrap(),
            b"not json",
            "the record is left for the operator to fix, not overwritten"
        );
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
}

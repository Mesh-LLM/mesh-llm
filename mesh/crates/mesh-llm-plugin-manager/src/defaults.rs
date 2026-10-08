//! Reviewed default plugin pins. A release bundles each default plugin's
//! release archive (`crate::bundled`), and a node installs a default only from
//! that bundled copy, after checking it against the pin here: when it starts,
//! and on `mesh-llm plugins install-defaults`. Nothing is ever downloaded, and
//! a release that bundles no copy installs none. The installed plugin metadata
//! records default ownership, so a new release's copy replaces only
//! default-managed installs.

use crate::catalog::PinnedRelease;
use crate::install::{
    InstallOutcome, PluginInstallOptions, PluginProgressReporter, install_bundled_default,
};
use crate::store::{InstalledPluginMetadata, PluginStore};
use std::collections::BTreeSet;

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
    /// The OpenAI exchange lifecycle hook: it sees every OpenAI exchange the
    /// node handles, prompts and answers included.
    OpenAiExchangeHook,
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
            Self::OpenAiExchangeHook => "OpenAI exchange hook",
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

/// The plugins a node installs from its release's bundled copy. Each entry
/// is added by its own PR, and bumping one is a one-entry change, made with
/// the release's own pin (`ci/bundled-plugins.json`) so the two agree.
/// Payment and wallet plugins are never on this list: a node pays or gets paid
/// only through a plugin its operator chose. A target with no pin (Windows
/// today: the plugin has no Windows build) reports it unsupported.
pub const DEFAULT_PLUGINS: &[DefaultPlugin] = &[DefaultPlugin {
    name: "capsules",
    version: "0.1.3",
    sha256: &[
        (
            "aarch64-apple-darwin",
            "90ad75f783b49371fe5e4d436dd306d57c6e3704611d276c25f75d2c4d739e64",
        ),
        (
            "x86_64-unknown-linux-gnu",
            "7a0a13ce2cb4210a48e937163f36cb41684d5399a402eff03ec389be44d03fb8",
        ),
        (
            "aarch64-unknown-linux-gnu",
            "992f1ebdf8a1ed624b0082440376daa1b597efeac0ea1fcf46c84f33eb145dd3",
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
    /// This install carries no bundled copy for this platform (a development
    /// build, or a package that dropped it). Nothing is downloaded instead.
    NotBundled,
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

/// Install the defaults from this release's bundled copy, each at its
/// reviewed pin. A default the operator runs from config, disabled, turned
/// off (with `plugins disable`, or deleted: a deleted default stays removed),
/// or already at this release's pin is left alone. Nothing is downloaded: a
/// missing bundled copy is reported, not fetched. Local and quick, so a node
/// can run it on every start.
pub fn provision_bundled_defaults(
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
        let outcome = match store.load_optional(default.name) {
            Err(error) => {
                DefaultPluginOutcome::NotInstalled(format!("read installed metadata: {error:#}"))
            }
            // The turned-off record only matters while nothing is installed:
            // an installed plugin is handled by its own record.
            Ok(None) if store.default_turned_off(default.name) => DefaultPluginOutcome::TurnedOff,
            Ok(current) => match plan(default, current.as_ref(), target) {
                Plan::SkipOperator => DefaultPluginOutcome::OperatorManaged,
                Plan::SkipDisabled => DefaultPluginOutcome::Disabled,
                Plan::SkipCurrent => DefaultPluginOutcome::AlreadyCurrent,
                Plan::NoPinForTarget => DefaultPluginOutcome::UnsupportedPlatform,
                Plan::Install(pin) => {
                    match install_bundled_default(default.name, pin, options, progress) {
                        Ok(Some(installed)) => DefaultPluginOutcome::Installed(Box::new(installed)),
                        Ok(None) => DefaultPluginOutcome::NotBundled,
                        Err(error) => DefaultPluginOutcome::NotInstalled(format!("{error:#}")),
                    }
                }
            },
        };
        outcomes.push((default.name, outcome));
    }
    outcomes
}

/// Turn every default off: an installed default-managed copy is disabled, and
/// a default that is not installed gets the turned-off record, so neither a
/// node start nor `plugins install-defaults` installs it until `plugins
/// enable NAME`. A default the operator installed or runs is left alone.
pub fn turn_off_defaults(
    defaults: &[DefaultPlugin],
    store: &PluginStore,
) -> anyhow::Result<Vec<&'static str>> {
    let mut turned_off = Vec::new();
    for default in defaults {
        match store.load_optional(default.name)? {
            Some(installed) if !installed.default_managed => continue,
            Some(_) => {
                store.set_enabled(default.name, false)?;
            }
            None => store.set_default_turned_off(default.name, true)?,
        }
        turned_off.push(default.name);
    }
    Ok(turned_off)
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

    /// A bundled copy of `notes` for Linux: a real plugin archive (a
    /// `plugin.toml` and an executable), and the pin it matches.
    fn bundled_notes(dir: &std::path::Path, version: &str, contents: &[u8]) -> DefaultPlugin {
        use flate2::{Compression, write::GzEncoder};
        let bundle = dir.join("bundle");
        std::fs::create_dir_all(&bundle).unwrap();
        let archive_path = bundle.join(crate::bundled::bundled_archive_name(
            "notes", version, LINUX,
        ));
        let mut archive = tar::Builder::new(GzEncoder::new(
            std::fs::File::create(&archive_path).unwrap(),
            Compression::default(),
        ));
        for (path, data) in [
            ("notes/plugin.toml", &b"name = \"notes\""[..]),
            ("notes/notes", contents),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            archive.append_data(&mut header, path, data).unwrap();
        }
        archive.into_inner().unwrap().finish().unwrap();
        let digest = crate::install::tests_sha256_file(&archive_path);
        let pins: &'static [(&'static str, &'static str)] =
            Box::leak(vec![(LINUX, &*Box::leak(digest.into_boxed_str()))].into_boxed_slice());
        DefaultPlugin {
            version: Box::leak(version.to_string().into_boxed_str()),
            sha256: pins,
            ..NOTES
        }
    }

    fn linux_options(
        root: &std::path::Path,
        bundle: Option<std::path::PathBuf>,
    ) -> PluginInstallOptions {
        PluginInstallOptions {
            store_root: root.to_path_buf(),
            install_root: root.join("installed"),
            // Never contacted: a default is installed only from the bundle.
            catalog_url: "http://127.0.0.1:9/unreachable".into(),
            target: crate::target::PluginTarget::from_os_arch("linux", "x86_64").unwrap(),
            bundled_plugins_dir: bundle,
        }
    }

    #[test]
    fn a_default_is_installed_from_the_bundled_copy_with_no_network() {
        let temp = tempfile::tempdir().unwrap();
        let notes = bundled_notes(temp.path(), "1.0.0", b"executable");
        let options = linux_options(temp.path(), Some(temp.path().join("bundle")));
        let outcomes =
            provision_bundled_defaults(&[notes], &BTreeSet::new(), &options, &mut |_| {});
        let [("notes", DefaultPluginOutcome::Installed(installed))] = outcomes.as_slice() else {
            panic!("{outcomes:?}");
        };
        assert!(installed.metadata.default_managed);
        assert_eq!(installed.metadata.installed_version, "1.0.0");
        assert!(installed.metadata.source_repository.starts_with("bundled:"));
        assert!(installed.metadata.install_path.join("notes").is_file());
        // Again: already current, nothing reinstalled.
        let again = provision_bundled_defaults(&[notes], &BTreeSet::new(), &options, &mut |_| {});
        assert!(matches!(
            again.as_slice(),
            [("notes", DefaultPluginOutcome::AlreadyCurrent)]
        ));
    }

    #[test]
    fn a_missing_bundled_copy_is_reported_and_nothing_is_downloaded() {
        let temp = tempfile::tempdir().unwrap();
        let notes = bundled_notes(temp.path(), "1.0.0", b"executable");
        for bundle in [None, Some(temp.path().join("elsewhere"))] {
            let options = linux_options(temp.path(), bundle);
            let mut events = 0;
            let outcomes =
                provision_bundled_defaults(&[notes], &BTreeSet::new(), &options, &mut |_| {
                    events += 1
                });
            assert!(matches!(
                outcomes.as_slice(),
                [("notes", DefaultPluginOutcome::NotBundled)]
            ));
            assert_eq!(events, 0, "nothing was fetched or extracted");
            assert!(
                PluginStore::new(temp.path())
                    .load_optional("notes")
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn a_bundled_copy_that_is_not_the_pinned_one_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let pinned = bundled_notes(temp.path(), "1.0.0", b"executable");
        // The bundle now holds a different archive under the same name.
        bundled_notes(temp.path(), "1.0.0", b"another executable");
        let options = linux_options(temp.path(), Some(temp.path().join("bundle")));
        let outcomes =
            provision_bundled_defaults(&[pinned], &BTreeSet::new(), &options, &mut |_| {});
        let [("notes", DefaultPluginOutcome::NotInstalled(reason))] = outcomes.as_slice() else {
            panic!("{outcomes:?}");
        };
        assert!(reason.contains("not the reviewed"), "{reason}");
        assert!(
            PluginStore::new(temp.path())
                .load_optional("notes")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_new_releases_bundled_copy_replaces_the_previous_default() {
        let temp = tempfile::tempdir().unwrap();
        let old = bundled_notes(temp.path(), "1.0.0", b"old executable");
        let options = linux_options(temp.path(), Some(temp.path().join("bundle")));
        provision_bundled_defaults(&[old], &BTreeSet::new(), &options, &mut |_| {});
        let new = bundled_notes(temp.path(), "1.1.0", b"new executable");
        let outcomes = provision_bundled_defaults(&[new], &BTreeSet::new(), &options, &mut |_| {});
        let [("notes", DefaultPluginOutcome::Installed(installed))] = outcomes.as_slice() else {
            panic!("{outcomes:?}");
        };
        assert_eq!(installed.metadata.installed_version, "1.1.0");
        assert_eq!(
            std::fs::read(installed.metadata.install_path.join("notes")).unwrap(),
            b"new executable"
        );
    }

    #[test]
    fn a_default_disabled_after_a_delete_stays_off() {
        let temp = tempfile::tempdir().unwrap();
        let notes = bundled_notes(temp.path(), "1.0.0", b"executable");
        let options = linux_options(temp.path(), Some(temp.path().join("bundle")));
        provision_bundled_defaults(&[notes], &BTreeSet::new(), &options, &mut |_| {});
        let store = PluginStore::new(temp.path());

        // plugins delete records the default as turned off; it stays removed.
        store.delete("notes").unwrap();
        store.set_default_turned_off("notes", true).unwrap();
        assert!(
            store.list().unwrap().is_empty(),
            "a turned-off default is not listed"
        );
        let mut events = 0;
        let outcomes =
            provision_bundled_defaults(&[notes], &BTreeSet::new(), &options, &mut |_| events += 1);
        assert!(matches!(
            outcomes.as_slice(),
            [("notes", DefaultPluginOutcome::TurnedOff)]
        ));
        assert_eq!(events, 0, "nothing was installed");
        assert!(store.load_optional("notes").unwrap().is_none());

        // plugins enable undoes it, and the next provisioning installs it.
        store.set_default_turned_off("notes", false).unwrap();
        let outcomes =
            provision_bundled_defaults(&[notes], &BTreeSet::new(), &options, &mut |_| {});
        assert!(matches!(
            outcomes.as_slice(),
            [("notes", DefaultPluginOutcome::Installed(_))]
        ));
    }

    #[test]
    fn an_installed_default_with_a_stale_turned_off_record_is_handled_normally() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store.set_default_turned_off("notes", true).unwrap();
        let mut record = installed("v1.0.0", true, true);
        record.install_path = temp.path().join("installed").join("notes");
        store.save(&record).unwrap();
        let options = linux_options(temp.path(), None);
        let outcomes =
            provision_bundled_defaults(&[NOTES], &BTreeSet::new(), &options, &mut |_| {});
        assert!(matches!(
            outcomes.as_slice(),
            [("notes", DefaultPluginOutcome::AlreadyCurrent)]
        ));
    }

    #[test]
    fn turning_defaults_off_disables_a_managed_copy_and_records_a_missing_one() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let managed = DefaultPlugin {
            name: "managed",
            ..NOTES
        };
        let operators = DefaultPlugin {
            name: "operators",
            ..NOTES
        };
        let missing = DefaultPlugin {
            name: "missing",
            ..NOTES
        };
        for (name, default_managed) in [("managed", true), ("operators", false)] {
            let mut record = installed("1.0.0", default_managed, true);
            record.name = name.into();
            record.install_path = temp.path().join("installed").join(name);
            store.save(&record).unwrap();
        }
        let turned = turn_off_defaults(&[managed, operators, missing], &store).unwrap();
        assert_eq!(turned, ["managed", "missing"]);
        assert!(!store.load("managed").unwrap().enabled);
        assert!(
            store.load("operators").unwrap().enabled,
            "an operator's own install is left alone"
        );
        assert!(store.default_turned_off("missing"));
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

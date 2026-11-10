//! What a host upgrade does with every installed plugin.
//!
//! The inventory is every record in the plugin store: the recommendation
//! status and set membership of a plugin never filter it, because a required
//! repair applies no matter how the plugin was chosen. Compatibility is
//! judged against the *target* host version from a consistent catalog
//! snapshot, using what the installed release recorded at install time and
//! the plugin protocol version it last negotiated; a catalog entry is the
//! only automatic update source, because its pin and digest are immutable.
//!
//! Unattended and attended callers share this: the plan says what must be
//! repaired, what may be updated, and what could not be established, and the
//! caller renders it for its audience.

use anyhow::Result;

use crate::catalog::PluginCatalog;
use crate::compatibility::{self as compat, Verdict};
use crate::install::{InstallOutcome, PluginInstallOptions, PluginProgressReporter};
use crate::store::{InstalledPluginMetadata, PluginStore};

/// What an upgrade can do with one installed plugin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginUpgradeVerdict {
    /// Compatible with the target host at its installed version, and the
    /// catalog pins nothing newer for this platform.
    KeepCurrent,
    /// Compatible, and a newer catalog-pinned release exists. Never applied
    /// automatically; the operator decides with `plugins update NAME`.
    UpdateAvailable { to: String },
    /// Incompatible with the target host, and a compatible catalog-pinned
    /// release can replace it. Applied automatically on every host upgrade
    /// path, attended or not.
    Repair { to: String, reason: String },
    /// Incompatible, and no verified usable update exists. The plugin must
    /// not run under the new host; the operator gets actionable guidance.
    RepairBlocked { reason: String },
    /// Compatibility with the target host cannot be established from what is
    /// recorded. No speculative updates; the host's existing checks still
    /// guard the plugin at launch.
    Unknown { reason: String },
}

/// One installed plugin's upgrade check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginUpgradeCheck {
    pub name: String,
    pub installed_version: String,
    /// Whether the installed plugin runs disabled; a repair keeps it
    /// disabled.
    pub enabled: bool,
    pub verdict: PluginUpgradeVerdict,
}

impl PluginUpgradeCheck {
    /// Whether this plugin blocks the managed upgrade until it is repaired
    /// or the operator intervenes.
    pub fn requires_repair(&self) -> bool {
        matches!(
            self.verdict,
            PluginUpgradeVerdict::Repair { .. } | PluginUpgradeVerdict::RepairBlocked { .. }
        )
    }
}

/// The check of every installed plugin against one target host version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginUpgradePlan {
    pub checks: Vec<PluginUpgradeCheck>,
    /// Whether the catalog could be read. `false` means absence of repair
    /// plans proves nothing: compatibility must then be established locally,
    /// or the upgrade stays unapplied.
    pub catalog_available: bool,
}

impl PluginUpgradePlan {
    pub fn required_repairs(&self) -> Vec<&PluginUpgradeCheck> {
        self.checks
            .iter()
            .filter(|check| check.requires_repair())
            .collect()
    }

    /// Compatible installed plugins with a newer catalog release: the
    /// attended notices, which list plugins inside and outside every set.
    pub fn update_notices(&self) -> Vec<&PluginUpgradeCheck> {
        self.checks
            .iter()
            .filter(|check| matches!(check.verdict, PluginUpgradeVerdict::UpdateAvailable { .. }))
            .collect()
    }

    /// Whether the catalog being unavailable leaves any plugin whose
    /// compatibility with the target cannot be established locally. Such an
    /// upgrade must not be applied: "no repairs found" would be a guess.
    pub fn has_unestablished_compatibility(&self) -> bool {
        self.checks
            .iter()
            .any(|check| matches!(check.verdict, PluginUpgradeVerdict::Unknown { .. }))
    }
}

/// Judge every installed plugin against `target_host_version`, resolving
/// updates from one catalog snapshot. `platform_target` is the target triple
/// of the machine doing the upgrade (a release must publish an archive for
/// it before anything can be installed), and `plugin_protocol_version` is
/// the protocol the *target* host speaks; a plugin whose last negotiated
/// version differs from it cannot work with that host, whatever its version
/// metadata says.
///
/// `catalog` is `None` when the catalog could not be fetched: then only what
/// the installed records retain locally can establish compatibility, and
/// anything else lands as [`PluginUpgradeVerdict::Unknown`].
pub fn plan_plugin_upgrades(
    store: &PluginStore,
    catalog: Option<&PluginCatalog>,
    platform_target: &str,
    target_host_version: &str,
    plugin_protocol_version: u32,
) -> Result<PluginUpgradePlan> {
    let installed = store.list()?;
    let mut checks = Vec::new();
    for metadata in &installed {
        checks.push(check_plugin(
            metadata,
            catalog,
            platform_target,
            target_host_version,
            plugin_protocol_version,
        ));
    }
    Ok(PluginUpgradePlan {
        checks,
        catalog_available: catalog.is_some(),
    })
}

fn check_plugin(
    metadata: &InstalledPluginMetadata,
    catalog: Option<&PluginCatalog>,
    platform_target: &str,
    target_host_version: &str,
    plugin_protocol_version: u32,
) -> PluginUpgradeCheck {
    let base = PluginUpgradeCheck {
        name: metadata.name.clone(),
        installed_version: metadata.installed_version.clone(),
        enabled: metadata.enabled,
        verdict: PluginUpgradeVerdict::KeepCurrent,
    };
    // A plugin whose protocol cannot work with the target host is
    // incompatible no matter what its version bounds say. An unrecorded
    // protocol version says nothing: negotiation at launch is the existing
    // check.
    let protocol_incompatible = metadata
        .last_protocol_version
        .is_some_and(|version| version != plugin_protocol_version);
    let retained_verdict = match (
        protocol_incompatible,
        compat::evaluate(metadata.host_compatibility.as_ref(), target_host_version),
    ) {
        (true, _) => Some(Verdict::Incompatible(format!(
            "plugin protocol {} does not match the host's {}",
            metadata.last_protocol_version.unwrap_or_default(),
            plugin_protocol_version
        ))),
        (false, verdict) => Some(verdict),
    };

    let Some(catalog) = catalog else {
        // Without a catalog, only local evidence counts. Established and
        // compatible stays; anything else is unknown, never "no updates".
        return match retained_verdict {
            Some(Verdict::Compatible) => base,
            Some(Verdict::Incompatible(reason)) => PluginUpgradeCheck {
                verdict: PluginUpgradeVerdict::RepairBlocked {
                    reason: format!(
                        "{reason}; the plugin catalog is unavailable, so no verified update \
                         source can be checked"
                    ),
                },
                ..base
            },
            Some(Verdict::Unknown(reason)) => PluginUpgradeCheck {
                verdict: PluginUpgradeVerdict::Unknown {
                    reason: format!(
                        "{reason}; the plugin catalog is unavailable, so compatibility with \
                         the target cannot be established"
                    ),
                },
                ..base
            },
            None => base,
        };
    };

    let Some(entry) = catalog.find_exact(&metadata.name) else {
        // No catalog entry: no verified update source and no release to
        // notice. Incompatibility is still known from local evidence.
        return match retained_verdict {
            Some(Verdict::Incompatible(reason)) => PluginUpgradeCheck {
                verdict: PluginUpgradeVerdict::RepairBlocked {
                    reason: format!(
                        "{reason}; '{}' is not in the plugin catalog, so there is no verified \
                         update source for it",
                        metadata.name
                    ),
                },
                ..base
            },
            _ => base,
        };
    };

    let pin = match entry.pinned_release(platform_target) {
        Ok(pin) => pin,
        Err(error) => {
            // No archive for this platform, or the entry pins nothing
            // installable. Unavailable is unavailable, for repair and for
            // notices alike.
            return match retained_verdict {
                Some(Verdict::Incompatible(reason)) => PluginUpgradeCheck {
                    verdict: PluginUpgradeVerdict::RepairBlocked {
                        reason: format!("{reason}; {}", error),
                    },
                    ..base
                },
                _ => base,
            };
        }
    };

    // The catalog pin is the newest release this host can act on. Its
    // compatibility decides whether it is a usable repair target or an
    // update notice.
    let release_compatible = |host: &str| {
        entry
            .compatibility
            .as_ref()
            .map(|declared| compat::evaluate(Some(declared), host))
            .unwrap_or(Verdict::Unknown(
                "entry declares no host compatibility".into(),
            ))
    };
    let pin_compatible_with_target = matches!(
        release_compatible(target_host_version),
        Verdict::Compatible | Verdict::Unknown(_)
    );
    let pin_version = pin.version.to_string();

    match retained_verdict {
        Some(Verdict::Incompatible(reason)) => {
            if pin_compatible_with_target {
                PluginUpgradeCheck {
                    verdict: PluginUpgradeVerdict::Repair {
                        to: pin_version,
                        reason,
                    },
                    ..base
                }
            } else {
                PluginUpgradeCheck {
                    verdict: PluginUpgradeVerdict::RepairBlocked {
                        reason: format!(
                            "{reason}; the catalog's release {pin_version} does not support \
                             MeshLLM {target_host_version}, and no other verified source exists"
                        ),
                    },
                    ..base
                }
            }
        }
        Some(Verdict::Unknown(reason)) => PluginUpgradeCheck {
            // Report unknown status and apply the existing checks; never
            // speculate with an automatic or advertised update.
            verdict: PluginUpgradeVerdict::Unknown { reason },
            ..base
        },
        // Compatible with the target. A compatible newer pin is a notice,
        // never an automatic update: the existence of a newer release is not
        // incompatibility. Unknown compatibility is reported as unknown; its
        // existing runtime checks still guard it, and nothing speculative is
        // installed or advertised.
        _ => {
            let installed_newer_or_equal =
                !crate::source_ref::version_string_newer(&pin_version, &metadata.installed_version);
            if installed_newer_or_equal {
                base
            } else {
                PluginUpgradeCheck {
                    verdict: PluginUpgradeVerdict::UpdateAvailable { to: pin_version },
                    ..base
                }
            }
        }
    }
}

/// Apply every planned repair: replace each incompatible managed plugin with
/// its verified compatible catalog release, preserving settings, data, and
/// enabled state. A failure is reported per plugin and stops nothing else,
/// so the caller can show every blocker; any failure means the managed
/// upgrade must not continue.
pub async fn repair_incompatible_plugins(
    plan: &PluginUpgradePlan,
    target_host_version: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<Vec<(String, InstallOutcome)>> {
    let store = PluginStore::new(&options.store_root);
    let mut repaired = Vec::new();
    for check in plan.required_repairs() {
        let PluginUpgradeVerdict::Repair { to, .. } = &check.verdict else {
            continue;
        };
        let current = store.load(&check.name)?;
        let outcome = crate::install::update_plugin_from_catalog(
            &current,
            target_host_version,
            options,
            progress,
        )
        .await?;
        if outcome.metadata.installed_version != *to {
            // The installer lands exactly the pin it verified; anything else
            // is a bug we refuse to paper over.
            anyhow::bail!(
                "repaired {} to {} but planned {}; refusing to continue",
                check.name,
                outcome.metadata.installed_version,
                to
            );
        }
        repaired.push((check.name.clone(), outcome));
    }
    Ok(repaired)
}

#[cfg(test)]
mod tests {
    use super::*;
    const TARGET: &str = "x86_64-unknown-linux-gnu";
    const PROTOCOL: u32 = 3;

    fn catalog(json: &str) -> PluginCatalog {
        PluginCatalog::parse_jsonl(json).unwrap()
    }

    fn installed(store: &PluginStore, name: &str, version: &str) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: name.into(),
            source_repository: format!("https://github.com/mesh-llm/{name}"),
            installed_version: version.into(),
            target_triple: TARGET.into(),
            downloaded_asset_name: format!("{name}.tar.gz"),
            install_path: store.root().join(name),
            enabled: true,
            default_managed: false,
            // Declared compatible with anything recent, so a test that wants
            // unknown compatibility overrides it explicitly.
            host_compatibility: Some(crate::compatibility::HostCompatibility {
                min: Some("0.1.0".into()),
                max: None,
            }),
            allowed_surfaces: None,
            manifest: None,
            last_protocol_version: Some(PROTOCOL),
            last_status: None,
            last_error: None,
            verified_sha256: None,
        }
    }

    fn entry(name: &str, version: &str, compatibility: &str) -> String {
        format!(
            r#"{{"name":"{name}","description":"{name}","github_url":"https://github.com/mesh-llm/{name}","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"{version}","sha256":{{"{TARGET}":"abababababababababababababababababababababababababababababababab"}}{compatibility}}}"#
        )
    }

    fn with_compatibility(
        metadata: &mut InstalledPluginMetadata,
        min: Option<&str>,
        max: Option<&str>,
    ) {
        metadata.host_compatibility = Some(crate::compatibility::HostCompatibility {
            min: min.map(str::to_string),
            max: max.map(str::to_string),
        });
    }

    #[test]
    fn an_incompatible_plugin_with_a_compatible_pin_is_a_repair() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut plugin = installed(&store, "blackboard", "v0.9.0");
        with_compatibility(&mut plugin, Some("1.0.0"), None);
        store.save(&plugin).unwrap();
        let catalog = catalog(&format!(
            "{}\n",
            entry(
                "blackboard",
                "v1.1.0",
                r#","compatibility":{"min":"0.1.0"}"#
            )
        ));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.81.0", PROTOCOL).unwrap();
        assert_eq!(plan.checks.len(), 1);
        assert_eq!(
            plan.checks[0].verdict,
            PluginUpgradeVerdict::Repair {
                to: "v1.1.0".into(),
                reason: "requires MeshLLM 1.0.0 or newer (host is 0.81.0)".into(),
            }
        );
        assert_eq!(plan.required_repairs().len(), 1);
        assert!(
            plan.checks[0].enabled,
            "enabled state is reported for the repair"
        );
    }

    #[test]
    fn a_protocol_mismatch_is_incompatible_regardless_of_version_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut plugin = installed(&store, "oldwire", "v2.0.0");
        // Declares itself fine with any host version…
        plugin.last_protocol_version = Some(PROTOCOL - 1);
        store.save(&plugin).unwrap();
        let catalog = catalog(&format!(
            "{}\n",
            entry("oldwire", "v3.0.0", r#","compatibility":{"min":"0.1.0"}"#)
        ));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.90.0", PROTOCOL).unwrap();
        let PluginUpgradeVerdict::Repair { reason, .. } = &plan.checks[0].verdict else {
            panic!("{:?}", plan.checks[0].verdict);
        };
        assert!(reason.contains("plugin protocol"), "{reason}");
    }

    #[test]
    fn a_compatible_plugin_with_a_newer_pin_is_a_notice_not_a_repair() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store
            .save(&installed(&store, "capsules", "v0.2.0"))
            .unwrap();
        let catalog = catalog(&format!(
            "{}\n",
            entry("capsules", "v0.2.1", r#","compatibility":{"min":"0.1.0"}"#)
        ));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.81.0", PROTOCOL).unwrap();
        assert_eq!(
            plan.checks[0].verdict,
            PluginUpgradeVerdict::UpdateAvailable {
                to: "v0.2.1".into(),
            }
        );
        assert!(plan.required_repairs().is_empty());
        assert_eq!(plan.update_notices().len(), 1);
    }

    #[test]
    fn a_plugin_outside_every_set_is_checked_the_same_way() {
        // Membership must not filter the inventory: a plugin in no set with
        // a newer release gets the same notice as a recommended one.
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store
            .save(&installed(&store, "blackboard", "v1.0.0"))
            .unwrap();
        store
            .save(&installed(&store, "capsules", "v0.2.0"))
            .unwrap();
        let catalog = catalog(&format!(
            "{}\n{}\n{{\"name\":\"recommended\",\"description\":\"Recommended\",\"plugins\":[\"capsules\"]}}",
            entry("capsules", "v0.2.1", r#","compatibility":{"min":"0.1.0"}"#),
            entry(
                "blackboard",
                "v1.1.0",
                r#","compatibility":{"min":"0.1.0"}"#
            ),
        ));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.81.0", PROTOCOL).unwrap();
        let noticed: Vec<&str> = plan
            .update_notices()
            .iter()
            .map(|check| check.name.as_str())
            .collect();
        assert_eq!(
            noticed,
            ["blackboard", "capsules"],
            "set membership does not filter"
        );
    }

    #[test]
    fn an_incompatible_plugin_with_no_usable_update_is_blocked() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut plugin = installed(&store, "legacy", "v1.0.0");
        with_compatibility(&mut plugin, Some("0.95.0"), None);
        store.save(&plugin).unwrap();
        // The catalog's only release is too old for the target host.
        let catalog = catalog(&format!(
            "{}\n",
            entry("legacy", "v1.0.0", r#","compatibility":{"max":"0.80.0"}"#)
        ));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.81.0", PROTOCOL).unwrap();
        let PluginUpgradeVerdict::RepairBlocked { reason } = &plan.checks[0].verdict else {
            panic!("{:?}", plan.checks[0].verdict);
        };
        assert!(
            reason.contains("does not support MeshLLM 0.81.0"),
            "{reason}"
        );
    }

    #[test]
    fn an_incompatible_plugin_without_a_catalog_entry_is_blocked_with_guidance() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut plugin = installed(&store, "indie", "v1.0.0");
        with_compatibility(&mut plugin, Some("0.99.0"), None);
        store.save(&plugin).unwrap();
        let catalog = catalog("");
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.90.0", PROTOCOL).unwrap();
        let PluginUpgradeVerdict::RepairBlocked { reason } = &plan.checks[0].verdict else {
            panic!("{:?}", plan.checks[0].verdict);
        };
        assert!(
            reason.contains("no catalog entry") || reason.contains("not in the plugin catalog"),
            "{reason}"
        );
    }

    #[test]
    fn unknown_compatibility_is_reported_and_speculates_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        // A legacy record with no compatibility metadata at all.
        let mut plugin = installed(&store, "legacy", "v1.0.0");
        plugin.host_compatibility = None;
        plugin.last_protocol_version = None;
        store.save(&plugin).unwrap();
        let catalog = catalog(&format!(
            "{}\n",
            entry("legacy", "v9.0.0", r#","compatibility":{"min":"0.1.0"}"#)
        ));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.90.0", PROTOCOL).unwrap();
        let PluginUpgradeVerdict::Unknown { reason } = &plan.checks[0].verdict else {
            panic!("{:?}", plan.checks[0].verdict);
        };
        assert!(
            reason.contains("no recorded host compatibility"),
            "{reason}"
        );
        assert!(plan.update_notices().is_empty(), "no speculative updates");
    }

    #[test]
    fn a_compatible_plugin_stays_at_its_version_when_the_catalog_cannot_be_read() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut plugin = installed(&store, "fine", "v1.0.0");
        with_compatibility(&mut plugin, Some("0.1.0"), None);
        store.save(&plugin).unwrap();
        let plan = plan_plugin_upgrades(&store, None, TARGET, "0.90.0", PROTOCOL).unwrap();
        assert!(!plan.catalog_available);
        assert_eq!(plan.checks[0].verdict, PluginUpgradeVerdict::KeepCurrent);
        assert!(!plan.has_unestablished_compatibility());
    }

    #[test]
    fn an_unknown_compatibility_with_no_catalog_stops_a_managed_upgrade() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut plugin = installed(&store, "legacy", "v1.0.0");
        plugin.host_compatibility = None;
        plugin.last_protocol_version = None;
        store.save(&plugin).unwrap();
        let plan = plan_plugin_upgrades(&store, None, TARGET, "0.90.0", PROTOCOL).unwrap();
        assert!(
            plan.has_unestablished_compatibility(),
            "compatibility cannot be established locally, so the upgrade must not claim success"
        );
        assert_eq!(plan.required_repairs().len(), 0);
    }

    #[test]
    fn an_incompatible_plugin_with_no_catalog_blocks_without_promising_updates() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut plugin = installed(&store, "broken", "v1.0.0");
        with_compatibility(&mut plugin, Some("0.99.0"), None);
        store.save(&plugin).unwrap();
        let plan = plan_plugin_upgrades(&store, None, TARGET, "0.90.0", PROTOCOL).unwrap();
        let PluginUpgradeVerdict::RepairBlocked { reason } = &plan.checks[0].verdict else {
            panic!("{:?}", plan.checks[0].verdict);
        };
        assert!(reason.contains("catalog is unavailable"), "{reason}");
    }

    #[test]
    fn a_disabled_incompatible_plugin_is_repaired_and_stays_disabled() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut plugin = installed(&store, "parked", "v1.0.0");
        plugin.enabled = false;
        with_compatibility(&mut plugin, Some("1.0.0"), None);
        store.save(&plugin).unwrap();
        let catalog = catalog(&format!(
            "{}\n",
            entry("parked", "v2.0.0", r#","compatibility":{"min":"0.1.0"}"#)
        ));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.90.0", PROTOCOL).unwrap();
        assert!(!plan.checks[0].enabled);
        assert_eq!(
            plan.checks[0].verdict,
            PluginUpgradeVerdict::Repair {
                to: "v2.0.0".into(),
                reason: "requires MeshLLM 1.0.0 or newer (host is 0.90.0)".into(),
            }
        );
    }

    #[test]
    fn a_newer_pin_for_an_unsupported_platform_is_not_an_update_notice() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store
            .save(&installed(&store, "capsules", "v0.2.0"))
            .unwrap();
        // The pin exists only for Windows; this platform has no release.
        let catalog = catalog(&format!(
            r#"{{"name":"capsules","description":"capsules","github_url":"https://github.com/mesh-llm/capsules","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v0.3.0","sha256":{{"x86_64-pc-windows-msvc":"{}"}},"compatibility":{{"min":"0.1.0"}}}}"#,
            "ab".repeat(32)
        ));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.90.0", PROTOCOL).unwrap();
        assert_eq!(plan.checks[0].verdict, PluginUpgradeVerdict::KeepCurrent);
    }

    #[test]
    fn a_host_at_the_pinned_version_is_not_told_to_update() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store
            .save(&installed(&store, "capsules", "v0.2.1"))
            .unwrap();
        let catalog = catalog(&format!("{}\n", entry("capsules", "v0.2.1", r#""#)));
        let plan =
            plan_plugin_upgrades(&store, Some(&catalog), TARGET, "0.90.0", PROTOCOL).unwrap();
        assert_eq!(plan.checks[0].verdict, PluginUpgradeVerdict::KeepCurrent);
    }

    #[test]
    fn an_empty_store_produces_no_checks_and_never_blocks() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let plan = plan_plugin_upgrades(&store, None, TARGET, "0.90.0", PROTOCOL).unwrap();
        assert!(plan.checks.is_empty());
        assert!(!plan.has_unestablished_compatibility());
    }
}

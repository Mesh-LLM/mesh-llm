//! Catalog-defined plugin sets: resolving what a named set means on this
//! machine, and installing its current eligible members once.
//!
//! A set resolves against one catalog snapshot per operation. Installing a
//! set installs its current missing members and does not subscribe the
//! machine to future membership changes: adding or removing a member later
//! never silently installs or uninstalls anything here. Set membership grants
//! no permissions and never enles, disles, or updates an installed
//! plugin.

use anyhow::{Context, Result};

use crate::catalog::{CatalogEntry, CatalogSet, PluginCatalog};
use crate::compatibility::{self as compat, Verdict};
use crate::install::{
    InstallOutcome, PluginInstallOptions, PluginProgressReporter, install_catalog_entry,
    install_pinned_entry,
};
use crate::store::PluginStore;

/// What one set member looks like on this machine, judged against one catalog
/// snapshot, the platform, and the host version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetMemberStatus {
    pub name: String,
    pub description: String,
    /// The release the catalog pins for this member, when it has a pin.
    pub version: Option<String>,
    pub availability: MemberAvailability,
}

/// Whether a member can be installed here, and why not when it cannot.
/// Incompatible and unverified releases display as unavailable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemberAvailability {
    Available,
    Unavailable(String),
}

/// A resolved set: every member and how it reads on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetResolution {
    pub name: String,
    pub description: String,
    pub members: Vec<SetMemberStatus>,
}

/// Resolve `set_name` from one catalog snapshot. Unknown sets are an error,
/// never an empty resolution.
pub fn resolve_set(
    catalog: &PluginCatalog,
    set_name: &str,
    target: &str,
    host_version: &str,
) -> Result<SetResolution> {
    let set = catalog.find_set(set_name).with_context(|| {
        format!("Unknown plugin set '{set_name}'. List availle sets: mesh-llm plugins sets list")
    })?;
    Ok(resolve_members(set, catalog, target, host_version))
}

fn resolve_members(
    set: &CatalogSet,
    catalog: &PluginCatalog,
    target: &str,
    host_version: &str,
) -> SetResolution {
    let members = set
        .plugins
        .iter()
        .map(|member| {
            let entry = catalog.find_exact(member);
            member_status(entry, member, target, host_version)
        })
        .collect();
    SetResolution {
        name: set.name.clone(),
        description: set.description.clone(),
        members,
    }
}

fn member_status(
    entry: Option<&CatalogEntry>,
    name: &str,
    target: &str,
    host_version: &str,
) -> SetMemberStatus {
    let Some(entry) = entry else {
        return SetMemberStatus {
            name: name.to_string(),
            description: String::new(),
            version: None,
            availability: MemberAvailability::Unavailable(
                "no catalog entry for this name".to_string(),
            ),
        };
    };
    let status = SetMemberStatus {
        name: entry.name.clone(),
        description: entry.description.clone(),
        version: entry.version.clone(),
        availability: MemberAvailability::Available,
    };
    match entry.pinned_release(target) {
        Err(error) => SetMemberStatus {
            availability: MemberAvailability::Unavailable(unavailable_reason(
                &error.to_string(),
                target,
            )),
            ..status
        },
        Ok(pin) => {
            if let Some(Verdict::Incompatible(reason)) = entry
                .compatibility
                .as_ref()
                .map(|declared| compat::evaluate(Some(declared), host_version))
            {
                return SetMemberStatus {
                    availability: MemberAvailability::Unavailable(format!(
                        "release {} does not support MeshLLM {host_version} ({reason})",
                        pin.version
                    )),
                    ..status
                };
            }
            status
        }
    }
}

/// The operator-facing reason a member is unavailable. A missing per-platform
/// pin is reported as exactly that, the way the release publishing would read.
fn unavailable_reason(error: &str, target: &str) -> String {
    if error.contains("no sha256") {
        format!("No release for {target}")
    } else {
        error.to_string()
    }
}

/// The one-line description an attended offer shows for a set, when the set
/// is worth offering: members and their resolvable versions. `None` when the
/// set is absent or has no members.
pub fn set_offer_summary(
    catalog: &PluginCatalog,
    set_name: &str,
) -> Option<(String, Vec<(String, String)>)> {
    let set = catalog.find_set(set_name)?;
    let members = set
        .plugins
        .iter()
        .map(|member| {
            let version = catalog
                .find_exact(member)
                .and_then(|entry| entry.version.clone())
                .unwrap_or_else(|| "unversioned".to_string());
            (member.clone(), version)
        })
        .collect();
    Some((set.description.clone(), members))
}

/// The plan installing a resolved set produces: the members to fetch, and
/// the members left as they are, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetInstallOutcome {
    pub members: Vec<MemberInstall>,
    pub skipped: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemberInstall {
    ToInstall { name: String },
}

/// Decide which members of a resolved set are installed now. An installed
/// member is never duplicated or updated by a set install, whatever its
/// state: preserving disabled state, settings, and data is the store's job,
/// and set membership is not an update path.
pub fn plan_set_install(
    resolution: &SetResolution,
    store: &PluginStore,
) -> Result<SetInstallOutcome> {
    let mut members = Vec::new();
    let mut skipped = Vec::new();
    for member in &resolution.members {
        match &member.availability {
            MemberAvailability::Unavailable(reason) => {
                skipped.push((member.name.clone(), reason.clone()));
                continue;
            }
            MemberAvailability::Available => {}
        }
        match store.load_optional(&member.name)? {
            Some(installed) => {
                skipped.push((
                    member.name.clone(),
                    format!(
                        "already installed {}{}",
                        installed.installed_version,
                        if installed.enabled {
                            ""
                        } else {
                            " (disabled; set membership does not enable it)"
                        }
                    ),
                ));
            }
            None => members.push(MemberInstall::ToInstall {
                name: member.name.clone(),
            }),
        }
    }
    Ok(SetInstallOutcome { members, skipped })
}

/// Install the planned members of a set. Every member is attempted, so a
/// partial failure still reports per-plugin results; the caller decides what
/// a partial install means for its exit status. Retrying is safe: members
/// that landed are skipped as already installed.
pub async fn install_set(
    outcome: &SetInstallOutcome,
    catalog: &PluginCatalog,
    host_version: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> SetInstallResult {
    let mut installed = Vec::new();
    let mut failed = Vec::new();
    for member in &outcome.members {
        let MemberInstall::ToInstall { name } = member;
        let result = install_member(name, catalog, host_version, options, progress).await;
        match result {
            Ok(outcome) => installed.push(outcome),
            Err(error) => failed.push((name.clone(), format!("{error:#}"))),
        }
    }
    SetInstallResult {
        installed,
        failed,
        skipped: outcome.skipped.clone(),
    }
}

async fn install_member(
    name: &str,
    catalog: &PluginCatalog,
    host_version: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<InstallOutcome> {
    let entry = catalog
        .find_exact(name)
        .with_context(|| format!("plugin '{name}' was not found in the catalog"))?;
    let store = PluginStore::new(&options.store_root);
    match store.load_optional(name)? {
        Some(existing) => {
            install_pinned_entry(&existing, entry, host_version, options, progress).await
        }
        None => install_catalog_entry(entry, host_version, options, progress).await,
    }
}

/// What installing a set did, per member.
#[derive(Debug, Clone, PartialEq)]
pub struct SetInstallResult {
    pub installed: Vec<InstallOutcome>,
    pub failed: Vec<(String, String)>,
    pub skipped: Vec<(String, String)>,
}

impl SetInstallResult {
    pub fn installed_count(&self) -> usize {
        self.installed.len()
    }

    pub fn is_complete(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Install a named set end to end against one catalog snapshot: resolve,
/// plan, and install. Errors name the set so an unknown selection reads as a
/// usage error rather than an empty success.
pub async fn install_named_set(
    catalog: &PluginCatalog,
    set_name: &str,
    target: &str,
    host_version: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<SetInstallResult> {
    let resolution = resolve_set(catalog, set_name, target, host_version)?;
    let store = PluginStore::new(&options.store_root);
    let outcome = plan_set_install(&resolution, &store)?;
    Ok(install_set(&outcome, catalog, host_version, options, progress).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InstalledPluginMetadata;

    const CATALOG: &str = r#"{"name":"capsules","description":"Audit and provenance records","github_url":"https://github.com/Mesh-LLM/capsules","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v0.2.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"},"compatibility":{"min":"0.81.0"}}
{"name":"blackboard","description":"Shared notes","github_url":"https://github.com/mesh-llm/blackboard","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v1.1.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"}}
{"name":"windows-only","description":"A plugin with no Linux build","github_url":"https://github.com/mesh-llm/windows-only","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v2.0.0","sha256":{"x86_64-pc-windows-msvc":"abababababababababababababababababababababababababababababababab"}}
{"name":"stale","description":"A plugin the new host dropped","github_url":"https://github.com/mesh-llm/stale","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v3.0.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"},"compatibility":{"max":"0.80.0"}}
{"name":"recommended","description":"Recommended plugins for MeshLLM","plugins":["capsules","blackboard","windows-only","stale"]}
"#;

    const TARGET: &str = "x86_64-unknown-linux-gnu";

    fn catalog() -> PluginCatalog {
        PluginCatalog::parse_jsonl(CATALOG).unwrap()
    }

    fn resolution(host: &str) -> SetResolution {
        resolve_set(&catalog(), "recommended", TARGET, host).unwrap()
    }

    #[test]
    fn an_unknown_set_is_an_error_not_an_empty_resolution() {
        let error = resolve_set(&catalog(), "missing", TARGET, "0.90.0").unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("Unknown plugin set 'missing'"),
            "{message}"
        );
        assert!(message.contains("plugins sets list"), "{message}");
    }

    #[test]
    fn members_show_versions_and_platform_availability() {
        let resolution = resolution("0.90.0");
        let find = |name: &str| {
            resolution
                .members
                .iter()
                .find(|member| member.name == name)
                .unwrap()
        };
        assert_eq!(find("capsules").version.as_deref(), Some("v0.2.0"));
        assert_eq!(find("capsules").availability, MemberAvailability::Available);
        assert_eq!(
            find("windows-only").availability,
            MemberAvailability::Unavailable("No release for x86_64-unknown-linux-gnu".into())
        );
        assert!(
            find("stale")
                .availability
                .unavailable_reason()
                .contains("does not support MeshLLM 0.90.0"),
        );
    }

    impl MemberAvailability {
        fn unavailable_reason(&self) -> String {
            match self {
                MemberAvailability::Unavailable(reason) => reason.clone(),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn a_member_compatible_with_an_older_host_is_unavailable_for_the_new_one() {
        let resolution = resolution("0.80.0");
        let stale = resolution
            .members
            .iter()
            .find(|member| member.name == "stale")
            .unwrap();
        assert_eq!(stale.availability, MemberAvailability::Available);
    }

    #[test]
    fn installed_members_are_skipped_whatever_their_state() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        for (name, enabled) in [("capsules", true), ("blackboard", false)] {
            store
                .save(&InstalledPluginMetadata {
                    name: name.into(),
                    source_repository: format!("https://github.com/mesh-llm/{name}"),
                    installed_version: "v1.0.0".into(),
                    target_triple: TARGET.into(),
                    downloaded_asset_name: format!("{name}.tar.gz"),
                    install_path: temp.path().join(name),
                    enabled,
                    default_managed: false,
                    host_compatibility: None,
                    allowed_surfaces: None,
                    manifest: None,
                    last_protocol_version: None,
                    last_status: None,
                    last_error: None,
                    verified_sha256: None,
                })
                .unwrap();
        }
        let plan = plan_set_install(&resolution("0.90.0"), &store).unwrap();
        assert!(
            plan.members.is_empty(),
            "installed and unavailable members are not installed: {:?}",
            plan.members
        );
        let skipped: Vec<(String, String)> = plan.skipped;
        let find = |name: &str| {
            skipped
                .iter()
                .find(|(skipped, _)| skipped == name)
                .unwrap()
                .1
                .clone()
        };
        assert_eq!(find("capsules"), "already installed v1.0.0");
        assert!(
            find("blackboard").contains("disabled"),
            "a disabled member stays disabled: {}",
            find("blackboard")
        );
        assert!(
            find("windows-only").contains("No release"),
            "{}",
            find("windows-only")
        );
        assert!(
            find("stale").contains("does not support"),
            "{}",
            find("stale")
        );
    }

    #[test]
    fn the_offer_summary_names_the_set_members_when_worth_offering() {
        let (description, members) = set_offer_summary(&catalog(), "recommended").unwrap();
        assert_eq!(description, "Recommended plugins for MeshLLM");
        assert_eq!(members[0], ("capsules".into(), "v0.2.0".into()));
        assert!(set_offer_summary(&catalog(), "missing").is_none());
    }

    #[tokio::test]
    async fn every_member_is_attempted_so_a_failure_still_reports_the_rest() {
        // A set with two installable members where one fails to install:
        // `install_set` must continue past the failure.
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let outcome = SetInstallOutcome {
            members: vec![
                MemberInstall::ToInstall { name: "a".into() },
                MemberInstall::ToInstall { name: "b".into() },
            ],
            skipped: Vec::new(),
        };
        let catalog = PluginCatalog::parse_jsonl(
            r#"{"name":"a","description":"A","github_url":"https://github.com/mesh-llm/a","author_email":"m@example.com","author_name":"A","version":"v1.0.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"}}
{"name":"b","description":"B","github_url":"https://github.com/mesh-llm/b","author_email":"m@example.com","author_name":"B","version":"v1.0.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"}}"#,
        )
        .unwrap();
        // Both fail here (no GitHub reachable in tests); what matters is that
        // both were attempted and reported, so a retry is informed.
        let options = PluginInstallOptions {
            store_root: store.root().to_path_buf(),
            install_root: temp.path().join("installed"),
            catalog_url: "http://127.0.0.1:9/unreachable".into(),
            target: crate::target::PluginTarget::from_os_arch("linux", "x86_64").unwrap(),
            host_version: "0.90.0".to_string(),
        };
        let result = install_set(&outcome, &catalog, "0.90.0", &options, &mut |_| {}).await;
        assert_eq!(result.failed.len(), 2, "{result:?}");
        assert_eq!(result.failed[0].0, "a");
        assert_eq!(result.failed[1].0, "b");
        assert!(!result.is_complete());
        let _ = store;
    }
}

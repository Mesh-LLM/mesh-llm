//! Catalog plugin sets: resolving what a named set means on this machine and
//! installing its members once.
//!
//! A set resolves against one catalog snapshot per operation. Installing a
//! set installs the members that are missing and then stops caring: it does
//! not subscribe the machine to future membership changes, and it never
//! updates, enables or re-enables a plugin that is already installed.
//! Design adapted from the catalog-sets design in Mesh-LLM/mesh-llm#2978.

use anyhow::{Context, Result};

use crate::{
    catalog::{CatalogEntry, PluginCatalog},
    install::{
        InstallOutcome, PluginInstallOptions, PluginProgressReporter, install_catalog_entry,
    },
    store::PluginStore,
};

/// Whether one set member can be installed on this platform, and why not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemberAvailability {
    Available,
    Unavailable(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetMember {
    pub name: String,
    /// The release the catalog pins for this member, when it pins one.
    pub version: Option<String>,
    pub availability: MemberAvailability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetResolution {
    pub name: String,
    pub description: String,
    pub members: Vec<SetMember>,
}

/// What installing a set did. Every member is attempted, so one failure never
/// hides the others; the caller decides what a partial result means.
#[derive(Debug, Default)]
pub struct SetInstallResult {
    pub installed: Vec<InstallOutcome>,
    /// `(member, reason)` for members that were attempted and failed.
    pub failed: Vec<(String, String)>,
    /// `(member, reason)` for members left alone: unavailable here, or
    /// already installed.
    pub skipped: Vec<(String, String)>,
}

/// Resolve `set_name` from one catalog snapshot for `target`. An unknown set
/// is an error, never an empty resolution.
pub fn resolve_set(catalog: &PluginCatalog, set_name: &str, target: &str) -> Result<SetResolution> {
    let set = catalog.find_set(set_name).with_context(|| {
        let known: Vec<&str> = catalog.sets().iter().map(|set| set.name.as_str()).collect();
        if known.is_empty() {
            format!(
                "the plugin catalog publishes no plugin set named '{set_name}' (it lists no sets)"
            )
        } else {
            format!(
                "the plugin catalog publishes no plugin set named '{set_name}'; available sets: {}",
                known.join(", ")
            )
        }
    })?;
    let members = set
        .plugins
        .iter()
        .map(|member| member_status(catalog.find_exact(member), member, target))
        .collect();
    Ok(SetResolution {
        name: set.name.clone(),
        description: set.description.clone(),
        members,
    })
}

fn member_status(entry: Option<&CatalogEntry>, name: &str, target: &str) -> SetMember {
    let Some(entry) = entry else {
        return SetMember {
            name: name.to_string(),
            version: None,
            availability: MemberAvailability::Unavailable("not in the plugin catalog".to_string()),
        };
    };
    let availability = match entry.pinned_release(target) {
        Ok(_) => MemberAvailability::Available,
        Err(error) => MemberAvailability::Unavailable(format!("{error:#}")),
    };
    SetMember {
        name: entry.name.clone(),
        version: entry.version.clone(),
        availability,
    }
}

/// Install the members of `resolution` that are available and not yet
/// installed, one at a time through `install_member`, which receives the
/// shared progress reporter. Retrying is safe: members that landed are
/// skipped as already installed.
pub async fn install_set_with<P>(
    resolution: &SetResolution,
    store: &PluginStore,
    progress: &mut P,
    mut install_member: impl AsyncFnMut(&str, &mut P) -> Result<InstallOutcome>,
) -> Result<SetInstallResult> {
    let mut result = SetInstallResult::default();
    for member in &resolution.members {
        if let MemberAvailability::Unavailable(reason) = &member.availability {
            result.skipped.push((member.name.clone(), reason.clone()));
            continue;
        }
        if let Some(installed) = store.load_optional(&member.name)? {
            result.skipped.push((
                member.name.clone(),
                format!("already installed ({})", installed.installed_version),
            ));
            continue;
        }
        match install_member(&member.name, progress).await {
            Ok(outcome) => result.installed.push(outcome),
            Err(error) => result
                .failed
                .push((member.name.clone(), format!("{error:#}"))),
        }
    }
    Ok(result)
}

/// Install a set from one catalog snapshot: every member is installed at the
/// release that snapshot pins.
pub async fn install_set(
    catalog: &PluginCatalog,
    set_name: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<SetInstallResult> {
    let resolution = resolve_set(catalog, set_name, options.target.triple())?;
    let store = PluginStore::new(&options.store_root);
    install_set_with(&resolution, &store, progress, async |name, progress| {
        let entry = catalog
            .find_exact(name)
            .with_context(|| format!("'{name}' is not in the plugin catalog"))?;
        install_catalog_entry(entry, options, progress).await
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InstalledPluginMetadata;
    use anyhow::bail;
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::TcpListener,
        path::PathBuf,
        thread,
    };
    use tempfile::TempDir;

    const TARGET: &str = "x86_64-unknown-linux-gnu";
    const SHA: &str = "abababababababababababababababababababababababababababababababab";

    fn entry_line(name: &str, pinned: bool) -> String {
        let pin = if pinned {
            format!(r#","version":"v1.0.0","sha256":{{"{TARGET}":"{SHA}"}}"#)
        } else {
            String::new()
        };
        format!(
            r#"{{"name":"{name}","description":"{name} plugin","github_url":"https://github.com/acme/{name}","author_email":"dev@example.com","author_name":"Acme"{pin}}}"#
        )
    }

    fn catalog(entries: &[(&str, bool)], sets: &str) -> PluginCatalog {
        let body = entries
            .iter()
            .map(|(name, pinned)| entry_line(name, *pinned))
            .collect::<Vec<_>>()
            .join("\n");
        PluginCatalog::parse_jsonl(&body)
            .unwrap()
            .with_sets(PluginCatalog::parse_sets_jsonl(sets).unwrap())
            .unwrap()
    }

    const RECOMMENDED: &str =
        r#"{"name":"recommended","description":"Start here","plugins":["alpha","beta","gamma"]}"#;

    fn installed(name: &str) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: name.to_string(),
            source_repository: format!("https://github.com/acme/{name}"),
            installed_version: "v1.0.0".to_string(),
            target_triple: TARGET.to_string(),
            downloaded_asset_name: format!("{name}.tar.gz"),
            install_path: PathBuf::from("/nonexistent"),
            enabled: true,
            manifest: None,
            last_protocol_version: None,
            last_status: None,
            last_error: None,
            verified_sha256: None,
        }
    }

    #[test]
    fn resolving_reports_availability_per_member() {
        let catalog = catalog(
            &[("alpha", true), ("beta", false)],
            r#"{"name":"recommended","description":"d","plugins":["alpha","beta"]}"#,
        );
        let resolution = resolve_set(&catalog, "recommended", TARGET).unwrap();
        assert_eq!(
            resolution.members[0].availability,
            MemberAvailability::Available
        );
        assert!(matches!(
            &resolution.members[1].availability,
            MemberAvailability::Unavailable(reason) if reason.contains("pins no version")
        ));
        let other = resolve_set(&catalog, "recommended", "aarch64-apple-darwin").unwrap();
        assert!(matches!(
            &other.members[0].availability,
            MemberAvailability::Unavailable(reason) if reason.contains("no sha256")
        ));
    }

    #[test]
    fn an_unknown_set_is_an_error_naming_the_available_ones() {
        let catalog = catalog(
            &[("alpha", true)],
            r#"{"name":"recommended","description":"d","plugins":["alpha"]}"#,
        );
        let error = resolve_set(&catalog, "nope", TARGET)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("'nope'") && error.contains("recommended"),
            "{error}"
        );
        let none = PluginCatalog::parse_jsonl(&entry_line("alpha", true)).unwrap();
        let error = resolve_set(&none, "recommended", TARGET)
            .unwrap_err()
            .to_string();
        assert!(error.contains("lists no sets"), "{error}");
    }

    /// One member fails, one lands, one is unavailable: every member is still
    /// attempted, and a retry installs only the member that failed.
    #[tokio::test]
    async fn a_partial_install_reports_each_member_and_a_retry_skips_what_landed() {
        let catalog = catalog(
            &[("alpha", true), ("beta", true), ("gamma", false)],
            RECOMMENDED,
        );
        let resolution = resolve_set(&catalog, "recommended", TARGET).unwrap();
        let temp = TempDir::new().unwrap();
        let store = PluginStore::new(temp.path());
        let mut attempts = Vec::new();

        let first = install_set_with(&resolution, &store, &mut (), async |name, _| {
            attempts.push(name.to_string());
            if name == "beta" {
                bail!("download failed");
            }
            let metadata = installed(name);
            store.save(&metadata)?;
            Ok(InstallOutcome {
                metadata,
                changed: true,
            })
        })
        .await
        .unwrap();
        assert_eq!(attempts, ["alpha", "beta"]);
        assert_eq!(first.installed.len(), 1);
        assert_eq!(
            first.failed,
            [("beta".to_string(), "download failed".to_string())]
        );
        assert_eq!(first.skipped.len(), 1);
        assert_eq!(first.skipped[0].0, "gamma");

        let mut retried = Vec::new();
        let second = install_set_with(&resolution, &store, &mut (), async |name, _| {
            retried.push(name.to_string());
            let metadata = installed(name);
            store.save(&metadata)?;
            Ok(InstallOutcome {
                metadata,
                changed: true,
            })
        })
        .await
        .unwrap();
        assert_eq!(retried, ["beta"]);
        assert!(
            second
                .skipped
                .iter()
                .any(|(name, reason)| name == "alpha" && reason.contains("already installed"))
        );
    }

    #[tokio::test]
    async fn a_set_install_never_touches_an_installed_member() {
        let catalog = catalog(
            &[("alpha", true)],
            r#"{"name":"recommended","description":"d","plugins":["alpha"]}"#,
        );
        let resolution = resolve_set(&catalog, "recommended", TARGET).unwrap();
        let temp = TempDir::new().unwrap();
        let store = PluginStore::new(temp.path());
        let mut disabled = installed("alpha");
        disabled.enabled = false;
        store.save(&disabled).unwrap();
        let result = install_set_with(
            &resolution,
            &store,
            &mut (),
            async |_, _| -> Result<InstallOutcome> {
                panic!("an installed member must not be reinstalled")
            },
        )
        .await
        .unwrap();
        assert!(result.installed.is_empty() && result.failed.is_empty());
        assert!(!store.load("alpha").unwrap().enabled);
    }

    /// Serve `routes` (path -> body) from a local server so catalog fetches
    /// never reach the network; unlisted paths answer 404.
    fn serve(routes: &'static [(&'static str, &'static str)]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut header = String::new();
                while reader.read_line(&mut header).is_ok() && header.trim() != "" {
                    header.clear();
                }
                let path = request_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/")
                    .to_string();
                let found = routes.iter().find(|(route, _)| *route == path);
                let (status, body) = match found {
                    Some((_, body)) => ("200 OK", *body),
                    None => ("404 Not Found", ""),
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = reader.read(&mut [0u8; 0]);
            }
        });
        base
    }

    const PLUGINS: &str = r#"{"name":"alpha","description":"A","github_url":"https://github.com/acme/alpha","author_email":"d@example.com","author_name":"Acme"}"#;

    #[tokio::test]
    async fn fetching_reads_plugins_and_sets_from_the_catalog_location() {
        let base = serve(&[
            ("/plugins.jsonl", PLUGINS),
            (
                "/plugin-sets.jsonl",
                r#"{"name":"recommended","description":"d","plugins":["alpha"]}"#,
            ),
        ]);
        let plugins_url = format!("{base}/plugins.jsonl");
        let sets_url = crate::catalog::sets_url_for(&plugins_url);
        let (catalog, problem) = PluginCatalog::fetch_with_sets(
            &reqwest::Client::new(),
            &plugins_url,
            sets_url.as_deref(),
        )
        .await
        .unwrap();
        assert_eq!(problem, None);
        assert_eq!(catalog.find_set("recommended").unwrap().plugins, ["alpha"]);
    }

    /// A missing or broken sets file leaves a usable plugin catalog with no
    /// sets and says why; it never fails the plugin catalog read.
    #[tokio::test]
    async fn a_broken_sets_file_leaves_the_plugin_catalog_readable() {
        let base = serve(&[("/plugins.jsonl", PLUGINS)]);
        let plugins_url = format!("{base}/plugins.jsonl");
        let sets_url = crate::catalog::sets_url_for(&plugins_url);
        let (catalog, problem) = PluginCatalog::fetch_with_sets(
            &reqwest::Client::new(),
            &plugins_url,
            sets_url.as_deref(),
        )
        .await
        .unwrap();
        assert!(catalog.find_exact("alpha").is_some());
        assert!(catalog.sets().is_empty());
        assert!(problem.unwrap().contains("404"));
    }
}

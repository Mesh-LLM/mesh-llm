//! `plugins install --set` and the plugin step of `mesh-llm setup`: installing
//! a catalog plugin set against one catalog snapshot.

use std::time::Duration;

use anyhow::{Result, bail};
use mesh_llm_plugin_manager::{
    PluginCatalog, PluginInstallOptions, PluginProgressReporter,
    catalog::sets_url_for,
    sets::{SetInstallResult, install_set},
};
use reqwest::Client;

/// The set setup installs unless the operator opts out.
pub const DEFAULT_SET: &str = "recommended";

/// Bound on reading the catalog and its sets file: setup must not hang on a
/// catalog host that accepts a connection and never answers.
const CATALOG_TIMEOUT: Duration = Duration::from_secs(30);
/// Bound on the whole default install in setup (release lookups and
/// downloads for every member).
const SETUP_INSTALL_TIMEOUT: Duration = Duration::from_secs(600);

/// What setup's plugin step ended with. Setup never fails on it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum SetupPluginsOutcome {
    #[default]
    NotRequested,
    /// The set resolved. Members may have been installed, left as they were
    /// (already installed), found unavailable on this platform, or failed.
    Finished {
        installed: Vec<String>,
        unavailable: Vec<(String, String)>,
        failed: Vec<(String, String)>,
    },
    /// The set could not be read or the install did not finish; nothing
    /// further was attempted.
    Unavailable(String),
}

async fn read_catalog(
    options: &PluginInstallOptions,
    timeout: Duration,
) -> Result<(PluginCatalog, Option<String>)> {
    let client = Client::builder().timeout(timeout).build()?;
    let sets_url = sets_url_for(&options.catalog_url);
    PluginCatalog::fetch_with_sets(&client, &options.catalog_url, sets_url.as_deref()).await
}

/// Why a catalog without the requested set has none: the reason the sets file
/// could not be read, which names a custom catalog URL when that is the cause.
fn no_set_reason(catalog: &PluginCatalog, set_name: &str, problem: Option<String>) -> String {
    match problem {
        Some(problem) => format!("the plugin catalog has no set '{set_name}' ({problem})"),
        None => {
            let known: Vec<&str> = catalog.sets().iter().map(|set| set.name.as_str()).collect();
            format!(
                "the plugin catalog has no set '{set_name}'; available sets: {}",
                known.join(", ")
            )
        }
    }
}

/// `mesh-llm plugins install --set NAME`: install the set's missing members.
/// Fails when the set is unknown or any member fails, after attempting all.
pub async fn install_set_command(
    set_name: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<SetInstallResult> {
    let (catalog, problem) = read_catalog(options, CATALOG_TIMEOUT).await?;
    if catalog.find_set(set_name).is_none() {
        bail!("{}", no_set_reason(&catalog, set_name, problem));
    }
    install_set(&catalog, set_name, options, progress).await
}

/// Setup's plugin step. Every failure (offline, a slow catalog host, no such
/// set, a failing member) becomes an outcome the caller reports with a retry
/// command: an optional plugin never fails setup.
pub async fn install_default_set_for_setup(
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> SetupPluginsOutcome {
    install_set_for_setup(
        DEFAULT_SET,
        options,
        progress,
        CATALOG_TIMEOUT,
        SETUP_INSTALL_TIMEOUT,
    )
    .await
}

async fn install_set_for_setup(
    set_name: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
    catalog_timeout: Duration,
    install_timeout: Duration,
) -> SetupPluginsOutcome {
    let work = async {
        let (catalog, problem) = match read_catalog(options, catalog_timeout).await {
            Ok(read) => read,
            Err(error) => return SetupPluginsOutcome::Unavailable(format!("{error:#}")),
        };
        if catalog.find_set(set_name).is_none() {
            return SetupPluginsOutcome::Unavailable(no_set_reason(&catalog, set_name, problem));
        }
        match install_set(&catalog, set_name, options, progress).await {
            Ok(result) => SetupPluginsOutcome::Finished {
                installed: result
                    .installed
                    .into_iter()
                    .map(|outcome| outcome.metadata.name)
                    .collect(),
                unavailable: result.unavailable,
                failed: result.failed,
            },
            Err(error) => SetupPluginsOutcome::Unavailable(format!("{error:#}")),
        }
    };
    match tokio::time::timeout(install_timeout, work).await {
        Ok(outcome) => outcome,
        Err(_) => SetupPluginsOutcome::Unavailable(format!(
            "timed out after {}s; plugins that finished installing before then stay installed (`mesh-llm plugins list`)",
            install_timeout.as_secs()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
    };
    use tempfile::TempDir;

    fn options(catalog_url: String, root: &std::path::Path) -> PluginInstallOptions {
        PluginInstallOptions {
            store_root: root.to_path_buf(),
            install_root: root.join("installed"),
            catalog_url,
            target: mesh_llm_plugin_manager::PluginTarget::current().unwrap(),
        }
    }

    /// Answer each request from `routes`, or never answer when `stall`.
    fn serve(routes: &'static [(&'static str, &'static str)], stall: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        thread::spawn(move || {
            let mut held = Vec::new();
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                if stall {
                    held.push(stream);
                    continue;
                }
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let mut header = String::new();
                while reader.read_line(&mut header).is_ok() && !header.trim().is_empty() {
                    header.clear();
                }
                let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (status, body) = match routes.iter().find(|(route, _)| *route == path) {
                    Some((_, body)) => ("200 OK", *body),
                    None => ("404 Not Found", ""),
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        base
    }

    const PLUGINS: &str = r#"{"name":"alpha","description":"A","github_url":"https://github.com/acme/alpha","author_email":"d@example.com","author_name":"Acme"}"#;

    #[tokio::test]
    async fn a_stalled_catalog_host_ends_setup_s_plugin_step_with_an_outcome() {
        let temp = TempDir::new().unwrap();
        let base = serve(&[], true);
        let outcome = install_set_for_setup(
            "recommended",
            &options(format!("{base}/plugins.jsonl"), temp.path()),
            &mut |_| {},
            Duration::from_millis(200),
            Duration::from_secs(30),
        )
        .await;
        assert!(
            matches!(outcome, SetupPluginsOutcome::Unavailable(_)),
            "{outcome:?}"
        );
    }

    #[tokio::test]
    async fn a_catalog_without_the_set_is_an_outcome_not_a_failure() {
        let temp = TempDir::new().unwrap();
        let base = serve(&[("/plugins.jsonl", PLUGINS)], false);
        let outcome = install_set_for_setup(
            "recommended",
            &options(format!("{base}/plugins.jsonl"), temp.path()),
            &mut |_| {},
            CATALOG_TIMEOUT,
            SETUP_INSTALL_TIMEOUT,
        )
        .await;
        let SetupPluginsOutcome::Unavailable(reason) = outcome else {
            panic!("expected Unavailable");
        };
        assert!(reason.contains("no set 'recommended'"), "{reason}");
    }

    #[tokio::test]
    async fn a_custom_catalog_url_says_sets_are_unavailable_for_it() {
        let temp = TempDir::new().unwrap();
        let base = serve(&[("/custom.json", PLUGINS)], false);
        let error = install_set_command(
            "recommended",
            &options(format!("{base}/custom.json"), temp.path()),
            &mut |_| {},
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("no plugin-sets.jsonl beside the catalog URL"),
            "{error}"
        );
    }

    /// The members have no release for this platform, so nothing downloads:
    /// the set install ends cleanly with every member unavailable.
    #[tokio::test]
    async fn a_set_whose_members_have_no_release_here_installs_nothing() {
        let temp = TempDir::new().unwrap();
        let base = serve(
            &[
                ("/plugins.jsonl", PLUGINS),
                (
                    "/plugin-sets.jsonl",
                    r#"{"name":"recommended","description":"d","plugins":["alpha"]}"#,
                ),
            ],
            false,
        );
        let outcome = install_set_for_setup(
            "recommended",
            &options(format!("{base}/plugins.jsonl"), temp.path()),
            &mut |_| {},
            CATALOG_TIMEOUT,
            SETUP_INSTALL_TIMEOUT,
        )
        .await;
        let SetupPluginsOutcome::Finished {
            installed,
            unavailable,
            failed,
        } = outcome
        else {
            panic!("expected Finished");
        };
        assert!(installed.is_empty() && failed.is_empty());
        assert_eq!(unavailable.len(), 1);
        assert_eq!(unavailable[0].0, "alpha");
    }
}

//! The capsules plugin in a release (Mesh-LLM/mesh-llm#2514).
//!
//! - `repository capsules-plugin-archive --out DIR` builds the plugin from
//!   this checkout in release mode and packages it for this machine's target
//!   with `scripts/package.sh` (deterministic for a given executable), after
//!   checking that `plugin.toml` carries mesh-llm's version and that the
//!   executable passes the shipped-binary native policy, writing the archive and a
//!   `<archive>.sha256` sidecar. The release job runs it before the host build
//!   and passes the digest to that build (`MESH_LLM_BUNDLED_CAPSULES_SHA256`,
//!   read by `mesh-llm-plugin-manager`'s build script), so the host pins the
//!   archive made from its own commit. `--web-bundle FILE` takes the
//!   Evidence page the release builds once from the same commit, as the
//!   plugin's whole `bundle/`, instead of running pnpm here.
//! - `repository capsules-bundle-check --bundle DIR` is the release check on
//!   a bundle's `plugins/`: the bundle's `mesh-llm` installs its defaults into
//!   a scratch store, finding `plugins/` beside itself as an installed node
//!   does, and every archive there must come out installed as a default. An archive that does not hash to the
//!   pin compiled into the host fails the install; one the host pins nothing
//!   for installs nothing. Either way the check fails.

use super::capsules_gate::{
    BUNDLE_ENTRY, PLUGIN_DIR, archive_name, build_plugin, build_web_ui, host_triple, package,
    plugin_version,
};
use super::{check_args::Grammar, check_report::CheckReport};
use crate::command::DynResult;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const ARCHIVE_GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository capsules-plugin-archive --out <dir> [--web-bundle <file>] [--root <path>]",
    values: &["--out", "--web-bundle", "--root"],
    flags: &[],
};

const CHECK_GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository capsules-bundle-check --bundle <dir>",
    values: &["--bundle"],
    flags: &[],
};

/// What the plugin store records for an installed plugin (the fields this
/// check reads of `mesh-llm-plugin-manager`'s `plugin-install.json`).
#[derive(Debug, Deserialize)]
struct Installed {
    downloaded_asset_name: String,
    enabled: bool,
    #[serde(default)]
    default_managed: bool,
}

const INSTALL_RECORD: &str = "plugin-install.json";

pub(super) fn archive(
    args: &[String],
    default_root: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    let parsed = match ARCHIVE_GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if !parsed.positionals.is_empty() {
        return ARCHIVE_GRAMMAR.error("no positional arguments").emit();
    }
    let Some(out) = parsed.last("--out").map(PathBuf::from) else {
        return ARCHIVE_GRAMMAR.error("--out is required").emit();
    };
    let root = match parsed.last("--root") {
        Some(root) => super::RepositoryRoot::resolve(Some(Path::new(root)))?
            .as_path()
            .to_path_buf(),
        None => default_root()?,
    };
    let plugin = root.join(PLUGIN_DIR);
    let version = plugin_version(&fs::read_to_string(plugin.join("plugin.toml"))?)?;
    let host_version = workspace_version(&fs::read_to_string(root.join("Cargo.toml"))?)?;
    if version != host_version {
        return CheckReport::failure(
            String::new(),
            format!(
                "plugin.toml declares {version} but mesh-llm is {host_version}: the host pins \
                 the archive by its own version (scripts/release-version.sh sets both)\n"
            ),
        )
        .emit();
    }
    match parsed.last("--web-bundle") {
        Some(bundle) => place_web_bundle(&plugin, Path::new(bundle))?,
        None => build_web_ui(&plugin, &[])?,
    }
    let binary = build_plugin(&root, &plugin, "release")?;
    // The same native policy as the host it ships beside: no forbidden
    // imports, and on Linux no GLIBC symbol above the declared floor.
    let checked = Command::new(std::env::current_exe()?)
        .args([
            "native",
            "verify-host-dependencies",
            "--max-glibc",
            "declared",
        ])
        .arg(&binary)
        .current_dir(&root)
        .status()?;
    if !checked.success() {
        return Err("the plugin executable fails the shipped-binary native policy".into());
    }
    let target = host_triple()?;
    let archive = package(&plugin, &binary, &version, &target, &out)?;
    let digest = hex::encode(Sha256::digest(fs::read(&archive)?));
    let line = format!("{digest}  {}\n", archive_name(&version, &target));
    fs::write(sidecar(&archive), &line)?;
    CheckReport::success(line).emit()
}

/// `[workspace.package] version` of the root manifest: mesh-llm's version.
fn workspace_version(manifest: &str) -> DynResult<String> {
    let manifest: toml::Value = toml::from_str(manifest)?;
    manifest
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "the root Cargo.toml has no [workspace.package] version".into())
}

/// Use an Evidence page built elsewhere from the same commit (the release
/// builds it once, in its own job, for every host producer): it becomes the
/// plugin's whole `bundle/`.
fn place_web_bundle(plugin: &Path, built: &Path) -> DynResult<()> {
    if built.file_name().and_then(|name| name.to_str()) != Some(BUNDLE_ENTRY) {
        return Err(format!("--web-bundle must be a built {BUNDLE_ENTRY}").into());
    }
    let bundle = plugin.join("bundle");
    if bundle.exists() {
        fs::remove_dir_all(&bundle)?;
    }
    fs::create_dir_all(&bundle)?;
    fs::copy(built, bundle.join(BUNDLE_ENTRY))?;
    Ok(())
}

fn sidecar(archive: &Path) -> PathBuf {
    let mut name = archive.as_os_str().to_owned();
    name.push(".sha256");
    PathBuf::from(name)
}

pub(super) fn bundle_check(args: &[String]) -> DynResult<()> {
    let parsed = match CHECK_GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if !parsed.positionals.is_empty() {
        return CHECK_GRAMMAR.error("no positional arguments").emit();
    }
    let Some(bundle) = parsed.last("--bundle") else {
        return CHECK_GRAMMAR.error("--bundle is required").emit();
    };
    check_bundle(Path::new(bundle))?.emit()
}

/// The bundle's host, which looks for `plugins/` beside itself
/// (`mesh_llm_plugin_manager::bundled::bundled_plugin_dir_candidates`).
const HOST: &str = "mesh-llm";
const PLUGINS: &str = "plugins";

fn check_bundle(bundle: &Path) -> DynResult<CheckReport> {
    let host = bundle.join(HOST);
    let plugins = bundle.join(PLUGINS);
    let archives = placed_archives(&plugins)?;
    if archives.is_empty() {
        return Ok(CheckReport::failure(
            String::new(),
            format!("{} holds no plugin archive\n", plugins.display()),
        ));
    }
    let scratch = tempfile::tempdir()?;
    let home = scratch.path().join("home");
    let store = scratch.path().join("store");
    fs::create_dir_all(&home)?;
    let output = Command::new(&host)
        .args(["--log-format", "json", "plugins", "install-defaults"])
        .env_clear()
        .envs(sandbox_env(std::env::vars_os(), &home, &store))
        .output()?;
    let printed = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        return Ok(CheckReport::failure(
            String::new(),
            format!(
                "the release host refused its bundled plugins ({}):\n{printed}",
                output.status
            ),
        ));
    }
    let missing = not_installed(&archives, &installed_defaults(&store)?);
    if !missing.is_empty() {
        return Ok(CheckReport::failure(
            String::new(),
            format!(
                "the release host installed no default from {}: it pins no archive by that \
                 name and digest\n{printed}",
                missing.join(", ")
            ),
        ));
    }
    Ok(CheckReport::success(format!(
        "the release host installed every bundled plugin at its compiled-in pin: {}\n",
        archives.join(", ")
    )))
}

/// The release host's environment for the check: this process's, without
/// any `MESH_LLM_*` override (a config file, a plugin store, a bundled
/// directory or an opt-out would each change what it installs) or home
/// directory, and with a scratch home, config and store.
fn sandbox_env(
    parent: impl Iterator<Item = (OsString, OsString)>,
    home: &Path,
    store: &Path,
) -> Vec<(OsString, OsString)> {
    const REPLACED: [&str; 5] = [
        "HOME",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    ];
    let mut env: Vec<(OsString, OsString)> = parent
        .filter(|(key, _)| {
            let key = key.to_string_lossy();
            !key.starts_with("MESH_LLM_") && !REPLACED.contains(&key.as_ref())
        })
        .collect();
    for (key, value) in [
        ("HOME", home.to_path_buf()),
        ("USERPROFILE", home.to_path_buf()),
        ("XDG_CONFIG_HOME", home.join(".config")),
        ("XDG_DATA_HOME", home.join(".local/share")),
        ("XDG_STATE_HOME", home.join(".local/state")),
        ("MESH_LLM_CONFIG", home.join(".mesh-llm/config.toml")),
        ("MESH_LLM_PLUGIN_DIR", store.to_path_buf()),
    ] {
        env.push((key.into(), value.into_os_string()));
    }
    env
}

/// The `.tar.gz` file names in a bundle's `plugins/`, sorted.
fn placed_archives(plugins: &Path) -> DynResult<Vec<String>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(plugins)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type()?.is_file() && name.ends_with(".tar.gz") {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// The archive names of the enabled, default-managed plugins in `store`.
fn installed_defaults(store: &Path) -> DynResult<Vec<String>> {
    let mut names = Vec::new();
    let Ok(entries) = fs::read_dir(store) else {
        return Ok(names);
    };
    for entry in entries {
        let record = entry?.path().join(INSTALL_RECORD);
        if !record.is_file() {
            continue;
        }
        let installed: Installed = serde_json::from_slice(&fs::read(&record)?)?;
        if installed.default_managed && installed.enabled {
            names.push(installed.downloaded_asset_name);
        }
    }
    Ok(names)
}

fn not_installed(archives: &[String], installed: &[String]) -> Vec<String> {
    archives
        .iter()
        .filter(|archive| !installed.contains(archive))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests;

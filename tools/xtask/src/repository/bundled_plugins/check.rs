//! `repository bundled-plugins-check --bundle DIR`: the bundle's own host
//! must install every archive in its `plugins/` as a default.

use super::super::{check_args::Grammar, check_report::CheckReport};
use crate::command::DynResult;
use serde::Deserialize;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::process::Command;

pub(super) const CHECK_GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository bundled-plugins-check --bundle <dir>",
    values: &["--bundle"],
    flags: &[],
};

/// The bundle's host, which looks for `plugins/` beside itself
/// (`mesh_llm_plugin_manager::bundled::bundled_plugin_dir_candidates`).
const HOST: &str = "mesh-llm";
pub(super) const PLUGINS: &str = "plugins";

/// What the plugin store records for an installed plugin (the fields this
/// check reads of `mesh-llm-plugin-manager`'s `plugin-install.json`).
#[derive(Debug, Deserialize)]
struct Installed {
    downloaded_asset_name: String,
    enabled: bool,
    #[serde(default)]
    default_managed: bool,
}

pub(super) const INSTALL_RECORD: &str = "plugin-install.json";

pub(in crate::repository) fn bundle_check(args: &[String]) -> DynResult<()> {
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

/// What `mesh-llm plugins install-defaults` prints for a default this host
/// pins for its platform but finds no bundled copy of
/// (`left_alone_line` for `DefaultPluginOutcome::NotBundled` in
/// `mesh-llm-commands/src/plugin.rs`). The command has no structured outcome,
/// so the check reads this line; a test keeps the two in step.
pub(super) const NOT_BUNDLED_LINE: &str = "carries no bundled copy of it";

/// The check runs both ways: every archive in `plugins/` must come out
/// installed, and the host must report no default it pins for this platform
/// as missing from the bundle. A bundle with no archive passes only when the
/// host pins nothing here.
pub(super) fn check_bundle(bundle: &Path) -> DynResult<CheckReport> {
    let host = bundle.join(format!("{HOST}{}", std::env::consts::EXE_SUFFIX));
    let archives = placed_archives(&bundle.join(PLUGINS))?;
    let scratch = tempfile::tempdir()?;
    let home = scratch.path().join("home");
    let store = scratch.path().join("store");
    fs::create_dir_all(&home)?;
    let output = Command::new(&host)
        .args(["--log-format", "json", "plugins", "install-defaults"])
        .env_clear()
        .envs(sandbox_env(std::env::vars_os(), &home, &store))
        .output()
        .map_err(|error| format!("run {}: {error}", host.display()))?;
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
    let unbundled: Vec<&str> = printed
        .lines()
        .filter(|line| line.contains(NOT_BUNDLED_LINE))
        .collect();
    if !unbundled.is_empty() {
        return Ok(CheckReport::failure(
            String::new(),
            format!(
                "the release host pins a default for this platform that the bundle does not \
                 carry:\n{}\n",
                unbundled.join("\n")
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
    if archives.is_empty() {
        return Ok(CheckReport::success(
            "the release host pins no default plugin for this platform; none is bundled\n".into(),
        ));
    }
    Ok(CheckReport::success(format!(
        "the release host installed every bundled plugin at its pin: {}\n",
        archives.join(", ")
    )))
}

/// The release host's environment for the check: this process's, without
/// any `MESH_LLM_*` override (a config file, a plugin store, a bundled
/// directory or an opt-out would each change what it installs) or home
/// directory, and with a scratch home, config and store. It runs as an
/// internal helper (`MESH_LLM_INTERNAL_HELPER`, as `mesh-llm update` runs a
/// staged binary), so a packaging run records and sends no analytics.
pub(super) fn sandbox_env(
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
        ("MESH_LLM_INTERNAL_HELPER", "1".into()),
    ] {
        env.push((key.into(), value.into_os_string()));
    }
    env
}

/// The `.tar.gz` file names in a bundle's `plugins/`, sorted. A bundle with
/// no `plugins/` holds none.
pub(super) fn placed_archives(plugins: &Path) -> DynResult<Vec<String>> {
    let mut names = Vec::new();
    let entries = match fs::read_dir(plugins) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(names),
        Err(error) => return Err(format!("read {}: {error}", plugins.display()).into()),
    };
    for entry in entries {
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
pub(super) fn installed_defaults(store: &Path) -> DynResult<Vec<String>> {
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

pub(super) fn not_installed(archives: &[String], installed: &[String]) -> Vec<String> {
    archives
        .iter()
        .filter(|archive| !installed.contains(archive))
        .cloned()
        .collect()
}

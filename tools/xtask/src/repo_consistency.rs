use crate::command::{DynResult, ensure_eq, run_command, trimmed_stderr_or_stdout};
use crate::{publish_consistency, release_targets, workflow_checks};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn check_attestation_default_version(repo_root: &Path) -> DynResult<()> {
    let runtime_version = resolve_runtime_version(repo_root)?;
    let default_node_version = host_runtime_package_version(repo_root)?;
    ensure_eq(
        runtime_version.as_str(),
        default_node_version.as_str(),
        "xtask release-attestation default node version",
    )
}

pub(crate) fn default_node_version() -> DynResult<String> {
    host_runtime_package_version(&repo_root()?)
}

fn resolve_runtime_version(repo_root: &Path) -> DynResult<String> {
    let runtime_lib = repo_root
        .join("crates")
        .join("mesh-llm-host-runtime")
        .join("src")
        .join("lib.rs");
    let contents = fs::read_to_string(runtime_lib)?;
    extract_runtime_version(repo_root, &contents)
}

fn extract_runtime_version(repo_root: &Path, contents: &str) -> DynResult<String> {
    const LITERAL_PREFIX: &str = "pub const VERSION: &str = \"";
    const CARGO_PKG_VERSION: &str = "pub const VERSION: &str = env!(\"CARGO_PKG_VERSION\");";
    const RELEASE_VERSION_ALIAS: &str = "pub const VERSION: &str = RELEASE_VERSION;";
    for line in contents.lines().map(str::trim) {
        if line == CARGO_PKG_VERSION {
            return host_runtime_package_version(repo_root);
        }
        if line == RELEASE_VERSION_ALIAS {
            return host_runtime_package_version(repo_root);
        }
        if let Some(rest) = line.strip_prefix(LITERAL_PREFIX) {
            return Ok(rest
                .strip_suffix("\";")
                .ok_or("malformed mesh-llm-host-runtime VERSION constant")?
                .to_string());
        }
    }
    Err("missing mesh-llm-host-runtime VERSION constant".into())
}

pub(crate) fn host_runtime_package_version(repo_root: &Path) -> DynResult<String> {
    let runtime_manifest = repo_root
        .join("crates")
        .join("mesh-llm-host-runtime")
        .join("Cargo.toml");
    let runtime_contents = fs::read_to_string(runtime_manifest)?;
    if let Some(version) = extract_manifest_string(&runtime_contents, "version")? {
        return Ok(version);
    }
    if has_manifest_bool(&runtime_contents, "version.workspace", true) {
        let workspace_manifest = repo_root.join("Cargo.toml");
        let workspace_contents = fs::read_to_string(workspace_manifest)?;
        return extract_manifest_string(&workspace_contents, "version")?
            .ok_or_else(|| "missing workspace package version".into());
    }
    Err("missing mesh-llm-host-runtime package version".into())
}

fn extract_manifest_string(contents: &str, key: &str) -> DynResult<Option<String>> {
    let prefix = format!("{key} = \"");
    for line in contents.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix(&prefix) {
            return Ok(Some(
                rest.strip_suffix('"')
                    .ok_or_else(|| format!("malformed manifest {key} value"))?
                    .to_string(),
            ));
        }
    }
    Ok(None)
}

fn has_manifest_bool(contents: &str, key: &str, expected: bool) -> bool {
    let expected_value = if expected { "true" } else { "false" };
    let expected_line = format!("{key} = {expected_value}");
    contents
        .lines()
        .map(str::trim)
        .any(|line| line == expected_line)
}

pub(crate) fn host_supports_shell_parity_checks() -> bool {
    !cfg!(windows)
}

pub(crate) fn repo_root() -> DynResult<PathBuf> {
    // CARGO_MANIFEST_DIR is <repo>/tools/xtask; go up two levels to reach the repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "could not determine repo root from xtask manifest directory".into())
}

#[derive(Debug, Deserialize)]
pub(crate) struct CargoMetadata {
    pub(crate) packages: Vec<CargoPackage>,
    pub(crate) workspace_members: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CargoPackage {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) manifest_path: PathBuf,
    #[serde(default)]
    pub(crate) dependencies: Vec<CargoDependency>,
    pub(crate) description: Option<String>,
    pub(crate) license: Option<String>,
    pub(crate) license_file: Option<String>,
    pub(crate) repository: Option<String>,
    pub(crate) readme: Option<String>,
    pub(crate) publish: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CargoDependency {
    pub(crate) name: String,
    pub(crate) req: String,
    pub(crate) kind: Option<String>,
    pub(crate) path: Option<PathBuf>,
}

pub(crate) fn workspace_metadata(repo_root: &Path, context: &str) -> DynResult<CargoMetadata> {
    let mut cargo = Command::new("cargo");
    cargo
        .current_dir(repo_root)
        .arg("metadata")
        .arg("--format-version=1")
        .arg("--no-deps");
    let output = run_command(&mut cargo)?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed while checking {context}: {}",
            trimmed_stderr_or_stdout(&output)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

pub(crate) fn workspace_package_names(repo_root: &Path) -> DynResult<BTreeSet<String>> {
    let metadata = workspace_metadata(repo_root, "CI crate lists")?;
    let workspace_members = metadata
        .workspace_members
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut names = BTreeSet::new();
    for package in metadata.packages {
        if workspace_members.contains(&package.id) {
            names.insert(package.name);
        }
    }

    if names.is_empty() {
        return Err("cargo metadata returned no workspace package names".into());
    }

    Ok(names)
}

pub(crate) fn script_workspace_members(
    repo_root: &Path,
    relative_path: &str,
) -> DynResult<BTreeSet<String>> {
    let contents = fs::read_to_string(repo_root.join(relative_path))?;
    let mut in_array = false;
    let mut members = BTreeSet::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if !in_array {
            if trimmed == "WORKSPACE_MEMBERS=(" {
                in_array = true;
            }
            continue;
        }

        if trimmed == ")" {
            return Ok(members);
        }

        let Some(member) = trimmed
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
        else {
            return Err(format!(
                "{relative_path} WORKSPACE_MEMBERS: expected quoted crate name, got `{trimmed}`"
            )
            .into());
        };
        if !members.insert(member.to_string()) {
            return Err(format!(
                "{relative_path} WORKSPACE_MEMBERS: duplicate crate name `{member}`"
            )
            .into());
        }
    }

    Err(format!("{relative_path}: missing WORKSPACE_MEMBERS array").into())
}

pub(crate) fn check_release_targets_command() -> DynResult<()> {
    release_targets::check_release_targets()
}

pub(crate) fn check_ci_crate_lists_command() -> DynResult<()> {
    let repo_root = repo_root()?;
    workflow_checks::check_ci_script_workspace_members(&repo_root)?;
    workflow_checks::check_ci_crate_test_coverage_files(&repo_root)?;
    check_attestation_default_version(&repo_root)?;
    println!("repo consistency checks passed: ci-crate-lists");
    Ok(())
}

pub(crate) fn check_publish_crates_command() -> DynResult<()> {
    let repo_root = repo_root()?;
    publish_consistency::check_publish_crates_consistency(&repo_root)?;
    println!("repo consistency checks passed: publish-crates");
    Ok(())
}

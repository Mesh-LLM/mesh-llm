use crate::command::DynResult;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub(super) const POLICY_PATH: &str = "ci/python-tooling-policy.json";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Policy {
    pub(super) schema_version: u32,
    pub(super) dependency_inputs: Vec<DependencyInput>,
    pub(super) external_source_descriptors: Vec<ExternalSourceDescriptor>,
    pub(super) external_script_pins: Vec<ExternalScriptPin>,
    pub(super) allowed_invocations: Vec<AllowedInvocation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DependencyInput {
    pub(super) manifest_path: String,
    pub(super) lock_path: String,
    pub(super) consumer_path: String,
    consumer_assignment: String,
    purpose: String,
    project_name: String,
    project_version: String,
    direct_dependencies: Vec<PackagePin>,
    locked_packages: Vec<PackagePin>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackagePin {
    name: String,
    version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExternalSourceDescriptor {
    pub(super) descriptor_path: String,
    repository: String,
    revision: String,
    manifest_sha256: String,
    purpose: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExternalScriptPin {
    pub(super) path: String,
    repository: String,
    revision: String,
    assignment: String,
    purpose: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AllowedInvocation {
    pub(super) path: String,
    pub(super) line: String,
    pub(super) occurrences: usize,
    purpose: String,
}

pub(super) fn validate_policy_shape(policy: &Policy) -> DynResult<()> {
    if policy.schema_version != 1 {
        return Err("unsupported Python tooling policy schema_version".into());
    }

    let mut exact_paths = BTreeSet::new();
    for path in policy
        .dependency_inputs
        .iter()
        .flat_map(|item| [&item.manifest_path, &item.lock_path])
        .chain(
            policy
                .dependency_inputs
                .iter()
                .map(|item| &item.consumer_path),
        )
        .chain(
            policy
                .external_source_descriptors
                .iter()
                .map(|item| &item.descriptor_path),
        )
        .chain(policy.external_script_pins.iter().map(|item| &item.path))
    {
        if !safe_relative_path(path) || !exact_paths.insert(path.as_str()) {
            return Err(
                format!("invalid or duplicate Python tooling exception path `{path}`").into(),
            );
        }
    }

    for input in &policy.dependency_inputs {
        if input.purpose.trim().is_empty()
            || input.project_name.trim().is_empty()
            || input.project_version.trim().is_empty()
            || input.consumer_assignment.trim().is_empty()
            || !safe_relative_path(&input.consumer_path)
        {
            return Err(format!(
                "Python dependency exception `{}` needs a purpose and exact package pins",
                input.manifest_path
            )
            .into());
        }
        unique_package_pins(&input.direct_dependencies, &input.manifest_path)?;
        unique_package_pins(&input.locked_packages, &input.lock_path)?;
    }

    for source in &policy.external_source_descriptors {
        if source.purpose.trim().is_empty()
            || source.repository.trim().is_empty()
            || !is_hex_revision(&source.revision, 40)
            || !is_hex_revision(&source.manifest_sha256, 64)
        {
            return Err(format!(
                "external source exception `{}` must pin a repository, full revision, and manifest digest",
                source.descriptor_path
            )
            .into());
        }
    }

    for source in &policy.external_script_pins {
        if source.purpose.trim().is_empty()
            || source.repository.trim().is_empty()
            || source.assignment.trim().is_empty()
            || !is_hex_revision(&source.revision, 40)
        {
            return Err(format!(
                "external script pin `{}` must name a purpose and immutable full revision",
                source.path
            )
            .into());
        }
    }

    let mut invocation_keys = BTreeSet::new();
    for allowed in &policy.allowed_invocations {
        if !safe_relative_path(&allowed.path)
            || allowed.line.trim().is_empty()
            || allowed.occurrences == 0
            || allowed.purpose.trim().is_empty()
            || !invocation_keys.insert((allowed.path.as_str(), allowed.line.as_str()))
        {
            return Err(format!(
                "invalid, duplicate, or unreviewed command exception in `{}`",
                allowed.path
            )
            .into());
        }
    }
    Ok(())
}

fn unique_package_pins(pins: &[PackagePin], path: &str) -> DynResult<()> {
    let mut names = BTreeSet::new();
    for pin in pins {
        if pin.name.trim().is_empty()
            || pin.version.trim().is_empty()
            || !names.insert(pin.name.to_ascii_lowercase())
        {
            return Err(format!("invalid or duplicate package pin in `{path}`").into());
        }
    }
    Ok(())
}

pub(super) fn validate_dependency_input(
    repo_root: &Path,
    input: &DependencyInput,
) -> DynResult<()> {
    let manifest_path = repo_root.join(&input.manifest_path);
    let lock_path = repo_root.join(&input.lock_path);
    let manifest: toml::Value = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
    let dependencies = manifest
        .get("project")
        .and_then(|project| project.get("dependencies"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| format!("`{}` lacks project.dependencies", input.manifest_path))?;
    let project = manifest
        .get("project")
        .ok_or_else(|| format!("`{}` lacks the project table", input.manifest_path))?;
    if project.get("name").and_then(toml::Value::as_str) != Some(input.project_name.as_str())
        || project.get("version").and_then(toml::Value::as_str)
            != Some(input.project_version.as_str())
    {
        return Err(format!(
            "`{}` project identity differs from the reviewed lock owner",
            input.manifest_path
        )
        .into());
    }
    let direct = dependencies
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("`{}` has a non-string dependency", input.manifest_path))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_direct: Vec<_> = input
        .direct_dependencies
        .iter()
        .map(|pin| format!("{}=={}", pin.name, pin.version))
        .collect();
    if direct != expected_direct {
        return Err(format!(
            "`{}` dependencies differ from the reviewed exact pins",
            input.manifest_path
        )
        .into());
    }

    let lock: toml::Value = toml::from_str(&fs::read_to_string(&lock_path)?)?;
    let packages = lock
        .get("package")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| format!("`{}` lacks locked package records", input.lock_path))?;
    let mut actual = BTreeSet::new();
    let mut virtual_project_count = 0;
    for package in packages {
        let name = package
            .get("name")
            .and_then(toml::Value::as_str)
            .ok_or_else(|| format!("`{}` has a package without a name", input.lock_path))?;
        let version = package
            .get("version")
            .and_then(toml::Value::as_str)
            .ok_or_else(|| format!("`{}` has a package without a version", input.lock_path))?;
        let package_source = package.get("source");
        if package_source
            .and_then(|source| source.get("virtual"))
            .and_then(toml::Value::as_str)
            == Some(".")
        {
            if name != input.project_name || version != input.project_version {
                return Err(format!(
                    "`{}` has an unexpected virtual project record `{name}=={version}`",
                    input.lock_path
                )
                .into());
            }
            virtual_project_count += 1;
            continue;
        }
        let source = package_source
            .and_then(|source| source.get("registry"))
            .and_then(toml::Value::as_str);
        if source != Some("https://pypi.org/simple") {
            return Err(format!(
                "`{}` package `{name}` must come from the pinned PyPI registry",
                input.lock_path
            )
            .into());
        }
        validate_package_hashes(package, &input.lock_path, name)?;
        actual.insert((name.to_owned(), version.to_owned()));
    }
    let expected: BTreeSet<_> = input
        .locked_packages
        .iter()
        .map(|pin| (pin.name.clone(), pin.version.clone()))
        .collect();
    if actual != expected || virtual_project_count != 1 {
        return Err(format!(
            "`{}` locked package set differs from the reviewed exact pins",
            input.lock_path
        )
        .into());
    }
    let project_directory = Path::new(&input.manifest_path)
        .parent()
        .and_then(Path::to_str)
        .ok_or_else(|| format!("`{}` has no project directory", input.manifest_path))?;
    let expected_binding = format!(
        "{}=\"$TRUSTED_ROOT/{project_directory}\"",
        input.consumer_assignment
    );
    let consumer = fs::read_to_string(repo_root.join(&input.consumer_path))?;
    if !consumer.lines().any(|line| line.trim() == expected_binding) {
        return Err(format!(
            "`{}` must bind {} to the reviewed project directory",
            input.consumer_path, input.consumer_assignment
        )
        .into());
    }
    Ok(())
}

fn validate_package_hashes(package: &toml::Value, lock_path: &str, name: &str) -> DynResult<()> {
    let mut hashes = Vec::new();
    let mut artifact_count = 0;
    if let Some(sdist) = package.get("sdist") {
        artifact_count += 1;
        if let Some(hash) = sdist.get("hash").and_then(toml::Value::as_str) {
            hashes.push(hash);
        }
    }
    if let Some(wheels) = package.get("wheels").and_then(toml::Value::as_array) {
        for wheel in wheels {
            artifact_count += 1;
            if let Some(hash) = wheel.get("hash").and_then(toml::Value::as_str) {
                hashes.push(hash);
            }
        }
    }
    if artifact_count == 0
        || hashes.len() != artifact_count
        || hashes.iter().any(|hash| {
            !hash
                .strip_prefix("sha256:")
                .is_some_and(|value| is_hex_revision(value, 64))
        })
    {
        return Err(format!(
            "`{lock_path}` package `{name}` must keep complete SHA-256 artifact hashes"
        )
        .into());
    }
    Ok(())
}

pub(super) fn validate_external_source_descriptors(
    repo_root: &Path,
    sources: &[ExternalSourceDescriptor],
) -> DynResult<()> {
    for source in sources {
        let descriptor: serde_json::Value =
            serde_json::from_slice(&fs::read(repo_root.join(&source.descriptor_path))?)?;
        if descriptor
            .get("repository")
            .and_then(serde_json::Value::as_str)
            != Some(source.repository.as_str())
            || descriptor
                .get("revision")
                .and_then(serde_json::Value::as_str)
                != Some(source.revision.as_str())
            || descriptor
                .get("manifest_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(source.manifest_sha256.as_str())
        {
            return Err(format!(
                "external source descriptor `{}` differs from its reviewed repository pin",
                source.descriptor_path
            )
            .into());
        }
    }
    Ok(())
}

pub(super) fn validate_external_script_pins(
    repo_root: &Path,
    sources: &[ExternalScriptPin],
) -> DynResult<()> {
    for source in sources {
        let text = fs::read_to_string(repo_root.join(&source.path))?;
        let expected = format!(
            "{}=\"${{{}:-git+{}@{}}}\"",
            source.assignment, source.assignment, source.repository, source.revision
        );
        if !text.lines().any(|line| line.trim() == expected) {
            return Err(format!(
                "external source default in `{}` differs from its immutable reviewed pin",
                source.path
            )
            .into());
        }
    }
    Ok(())
}

fn safe_relative_path(path: &str) -> bool {
    let path = PathBuf::from(path);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

fn is_hex_revision(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

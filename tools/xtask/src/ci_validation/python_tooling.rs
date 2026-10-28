//! Bounded repository inventory for Python source, package inputs and execution.
//!
//! This static check inventories non-ignored files, inspects maintained automation
//! entrypoints, and catches direct literal interpreter launches in production Rust.
//! It does not parse Python or prove semantic absence of interpreter emulation.

mod invocations;
mod policy;
#[cfg(test)]
mod tests;

use crate::command::DynResult;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use policy::{POLICY_PATH, Policy};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

const GIT_ROSTER_LIMIT: usize = 16 * 1024 * 1024;
const GIT_DIAGNOSTIC_LIMIT: usize = 64 * 1024;

pub(crate) fn check(repo_root: &Path) -> DynResult<()> {
    let policy: Policy = serde_json::from_slice(&fs::read(repo_root.join(POLICY_PATH))?)?;
    policy::validate_policy_shape(&policy)?;
    let paths = repository_paths(repo_root)?;
    check_paths(repo_root, &paths, &policy)
}

fn repository_paths(repo_root: &Path) -> DynResult<BTreeSet<String>> {
    let cwd = repo_root.canonicalize()?;
    let executable = find_git_executable()?;
    let environment = git_environment()?;
    let null_path = if cfg!(windows) { "NUL" } else { "/dev/null" };
    let arguments = [
        "-c".to_owned(),
        "core.fsmonitor=false".to_owned(),
        "-c".to_owned(),
        "core.untrackedCache=false".to_owned(),
        "-c".to_owned(),
        format!("core.hooksPath={null_path}"),
        "-c".to_owned(),
        format!("core.excludesFile={null_path}"),
        "ls-files".to_owned(),
        "--cached".to_owned(),
        "--others".to_owned(),
        "--exclude-standard".to_owned(),
        "-z".to_owned(),
    ]
    .into_iter()
    .map(|argument| Value::Public(argument.into()))
    .collect();
    let report = process::supervise_raw(
        &ProcessSpec {
            executable,
            arguments,
            cwd,
            environment,
        },
        &Limits {
            execution: Duration::from_secs(30),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: GIT_DIAGNOSTIC_LIMIT,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(GIT_ROSTER_LIMIT),
            stderr: None,
        },
    )?;
    if !report.process.success() {
        return Err(format!(
            "Git source inventory did not exit cleanly within its process bounds: {}",
            String::from_utf8_lossy(&report.process.stderr.bytes_retained).trim()
        )
        .into());
    }
    let output = report
        .stdout
        .ok_or("Git source inventory raw stdout capture missing")?;
    if report.process.stdout.bytes_seen != output.as_bytes().len() as u64 {
        return Err("Git source inventory raw stdout capture was incomplete".into());
    }
    let bytes = output.as_bytes();
    if !bytes.is_empty() && bytes.last() != Some(&0) {
        return Err("Git source inventory returned an unterminated NUL record".into());
    }
    let mut paths = BTreeSet::new();
    let entries = if bytes.is_empty() {
        &[][..]
    } else {
        &bytes[..bytes.len() - 1]
    };
    for raw in entries.split(|byte| *byte == 0) {
        if raw.is_empty() {
            return Err("Git source inventory returned an empty path record".into());
        }
        let path = std::str::from_utf8(raw)
            .map_err(|_| "Python tooling inventory encountered a non-UTF-8 repository path")?;
        if repo_root.join(path).is_file() {
            paths.insert(path.to_owned());
        }
    }
    Ok(paths)
}

fn find_git_executable() -> DynResult<PathBuf> {
    let path = std::env::var_os("PATH").ok_or("Git PATH is unavailable")?;
    let cwd = std::env::current_dir()?;
    let name = if cfg!(windows) { "git.exe" } else { "git" };
    for directory in std::env::split_paths(&path) {
        let directory = if directory.is_absolute() {
            directory
        } else {
            cwd.join(directory)
        };
        let candidate = directory.join(name);
        let Ok(metadata) = fs::metadata(&candidate) else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 == 0 {
                continue;
            }
        }
        return Ok(candidate.canonicalize()?);
    }
    Err("an existing Git executable was not found on PATH".into())
}

fn git_environment() -> DynResult<BTreeMap<OsString, Value>> {
    let mut environment = BTreeMap::new();
    let path = std::env::var_os("PATH").ok_or("Git PATH is unavailable")?;
    environment.insert("PATH".into(), Value::Public(path));
    if let Some(home) = std::env::var_os("HOME") {
        environment.insert("HOME".into(), Value::Public(home));
    }
    for (key, value) in [
        ("GIT_CONFIG_NOSYSTEM", OsString::from("1")),
        (
            "GIT_CONFIG_GLOBAL",
            OsString::from(if cfg!(windows) { "NUL" } else { "/dev/null" }),
        ),
        ("GIT_OPTIONAL_LOCKS", OsString::from("0")),
        ("GIT_NO_REPLACE_OBJECTS", OsString::from("1")),
    ] {
        environment.insert(key.into(), Value::Public(value));
    }
    #[cfg(windows)]
    for key in ["SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE"] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), Value::Public(value));
        }
    }
    Ok(environment)
}

fn check_paths(repo_root: &Path, paths: &BTreeSet<String>, policy: &Policy) -> DynResult<()> {
    let manifest_paths = validate_inputs(repo_root, paths, policy)?;
    let mut observed_manifests = BTreeSet::new();
    let mut observed_invocations = BTreeMap::<(String, String), usize>::new();

    for relative in paths {
        let path = Path::new(relative);
        let absolute = repo_root.join(path);
        if invocations::is_python_source(path) {
            return Err(format!(
                "repository Python source is forbidden: {relative}; keep SDK and upstream source in pinned external checkouts"
            )
            .into());
        }
        if invocations::is_python_package_manifest(path, &absolute)? {
            if !manifest_paths.contains(relative.as_str()) {
                return Err(format!(
                    "unreviewed Python package input {relative}; add only a precise pinned exception to {POLICY_PATH}"
                )
                .into());
            }
            observed_manifests.insert(relative.as_str());
        }

        let contents = fs::read(&absolute)?;
        if invocations::has_python_shebang(&contents) {
            return Err(
                format!("Python shebang is forbidden in repository file {relative}").into(),
            );
        }
        check_file_invocations(
            relative,
            path,
            &absolute,
            &contents,
            policy,
            &mut observed_invocations,
        )?;
        if invocations::is_production_rust_source(path) {
            invocations::check_rust_process_invocations(relative, &contents)?;
        }
    }

    for expected in &manifest_paths {
        if !observed_manifests.contains(expected.as_str()) {
            return Err(format!(
                "stale Python dependency exception {expected} is not an active package input"
            )
            .into());
        }
    }
    check_invocation_counts(policy, &observed_invocations)
}

fn validate_inputs(
    repo_root: &Path,
    paths: &BTreeSet<String>,
    policy: &Policy,
) -> DynResult<BTreeSet<String>> {
    let mut manifests = BTreeSet::new();
    for input in &policy.dependency_inputs {
        manifests.insert(input.manifest_path.clone());
        manifests.insert(input.lock_path.clone());
        policy::validate_dependency_input(repo_root, input)?;
        if !paths.contains(&input.consumer_path) {
            return Err(format!(
                "Python dependency consumer {} is not an active repository file",
                input.consumer_path
            )
            .into());
        }
    }
    for input in &policy.external_dependency_inputs {
        manifests.insert(input.manifest_path.clone());
        manifests.insert(input.lock_path.clone());
        policy::validate_pinned_dependency_input(repo_root, input)?;
        if !paths.contains(&input.consumer_path) {
            return Err(format!(
                "external dependency consumer {} is not an active repository file",
                input.consumer_path
            )
            .into());
        }
    }
    policy::validate_external_source_descriptors(repo_root, &policy.external_source_descriptors)?;
    policy::validate_external_script_pins(repo_root, &policy.external_script_pins)?;
    for source in &policy.external_source_descriptors {
        if !paths.contains(&source.descriptor_path) {
            return Err(format!(
                "pinned external source descriptor {} is not an active repository file",
                source.descriptor_path
            )
            .into());
        }
    }
    for source in &policy.external_script_pins {
        if !paths.contains(&source.path) {
            return Err(format!(
                "pinned external source declaration {} is not an active repository file",
                source.path
            )
            .into());
        }
    }
    for path in &manifests {
        if !paths.contains(path) {
            return Err(
                format!("pinned Python dependency input {path} is not an active file").into(),
            );
        }
    }
    Ok(manifests)
}

fn check_file_invocations(
    relative: &str,
    path: &Path,
    absolute: &Path,
    contents: &[u8],
    policy: &Policy,
    observed: &mut BTreeMap<(String, String), usize>,
) -> DynResult<()> {
    if invocations::is_javascript_source(path) {
        let source = std::str::from_utf8(contents)?;
        let invocations = invocations::javascript_source_python_invocations(source)
            .map_err(|error| format!("cannot inspect executable source {relative}: {error}"))?;
        for invocation in invocations {
            record_invocation(
                relative,
                &invocation,
                "JavaScript process call",
                policy,
                observed,
            )?;
        }
        return Ok(());
    }
    let Some(lines) = invocations::executable_source(path, absolute, contents)
        .map_err(|error| format!("cannot inspect executable source {relative}: {error}"))?
    else {
        return Ok(());
    };
    let mut heredoc = None;
    let mut previous_run_command = None;
    for (location, line) in lines {
        if location.starts_with("run command ") {
            let command = location
                .split_once(" line ")
                .map_or(location.as_str(), |(command, _)| command);
            if previous_run_command.as_deref() != Some(command) {
                heredoc = None;
                previous_run_command = Some(command.to_owned());
            }
        }
        let normalized = line.trim();
        if normalized.is_empty() || normalized.starts_with('#') {
            continue;
        }
        let invokes_python =
            invocations::has_python_tool_invocation_in_line(normalized, &mut heredoc);
        if invokes_python {
            record_invocation(relative, normalized, &location, policy, observed)?;
        }
    }
    Ok(())
}

fn record_invocation(
    relative: &str,
    line: &str,
    location: &str,
    policy: &Policy,
    observed: &mut BTreeMap<(String, String), usize>,
) -> DynResult<()> {
    let key = (relative.to_owned(), line.to_owned());
    let Some(allowed) = policy
        .allowed_invocations
        .iter()
        .find(|entry| entry.path == relative && entry.line == line)
    else {
        return Err(format!(
            "unreviewed Python tooling invocation in {relative} at {location}: {line}"
        )
        .into());
    };
    let count = observed.entry(key).or_default();
    *count += 1;
    if *count > allowed.occurrences {
        return Err(format!(
            "Python tooling exception in {relative} exceeds its reviewed occurrence count"
        )
        .into());
    }
    Ok(())
}

fn check_invocation_counts(
    policy: &Policy,
    observed: &BTreeMap<(String, String), usize>,
) -> DynResult<()> {
    for allowed in &policy.allowed_invocations {
        let key = (allowed.path.clone(), allowed.line.clone());
        if observed.get(&key).copied().unwrap_or(0) != allowed.occurrences {
            return Err(format!(
                "Python tooling exception in {} no longer matches its reviewed occurrence count",
                allowed.path
            )
            .into());
        }
    }
    Ok(())
}

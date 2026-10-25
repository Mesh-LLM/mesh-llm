//! Native custody of the two fixed external Python SDK/research checkouts.
use super::hf_certify::admission::{digest, read};
use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
    time::Duration,
};

pub(super) fn git(root: &Path, args: &[&str]) -> DynResult<Vec<u8>> {
    git_capture(root, args, false)
}

fn git_capture(root: &Path, args: &[&str], research_roster: bool) -> DynResult<Vec<u8>> {
    let logs = tempfile::tempdir()?;
    let stdout = logs.path().join("stdout");
    let name = if cfg!(windows) { "git.exe" } else { "git" };
    let executable = std::env::split_paths(&std::env::var_os("PATH").ok_or("Git PATH missing")?)
        .map(|p| p.join(name))
        .find(|p| p.is_absolute() && p.is_file())
        .ok_or("existing Git executable required")?
        .canonicalize()?;
    let mut environment: BTreeMap<_, _> = [
        ("GIT_MASTER", "1"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
        ("GIT_CONFIG_GLOBAL", "/dev/null"),
        ("GIT_NO_REPLACE_OBJECTS", "1"),
        ("GIT_OPTIONAL_LOCKS", "0"),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), Value::Public(value.into())))
    .collect();
    environment.insert(
        "PATH".into(),
        Value::Public(std::env::join_paths([
            executable.parent().ok_or("Git executable parent")?,
            Path::new("/usr/bin"),
            Path::new("/bin"),
        ])?),
    );
    #[cfg(windows)]
    for key in ["SYSTEMROOT", "WINDIR"] {
        let value = std::env::var_os(key).ok_or("Windows Git requires native system directory")?;
        if !Path::new(&value).is_absolute() || !Path::new(&value).is_dir() {
            return Err("Windows Git native system directory refused".into());
        }
        environment.insert(key.into(), Value::Public(value));
    }
    let spec = ProcessSpec {
        executable,
        arguments: [
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
        ]
        .into_iter()
        .chain(args.iter().copied())
        .map(|s| Value::Public(s.into()))
        .collect(),
        cwd: root.into(),
        environment,
    };
    let limits = Limits {
        execution: Duration::from_secs(10),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    if research_roster {
        // Source paths are data: diagnostic redaction suppresses legitimate tokenizer paths.
        // Only this fixed roster command uses raw bytes, admitted by the supervisor after EOF.
        if args != ["ls-files"] {
            return Err("external research Git data command refused".into());
        }
        let report = process::supervise_raw(
            &spec,
            &limits,
            &Cancellation::default(),
            process::RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: None,
            },
        )?;
        if !report.process.success() || !report.process.stdout.line_capture_complete {
            return Err("external research Git ls-files admission failed".into());
        }
        let bytes = report
            .stdout
            .ok_or("external research Git ls-files capture missing")?;
        if bytes.as_bytes().len() as u64 != report.process.stdout.bytes_seen {
            return Err("external research Git ls-files capture incomplete".into());
        }
        return Ok(bytes.as_bytes().to_vec());
    }
    let report = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles {
            stdout: Some(stdout.clone()),
            stderr: None,
        },
    )?;
    if !report.success()
        || report.stdout.truncated
        || !report.stdout.line_capture_complete
        || report.stdout.suppressed_lines != 0
    {
        return Err("external SDK Git admission failed".into());
    }
    read(&stdout, 65536)
}

pub(super) fn checkout(root: &Path, revision: &str) -> DynResult<PathBuf> {
    if !root.is_absolute()
        || revision.len() != 40
        || !revision.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("external SDK source pin refused".into());
    }
    let root = root.canonicalize()?;
    let top = String::from_utf8(git(&root, &["rev-parse", "--show-toplevel"])?)?;
    if Path::new(top.trim_end()).canonicalize()? != root
        || git(&root, &["rev-parse", "HEAD"])? != format!("{revision}\n").as_bytes()
        || !git(
            &root,
            &[
                "diff",
                "--cached",
                "--no-ext-diff",
                "--no-textconv",
                "--exit-code",
                "HEAD",
                "--",
            ],
        )?
        .is_empty()
    {
        return Err("external SDK checkout identity/drift refused".into());
    }
    if !git(&root, &["ls-files", "--others", "--exclude-standard"])?.is_empty() {
        return Err("external SDK has unbound untracked source".into());
    }
    Ok(root)
}

pub(super) fn files(
    root: &Path,
    manifest: &str,
    expected: &BTreeMap<String, String>,
    maximum: usize,
    leaf_bytes: u64,
) -> DynResult<Vec<(PathBuf, String)>> {
    files_with_roster(root, manifest, expected, maximum, leaf_bytes, false)
}

pub(super) fn research_files(
    root: &Path,
    expected: &BTreeMap<String, String>,
) -> DynResult<Vec<(PathBuf, String)>> {
    files_with_roster(
        root,
        "project-inputs.json",
        expected,
        4096,
        16 * 1048576,
        true,
    )
}

fn files_with_roster(
    root: &Path,
    manifest: &str,
    expected: &BTreeMap<String, String>,
    maximum: usize,
    leaf_bytes: u64,
    research_roster: bool,
) -> DynResult<Vec<(PathBuf, String)>> {
    if expected.is_empty() || expected.len() > maximum {
        return Err("external SDK tracked roster bound refused".into());
    }
    let mut expected_paths = expected.keys().map(String::as_str).collect::<Vec<_>>();
    expected_paths.push(manifest);
    expected_paths.sort();
    let tracked = git_capture(root, &["ls-files"], research_roster)?;
    if std::str::from_utf8(&tracked)?.lines().collect::<Vec<_>>() != expected_paths {
        return Err("external SDK tracked roster refused".into());
    }
    let mut pins = Vec::new();
    for (relative, hash) in expected {
        if Path::new(relative)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err("external SDK file path refused".into());
        }
        let path =
            Path::new(relative)
                .components()
                .fold(root.to_path_buf(), |mut path, component| {
                    path.push(component.as_os_str());
                    path
                });
        if path.canonicalize()? != path || digest(&read(&path, leaf_bytes)?) != *hash {
            return Err("external SDK file custody refused".into());
        }
        pins.push((path, hash.clone()));
    }
    Ok(pins)
}

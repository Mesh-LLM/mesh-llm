use crate::command::DynResult;
use crate::process::{self, Completion, Limits, ProcessSpec, Readiness, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

pub(super) fn git_head_with_budget(root: &Path, budget: Duration) -> DynResult<String> {
    let path = std::env::var_os("PATH").ok_or("Git PATH unavailable")?;
    let name = if cfg!(windows) { "git.exe" } else { "git" };
    let executable = std::env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
        .ok_or("Git executable unavailable")?
        .canonicalize()?;
    let mut environment: BTreeMap<_, _> = ["PATH", "SystemRoot", "WINDIR", "TMP", "TEMP"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name.into(), Value::Public(value))))
        .collect();
    for (name, value) in [
        ("GIT_MASTER", "1"),
        ("GIT_OPTIONAL_LOCKS", "0"),
        ("GIT_NO_LAZY_FETCH", "1"),
        ("GIT_NO_REPLACE_OBJECTS", "1"),
        ("GIT_TERMINAL_PROMPT", "0"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
        (
            "GIT_CONFIG_GLOBAL",
            if cfg!(windows) { "NUL" } else { "/dev/null" },
        ),
        ("LC_ALL", "C"),
    ] {
        environment.insert(name.into(), Value::Public(value.into()));
    }
    let spec = ProcessSpec {
        executable,
        arguments: ["rev-parse", "HEAD"]
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect(),
        cwd: root.to_path_buf(),
        environment,
    };
    let limits = Limits {
        execution: budget,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let output = process::supervise(&spec, &limits, &Default::default(), Default::default())?;
    if !output.success() || output.stdout.truncated {
        return Err("candidate source SHA unavailable".into());
    }
    Ok(String::from_utf8(output.stdout.bytes_retained)?)
}

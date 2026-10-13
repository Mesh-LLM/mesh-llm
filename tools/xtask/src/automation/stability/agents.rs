use super::{
    evidence,
    options::{Agent, Options},
    reports::{CommandRow, CommandStatus},
};
use crate::{
    command::DynResult,
    process::{
        Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value, supervise,
    },
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) fn environment(agent: Agent, options: &Options) -> BTreeMap<String, String> {
    let work = options.output.join("agent-smokes").join(agent.name());
    let mut values = BTreeMap::from([
        ("MESH_AGENT_BASE_URL".into(), options.base.to_string()),
        ("MESH_OPENCODE_BASE_URL".into(), options.base.to_string()),
    ]);
    let prefix = agent.name().to_uppercase();
    values.insert(
        format!("{prefix}_SMOKE_WORK_DIR"),
        work.display().to_string(),
    );
    for (suffix, filename) in [
        ("OUTPUT", format!("{}-output.jsonl", agent.name())),
        ("ERROR_LOG", format!("{}-stderr.log", agent.name())),
    ] {
        values.insert(
            format!("{prefix}_SMOKE_{suffix}"),
            work.join(filename).display().to_string(),
        );
    }
    if agent == Agent::Opencode {
        for (suffix, filename) in [
            ("TURN1_OUTPUT", "opencode-turn1.jsonl"),
            ("TURN2_OUTPUT", "opencode-turn2.jsonl"),
            ("SURFACE_LOG", "openai-surface.jsonl"),
            ("SURFACE_PROXY_LOG", "openai-surface-proxy.log"),
        ] {
            values.insert(
                format!("OPENCODE_SMOKE_{suffix}"),
                work.join(filename).display().to_string(),
            );
        }
    }
    values
}

pub(super) fn run(
    root: &Path,
    options: &Options,
    cancellation: &Cancellation,
) -> DynResult<Vec<CommandRow>> {
    let mut rows = Vec::new();
    for agent in &options.agents {
        if cancellation.is_cancelled() {
            break;
        }
        rows.push(run_one(root, *agent, options, cancellation)?);
    }
    Ok(rows)
}

fn run_one(
    root: &Path,
    agent: Agent,
    options: &Options,
    cancellation: &Cancellation,
) -> DynResult<CommandRow> {
    let log = format!("logs/{}-agent-smoke.log", agent.name());
    let mut row = CommandRow {
        name: format!("{}-agent-smoke", agent.name()),
        status: CommandStatus::Prereq,
        exit_code: 0,
        elapsed_ms: 0,
        log,
        detail: format!("Prerequisite command not found on PATH: {}", agent.name()),
    };
    if find(agent.name()).is_none() {
        evidence::bytes(&options.output.join(&row.log), row.detail.as_bytes())?;
        return Ok(row);
    }
    let Some(bash) = find("bash") else {
        row.status = CommandStatus::Fail;
        row.exit_code = 1;
        row.detail = "agent smoke requires Bash".into();
        evidence::bytes(&options.output.join(&row.log), row.detail.as_bytes())?;
        return Ok(row);
    };
    let script = root
        .join("scripts")
        .join(format!("ci-{}-smoke.sh", agent.name()));
    let mut env: BTreeMap<OsString, Value> = std::env::vars_os()
        .map(|(key, value)| {
            let name = key.to_string_lossy().to_uppercase();
            let secret = ["TOKEN", "PASSWORD", "SECRET", "API_KEY"]
                .iter()
                .any(|part| name.contains(part));
            let value = if secret && !value.is_empty() {
                Value::Secret(value)
            } else {
                Value::Public(value)
            };
            (key, value)
        })
        .collect();
    for (key, value) in environment(agent, options) {
        env.insert(key.into(), Value::Public(value.into()));
    }
    let spec = ProcessSpec {
        executable: bash,
        arguments: vec![Value::Public(script.into_os_string())],
        cwd: root.to_path_buf(),
        environment: env,
    };
    let limits = Limits {
        execution: options.agent_timeout,
        graceful_shutdown: Duration::from_secs(2),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = match supervise(&spec, &limits, cancellation, OutputFiles::default()) {
        Ok(report) => report,
        Err(error) => {
            row.status = CommandStatus::Fail;
            row.exit_code = 1;
            row.detail = format!("agent smoke could not complete: {error}");
            evidence::bytes(&options.output.join(&row.log), row.detail.as_bytes())?;
            return Ok(row);
        }
    };
    row.status = if report.success() {
        CommandStatus::Pass
    } else {
        CommandStatus::Fail
    };
    row.exit_code = report.status.and_then(|status| status.code()).unwrap_or(1);
    if row.status == CommandStatus::Fail && row.exit_code == 0 {
        row.exit_code = 1;
    }
    row.elapsed_ms = u64::try_from(report.elapsed.as_millis()).unwrap_or(u64::MAX);
    row.detail = format!(
        "outcome={:?}, cleanup_complete={}, stdout_truncated={}, stderr_truncated={}",
        report.outcome, report.cleanup.complete, report.stdout.truncated, report.stderr.truncated
    );
    let mut bytes = report.stdout.bytes_retained;
    bytes.extend_from_slice(b"\n--- stderr ---\n");
    bytes.extend_from_slice(&report.stderr.bytes_retained);
    evidence::bytes(&options.output.join(&row.log), &bytes)?;
    Ok(row)
}

fn find(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).find_map(|directory| {
        let path = directory.join(name);
        let metadata = std::fs::metadata(&path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 == 0 {
                return None;
            }
        }
        path.canonicalize().ok()
    })
}

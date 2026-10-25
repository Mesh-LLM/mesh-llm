//! Bounded package-tool execution, attempt logs and success stamps.
use super::policy::{Decision, Environment, decide};
use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

/// Explicit native launcher. A Node executable plus pnpm's JavaScript entry
/// point can be used on Windows instead of executing pnpm.cmd through a shell.
pub(super) struct Tool {
    pub executable: PathBuf,
    pub prefix: Vec<OsString>,
    pub environment: BTreeMap<OsString, (OsString, bool)>,
}

pub(super) fn run(
    ui: &Path,
    environment: &Environment,
    prepare_tool: impl FnOnce() -> DynResult<Tool>,
    timeout: Duration,
    cancellation: &Cancellation,
    logs: &Path,
) -> DynResult<Decision> {
    if timeout.is_zero() || timeout > Duration::from_secs(3600) {
        return Err("UI build timeout must be positive and at most one hour".into());
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or("UI deadline overflow")?;
    let ui = ui.canonicalize()?;
    if cancellation.is_cancelled() {
        return Err(super::failure::Admission::Cancelled.into());
    }
    if decide(&ui, environment)? == Decision::Reuse {
        return Ok(Decision::Reuse);
    }
    let _lock = acquire_build_lock(&ui)?;
    if cancellation.is_cancelled() {
        return Err(super::failure::Admission::Cancelled.into());
    }
    // Another producer may have completed between the first check and locking.
    let decision = decide(&ui, environment)?;
    if decision == Decision::Reuse {
        return Ok(decision);
    }
    let admitted_stamp = environment.stamp_for(&ui)?;
    let tool = prepare_tool()?;
    fs::create_dir_all(logs)?;
    let logs = tempfile::Builder::new()
        .prefix("attempt-")
        .tempdir_in(logs.canonicalize()?)?
        .keep();
    let stamp = ui.join("dist/.mesh-llm-ui-build-env");
    // A failed build may modify dist and its timestamp. Invalidate the previous
    // success stamp before invoking any producer so partial output cannot reuse it.
    match fs::remove_file(&stamp) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    if decision == Decision::InstallAndBuild {
        execute(
            &tool,
            &ui,
            environment,
            Step {
                arguments: &["install", "--frozen-lockfile"],
                label: "install",
            },
            deadline,
            cancellation,
            &logs,
        )?;
    }
    execute(
        &tool,
        &ui,
        environment,
        Step {
            arguments: &["run", "build"],
            label: "build",
        },
        deadline,
        cancellation,
        &logs,
    )?;
    if !super::policy::has_regular_file(&ui.join("dist"))? {
        return Err("UI build succeeded without regular output files".into());
    }
    if environment.stamp_for(&ui)? != admitted_stamp {
        return Err(
            "UI build environment files changed during production; rebuild required".into(),
        );
    }
    fs::write(stamp, admitted_stamp)?;
    Ok(decision)
}

struct Step<'a> {
    arguments: &'a [&'a str],
    label: &'a str,
}

fn execute(
    tool: &Tool,
    ui: &Path,
    build: &Environment,
    step: Step<'_>,
    deadline: Instant,
    cancellation: &Cancellation,
    logs: &Path,
) -> DynResult<()> {
    let Step { arguments, label } = step;
    let execution = deadline
        .checked_duration_since(Instant::now())
        .ok_or(super::failure::Admission::Deadline)?;
    let mut environment = tool
        .environment
        .iter()
        .map(|(name, (value, secret))| {
            (
                name.clone(),
                if *secret {
                    Value::Secret(value.clone())
                } else {
                    Value::Public(value.clone())
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (name, value) in build.projected_variables() {
        environment.insert(name.into(), Value::Public(value.into()));
    }
    environment.insert(
        "ONNXRUNTIME_NODE_INSTALL_CUDA".into(),
        Value::Public("skip".into()),
    );
    let report = process::supervise(
        &ProcessSpec {
            executable: tool.executable.clone(),
            cwd: ui.to_path_buf(),
            environment,
            arguments: tool
                .prefix
                .iter()
                .cloned()
                .chain(arguments.iter().map(|argument| OsString::from(*argument)))
                .map(Value::Public)
                .collect(),
        },
        &Limits {
            execution,
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        OutputFiles {
            stdout: Some(logs.join(format!("{label}.stdout.log"))),
            stderr: Some(logs.join(format!("{label}.stderr.log"))),
        },
    )?;
    if !report.cleanup.complete || !report.success() {
        return Err(super::failure::Failure {
            step: label.to_owned(),
            report: Box::new(report),
            logs: logs.to_path_buf(),
        }
        .into());
    }
    Ok(())
}

fn acquire_build_lock(ui: &Path) -> DynResult<fs::File> {
    let path = ui.join(".mesh-llm-ui-build.lock");
    if let Ok(metadata) = fs::symlink_metadata(&path)
        && !metadata.is_file()
    {
        return Err("UI build lock must be a regular file".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW).mode(0o600);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("UI build lock must be a regular file".into());
    }
    file.try_lock()
        .map_err(|error| format!("UI build is busy or locking failed: {error}"))?;
    Ok(file)
}

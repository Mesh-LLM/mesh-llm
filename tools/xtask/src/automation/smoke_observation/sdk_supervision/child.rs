use super::super::super::{command_interrupt::Interrupt, hf_certify::admission};
use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
pub(super) struct Admitted {
    pub spec: ProcessSpec,
    pub pins: Vec<(PathBuf, String)>,
    pub home: tempfile::TempDir,
}
fn pin(path: &Path) -> DynResult<String> {
    Ok(admission::digest(&admission::read(
        &path.canonicalize()?,
        128 * 1048576,
    )?))
}
pub(super) fn admit(
    root: &Path,
    client: &str,
    python: &Path,
    base: &str,
    model: Option<&str>,
) -> DynResult<Admitted> {
    let (script, lock) = match client {
        "openai" => (
            "ci-openai-python-smoke.py",
            "ci/required-sdk-python/requirements.lock",
        ),
        "langchain" => (
            "ci-langchain-openai-smoke.py",
            "ci/required-sdk-python/requirements.lock",
        ),
        "litellm" => (
            "ci-litellm-smoke.py",
            "ci/required-sdk-python/requirements.lock",
        ),
        "embeddings" => ("ci-openai-embeddings-smoke.py", "ci/canary-python/uv.lock"),
        _ => return Err("SDK owner refuses unknown client".into()),
    };
    if !python.is_absolute() || !std::fs::metadata(python)?.is_file() {
        return Err("SDK needs absolute existing interpreter".into());
    }
    if (client == "openai") != model.is_none()
        || model.is_some_and(|m| m.is_empty() || m.len() > 4096 || m.chars().any(char::is_control))
    {
        return Err("SDK model options refused".into());
    }
    let script = root.join("scripts").join(script);
    let paths = [python.to_path_buf(), script.clone(), root.join(lock)];
    let pins = paths
        .into_iter()
        .map(|p| Ok((p.clone(), pin(&p)?)))
        .collect::<DynResult<Vec<_>>>()?;
    let home = tempfile::Builder::new().prefix("retained-sdk-").tempdir()?;
    let private = home.path().canonicalize()?;
    let mut environment = BTreeMap::new();
    for (key, value) in [
        ("HOME", private.as_os_str()),
        ("TMPDIR", private.as_os_str()),
    ] {
        environment.insert(OsString::from(key), Value::Public(value.into()));
    }
    environment.insert("PATH".into(), Value::Public("/usr/bin:/bin".into()));
    // A setup-python interpreter links its own libpython through this path;
    // without it the loader picks the system libpython and extensions fail.
    if let Some(path) = std::env::var_os("LD_LIBRARY_PATH").filter(|path| !path.is_empty()) {
        environment.insert("LD_LIBRARY_PATH".into(), Value::Public(path));
    }
    environment.insert(
        "NO_PROXY".into(),
        Value::Public("127.0.0.1,localhost".into()),
    );
    let mut arguments: Vec<Value> = [
        OsString::from("-I"),
        script.into_os_string(),
        "--base-url".into(),
        base.into(),
    ]
    .into_iter()
    .map(Value::Public)
    .collect();
    if let Some(model) = model {
        arguments.extend([Value::Public("--model".into()), Value::Public(model.into())]);
    }
    Ok(Admitted {
        spec: ProcessSpec {
            executable: python.to_path_buf(),
            arguments,
            cwd: root.into(),
            environment,
        },
        pins,
        home,
    })
}
fn custody(input: &Admitted) -> bool {
    input
        .pins
        .iter()
        .all(|(p, h)| pin(p).is_ok_and(|actual| actual == *h))
}
pub(super) fn execute(
    input: &Admitted,
    deadline: Instant,
    cancel: &Cancellation,
    files: OutputFiles,
) -> DynResult<process::ProcessReport> {
    if cancel.is_cancelled() || !custody(input) {
        return Err("SDK source/interpreter/lock custody refused before child".into());
    }
    let remaining = deadline.saturating_duration_since(Instant::now());
    let execution = remaining
        .checked_sub(Duration::from_secs(3))
        .filter(|d| !d.is_zero())
        .ok_or("SDK cleanup reserve unavailable")?;
    Ok(process::supervise(
        &input.spec,
        &Limits {
            execution,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        files,
    )?)
}
pub(super) fn run(input: Admitted, receipt: &Path, deadline: Instant) -> DynResult<()> {
    let fresh = matches!(std::fs::symlink_metadata(receipt), Err(e) if e.kind() == std::io::ErrorKind::NotFound);
    if !receipt.is_absolute() || !fresh {
        return Err("SDK receipt must be absolute and fresh".into());
    }
    let parent = receipt
        .parent()
        .ok_or("SDK receipt parent")?
        .canonicalize()?;
    let receipt = parent.join(receipt.file_name().ok_or("SDK receipt leaf")?);
    let log = tempfile::Builder::new()
        .prefix("sdk-capture-")
        .tempdir_in(&parent)?;
    let root = log.path().canonicalize()?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let result = execute(
        &input,
        deadline,
        &cancel,
        OutputFiles {
            stdout: Some(root.join("stdout.log")),
            stderr: Some(root.join("stderr.log")),
        },
    );
    let finished = interrupt.finish();
    let unchanged = custody(&input);
    let observed = result.as_ref().ok();
    let child_success = observed.is_some_and(|r| {
        r.success()
            && r.stdout.line_capture_complete
            && r.stderr.line_capture_complete
            && !r.stdout.truncated
            && !r.stderr.truncated
    });
    let process=observed.map(|r|serde_json::json!({"outcome":format!("{:?}",r.outcome),"exit_code":r.status.and_then(|s|s.code()),"cleanup_complete":r.cleanup.complete,"forced":r.cleanup.forced,"stdout_bytes":r.stdout.bytes_seen,"stderr_bytes":r.stderr.bytes_seen,"stdout_truncated":r.stdout.truncated,"stderr_truncated":r.stderr.truncated,"stdout_suppressed_lines":r.stdout.suppressed_lines,"stderr_suppressed_lines":r.stderr.suppressed_lines}));
    // The durable record is child observations, never SDK qualification or final success.
    let value = serde_json::json!({"schema_version":1,"status":"SDK_CHILD_OBSERVED","sdk_qualified":false,"child_success":child_success,"source_unchanged":unchanged,"pins":input.pins,"process":process,"sanitized_logs":root,"error":if result.is_err(){Some("child setup/supervision failed")}else{None},"terminal_refused_before_publication":finished.is_err()||cancel.is_cancelled()||Instant::now()>=deadline});
    admission::publish(&receipt, &value)?;
    // Keep bounded diagnostic files on failure too; caller owns evidence parent.
    let _retained = log.keep();
    input.home.close()?;
    if !child_success
        || !unchanged
        || finished.is_err()
        || cancel.is_cancelled()
        || Instant::now() >= deadline
    {
        return Err("SDK child or terminal admission failed; observations retained".into());
    }
    Ok(())
}

//! Closed whole OpenAI compatibility wrapper ownership; actual SDK leaves stay unchanged.
#[path = "sdk_compat/admission.rs"]
mod admission;
#[cfg(all(test, unix))]
#[path = "sdk_compat/tests.rs"]
mod tests;
use super::{
    command_interrupt::Interrupt,
    private_state::{FinishError, PrivateState},
};
use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{
    fs,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

pub(crate) const USAGE: &str = "automation sdk-compat run --binary ABS --native-runtime-root ABS --model ABS --python ABS --node ABS --node-modules ABS --state-parent ABS --output ABS --device CPU|MTL0|CUDA0 --api-port N --console-port N [--cuda-visible-devices UUID] [--timeout-secs 1..3600]";

#[cfg(unix)]
pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(crate::cli_output::stdout(), "{USAGE}")?;
        return Ok(());
    }
    let deadline_started = Instant::now();
    let input = admission::Input::admit(root, args)?;
    let deadline = deadline_started + input.timeout;
    let interrupt = Interrupt::install()?;
    let result = execute(root, &input, deadline, &interrupt.cancellation());
    let interruption = interrupt.finish();
    result?;
    interruption?;
    Ok(())
}
#[cfg(unix)]
fn execute(
    root: &Path,
    input: &admission::Input,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    fs::create_dir(&input.output)?;
    let state = PrivateState::create(&input.parent, "sdk-compat")?;
    let result = (|| {
        state.prepare()?;
        let sdk_evidence = state.root().join("sdk-evidence");
        fs::create_dir(&sdk_evidence)?;
        let spec = specification(root, input, &state, &sdk_evidence)?;
        input.custody()?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        let execution = remaining
            .checked_sub(Duration::from_secs(10))
            .filter(|d| !d.is_zero())
            .ok_or("SDK compatibility cleanup reserve unavailable")?;
        let report = process::supervise(
            &spec,
            &Limits {
                execution,
                graceful_shutdown: Duration::from_secs(5),
                forced_shutdown: Duration::from_secs(3),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            cancel,
            OutputFiles {
                stdout: Some(input.output.join("stdout.log")),
                stderr: Some(input.output.join("stderr.log")),
            },
        )?;
        preserve_sdk_evidence(&sdk_evidence, &input.output)?;
        input.custody()?;
        let passed = report.success()
            && !report.cleanup.forced
            && report.cleanup.complete
            && report.cleanup.failure.is_none()
            && !cancel.is_cancelled();
        fs::write(
            input.output.join("receipt.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema":1, "owner":"sdk_compat", "status":if passed {"execution_passed_cleanup_pending"} else {"rejected"},
                "exit_code":report.status.and_then(|s|s.code()), "cleanup_complete":report.cleanup.complete,
                "forced":report.cleanup.forced, "source_unchanged":true,
                "sdk_qualified":false, "pins":input.pins(), "capture_mapping":"original private sanitized_logs basename is retained beside each unchanged SDK receipt", "scope":"actual fixed whole wrapper execution; environment/model admission remains separate"
            }))?,
        )?;
        if !passed {
            return Err("SDK compatibility execution or cleanup refused".into());
        }
        Ok(())
    })();
    match state.finish(result) {
        Ok(()) => {
            let path = input.output.join("receipt.json");
            let mut receipt: serde_json::Value =
                serde_json::from_slice(&admission::bounded_file(&path, 1048576)?)?;
            receipt["status"] = "passed".into();
            receipt["private_state_cleanup_complete"] = true.into();
            fs::write(path, serde_json::to_vec_pretty(&receipt)?)?;
            Ok(())
        }
        Err(FinishError::Prior(error)) => Err(error),
        Err(FinishError::Deletion { .. }) => {
            Err("SDK compatibility private state cleanup failed".into())
        }
    }
}

#[cfg(unix)]
fn specification(
    root: &Path,
    input: &admission::Input,
    state: &PrivateState,
    sdk_evidence: &Path,
) -> DynResult<ProcessSpec> {
    let mut env = state.environment(&input.native);
    env.retain(|key, _| {
        [
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "MESH_LLM_CONFIG",
            "MESH_LLM_RUNTIME_ROOT",
            "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_RUNTIME_DIR",
            "TMPDIR",
            "TEMP",
            "TMP",
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
        ]
        .iter()
        .any(|allowed| key == *allowed)
    });
    env.insert(
        "PATH".into(),
        Value::Public(
            format!(
                "{}:/usr/bin:/bin",
                input.node.parent().ok_or("Node parent missing")?.display()
            )
            .into(),
        ),
    );
    for (key, value) in [
        ("MESH_REQUIRED_SDK_PYTHON", input.python.as_os_str()),
        ("MESH_LLM_AUTOMATION_BIN", input.controller.as_os_str()),
        ("NODE_PATH", input.node_modules.as_os_str()),
        ("MESH_COMPAT_STATE_PARENT", state.root().as_os_str()),
        ("MESH_COMPAT_SDK_EVIDENCE_PARENT", sdk_evidence.as_os_str()),
    ] {
        env.insert(key.into(), Value::Public(value.into()));
    }
    for (key, value) in [
        ("MESH_COMPAT_DEVICE", input.device.clone()),
        ("MESH_COMPAT_API_PORT", input.api.to_string()),
        ("MESH_COMPAT_CONSOLE_PORT", input.console.to_string()),
        (
            "MESH_COMPAT_LOG",
            state.root().join("server.log").display().to_string(),
        ),
        ("MESH_COMPAT_MAX_WAIT", "180".into()),
        ("MESH_COMPAT_SDK_TIMEOUT_SECS", "240".into()),
        ("MESH_RELEASE_ATTESTATION_EXPECTED_STATUS", "missing".into()),
    ] {
        env.insert(key.into(), Value::Public(value.into()));
    }
    match &input.cuda {
        Some(value) => {
            env.insert("CUDA_VISIBLE_DEVICES".into(), Value::Secret(value.into()));
        }
        None => {
            env.remove(std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES"));
        }
    }
    Ok(ProcessSpec {
        executable: "/bin/bash".into(),
        cwd: root.into(),
        environment: env,
        arguments: vec![
            Value::Public(input.wrapper.as_os_str().into()),
            Value::Public(input.binary.as_os_str().into()),
            Value::Public(
                input
                    .binary
                    .parent()
                    .ok_or("binary parent missing")?
                    .as_os_str()
                    .into(),
            ),
            Value::Public(input.model.as_os_str().into()),
        ],
    })
}

#[cfg(unix)]
fn preserve_sdk_evidence(source: &Path, output: &Path) -> DynResult<()> {
    let mut dirs = fs::read_dir(source)?.collect::<Result<Vec<_>, _>>()?;
    if dirs.len() > 1 {
        return Err("unexpected compatibility SDK evidence roster".into());
    }
    let Some(dir) = dirs.pop() else {
        return Ok(());
    };
    if !dir.file_type()?.is_dir() {
        return Err("SDK evidence directory refused".into());
    }
    for client in ["openai", "litellm", "langchain"] {
        let receipt = dir.path().join(format!("{client}-sdk.json"));
        if receipt.exists() {
            let bytes = admission::bounded_file(&receipt, 1048576)?;
            preserve_capture(&dir.path(), output, &bytes)?;
            fs::write(output.join(format!("{client}-sdk.json")), bytes)?;
        }
    }
    Ok(())
}

fn preserve_capture(source: &Path, output: &Path, receipt: &[u8]) -> DynResult<()> {
    let record: serde_json::Value = serde_json::from_slice(receipt)?;
    let original = record
        .get("sanitized_logs")
        .and_then(serde_json::Value::as_str)
        .ok_or("SDK capture locator missing")?;
    let original = Path::new(original);
    let name = original.file_name().ok_or("SDK capture leaf missing")?;
    if original.parent() != Some(source)
        || !name.to_string_lossy().starts_with("sdk-capture-")
        || !fs::symlink_metadata(original)?.is_dir()
    {
        return Err("SDK capture locator outside owned evidence".into());
    }
    let destination = output.join(name);
    fs::create_dir(&destination)?;
    let entries = fs::read_dir(original)?.collect::<Result<Vec<_>, _>>()?;
    if entries.len() != 2
        || entries
            .iter()
            .any(|e| e.file_name() != "stdout.log" && e.file_name() != "stderr.log")
    {
        return Err("SDK capture roster refused".into());
    }
    for leaf in ["stdout.log", "stderr.log"] {
        fs::write(
            destination.join(leaf),
            admission::bounded_file(&original.join(leaf), 1048576)?,
        )?;
    }
    Ok(())
}

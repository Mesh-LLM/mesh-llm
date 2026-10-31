use crate::{
    command::DynResult,
    process::{self, ProcessSpec, Value},
};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) fn executable(name: &str) -> DynResult<PathBuf> {
    if Path::new(name).is_absolute() {
        return Ok(name.into());
    }
    std::env::split_paths(&std::env::var_os("PATH").ok_or("PATH missing")?)
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| format!("missing executable: {name}").into())
}
pub(super) fn spec(root: &Path, executable: &str, args: &[String]) -> DynResult<ProcessSpec> {
    let mut environment: std::collections::BTreeMap<_, _> = std::env::vars_os()
        .filter(|(key, _)| {
            matches!(
                key.to_str(),
                Some(
                    "PATH"
                        | "HOME"
                        | "TMPDIR"
                        | "CUDA_HOME"
                        | "CUDA_PATH"
                        | "SSL_CERT_FILE"
                        | "SSL_CERT_DIR"
                        | "HF_HOME"
                        | "HF_HUB_CACHE"
                )
            )
        })
        .map(|(key, value)| (key, Value::Public(value)))
        .collect();
    environment.insert("PYTHONDONTWRITEBYTECODE".into(), Value::Public("1".into()));
    environment.insert("HF_HUB_OFFLINE".into(), Value::Public("1".into()));
    environment.insert(
        "HF_HUB_CACHE".into(),
        Value::Public(super::agentic_config::tokenizer_hub_cache(root).into_os_string()),
    );
    Ok(ProcessSpec {
        executable: self::executable(executable)?,
        arguments: args.iter().map(|arg| Value::Public(arg.into())).collect(),
        cwd: root.into(),
        environment,
    })
}
pub(super) fn limits(seconds: u64) -> process::Limits {
    process::Limits {
        execution: Duration::from_secs(seconds),
        graceful_shutdown: Duration::from_secs(15),
        forced_shutdown: Duration::from_secs(5),
        retained_bytes_per_stream: 65536,
        readiness: process::Readiness::None,
        completion: process::Completion::Exit,
    }
}
pub(super) fn run(spec: &ProcessSpec, seconds: u64, log: &Path) -> DynResult<()> {
    capture(spec, seconds, log, None).map(|_| ())
}

/// Parse trusted public CLI/API JSON without interpreting redacted diagnostics
/// as machine data. Raw bytes remain bounded in memory and are never logged.
pub(super) fn public_json(
    spec: &ProcessSpec,
    seconds: u64,
    log: &Path,
) -> DynResult<serde_json::Value> {
    let report = capture(
        spec,
        seconds,
        log,
        std::num::NonZeroUsize::new(4 * 1024 * 1024),
    )?;
    let bytes = report
        .stdout
        .as_ref()
        .ok_or("public JSON capture absent")?
        .as_bytes();
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(serde_json::Value::Null);
    }
    Ok(serde_json::from_slice(bytes)?)
}

fn capture(
    spec: &ProcessSpec,
    seconds: u64,
    log: &Path,
    stdout: Option<std::num::NonZeroUsize>,
) -> DynResult<process::RawProcessReport> {
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = process::supervise_raw_with_files(
        spec,
        &limits(seconds),
        &interrupt.cancellation(),
        process::OutputFiles {
            stdout: Some(log.with_extension("stdout.log")),
            stderr: Some(log.with_extension("stderr.log")),
        },
        process::RawCaptureOptions {
            stdout,
            stderr: None,
        },
    );
    let finish = interrupt.finish();
    let report = result?;
    finish?;
    if !report.process.success() {
        return Err(format!(
            "AgentX child failed: {}; see {}",
            spec.executable.display(),
            log.display()
        )
        .into());
    }
    Ok(report)
}
pub(super) fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).into()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_public_json_survives_redaction_without_disabling_log_redaction() {
        let root = tempfile::tempdir().unwrap();
        let log = root.path().join("public-json");
        let spec = spec(
            root.path(),
            "sh",
            &strings(&[
                "-c",
                r#"printf '%s' '{"context_tokens":8,"model_cache":"/models/tokenizer"}'"#,
            ]),
        )
        .unwrap();
        let value = public_json(&spec, 10, &log).unwrap();
        assert_eq!(value["context_tokens"], 8);
        assert_eq!(value["model_cache"], "/models/tokenizer");
        let diagnostic = std::fs::read_to_string(log.with_extension("stdout.log")).unwrap();
        assert!(diagnostic.contains("[output line suppressed]"));
        assert!(!diagnostic.contains("/models/tokenizer"));
    }
}

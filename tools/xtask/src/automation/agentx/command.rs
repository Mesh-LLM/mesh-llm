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
    let environment = std::env::vars_os()
        .filter(|(key, _)| {
            matches!(
                key.to_str(),
                Some(
                    "PATH"
                        | "HOME"
                        | "TMPDIR"
                        | "LD_LIBRARY_PATH"
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
    let interrupt = crate::command_interrupt::Interrupt::install()?;
    let result = process::supervise_raw_with_files(
        spec,
        &limits(seconds),
        &interrupt.cancellation(),
        process::OutputFiles {
            stdout: Some(log.with_extension("stdout.log")),
            stderr: Some(log.with_extension("stderr.log")),
        },
        process::RawCaptureOptions {
            stdout: None,
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
    Ok(())
}
pub(super) fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).into()).collect()
}

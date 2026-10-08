//! Bounded native Just parsing under the inventory command's interrupt scope.
use crate::command::DynResult;
use crate::command_interrupt::Interrupt;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

const JSON_LIMIT: usize = 32 * 1024 * 1024;
const DIAGNOSTIC_LIMIT: usize = 1024 * 1024;
thread_local! { static ACTIVE: RefCell<Option<Cancellation>> = const { RefCell::new(None) }; }

struct Reset(Option<Cancellation>);
impl Drop for Reset {
    fn drop(&mut self) {
        ACTIVE.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

pub(super) fn operation<T>(body: impl FnOnce() -> DynResult<T>) -> DynResult<T> {
    // An existing unrelated signal owner is an error, never silently bypassed.
    let interrupt = Interrupt::install()?;
    let previous = ACTIVE.with(|slot| slot.replace(Some(interrupt.cancellation())));
    let reset = Reset(previous);
    let result = body();
    drop(reset);
    // Final interruption wins even if the body reached its last successful check.
    interrupt.finish()?;
    result
}

pub(super) fn dump(root: &Path) -> DynResult<Vec<u8>> {
    let cancellation = ACTIVE
        .with(|slot| slot.borrow().clone())
        .unwrap_or_default();
    check(&cancellation)?;
    let cwd = root.canonicalize()?;
    let environment = inherited_environment();
    let spec = ProcessSpec {
        executable: tool(&cwd, std::env::var_os("PATH"))?,
        arguments: ["--justfile", "Justfile", "--dump", "--dump-format", "json"]
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        cwd,
        environment,
    };
    invoke(
        &spec,
        &limits(Duration::from_secs(120)),
        &cancellation,
        JSON_LIMIT,
    )
}

fn inherited_environment() -> BTreeMap<OsString, Value> {
    std::env::vars_os()
        .map(|(key, value)| {
            let name = key.to_string_lossy();
            let secret = !value.is_empty()
                && ["_TOKEN", "_KEY", "_SECRET", "_PASSWORD"]
                    .iter()
                    .any(|suffix| name.ends_with(suffix));
            (
                key,
                if secret {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                },
            )
        })
        .collect()
}

fn limits(execution: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: Duration::from_secs(2),
        forced_shutdown: Duration::from_secs(5),
        retained_bytes_per_stream: DIAGNOSTIC_LIMIT,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

fn check(cancellation: &Cancellation) -> DynResult<()> {
    if cancellation.is_cancelled() {
        return Err("Just recipe: native dump cancelled".into());
    }
    Ok(())
}

fn invoke(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    json_limit: usize,
) -> DynResult<Vec<u8>> {
    check(cancellation)?;
    let report = process::supervise_raw(
        spec,
        limits,
        cancellation,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(json_limit),
            stderr: NonZeroUsize::new(DIAGNOSTIC_LIMIT),
        },
    )?;
    if report.process.failure.is_some() || !report.process.cleanup.complete {
        return Err(format!(
            "Just recipe: native dump supervision failed: {:?}",
            report.process
        )
        .into());
    }
    check(cancellation)?;
    if !report.process.success() {
        if report.process.outcome == process::Outcome::Exited {
            let stderr = String::from_utf8_lossy(&report.process.stderr.bytes_retained);
            let stdout = String::from_utf8_lossy(&report.process.stdout.bytes_retained);
            let diagnostic = if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            };
            return Err(format!("Just recipe: parse failed: {diagnostic}").into());
        }
        return Err(format!(
            "Just recipe: native dump stopped: {:?}",
            report.process.outcome
        )
        .into());
    }
    check(cancellation)?;
    report
        .stdout
        .map(|bytes| bytes.as_bytes().to_vec())
        .ok_or_else(|| "Just recipe: native dump stdout missing".into())
}

fn tool(root: &Path, path: Option<OsString>) -> DynResult<PathBuf> {
    #[cfg(unix)]
    let search = path.unwrap_or_else(|| "/usr/bin:/bin".into());
    #[cfg(windows)]
    let search = path.unwrap_or_default();
    #[cfg(unix)]
    let name = "just";
    #[cfg(windows)]
    let name = "just.exe";
    let directories = std::env::split_paths(&search);
    for directory in directories {
        let candidate = if directory.is_absolute() {
            directory.join(name)
        } else {
            root.join(directory).join(name)
        };
        let Ok(metadata) = candidate.metadata() else {
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
        // Preserve the PATH-selected spelling: symlink-sensitive native tools stay intact.
        return Ok(candidate);
    }
    Err("Just recipe: native just executable unavailable on PATH".into())
}

#[cfg(all(test, unix))]
#[path = "just_process_tests.rs"]
mod tests;

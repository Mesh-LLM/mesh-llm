use crate::automation::command_interrupt::Interrupt;
use crate::command::DynResult;
use crate::process::{self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    time::Duration,
};

thread_local! { static ACTIVE: RefCell<Option<crate::process::Cancellation>> = const { RefCell::new(None) }; }

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
    }
}

pub(super) fn operation<T>(body: impl FnOnce() -> DynResult<T>) -> DynResult<T> {
    let interrupt = Interrupt::install()?;
    ACTIVE.with(|slot| *slot.borrow_mut() = Some(interrupt.cancellation()));
    let guard = ActiveGuard;
    let result = body();
    drop(guard);
    interrupt.finish()?;
    result
}

pub(super) fn cancellation() -> crate::process::Cancellation {
    ACTIVE
        .with(|slot| slot.borrow().clone())
        .unwrap_or_default()
}
pub(super) fn check() -> DynResult<()> {
    if cancellation().is_cancelled() {
        Err("package operation interrupted".into())
    } else {
        Ok(())
    }
}

/// Rollback is bounded even when the triggering command was cancelled.
pub(super) fn cleanup<T>(body: impl FnOnce() -> T) -> T {
    let previous = ACTIVE.with(|slot| slot.replace(Some(crate::process::Cancellation::default())));
    struct Reset(Option<crate::process::Cancellation>);
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _reset = Reset(previous);
    body()
}

/// Fixed owner-selected Git probes or scratch restoration, without credentials/hooks.
pub(super) fn git(
    root: &Path,
    arguments: &[OsString],
    capture: Option<&Path>,
) -> DynResult<Vec<u8>> {
    let mut args = vec![
        "--no-optional-locks".into(),
        "-c".into(),
        "core.hooksPath=/dev/null".into(),
    ];
    args.extend_from_slice(arguments);
    let environment: BTreeMap<_, _> = [
        ("PATH", "/usr/bin:/bin"),
        ("LC_ALL", "C"),
        ("GIT_MASTER", "1"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
        ("GIT_CONFIG_GLOBAL", "/dev/null"),
        ("GIT_TERMINAL_PROMPT", "0"),
        ("GIT_ALLOW_PROTOCOL", "file"),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), Value::Public(value.into())))
    .collect();
    let owns = ACTIVE.with(|slot| slot.borrow().is_none());
    let interrupt = if owns {
        Some(Interrupt::install()?)
    } else {
        None
    };
    let cancellation = interrupt
        .as_ref()
        .map(Interrupt::cancellation)
        .unwrap_or_else(cancellation);
    let report = process::supervise(
        &ProcessSpec {
            executable: PathBuf::from("/usr/bin/git"),
            arguments: args.into_iter().map(Value::Public).collect(),
            cwd: root.canonicalize()?,
            environment,
        },
        &Limits {
            execution: Duration::from_secs(60),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 4 * 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &cancellation,
        OutputFiles {
            stdout: capture.map(Path::to_path_buf),
            stderr: None,
        },
    );
    if let Some(interrupt) = interrupt {
        interrupt.finish()?;
    }
    let report = report?;
    if !report.success()
        || report.stderr.truncated
        || (capture.is_none() && report.stdout.truncated)
    {
        return Err(format!(
            "canary package Git admission failed: {:?}, status {:?}",
            report.outcome, report.status
        )
        .into());
    }
    Ok(report.stdout.bytes_retained)
}

pub(super) fn text(root: &Path, args: &[&str]) -> DynResult<String> {
    let args = args.iter().map(OsString::from).collect::<Vec<_>>();
    Ok(String::from_utf8(git(root, &args, None)?)?
        .trim()
        .to_owned())
}

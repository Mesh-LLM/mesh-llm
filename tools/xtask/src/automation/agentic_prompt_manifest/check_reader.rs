//! Probe the exact optional reader selected by the prompt frontend before downloads.
use super::parquet_input::{reader_binary, reader_environment};
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process};
use std::{num::NonZeroUsize, time::Duration};

pub(super) fn run() -> DynResult<()> {
    let reader = reader_binary()?;
    if !reader.is_file() {
        return Err("optional trajectory reader must be a regular executable file".into());
    }
    let directory = tempfile::tempdir()?;
    let interrupt = Interrupt::install()?;
    let result = process::supervise_raw(
        &process::ProcessSpec {
            executable: reader,
            cwd: directory.path().to_path_buf(),
            arguments: vec![process::Value::Public("--help".into())],
            environment: reader_environment(),
        },
        &process::Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &interrupt.cancellation(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(4096),
            stderr: NonZeroUsize::new(4096),
        },
    );
    let interrupt_result = interrupt.finish();
    let close_result = directory.close();
    let raw = result?;
    interrupt_result?;
    close_result?;
    let report = raw.process;
    let stdout = raw.stdout.ok_or("missing reader preflight stdout")?;
    let stderr = raw.stderr.ok_or("missing reader preflight stderr")?;
    if report.outcome != process::Outcome::Exited
        || !report.status.is_some_and(|status| status.success())
        || report.failure.is_some()
        || !report.cleanup.complete
        || report.cleanup.forced
        || report.cleanup.graceful_signal_failed
        || report.cleanup.failure.is_some()
        || stdout.as_bytes().len() as u64 != report.stdout.bytes_seen
        || stderr.as_bytes().len() as u64 != report.stderr.bytes_seen
        || stdout.as_bytes() != b"trajectory-reader --request FILE --response FILE\n"
    {
        return Err(format!("trajectory reader preflight failed: {:?}", report.outcome).into());
    }
    println!("trajectory reader executable ready; corpus validation remains separate");
    Ok(())
}

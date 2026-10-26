//! Forward only the supervisor's redacted files, outside its bounded callbacks.
use crate::process::{self, Cancellation, Limits, OutputFiles, ProcessReport, ProcessSpec};
use std::{
    fs::File,
    io::{self, Read, Write},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

pub(super) fn supervise(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    files: OutputFiles,
) -> Result<ProcessReport, process::Failure> {
    let complete = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let forward = scope.spawn(|| {
            let mut stdout = None;
            let mut stderr = None;
            loop {
                drain(
                    files.stdout.as_deref(),
                    &mut stdout,
                    &mut crate::cli_output::stdout(),
                )?;
                drain(
                    files.stderr.as_deref(),
                    &mut stderr,
                    &mut crate::cli_output::stderr(),
                )?;
                if complete.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            // The completion flag is published only after capture finished.
            drain(
                files.stdout.as_deref(),
                &mut stdout,
                &mut crate::cli_output::stdout(),
            )?;
            drain(
                files.stderr.as_deref(),
                &mut stderr,
                &mut crate::cli_output::stderr(),
            )
        });
        let done = Done(&complete);
        let report = process::supervise(
            spec,
            limits,
            cancellation,
            OutputFiles {
                stdout: files.stdout.clone(),
                stderr: files.stderr.clone(),
            },
        );
        drop(done);
        match forward.join() {
            Ok(Ok(())) => {
                if let Ok(report) = &report
                    && (report.stdout.truncated || report.stderr.truncated)
                {
                    writeln!(crate::cli_output::stderr(),
                        "canary live output was truncated; complete gate logs and CTest results are in the build evidence artifact")
                        .map_err(|_| process::Failure::InvalidSpec("canary truncation notice failed"))?;
                }
                report
            }
            Ok(Err(_)) | Err(_) => Err(process::Failure::InvalidSpec(
                "canary log forwarding failed",
            )),
        }
    })
}

struct Done<'a>(&'a AtomicBool);
impl Drop for Done<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn drain(
    path: Option<&std::path::Path>,
    reader: &mut Option<File>,
    sink: &mut impl Write,
) -> io::Result<()> {
    if reader.is_none() {
        let Some(path) = path else {
            return Ok(());
        };
        match File::open(path) {
            Ok(file) => *reader = Some(file),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        }
    }
    let Some(reader) = reader.as_mut() else {
        return Ok(());
    };
    let mut bytes = [0_u8; 8192];
    loop {
        let count = reader.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        sink.write_all(&bytes[..count])?;
    }
    sink.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forwarding_preserves_incremental_bytes_without_repeating_prior_content() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("output");
        let mut reader = None;
        let mut bytes = Vec::new();
        drain(Some(&path), &mut reader, &mut bytes).unwrap();
        std::fs::write(&path, b"first\n").unwrap();
        drain(Some(&path), &mut reader, &mut bytes).unwrap();
        File::options()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"second\n")
            .unwrap();
        drain(Some(&path), &mut reader, &mut bytes).unwrap();
        drain(Some(&path), &mut reader, &mut bytes).unwrap();
        assert_eq!(bytes, b"first\nsecond\n");
    }

    #[cfg(unix)]
    #[test]
    fn build_output_keeps_terminal_failure_beyond_old_live_log_bound() {
        use crate::process::{Completion, Readiness, Value};
        let directory = tempfile::tempdir().unwrap();
        let mut output = b"compiler progress\n".repeat(5000);
        output.extend_from_slice(b"native test failed: final diagnostic\n");
        std::fs::write(directory.path().join("fixture-output"), &output).unwrap();
        let log = directory.path().join("stdout.log");
        let report = supervise(
            &ProcessSpec {
                executable: "/bin/sh".into(),
                cwd: directory.path().into(),
                arguments: ["-c", "/bin/cat fixture-output; exit 73"]
                    .map(|arg| Value::Public(arg.into()))
                    .into(),
                environment: Default::default(),
            },
            &Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: super::super::WRAPPER_LOG_BYTES,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles {
                stdout: Some(log.clone()),
                stderr: None,
            },
        )
        .unwrap();
        assert_eq!(report.status.unwrap().code(), Some(73));
        assert!(report.cleanup.complete);
        assert!(!report.stdout.truncated);
        assert_eq!(std::fs::read(log).unwrap(), output);
    }
}

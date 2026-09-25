//! Bounded child processes with piped stdin and fully captured streams.

use crate::command::DynResult;
use std::io::{ErrorKind, Read, Write};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const DEADLINE: Duration = Duration::from_secs(600);
const POLL: Duration = Duration::from_millis(10);

/// Everything a caller can observe from one process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Captured {
    pub(super) stdout: Vec<u8>,
    pub(super) stderr: Vec<u8>,
    pub(super) code: Option<i32>,
}

type Pipe = JoinHandle<std::io::Result<Vec<u8>>>;

fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> Pipe {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            pipe.read_to_end(&mut bytes)?;
        }
        Ok(bytes)
    })
}

fn collect(pipe: Pipe) -> DynResult<Vec<u8>> {
    Ok(pipe.join().map_err(|_| "stream reader panicked")??)
}

fn wait_bounded(child: &mut Child, program: &str) -> DynResult<ExitStatus> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if started.elapsed() > DEADLINE {
            child.kill()?;
            child.wait()?;
            return Err(
                format!("{program} exceeded {}s and was stopped", DEADLINE.as_secs()).into(),
            );
        }
        thread::sleep(POLL);
    }
}

/// Runs `command` to completion within the deadline. An early-exiting child
/// may close stdin before reading it all; that broken pipe is not an error.
pub(super) fn run_bounded(command: &mut Command, stdin: &[u8]) -> DynResult<Captured> {
    let program = format!("{}", command.get_program().to_string_lossy());
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("unable to start {program}: {error}"))?;
    let mut pipe = child.stdin.take().ok_or("missing stdin pipe")?;
    let input = stdin.to_vec();
    let writer = thread::spawn(move || match pipe.write_all(&input) {
        Err(error) if error.kind() != ErrorKind::BrokenPipe => Err(error),
        _ => Ok(()),
    });
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let status = wait_bounded(&mut child, &program)?;
    writer.join().map_err(|_| "stdin writer panicked")??;
    Ok(Captured {
        stdout: collect(stdout)?,
        stderr: collect(stderr)?,
        code: status.code(),
    })
}

/// The first byte where two buffers differ, or `None` when they are equal.
pub(super) fn first_difference(expected: &[u8], actual: &[u8]) -> Option<String> {
    if expected == actual {
        return None;
    }
    let offset = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap_or(expected.len().min(actual.len()));
    let byte = |bytes: &[u8]| {
        bytes
            .get(offset)
            .map_or_else(|| "end".to_owned(), |value| format!("0x{value:02x}"))
    };
    Some(format!(
        "first differing byte at offset {offset} (expected {}, actual {}; lengths {} and {})",
        byte(expected),
        byte(actual),
        expected.len(),
        actual.len()
    ))
}

/// Status, stdout and stderr compared in that order.
pub(super) fn process_difference(expected: &Captured, actual: &Captured) -> Option<String> {
    if expected.code != actual.code {
        return Some(format!("status {:?} != {:?}", expected.code, actual.code));
    }
    if let Some(detail) = first_difference(&expected.stdout, &actual.stdout) {
        return Some(format!("stdout: {detail}"));
    }
    first_difference(&expected.stderr, &actual.stderr).map(|detail| format!("stderr: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_shadow_first_difference_names_offset_and_bytes() {
        assert_eq!(first_difference(b"abc", b"abc"), None);
        let detail = first_difference(b"abc", b"abd").unwrap_or_default();
        assert!(detail.contains("offset 2") && detail.contains("0x63") && detail.contains("0x64"));
        let shorter = first_difference(b"abc", b"ab").unwrap_or_default();
        assert!(shorter.contains("offset 2") && shorter.contains("actual end"));
    }
}

//! The process adapter for `release notes-link`: the only place its `git`
//! and `gh` children are spawned. Tests substitute [`ReleaseHost`] or put
//! stub executables first on `PATH`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How a child ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    Code(i32),
    Signal(i32),
    /// Killed after its timeout, like `subprocess.run(timeout=...)`.
    TimedOut,
}

/// A finished child with its raw captured streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProcessOutput {
    pub(crate) exit: Exit,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}

/// A spawn failure and the file name Python's `OSError` would name: the
/// working directory when it cannot be entered, else the program.
#[derive(Debug)]
pub(crate) struct SpawnFailure {
    pub(crate) error: std::io::Error,
    pub(crate) filename: PathBuf,
}

pub(crate) type Spawned = Result<ProcessOutput, SpawnFailure>;

/// Runs `git` and `gh` with captured output and inherited stdin.
pub(crate) trait ReleaseHost {
    fn git(&mut self, args: &[String], cwd: Option<&Path>) -> Spawned;
    fn gh(&mut self, args: &[String], timeout: Duration) -> Spawned;
}

/// The executables found on `PATH`.
pub(crate) struct SystemHost;

impl ReleaseHost for SystemHost {
    fn git(&mut self, args: &[String], cwd: Option<&Path>) -> Spawned {
        spawn("git", args, cwd, None)
    }

    fn gh(&mut self, args: &[String], timeout: Duration) -> Spawned {
        spawn("gh", args, None, Some(timeout))
    }
}

fn cwd_failure(cwd: &Path) -> Option<SpawnFailure> {
    let failure = |error| SpawnFailure {
        error,
        filename: cwd.to_path_buf(),
    };
    match std::fs::metadata(cwd) {
        Err(error) => Some(failure(error)),
        Ok(metadata) if !metadata.is_dir() => Some(failure(std::io::Error::from_raw_os_error(20))),
        Ok(_) => None,
    }
}

fn spawn(program: &str, args: &[String], cwd: Option<&Path>, timeout: Option<Duration>) -> Spawned {
    if let Some(failure) = cwd.and_then(cwd_failure) {
        return Err(failure);
    }
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let failure = |error| SpawnFailure {
        error,
        filename: PathBuf::from(program),
    };
    let mut child = command.spawn().map_err(failure)?;
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let deadline = timeout.map(|limit| Instant::now() + limit);
    let exit = loop {
        if let Some(status) = child.try_wait().map_err(failure)? {
            break exit_of(status);
        }
        if deadline.is_some_and(|end| Instant::now() >= end) {
            let _killed = child.kill();
            let _reaped = child.wait();
            break Exit::TimedOut;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    Ok(ProcessOutput {
        exit,
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

#[cfg(unix)]
fn exit_of(status: std::process::ExitStatus) -> Exit {
    use std::os::unix::process::ExitStatusExt as _;
    match (status.code(), status.signal()) {
        (Some(code), _) => Exit::Code(code),
        (None, Some(signal)) => Exit::Signal(signal),
        (None, None) => Exit::Code(1),
    }
}

#[cfg(not(unix))]
fn exit_of(status: std::process::ExitStatus) -> Exit {
    Exit::Code(status.code().unwrap_or(1))
}

fn drain(pipe: Option<impl Read + Send + 'static>) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            let _partial = pipe.read_to_end(&mut bytes);
        }
        bytes
    })
}

/// `subprocess`'s text mode: strict UTF-8, then universal newlines. Stdout
/// is decoded before stderr, so its error wins.
pub(crate) fn text_streams(output: &ProcessOutput) -> Result<(String, String), String> {
    use crate::prepared_input::text_io::decode_utf8;
    let translate = |text: String| text.replace("\r\n", "\n").replace('\r', "\n");
    let stdout = decode_utf8(output.stdout.clone())?;
    let stderr = decode_utf8(output.stderr.clone())?;
    Ok((translate(stdout), translate(stderr)))
}

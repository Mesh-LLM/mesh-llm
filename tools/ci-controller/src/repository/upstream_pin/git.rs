//! Git invocation for the pin guard, with the legacy fail-closed wording for
//! failures and timeouts.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A condition under which the guard cannot prove the pin update safe.
#[derive(Debug)]
pub(super) struct PinGuardError(pub(super) String);

pub(super) struct GitOutput {
    pub(super) code: Option<i32>,
    pub(super) stdout: String,
    stderr: String,
}

impl GitOutput {
    pub(super) fn success(&self) -> bool {
        self.code == Some(0)
    }

    /// `stderr.strip() or stdout.strip() or "no git output"`.
    pub(super) fn detail(&self) -> &str {
        [self.stderr.trim(), self.stdout.trim()]
            .into_iter()
            .find(|text| !text.is_empty())
            .unwrap_or("no git output")
    }
}

/// Runs `git -C repo args...`; a timeout is fatal and fails closed.
pub(super) fn run(
    repo: &Path,
    args: &[&str],
    timeout: Option<Duration>,
) -> Result<GitOutput, PinGuardError> {
    let spawn_error =
        |error: std::io::Error| PinGuardError(format!("git {} failed: {error}", args.join(" ")));
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(spawn_error)?;
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let deadline = timeout.map(|limit| (Instant::now() + limit, limit));
    let status = loop {
        if let Some(status) = child.try_wait().map_err(spawn_error)? {
            break status;
        }
        if let Some((end, limit)) = deadline
            && Instant::now() >= end
        {
            let _killed = child.kill();
            let _reaped = child.wait();
            return Err(PinGuardError(format!(
                "git {} timed out after {} seconds; the guard cannot prove llama.cpp upstream ancestry and will fail closed",
                args.join(" "),
                limit.as_secs()
            )));
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Ok(GitOutput {
        code: status.code(),
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

/// Like [`run`] without a timeout, turning a nonzero status into an error.
pub(super) fn checked(repo: &Path, args: &[&str]) -> Result<GitOutput, PinGuardError> {
    let output = run(repo, args, None)?;
    if !output.success() {
        return Err(PinGuardError(format!(
            "git {} failed: {}",
            args.join(" "),
            output.detail()
        )));
    }
    Ok(output)
}

fn drain(pipe: Option<impl Read + Send + 'static>) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            let _partial = pipe.read_to_end(&mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

//! Closed lifecycle for this executable's package-acquisition worker only.
//! Capture descriptors avoid pipe-drain threads and inherited-writer hangs.
use anyhow::{Context as _, Result, bail, ensure};
use std::{
    fs::File,
    io::{Read as _, Seek as _},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};
const STDOUT_CAP: u64 = 64 * 1024;
const STDERR_CAP: u64 = 256 * 1024;
struct Capture {
    file: tempfile::NamedTempFile,
    cap: u64,
}
impl Capture {
    fn new(directory: &std::path::Path, cap: u64) -> Result<Self> {
        Ok(Self {
            file: tempfile::NamedTempFile::new_in(directory)?,
            cap,
        })
    }
    fn descriptor(&self) -> Result<File> {
        Ok(self.file.as_file().try_clone()?)
    }
    fn check(&self) -> Result<()> {
        ensure!(
            self.file.as_file().metadata()?.len() <= self.cap,
            "acquisition worker capture exceeded bound"
        );
        Ok(())
    }
    fn bytes(&mut self) -> Result<Vec<u8>> {
        self.check()?;
        self.file.as_file_mut().rewind()?;
        let mut bytes = Vec::new();
        self.file
            .as_file_mut()
            .take(self.cap + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= self.cap,
            "worker capture grew after cleanup"
        );
        Ok(bytes)
    }
}
struct OwnedChild {
    child: Child,
    cleanup_required: bool,
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.cleanup_required {
            let _ = terminate(&mut self.child);
        }
    }
}
pub(super) fn run(mut command: Command, budget: Duration) -> Result<Vec<u8>> {
    ensure!(!budget.is_zero(), "worker budget must be positive");
    let until = Instant::now()
        .checked_add(budget)
        .ok_or_else(|| anyhow::anyhow!("worker deadline overflow"))?;
    let reserve = (budget / 4).min(Duration::from_millis(500));
    let execution_until = until - reserve;
    let directory = tempfile::tempdir()?;
    let mut stdout = Capture::new(directory.path(), STDOUT_CAP)?;
    let mut stderr = Capture::new(directory.path(), STDERR_CAP)?;
    command
        .stdin(Stdio::null())
        .stdout(stdout.descriptor()?)
        .stderr(stderr.descriptor()?);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    let mut owned = OwnedChild {
        child: command
            .spawn()
            .context("spawn fixed package acquisition worker")?,
        cleanup_required: true,
    };
    let pid = owned.child.id();
    let result = poll(&mut owned.child, &stdout, &stderr, execution_until);
    // Run cleanup on success, deadline, capture refusal and polling errors alike.
    let cleanup = cleanup(&mut owned.child, until);
    if cleanup.is_ok() {
        owned.cleanup_required = false;
    }
    let (status, reaped) = match (result, cleanup) {
        (Ok(status), Ok(())) => (status, true),
        (Err(error), Ok(())) => return Err(error.context(format!("worker_pid={pid}; reaped=true"))),
        (result, Err(error)) => {
            return Err(error.context(format!(
                "worker_pid={pid}; cleanup unconfirmed; original={result:?}"
            )));
        }
    };
    let bytes = stdout.bytes()?;
    let errors = stderr.bytes()?;
    ensure!(
        status.success(),
        "package worker failed; worker_pid={pid}; reaped={reaped}; stderr={}",
        String::from_utf8_lossy(&errors)
    );
    ensure!(
        Instant::now() < until,
        "acquisition completion exceeded overall deadline"
    );
    Ok(bytes)
}
fn poll(
    child: &mut Child,
    stdout: &Capture,
    stderr: &Capture,
    until: Instant,
) -> Result<ExitStatus> {
    loop {
        stdout.check()?;
        stderr.check()?;
        if Instant::now() >= until {
            bail!("owned acquisition worker deadline expired");
        }
        if let Some(status) = child
            .try_wait()
            .context("poll package acquisition worker")?
        {
            return Ok(status);
        }
        thread::park_timeout(
            until
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10)),
        );
    }
}
fn cleanup(child: &mut Child, until: Instant) -> Result<()> {
    terminate(child)?;
    loop {
        if child
            .try_wait()
            .context("reap package acquisition worker")?
            .is_some()
        {
            return Ok(());
        }
        ensure!(
            Instant::now() < until,
            "package acquisition worker reaping exceeded overall deadline"
        );
        thread::park_timeout(
            until
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(5)),
        );
    }
}
#[cfg(unix)]
fn terminate(child: &mut Child) -> Result<()> {
    let group = libc::pid_t::try_from(child.id()).context("worker PID overflow")?;
    // SAFETY: run establishes this freshly spawned child's private process group.
    if unsafe { libc::kill(-group, libc::SIGKILL) } == -1 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            let _ = child.kill();
            return Err(error.into());
        }
    }
    // Direct-child fallback covers a worker which left its original group.
    if child.try_wait()?.is_none() {
        child.kill().context("terminate direct package worker")?;
    }
    Ok(())
}
#[cfg(not(unix))]
fn terminate(child: &mut Child) -> Result<()> {
    // Windows std::process kill terminates the worker and all its native threads.
    // No job object is claimed; separately spawned descendants are not covered.
    if child.try_wait()?.is_none() {
        child.kill().context("terminate direct package worker")?;
    }
    Ok(())
}
#[cfg(test)]
#[path = "worker_lifecycle_tests.rs"]
mod tests;

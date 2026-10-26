//! The Windows half of `mesh-llm service-host`: the log file, the
//! `mesh-llm serve` child, and the process-lifetime pieces from
//! `mesh_llm_system::process_lifetime`.

use super::service_host::render_service_host_record;
use super::service_paths::{
    WINDOWS_SERVICE_HOST_RECORD, WINDOWS_SERVICE_LOG_FILE, windows_service_log_dir,
};
use super::service_supervisor::{
    Backoff, ConsoleHost, FIRST_RESTART_DELAY, HostOutcome, MAX_RESTARTS, RunningServe, STABLE_RUN,
    ServeExit, ServeLauncher, note, supervise,
};
use anyhow::{Context, Result, bail};
use mesh_llm_system::process_lifetime::{ParentProcess, end_children_with_current_process};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

const PREVIOUS_LOG_FILE: &str = "service.previous.log";
/// A log file is set aside once it would grow past this, and only the one
/// before it is kept, so a node that runs for weeks keeps its logs under
/// twice this size.
const LOG_ROTATE_BYTES: u64 = 10 * 1024 * 1024;

pub(super) fn run() -> Result<()> {
    let binary = std::env::current_exe().context("could not determine the mesh-llm binary")?;
    let home = dirs::home_dir().context("could not determine the home directory")?;
    let local_data = dirs::data_local_dir().context("could not determine %LOCALAPPDATA%")?;
    let log_dir = windows_service_log_dir(&local_data);
    let mut log = ServiceLog::open(&log_dir, LOG_ROTATE_BYTES)?;

    // From here on the log is the only place a failure can be reported.
    if let Err(error) = record_this_host(&log_dir) {
        note(
            &mut log,
            &format!("warning: could not record this service host for uninstall: {error:#}"),
        );
    }
    let result = host(binary, home, &mut log);
    if let Err(error) = &result {
        note(&mut log, &format!("stopping: {error:#}"));
    }
    result
}

fn host(binary: PathBuf, home: PathBuf, log: &mut ServiceLog) -> Result<()> {
    // Children inherit the job, so `mesh-llm serve` and anything it starts end
    // when this process does, however it ends.
    end_children_with_current_process().context("put the service host in a job")?;
    // Without the console host to watch, ending the task could not stop the node.
    let Some(console_host) = ParentProcess::open().context("watch the console host")? else {
        note(log, "the scheduled task had already ended; not starting");
        return Ok(());
    };
    let mut launcher = ServeCommand {
        binary,
        home,
        log: log.clone(),
    };
    let mut backoff = Backoff::new(FIRST_RESTART_DELAY, MAX_RESTARTS, STABLE_RUN);
    match supervise(&mut launcher, &TaskConsole(console_host), &mut backoff, log)? {
        HostOutcome::ServeExited | HostOutcome::TaskEnded => Ok(()),
        HostOutcome::GaveUp(_) => bail!("mesh-llm serve kept failing"),
    }
}

fn record_this_host(log_dir: &Path) -> Result<()> {
    let started_at_unix = mesh_llm_system::process::current_process_start_time_unix()?;
    fs::write(
        log_dir.join(WINDOWS_SERVICE_HOST_RECORD),
        render_service_host_record(std::process::id(), started_at_unix),
    )?;
    Ok(())
}

/// The service log, written by the host and, through it, by the node.
#[derive(Clone)]
struct ServiceLog(Arc<Mutex<RotatingLog>>);

struct RotatingLog {
    dir: PathBuf,
    file: File,
    len: u64,
    /// The log is set aside once a write would take it past this.
    rotate_at: u64,
    /// How much the log may grow between attempts to set it aside.
    interval: u64,
}

impl RotatingLog {
    /// Starts a new log, keeping the one before beside it.
    ///
    /// Returns why the log could not be set aside, if it could not: something
    /// else then holds the file open (an editor, a scanner). The caller keeps
    /// the file it already has and reports the failure in the log.
    fn rotate(&mut self) -> io::Result<Option<io::Error>> {
        let current = self.dir.join(WINDOWS_SERVICE_LOG_FILE);
        match fs::rename(&current, self.dir.join(PREVIOUS_LOG_FILE)) {
            // A log that is already gone needs no setting aside.
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Ok(Some(error)),
        }
        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&current)?;
        self.len = self.file.metadata()?.len();
        self.rotate_at = self.interval;
        Ok(None)
    }
}

impl ServiceLog {
    fn open(dir: &Path, interval: u64) -> Result<Self> {
        fs::create_dir_all(dir)
            .with_context(|| format!("create the service log directory {}", dir.display()))?;
        let (file, len) = start_log_file(dir).with_context(|| {
            format!(
                "open the service log {}",
                dir.join(WINDOWS_SERVICE_LOG_FILE).display()
            )
        })?;
        Ok(Self(Arc::new(Mutex::new(RotatingLog {
            dir: dir.to_path_buf(),
            file,
            len,
            rotate_at: interval,
            interval,
        }))))
    }
}

// The log before is kept beside the new one: the previous session's at each
// start, the earlier part of this one at each rotation.
fn start_log_file(dir: &Path) -> io::Result<(File, u64)> {
    let current = dir.join(WINDOWS_SERVICE_LOG_FILE);
    if current.exists() {
        let _ = fs::rename(&current, dir.join(PREVIOUS_LOG_FILE));
    }
    let file = OpenOptions::new().create(true).append(true).open(current)?;
    // Not zero when the old log could not be set aside.
    let len = file.metadata()?.len();
    Ok((file, len))
}

impl Write for ServiceLog {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut log = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let after = log.len + bytes.len() as u64;
        let kept = if log.len > 0 && after > log.rotate_at {
            log.rotate()?
        } else {
            None
        };
        log.file.write_all(bytes)?;
        log.len += bytes.len() as u64;
        if let Some(error) = kept {
            // Reported into the file directly: going through this writer would
            // rotate again and recurse.
            let line = format!("warning: could not set the previous service log aside: {error}\n");
            if log.file.write_all(line.as_bytes()).is_ok() {
                log.len += line.len() as u64;
            }
            // Give the holder another interval to let go of the file instead
            // of reopening it at every write.
            log.rotate_at = log.len + log.interval;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .file
            .flush()
    }
}

/// Copies one output stream of the node into the log until it closes.
///
/// A log that can no longer be written (a full disk, a rotation that cannot
/// reopen the file) must not stop the pipe being drained: once the pipe buffer
/// fills, `mesh-llm serve` blocks on its next stdout/stderr write and the node
/// stalls with nothing to say why. The rest of the output is read away instead,
/// so the node keeps running without those logs.
fn forward(
    mut output: impl Read + Send + 'static,
    mut log: impl Write + Send + 'static,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        if io::copy(&mut output, &mut log).is_err() {
            let _ = io::copy(&mut output, &mut io::sink());
        }
    })
}

struct TaskConsole(ParentProcess);

impl ConsoleHost for TaskConsole {
    fn gone_within(&self, timeout: Duration) -> bool {
        self.0.gone_within(timeout)
    }
}

struct ServeCommand {
    binary: PathBuf,
    home: PathBuf,
    log: ServiceLog,
}

impl ServeLauncher for ServeCommand {
    type Running = Child;

    fn launch(&mut self) -> io::Result<Child> {
        let mut serve = Command::new(&self.binary)
            .arg("serve")
            .current_dir(&self.home)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        // Through the host rather than straight into the file, so that the
        // log can be rotated while the node runs.
        if let Some(stdout) = serve.stdout.take() {
            forward(stdout, self.log.clone());
        }
        if let Some(stderr) = serve.stderr.take() {
            forward(stderr, self.log.clone());
        }
        Ok(serve)
    }
}

impl RunningServe for Child {
    fn try_exit(&mut self) -> io::Result<Option<ServeExit>> {
        Ok(self.try_wait()?.map(|status| {
            if status.success() {
                ServeExit::Success
            } else {
                ServeExit::Failure(status.code())
            }
        }))
    }

    fn stop(&mut self) {
        let _ = self.kill();
        let _ = self.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn wait_for_exit(serve: &mut Child) -> ServeExit {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(exit) = serve.try_exit().expect("try_exit") {
                return exit;
            }
            assert!(Instant::now() < deadline, "the child did not exit");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn cmd(args: &[&str]) -> Child {
        Command::new("cmd")
            .args(["/d", "/c"])
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn cmd")
    }

    fn read_log(log_dir: &Path, name: &str) -> String {
        fs::read_to_string(log_dir.join(name)).expect("read log")
    }

    #[test]
    fn a_child_reports_its_exit_code() {
        assert_eq!(
            wait_for_exit(&mut cmd(&["exit", "3"])),
            ServeExit::Failure(Some(3))
        );
        assert_eq!(wait_for_exit(&mut cmd(&["exit", "0"])), ServeExit::Success);
    }

    #[test]
    fn stopping_a_running_child_ends_it() {
        let mut serve = cmd(&["ping", "-n", "30", "127.0.0.1"]);
        assert_eq!(serve.try_exit().expect("try_exit"), None);

        serve.stop();

        assert!(serve.try_wait().expect("try_wait").is_some());
    }

    #[test]
    fn opening_the_log_keeps_the_previous_session() {
        let temp = tempfile::tempdir().expect("tempdir");
        let log_dir = temp.path().join("logs");
        fs::create_dir_all(&log_dir).expect("log dir");
        fs::write(log_dir.join(WINDOWS_SERVICE_LOG_FILE), "last logon\n").expect("seed log");

        let mut log = ServiceLog::open(&log_dir, LOG_ROTATE_BYTES).expect("open log");
        note(&mut log, "this logon");
        drop(log);

        assert_eq!(read_log(&log_dir, PREVIOUS_LOG_FILE), "last logon\n");
        let current = read_log(&log_dir, WINDOWS_SERVICE_LOG_FILE);
        assert!(current.ends_with("] this logon\n"), "{current}");
        assert!(!current.contains("last logon"));
    }

    #[test]
    fn the_log_is_set_aside_once_it_would_pass_its_size() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut log = ServiceLog::open(temp.path(), 16).expect("open log");

        log.write_all(b"first 0123456\n").expect("write");
        log.write_all(b"second 012345\n").expect("write");
        assert_eq!(read_log(temp.path(), PREVIOUS_LOG_FILE), "first 0123456\n");
        assert_eq!(
            read_log(temp.path(), WINDOWS_SERVICE_LOG_FILE),
            "second 012345\n"
        );

        log.write_all(b"third 0123456\n").expect("write");
        assert_eq!(read_log(temp.path(), PREVIOUS_LOG_FILE), "second 012345\n");
        assert_eq!(
            read_log(temp.path(), WINDOWS_SERVICE_LOG_FILE),
            "third 0123456\n"
        );
    }

    #[test]
    fn a_log_that_cannot_be_set_aside_is_retried_after_another_interval() {
        let temp = tempfile::tempdir().expect("tempdir");
        // A rename cannot replace a directory, so the log cannot be set aside.
        fs::create_dir(temp.path().join(PREVIOUS_LOG_FILE)).expect("blocker");
        let mut log = ServiceLog::open(temp.path(), 16).expect("open log");

        log.write_all(b"first 0123456\n").expect("write");
        log.write_all(b"second 012345\n").expect("write");
        let kept = read_log(temp.path(), WINDOWS_SERVICE_LOG_FILE);
        assert!(kept.starts_with("first 0123456\n"), "{kept}");
        assert!(kept.contains("second 012345\n"), "{kept}");
        assert!(
            kept.contains("could not set the previous service log aside"),
            "{kept}"
        );

        fs::remove_dir(temp.path().join(PREVIOUS_LOG_FILE)).expect("unblock");
        // Rotation waits for another interval of growth rather than reopening
        // the file at every write, so this one still lands in the same log.
        log.write_all(b"third 0123456\n").expect("write");
        assert!(!temp.path().join(PREVIOUS_LOG_FILE).exists());
        log.write_all(b"fourth 0123456\n").expect("write");

        let previous = read_log(temp.path(), PREVIOUS_LOG_FILE);
        assert!(previous.ends_with("third 0123456\n"), "{previous}");
        assert_eq!(
            read_log(temp.path(), WINDOWS_SERVICE_LOG_FILE),
            "fourth 0123456\n"
        );
    }

    /// A log that has stopped accepting bytes, as a full disk would.
    struct FailingLog {
        written: usize,
        fails_after: usize,
    }

    impl Write for FailingLog {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let room = self.fails_after.saturating_sub(self.written);
            if room == 0 {
                return Err(io::Error::other("no space left on device"));
            }
            let accepted = room.min(bytes.len());
            self.written += accepted;
            Ok(accepted)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// The node's output stream, counting what has been read from it.
    struct NodeOutput {
        bytes: io::Cursor<Vec<u8>>,
        read: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Read for NodeOutput {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let count = self.bytes.read(buf)?;
            self.read
                .fetch_add(count, std::sync::atomic::Ordering::SeqCst);
            Ok(count)
        }
    }

    #[test]
    fn a_log_that_cannot_be_written_still_drains_the_node_output() {
        const OUTPUT: usize = 64 * 1024;
        let read = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let output = NodeOutput {
            bytes: io::Cursor::new(vec![b'.'; OUTPUT]),
            read: Arc::clone(&read),
        };
        let log = FailingLog {
            written: 0,
            fails_after: 4096,
        };

        forward(output, log).join().expect("copy thread");

        // The node's pipe is never left full, so its next write returns.
        assert_eq!(
            read.load(std::sync::atomic::Ordering::SeqCst),
            OUTPUT,
            "the whole output should be read away"
        );
    }

    #[test]
    fn the_node_output_reaches_the_log() {
        let temp = tempfile::tempdir().expect("tempdir");
        let log = ServiceLog::open(temp.path(), LOG_ROTATE_BYTES).expect("open log");
        let mut child = Command::new("cmd")
            .args(["/d", "/c", "echo", "out&", "echo", "err", "1>&2"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn cmd");
        let copies = [
            forward(child.stdout.take().expect("stdout"), log.clone()),
            forward(child.stderr.take().expect("stderr"), log.clone()),
        ];
        child.wait().expect("wait for cmd");
        for copy in copies {
            copy.join().expect("copy thread");
        }

        let written = read_log(temp.path(), WINDOWS_SERVICE_LOG_FILE);
        assert!(written.contains("out"), "{written}");
        assert!(written.contains("err"), "{written}");
    }
}

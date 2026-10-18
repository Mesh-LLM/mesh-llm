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
    rotate_at: u64,
}

impl ServiceLog {
    fn open(dir: &Path, rotate_at: u64) -> Result<Self> {
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
            rotate_at,
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
        if log.len > 0 && log.len + bytes.len() as u64 > log.rotate_at {
            (log.file, log.len) = start_log_file(&log.dir)?;
        }
        log.file.write_all(bytes)?;
        log.len += bytes.len() as u64;
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
fn forward(mut output: impl Read + Send + 'static, mut log: ServiceLog) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let _ = io::copy(&mut output, &mut log);
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

//! The Windows half of `mesh-llm service-host`: the log file, the
//! `mesh-llm serve` child, and the process-lifetime pieces from
//! `mesh_llm_system::process_lifetime`.

use super::service_host::{
    Backoff, ConsoleHost, FIRST_RESTART_DELAY, HostOutcome, MAX_RESTARTS, RunningServe, STABLE_RUN,
    ServeExit, ServeLauncher, note, supervise,
};
use super::service_paths::{WINDOWS_SERVICE_LOG_FILE, windows_service_log_dir};
use anyhow::{Context, Result, bail};
use mesh_llm_system::process_lifetime::{ParentProcess, end_children_with_current_process};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const PREVIOUS_LOG_FILE: &str = "service.previous.log";

pub(super) fn run() -> Result<()> {
    let binary = std::env::current_exe().context("could not determine the mesh-llm binary")?;
    let home = dirs::home_dir().context("could not determine the home directory")?;
    let local_data = dirs::data_local_dir().context("could not determine %LOCALAPPDATA%")?;
    let mut log = open_log(&windows_service_log_dir(&local_data))?;

    // From here on the log is the only place a failure can be reported.
    let result = host(binary, home, &mut log);
    if let Err(error) = &result {
        note(&mut log, &format!("stopping: {error:#}"));
    }
    result
}

fn host(binary: PathBuf, home: PathBuf, log: &mut File) -> Result<()> {
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
        log: log
            .try_clone()
            .context("share the log with mesh-llm serve")?,
    };
    let mut backoff = Backoff::new(FIRST_RESTART_DELAY, MAX_RESTARTS, STABLE_RUN);
    match supervise(&mut launcher, &TaskConsole(console_host), &mut backoff, log)? {
        HostOutcome::ServeExited | HostOutcome::TaskEnded => Ok(()),
        HostOutcome::GaveUp(_) => bail!("mesh-llm serve kept failing"),
    }
}

// The previous session's log is kept beside the new one, so a log grows for
// one logon at most.
fn open_log(log_dir: &Path) -> Result<File> {
    fs::create_dir_all(log_dir)
        .with_context(|| format!("create the service log directory {}", log_dir.display()))?;
    let log_path = log_dir.join(WINDOWS_SERVICE_LOG_FILE);
    if log_path.exists() {
        let _ = fs::rename(&log_path, log_dir.join(PREVIOUS_LOG_FILE));
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("open the service log {}", log_path.display()))
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
    log: File,
}

impl ServeLauncher for ServeCommand {
    type Running = Child;

    fn launch(&mut self) -> io::Result<Child> {
        Command::new(&self.binary)
            .arg("serve")
            .current_dir(&self.home)
            .stdin(Stdio::null())
            .stdout(self.log.try_clone()?)
            .stderr(self.log.try_clone()?)
            .spawn()
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

        let mut log = open_log(&log_dir).expect("open log");
        note(&mut log, "this logon");
        drop(log);

        assert_eq!(
            fs::read_to_string(log_dir.join(PREVIOUS_LOG_FILE)).expect("previous log"),
            "last logon\n"
        );
        let current = fs::read_to_string(log_dir.join(WINDOWS_SERVICE_LOG_FILE)).expect("log");
        assert!(current.ends_with("] this logon\n"), "{current}");
        assert!(!current.contains("last logon"));
    }
}

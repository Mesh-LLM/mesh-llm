use crate::process::*;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;
use std::time::Duration;

pub fn spec(root: &Path, mode: &str) -> ProcessSpec {
    let mut environment = BTreeMap::from([
        (
            OsString::from("MIGRATION_PROCESS_MODE"),
            Value::Public(mode.into()),
        ),
        (
            OsString::from("MIGRATION_PROCESS_ROOT"),
            Value::Public(root.into()),
        ),
    ]);
    for key in ["SYSTEMROOT", "WINDIR"] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), Value::Public(value));
        }
    }
    ProcessSpec {
        executable: std::env::current_exe().expect("test executable"),
        arguments: ["--exact", "migration_process_fixture", "--nocapture"]
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect(),
        cwd: root.to_path_buf(),
        environment,
    }
}

pub fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(5),
        graceful_shutdown: Duration::from_millis(100),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 32768,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

pub fn ready(limits: &mut Limits, marker: &[u8]) {
    limits.readiness = Readiness::Line {
        stream: Stream::Stdout,
        bytes: marker.to_vec(),
        deadline: Duration::from_secs(2),
    };
}

pub fn assert_stopped(root: &Path, modes: &[&str]) {
    for mode in modes {
        let pid: u32 = std::fs::read_to_string(root.join(format!("{mode}.pid")))
            .expect("recorded PID")
            .parse()
            .expect("numeric PID");
        assert!(!alive(pid), "owned {mode} PID {pid} survived");
        eprintln!("cleanup receipt: mode={mode} pid={pid} live=false");
    }
}

#[cfg(unix)]
pub fn alive(pid: u32) -> bool {
    // SAFETY: signal zero performs only a liveness query for this recorded PID.
    unsafe { libc::kill(i32::try_from(pid).expect("PID range"), 0) == 0 }
}

#[cfg(windows)]
pub fn alive(pid: u32) -> bool {
    use windows_sys::Win32::System::Threading::*;
    // SAFETY: query-only process access, closed below on every non-null path.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        return false;
    }
    let mut code = 0;
    // SAFETY: writable exit code and valid query handle.
    let queried = unsafe { GetExitCodeProcess(handle, &mut code) };
    // SAFETY: relinquishes this function's one owned process handle.
    assert_ne!(
        unsafe { windows_sys::Win32::Foundation::CloseHandle(handle) },
        0
    );
    assert_ne!(queried, 0);
    code == 259
}

pub struct Sentinel(std::process::Child);

impl Sentinel {
    pub fn new(root: &Path) -> Self {
        let mut command =
            std::process::Command::new(std::env::current_exe().expect("test executable"));
        command
            .args(["--exact", "migration_process_fixture", "--nocapture"])
            .env("MIGRATION_PROCESS_MODE", "sentinel")
            .env("MIGRATION_PROCESS_ROOT", root)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        Self(command.spawn().expect("sentinel spawn"))
    }
    pub fn assert_alive(&mut self) {
        assert!(self.0.try_wait().expect("sentinel status").is_none());
        eprintln!("sentinel receipt: pid={} live=true", self.0.id());
    }
}

impl Drop for Sentinel {
    fn drop(&mut self) {
        self.0.kill().expect("sentinel stop");
        self.0.wait().expect("sentinel reap");
    }
}

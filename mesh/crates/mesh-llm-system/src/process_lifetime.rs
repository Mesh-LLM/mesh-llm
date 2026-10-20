//! Tie a process tree's lifetime to one process, on Windows.
//!
//! Windows does not end a process's children when the process ends, and a
//! scheduled task that is ended only loses its own process. The Windows
//! background service host uses two pieces to stop the node with the task: a
//! kill-on-close job that takes its children down when it exits, and a watch
//! on the console host that started it. The GPU energy sampler uses a third,
//! a job for one child, so its `nvidia-smi` ends with the node.

use std::io;
use std::os::windows::io::{AsRawHandle, FromRawHandle, IntoRawHandle, OwnedHandle, RawHandle};
use std::time::Duration;
use windows_sys::Win32::Foundation::{
    ERROR_INVALID_PARAMETER, FILETIME, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentProcessId, GetProcessTimes, OpenProcess,
    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, WaitForSingleObject,
};

/// Puts the current process in a kill-on-close job, so its children, which
/// inherit the job, end when it does, however it ends.
///
/// The job handle stays open until this process exits: closing it earlier
/// would end this process as well, since it is in the job. The handle is not
/// inheritable, so no child keeps the job open after this process.
pub fn end_children_with_current_process() -> io::Result<()> {
    let job = kill_on_close_job()?;
    // SAFETY: the job handle is owned and valid; the current-process
    // pseudo-handle needs no closing.
    if unsafe { AssignProcessToJobObject(job.as_raw_handle() as HANDLE, GetCurrentProcess()) } == 0
    {
        return Err(io::Error::last_os_error());
    }
    // Left to the process exit to close, as documented above.
    let _ = job.into_raw_handle();
    Ok(())
}

/// Ends one child process when the current process ends, however it ends,
/// for a helper that would otherwise outlive a killed node: `nvidia-smi
/// --loop-ms` keeps running after its parent is gone and its stdout is
/// broken. The child goes alone into a kill-on-close job whose only handle
/// this process keeps until it exits.
pub fn end_with_current_process(child: RawHandle) -> io::Result<()> {
    let job = kill_on_close_job()?;
    // SAFETY: the job handle is owned and valid, and `child` is a live process
    // handle the caller owns.
    if unsafe { AssignProcessToJobObject(job.as_raw_handle() as HANDLE, child as HANDLE) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // Closing it earlier would end the child.
    let _ = job.into_raw_handle();
    Ok(())
}

fn kill_on_close_job() -> io::Result<OwnedHandle> {
    // SAFETY: null attributes and name create an unnamed, non-inheritable job;
    // the null return is checked before use.
    let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if job.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `job` is a fresh handle owned from here on.
    let job = unsafe { OwnedHandle::from_raw_handle(job as RawHandle) };
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    let size = u32::try_from(std::mem::size_of_val(&limits))
        .map_err(|_| io::Error::other("job limits do not fit in u32"))?;
    // SAFETY: the structure matches the information class and outlives the call.
    let set = unsafe {
        SetInformationJobObject(
            job.as_raw_handle() as HANDLE,
            JobObjectExtendedLimitInformation,
            (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            size,
        )
    };
    if set == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(job)
}

/// The process that started the current one.
pub struct ParentProcess(OwnedHandle);

impl ParentProcess {
    /// Opens the parent for waiting; `None` when it has already exited.
    pub fn open() -> io::Result<Option<Self>> {
        // SAFETY: a plain query of the current process id.
        let parent_id = parent_process_id(unsafe { GetCurrentProcessId() })?;
        // SAFETY: opening by id takes values only; the null return is checked.
        let handle = unsafe {
            OpenProcess(
                PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                0,
                parent_id,
            )
        };
        if handle.is_null() {
            let error = io::Error::last_os_error();
            // No process has that id any more.
            return if error.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32) {
                Ok(None)
            } else {
                Err(error)
            };
        }
        // SAFETY: `handle` is a fresh handle owned from here on.
        let parent = Self(unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) });
        // An id is reused once its process has exited, so a process created
        // after this one is not the parent.
        // SAFETY: the current-process pseudo-handle is always valid.
        let own_start = creation_time(unsafe { GetCurrentProcess() })?;
        if creation_time(parent.0.as_raw_handle() as HANDLE)? > own_start {
            return Ok(None);
        }
        Ok(Some(parent))
    }

    /// Waits up to `timeout`; true once the parent has exited.
    pub fn gone_within(&self, timeout: Duration) -> bool {
        // INFINITE is u32::MAX, so a longer wait is capped just below it.
        let millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX - 1);
        // SAFETY: the handle is owned by `self` and carries SYNCHRONIZE.
        unsafe { WaitForSingleObject(self.0.as_raw_handle() as HANDLE, millis) == WAIT_OBJECT_0 }
    }
}

fn parent_process_id(own_id: u32) -> io::Result<u32> {
    // SAFETY: a process snapshot takes values only; the invalid return is checked.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `snapshot` is a fresh handle owned from here on.
    let snapshot = unsafe { OwnedHandle::from_raw_handle(snapshot as RawHandle) };
    let mut entry = PROCESSENTRY32W {
        dwSize: u32::try_from(std::mem::size_of::<PROCESSENTRY32W>())
            .map_err(|_| io::Error::other("PROCESSENTRY32W does not fit in u32"))?,
        ..Default::default()
    };
    // SAFETY: `entry.dwSize` is set and the entry outlives each call.
    let mut found = unsafe { Process32FirstW(snapshot.as_raw_handle() as HANDLE, &mut entry) };
    while found != 0 {
        if entry.th32ProcessID == own_id {
            return Ok(entry.th32ParentProcessID);
        }
        // SAFETY: as above.
        found = unsafe { Process32NextW(snapshot.as_raw_handle() as HANDLE, &mut entry) };
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "the current process is missing from the process snapshot",
    ))
}

/// Creation time in 100-nanosecond ticks, finer than the seconds
/// `process_started_at_unix` reports.
fn creation_time(process: HANDLE) -> io::Result<u64> {
    let mut created = FILETIME::default();
    let mut exited = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    // SAFETY: the handle carries query rights and the out-pointers outlive the call.
    let ok = unsafe { GetProcessTimes(process, &mut created, &mut exited, &mut kernel, &mut user) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command, Stdio};

    const HOLDER_PID_FILE: &str = "MESH_LLM_PROCESS_LIFETIME_PID_FILE";
    const TIED_CHILD_PID_FILE: &str = "MESH_LLM_PROCESS_LIFETIME_TIED_PID_FILE";

    #[test]
    fn the_parent_process_is_watched_and_still_running() {
        let parent = ParentProcess::open()
            .expect("open the parent")
            .expect("the test runner's parent is running");

        assert!(!parent.gone_within(Duration::ZERO));
    }

    // A child process puts itself in the job and starts a long grandchild;
    // once the child exits, the grandchild must be gone too. Done in a child
    // so that the test runner itself never joins a kill-on-close job.
    #[test]
    fn ending_the_job_holder_ends_its_children() {
        if let Some(pid_file) = std::env::var_os(HOLDER_PID_FILE) {
            end_children_with_current_process().expect("join the job");
            leave_running(spawn_grandchild(), pid_file);
            return;
        }
        assert_grandchild_ends_with_holder(
            "process_lifetime::tests::ending_the_job_holder_ends_its_children",
            HOLDER_PID_FILE,
        );
    }

    // The same, with only the grandchild in a job: what the energy sampler
    // does for `nvidia-smi`, while the node itself stays out of any job.
    #[test]
    fn a_child_tied_to_the_current_process_ends_with_it() {
        if let Some(pid_file) = std::env::var_os(TIED_CHILD_PID_FILE) {
            let grandchild = spawn_grandchild();
            end_with_current_process(std::os::windows::io::AsRawHandle::as_raw_handle(
                &grandchild,
            ))
            .expect("tie the grandchild");
            leave_running(grandchild, pid_file);
            return;
        }
        assert_grandchild_ends_with_holder(
            "process_lifetime::tests::a_child_tied_to_the_current_process_ends_with_it",
            TIED_CHILD_PID_FILE,
        );
    }

    fn spawn_grandchild() -> Child {
        Command::new("ping")
            .args(["-n", "60", "127.0.0.1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn the grandchild")
    }

    // The pid travels through a file, not a pipe: the grandchild inherits
    // every inheritable handle, so a pipe would stay open as long as it lives.
    fn leave_running(grandchild: Child, pid_file: std::ffi::OsString) {
        let pid = grandchild.id();
        // Dropped, never waited on: it has to outlive this process for the
        // job to be what ends it.
        drop(grandchild);
        std::fs::write(pid_file, pid.to_string()).expect("write the pid");
    }

    fn assert_grandchild_ends_with_holder(test: &str, pid_env: &str) {
        let temp = tempfile::tempdir().expect("tempdir");
        let pid_file = temp.path().join("grandchild.pid");
        let status = Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", test, "--test-threads=1"])
            .env(pid_env, &pid_file)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("run the job holder");
        assert!(status.success(), "the job holder failed: {status}");
        let grandchild: u32 = std::fs::read_to_string(&pid_file)
            .expect("the job holder wrote the grandchild pid")
            .trim()
            .parse()
            .expect("a pid");

        // Matching the image name keeps a reused id from passing for the
        // grandchild.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while crate::process::process_name_matches(grandchild, "PING") {
            assert!(
                std::time::Instant::now() < deadline,
                "the grandchild {grandchild} outlived the job holder"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

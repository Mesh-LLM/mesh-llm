/// Tolerance (in seconds) when comparing a recorded start time against the
/// live process start time. A difference of up to this many seconds is treated
/// as the same process.
pub const START_TIME_TOLERANCE_SECS: i64 = 2;

/// Liveness state inferred from whether the process comm is readable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Liveness {
    /// Process is alive because comm was readable.
    Alive,
    /// Process is gone because the PID was not found.
    Dead,
    /// Liveness could not be determined.
    Unknown,
}

#[cfg(target_os = "linux")]
mod platform {
    use std::sync::OnceLock;

    static BTIME: OnceLock<i64> = OnceLock::new();

    fn btime() -> i64 {
        *BTIME.get_or_init(|| {
            (|| -> anyhow::Result<i64> {
                let content = std::fs::read_to_string("/proc/stat")?;
                for line in content.lines() {
                    if let Some(rest) = line.strip_prefix("btime ") {
                        return Ok(rest.trim().parse()?);
                    }
                }
                anyhow::bail!("btime line not found in /proc/stat")
            })()
            .unwrap_or(0)
        })
    }

    pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
        let path = format!("/proc/{pid}/comm");
        match std::fs::read_to_string(&path) {
            Ok(s) => Ok(Some(s.trim().to_string())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                tracing::debug!(pid, path, "permission denied reading process comm");
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
        let path = format!("/proc/{pid}/exe");
        match std::fs::read_link(&path) {
            Ok(target) => Ok(target
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                tracing::debug!(
                    pid,
                    path,
                    "permission denied reading process executable path"
                );
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
        let path = format!("/proc/{pid}/stat");
        let content = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                tracing::debug!(pid, "permission denied reading /proc/{pid}/stat");
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };

        let rparen = content
            .rfind(')')
            .ok_or_else(|| anyhow::anyhow!("malformed /proc/{pid}/stat: no closing ')' found"))?;
        let after_comm = content.get(rparen + 2..).unwrap_or("");
        let fields: Vec<&str> = after_comm.split_whitespace().collect();

        let starttime_ticks: u64 = fields
            .get(19)
            .ok_or_else(|| anyhow::anyhow!("starttime field missing in /proc/{pid}/stat"))?
            .parse()
            .map_err(|e| anyhow::anyhow!("failed to parse starttime in /proc/{pid}/stat: {e}"))?;

        let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if clk_tck <= 0 {
            anyhow::bail!("sysconf(_SC_CLK_TCK) returned {clk_tck}");
        }

        let bt = btime();
        if bt == 0 {
            anyhow::bail!("could not determine boot time from /proc/stat");
        }

        Ok(Some(bt + (starttime_ticks as i64 / clk_tck)))
    }
}

#[cfg(target_os = "macos")]
mod platform {
    //! Process facts from the kernel's process table, without spawning `ps`.
    //! Both names are the ones the process was started under, so renaming or
    //! deleting its executable while it runs doesn't change them, as it didn't
    //! for `ps`.

    use std::io;

    /// The name the kernel recorded at exec, cut to 16 bytes like Linux's
    /// `/proc/{pid}/comm`; `None` once no process has the id. Any account
    /// may read it.
    pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
        let Ok(pid) = libc::pid_t::try_from(pid) else {
            return Ok(None);
        };
        // SAFETY: proc_bsdshortinfo is a C POD whose integer/byte fields admit zero.
        let mut info: libc::proc_bsdshortinfo = unsafe { std::mem::zeroed() };
        let size = i32::try_from(std::mem::size_of_val(&info))?;
        // SAFETY: the short BSD-info flavor matches the aligned, writable info buffer.
        let read = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDT_SHORTBSDINFO,
                0,
                (&mut info as *mut libc::proc_bsdshortinfo).cast(),
                size,
            )
        };
        if read != size {
            let error = io::Error::last_os_error();
            if read == 0 && error.raw_os_error() == Some(libc::ESRCH) {
                return Ok(None);
            }
            anyhow::bail!("proc_pidinfo({pid}) failed: {error}");
        }
        // The field is not terminated when the name fills it.
        let bytes: Vec<u8> = info
            .pbsi_comm
            .iter()
            .take_while(|&&byte| byte != 0)
            .map(|&byte| byte as u8)
            .collect();
        Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
    }

    pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
        let Ok(pid) = libc::pid_t::try_from(pid) else {
            return Ok(None);
        };
        // SAFETY: proc_bsdinfo is a C POD whose integer/byte fields admit zero.
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size = i32::try_from(std::mem::size_of_val(&info))?;
        // SAFETY: the BSD-info flavor matches the aligned, writable info buffer.
        let read = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDTBSDINFO,
                0,
                (&mut info as *mut libc::proc_bsdinfo).cast(),
                size,
            )
        };
        if read != size {
            let error = io::Error::last_os_error();
            if read == 0 && error.raw_os_error() == Some(libc::ESRCH) {
                return Ok(None);
            }
            anyhow::bail!("proc_pidinfo({pid}) failed: {error}");
        }
        Ok(Some(i64::try_from(info.pbi_start_tvsec)?))
    }

    /// The basename of the path the process was executed through, the name
    /// `ps -o comm=` showed, from `KERN_PROCARGS2`. The kernel keeps that
    /// path from exec, unlike `proc_pidpath`, which follows the file. Only
    /// the process's own account may read it; elsewhere, and once no process
    /// has the id, the kernel answers `EINVAL`, and this is `None`.
    pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
        let Ok(pid) = libc::c_int::try_from(pid) else {
            return Ok(None);
        };
        let mut mib = [libc::CTL_KERN, libc::KERN_PROCARGS2, pid];
        let mut buffer = vec![0u8; arguments_size()?];
        let mut length = buffer.len();
        // SAFETY: the buffer is writable for `length` bytes, and sysctl
        // writes at most that many and stores the count back in `length`.
        let status = unsafe {
            libc::sysctl(
                mib.as_mut_ptr(),
                u32::try_from(mib.len())?,
                buffer.as_mut_ptr().cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 {
            let error = io::Error::last_os_error();
            if matches!(
                error.raw_os_error(),
                Some(libc::EINVAL | libc::ESRCH | libc::EPERM)
            ) {
                return Ok(None);
            }
            anyhow::bail!("sysctl KERN_PROCARGS2 for {pid} failed: {error}");
        }
        buffer.truncate(length);
        Ok(super::exec_path_name(&buffer))
    }

    /// The largest argument area the kernel hands out, `KERN_ARGMAX`.
    fn arguments_size() -> anyhow::Result<usize> {
        let mut mib = [libc::CTL_KERN, libc::KERN_ARGMAX];
        let mut size: libc::c_int = 0;
        let mut length = std::mem::size_of_val(&size);
        // SAFETY: `size` is writable for `length` bytes.
        let status = unsafe {
            libc::sysctl(
                mib.as_mut_ptr(),
                u32::try_from(mib.len())?,
                (&mut size as *mut libc::c_int).cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 {
            anyhow::bail!("sysctl KERN_ARGMAX failed: {}", io::Error::last_os_error());
        }
        Ok(usize::try_from(size)?)
    }
}

/// The executable's basename from a `KERN_PROCARGS2` record: the argument
/// count as a native `int`, then the path the process was executed through,
/// NUL-terminated. Kept outside the macOS module so the parsing runs on every
/// CI runner.
#[cfg(any(test, target_os = "macos"))]
fn exec_path_name(record: &[u8]) -> Option<String> {
    let path = record.get(std::mem::size_of::<std::ffi::c_int>()..)?;
    let path = &path[..path.iter().position(|&byte| byte == 0)?];
    let name = std::path::Path::new(std::str::from_utf8(path).ok()?).file_name()?;
    Some(name.to_string_lossy().into_owned())
}

#[cfg(windows)]
mod platform {
    //! Process facts through the Win32 process API. Every query opens the
    //! process with `PROCESS_QUERY_LIMITED_INFORMATION`, which the same user
    //! holds without any privilege, plus `PROCESS_SYNCHRONIZE` to tell an
    //! exited process apart; when that is refused, with the query right
    //! alone.
    //!
    //! Same contract as the Unix modules: a PID that names no process is
    //! `Ok(None)`, and so is one we are not allowed to query, as a `/proc`
    //! read denied on Linux; any other failure is an error, which
    //! `process_liveness` reports as `Unknown` rather than `Dead`.

    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER, FILETIME, GetLastError, HANDLE,
        WAIT_OBJECT_0,
    };
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, GetProcessTimes, OpenProcess, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, QueryFullProcessImageNameW,
        WaitForSingleObject,
    };

    /// The exit code a process reports while it is still running, and also a
    /// code a process may exit with, hence only a fallback.
    const STILL_ACTIVE: u32 = 259;

    /// Owned process handle, closed on drop.
    struct Process {
        handle: HANDLE,
        /// The handle carries `PROCESS_SYNCHRONIZE`, so it may be waited on.
        waitable: bool,
    }

    impl Drop for Process {
        fn drop(&mut self) {
            // SAFETY: the handle came from a successful `OpenProcess` and is
            // closed exactly once, here.
            unsafe { CloseHandle(self.handle) };
        }
    }

    /// Open `pid` for querying: `Ok(None)` when no such process exists or it
    /// cannot be queried, `Err` for anything else.
    fn open(pid: u32) -> anyhow::Result<Option<Process>> {
        // SYNCHRONIZE lets `has_exited` wait on the handle; a process that
        // grants query rights only is still opened, without it.
        for access in [
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE,
            PROCESS_QUERY_LIMITED_INFORMATION,
        ] {
            // SAFETY: a plain Win32 call taking values; the null return is
            // checked before the handle is used.
            let handle = unsafe { OpenProcess(access, 0, pid) };
            if !handle.is_null() {
                let process = Process {
                    handle,
                    waitable: access & PROCESS_SYNCHRONIZE != 0,
                };
                // An exited process stays openable while any handle still
                // references it, but names no running process any more, and
                // its image name query fails (Win32 error 31, measured).
                return Ok((!has_exited(&process)).then_some(process));
            }
            // SAFETY: reads the calling thread's last error and takes nothing.
            match unsafe { GetLastError() } {
                ERROR_INVALID_PARAMETER => return Ok(None),
                ERROR_ACCESS_DENIED => {}
                code => {
                    return Err(anyhow::anyhow!(
                        "OpenProcess({pid}) failed with Win32 error {code}"
                    ));
                }
            }
        }
        tracing::debug!(pid, "access denied opening process for query");
        Ok(None)
    }

    /// A process handle is signaled once the process has exited. Without the
    /// right to wait, the exit code is the fallback, which a process that
    /// exited with code 259 would pass as still running.
    fn has_exited(process: &Process) -> bool {
        if process.waitable {
            // SAFETY: the handle carries SYNCHRONIZE; a zero timeout only polls.
            return unsafe { WaitForSingleObject(process.handle, 0) } == WAIT_OBJECT_0;
        }
        let mut code = 0u32;
        // SAFETY: the handle carries query rights and `code` outlives the call.
        let ok = unsafe { GetExitCodeProcess(process.handle, &mut code) };
        ok != 0 && code != STILL_ACTIVE
    }

    /// Full image path of the process, as the kernel reports it.
    fn image_path(process: &Process) -> anyhow::Result<String> {
        let mut buffer = vec![0u16; 32 * 1024];
        let mut length = buffer.len() as u32;
        // SAFETY: the buffer outlives the call and `length` carries its
        // capacity in UTF-16 units; the API writes at most that many and
        // stores the written length back.
        let ok = unsafe {
            QueryFullProcessImageNameW(
                process.handle,
                PROCESS_NAME_WIN32,
                buffer.as_mut_ptr(),
                &mut length,
            )
        };
        if ok == 0 {
            // SAFETY: as above.
            let code = unsafe { GetLastError() };
            anyhow::bail!("QueryFullProcessImageNameW failed with Win32 error {code}");
        }
        Ok(String::from_utf16_lossy(&buffer[..length as usize]))
    }

    /// The image file name without its extension, which is the shape
    /// `binary_process_name` derives from a configured binary on Windows.
    fn image_stem(process: &Process) -> anyhow::Result<Option<String>> {
        let path = image_path(process)?;
        Ok(std::path::Path::new(&path)
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned()))
    }

    pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
        match open(pid)? {
            Some(process) => image_stem(&process),
            None => Ok(None),
        }
    }

    pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
        process_comm(pid)
    }

    pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
        let Some(process) = open(pid)? else {
            return Ok(None);
        };
        let zero = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let (mut creation, mut exit, mut kernel, mut user) = (zero, zero, zero, zero);
        // SAFETY: four valid out-pointers to stack values that outlive the call.
        let ok = unsafe {
            GetProcessTimes(
                process.handle,
                &mut creation,
                &mut exit,
                &mut kernel,
                &mut user,
            )
        };
        if ok == 0 {
            // SAFETY: as above.
            let code = unsafe { GetLastError() };
            anyhow::bail!("GetProcessTimes({pid}) failed with Win32 error {code}");
        }
        Ok(Some(super::filetime_to_unix(
            creation.dwHighDateTime,
            creation.dwLowDateTime,
        )))
    }

    /// Whether this user is refused even the least query on `pid`, which
    /// `open` reports as no process. A user can always query its own
    /// processes, so such a process belongs to another account, such as a
    /// service (measured on Windows 11: none of a user's 168 processes was
    /// refused, all 191 refused ones ran under other accounts).
    pub fn process_query_denied(pid: u32) -> bool {
        // SAFETY: a plain Win32 call taking values; the null return is
        // checked before the handle is used.
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if !handle.is_null() {
            // SAFETY: a handle just returned by `OpenProcess`, closed once.
            unsafe { CloseHandle(handle) };
            return false;
        }
        // SAFETY: reads the calling thread's last error and takes nothing.
        let code = unsafe { GetLastError() };
        code == ERROR_ACCESS_DENIED
    }
}

/// Convert a Win32 `FILETIME`, 100-nanosecond ticks since 1601-01-01 UTC, to
/// Unix seconds. Kept outside the Windows module so the arithmetic runs on
/// every CI runner.
#[cfg(any(test, windows))]
fn filetime_to_unix(high: u32, low: u32) -> i64 {
    const TICKS_PER_SECOND: i64 = 10_000_000;
    const SECONDS_FROM_1601_TO_1970: i64 = 11_644_473_600;
    let ticks = ((high as i64) << 32) | (low as i64);
    ticks / TICKS_PER_SECOND - SECONDS_FROM_1601_TO_1970
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod platform {
    pub fn process_comm(_pid: u32) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    pub fn process_started_at_unix(_pid: u32) -> anyhow::Result<Option<i64>> {
        Ok(None)
    }

    pub fn process_executable_name(_pid: u32) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
}

/// Read the command name of the given process.
pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
    platform::process_comm(pid)
}

/// Read the Unix start time of the given process.
pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
    platform::process_started_at_unix(pid)
}

/// Read the executable basename for the given process when available.
pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
    platform::process_executable_name(pid)
}

/// Determine liveness of a process by attempting to read its comm.
pub fn process_liveness(pid: u32) -> Liveness {
    match process_comm(pid) {
        Ok(Some(_)) => Liveness::Alive,
        Ok(None) => Liveness::Dead,
        Err(_) => Liveness::Unknown,
    }
}

/// Returns true iff the live process name matches the expected spawned binary.
pub fn process_name_matches(pid: u32, expected_comm: &str) -> bool {
    process_executable_name(pid)
        .ok()
        .flatten()
        .is_some_and(|name| names_match(&name, expected_comm))
        || process_comm(pid)
            .ok()
            .flatten()
            .is_some_and(|name| names_match(&name, expected_comm))
}

/// Compares a live process name with the expected one.
///
/// The two sides reach the name through different APIs. The expected name is
/// the file stem of what `owner.json` recorded from `std::env::current_exe`,
/// which on Windows reports the path as it was handed to `CreateProcess`, so
/// as the caller spelled it. The live name comes from
/// `QueryFullProcessImageNameW`, which reports the canonical path on disk.
/// Windows opens files without regard to case, so launching through a
/// differently cased path is ordinary and makes the two disagree: measured
/// here, one side answered `PYTHON` while the other answered `python` for the
/// same process. Folding ASCII case covers that, the executable stems compared
/// here being ASCII, and every other platform keeps the exact comparison its
/// filesystem calls for.
#[cfg(windows)]
fn names_match(live: &str, expected: &str) -> bool {
    live.eq_ignore_ascii_case(expected)
}

#[cfg(not(windows))]
fn names_match(live: &str, expected: &str) -> bool {
    live == expected
}

/// Returns true iff the live process matches the expected name and start time.
pub fn validate_pid_matches(pid: u32, expected_comm: &str, expected_started_at_unix: i64) -> bool {
    match process_started_at_unix(pid) {
        Ok(Some(t)) => {
            process_name_matches(pid, expected_comm)
                && (t - expected_started_at_unix).abs() <= START_TIME_TOLERANCE_SECS
        }
        _ => false,
    }
}

/// What a pid is now, compared with a recorded process, for deciding whether
/// stopping it is ours to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PidIdentity {
    /// The recorded process.
    Recorded,
    /// Another process now holds the id.
    Other,
    /// No process can be queried under the id.
    Gone,
    /// What the id names could not be read.
    Unknown,
}

/// Compares the process under `pid` with a recorded name and, when given,
/// start time. Unlike [`validate_pid_matches`], a lookup that fails is
/// `Unknown`, not a mismatch, so a caller never mistakes a process it could
/// not read for someone else's.
pub fn pid_identity(
    pid: u32,
    expected_comm: &str,
    expected_started_at_unix: Option<i64>,
) -> PidIdentity {
    let names = [process_executable_name(pid), process_comm(pid)];
    let readable: Vec<&String> = names
        .iter()
        .filter_map(|name| name.as_ref().ok()?.as_ref())
        .collect();
    if !readable.iter().any(|name| names_match(name, expected_comm)) {
        return if !readable.is_empty() {
            PidIdentity::Other
        } else if names.iter().any(Result::is_err) {
            PidIdentity::Unknown
        } else if query_denied(pid) {
            // Still running, but under another account: not the user's node.
            PidIdentity::Other
        } else {
            PidIdentity::Gone
        };
    }
    let Some(expected) = expected_started_at_unix else {
        return PidIdentity::Recorded;
    };
    match process_started_at_unix(pid) {
        Ok(Some(started)) if (started - expected).abs() <= START_TIME_TOLERANCE_SECS => {
            PidIdentity::Recorded
        }
        Ok(Some(_)) => PidIdentity::Other,
        Ok(None) => PidIdentity::Gone,
        Err(_) => PidIdentity::Unknown,
    }
}

/// Whether the name lookups' "no process" for `pid` was a refusal instead,
/// which only Windows reports that way.
fn query_denied(pid: u32) -> bool {
    #[cfg(windows)]
    {
        platform::process_query_denied(pid)
    }
    #[cfg(not(windows))]
    {
        let _ = pid;
        false
    }
}

/// Returns the Unix start time of the current process.
pub fn current_process_start_time_unix() -> anyhow::Result<i64> {
    process_started_at_unix(std::process::id())?
        .ok_or_else(|| anyhow::anyhow!("could not determine start time of current process"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn exec_path_name_reads_the_path_after_the_argument_count() {
        let mut record = 2i32.to_ne_bytes().to_vec();
        record.extend_from_slice(b"/opt/mesh/bin/mesh-llm\0\0\0mesh-llm\0serve\0");
        assert_eq!(super::exec_path_name(&record).as_deref(), Some("mesh-llm"));

        let mut relative = 1i32.to_ne_bytes().to_vec();
        relative.extend_from_slice(b"./llama-server\0");
        assert_eq!(
            super::exec_path_name(&relative).as_deref(),
            Some("llama-server")
        );

        assert_eq!(super::exec_path_name(&[0, 0]), None);
        assert_eq!(super::exec_path_name(&1i32.to_ne_bytes()), None);
    }

    #[test]
    fn filetime_ticks_at_the_unix_epoch_convert_to_zero() {
        // 1970-01-01T00:00:00Z is 116_444_736_000_000_000 ticks after
        // 1601-01-01, which is 0x019DB1DE_D53E8000 split into its two halves.
        assert_eq!(super::filetime_to_unix(0x019D_B1DE, 0xD53E_8000), 0);
        // One second later, ten million ticks further.
        assert_eq!(
            super::filetime_to_unix(0x019D_B1DE, 0xD53E_8000 + 10_000_000),
            1
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_reports_the_current_process() {
        let pid = std::process::id();
        let comm = super::process_comm(pid)
            .unwrap()
            .expect("the current process must be visible to itself");
        let exe = std::env::current_exe().unwrap();
        let exe_name = exe.file_name().and_then(|n| n.to_str()).unwrap();
        assert_eq!(
            super::process_executable_name(pid).unwrap().as_deref(),
            Some(exe_name)
        );
        // The kernel keeps the first 16 bytes; a test binary's name is longer.
        assert!(
            !comm.is_empty() && comm.len() <= 16 && exe_name.starts_with(&comm),
            "comm {comm:?} for {exe_name:?}"
        );
        let started = super::process_started_at_unix(pid)
            .unwrap()
            .expect("the current process has a start time");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert!(
            (0..=3600).contains(&(now - started)),
            "started {started}, now {now}"
        );
    }

    // launchd runs as root. Its name is readable from any account, so
    // liveness and the name check work; its start time is not, and an error
    // keeps an identity check from taking it for one of ours.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_reports_a_process_of_another_account() {
        assert_eq!(super::process_comm(1).unwrap().as_deref(), Some("launchd"));
        assert_eq!(super::process_liveness(1), super::Liveness::Alive);
        assert_ne!(
            super::pid_identity(1, "launchd", Some(0)),
            super::PidIdentity::Recorded
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_reports_the_current_process() {
        let pid = std::process::id();
        let comm = super::process_comm(pid)
            .unwrap()
            .expect("the current process must be visible to itself");
        assert!(!comm.is_empty());
        assert!(
            !comm.to_ascii_lowercase().ends_with(".exe"),
            "comm is the image stem, got {comm}"
        );
        let started = super::process_started_at_unix(pid)
            .unwrap()
            .expect("the current process has a start time");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert!(
            (0..3600).contains(&(now - started)),
            "started at {started}, now {now}"
        );
        assert_eq!(super::process_liveness(pid), super::Liveness::Alive);
    }

    #[cfg(windows)]
    #[test]
    fn windows_reports_a_missing_pid_as_dead() {
        // PID 0 names the System Idle Process, and `OpenProcess` documents
        // that it fails there with `ERROR_INVALID_PARAMETER`, which this
        // module reads as "no such process". An arbitrary large value would
        // not do as well: the kernel allocates process ids through the handle
        // manager, which ignores their low two bits, so a value like 999_999
        // reaches 999_996 and would open it on a machine that happened to run
        // it.
        assert_eq!(super::process_comm(0).unwrap(), None);
        assert_eq!(super::process_liveness(0), super::Liveness::Dead);
    }

    #[test]
    fn a_name_that_differs_only_in_case_matches_where_the_filesystem_does() {
        assert!(super::names_match("mesh-llm", "mesh-llm"));
        assert!(!super::names_match("mesh-llm", "other"));
        // `current_exe` spells the binary the way it was launched, so a
        // Windows caller reaching it through an upper-cased path records a
        // name the canonical one does not equal.
        assert_eq!(
            super::names_match("mesh-llm", "MESH-LLM"),
            cfg!(windows),
            "case folding must follow the platform's filesystem"
        );
    }
}

#[cfg(test)]
mod pid_identity_tests {
    use super::*;

    fn own_identity() -> (u32, String, i64) {
        let pid = std::process::id();
        let comm = process_comm(pid)
            .expect("read own name")
            .expect("own process is running");
        let started = current_process_start_time_unix().expect("own start time");
        (pid, comm, started)
    }

    #[test]
    fn the_running_process_is_the_recorded_one() {
        let (pid, comm, started) = own_identity();

        assert_eq!(
            pid_identity(pid, &comm, Some(started)),
            PidIdentity::Recorded
        );
        assert_eq!(pid_identity(pid, &comm, None), PidIdentity::Recorded);
    }

    #[test]
    fn another_name_or_start_time_is_another_process() {
        let (pid, comm, started) = own_identity();

        assert_eq!(
            pid_identity(pid, &comm, Some(started - 3600)),
            PidIdentity::Other
        );
        assert_eq!(
            pid_identity(pid, "not-the-recorded-binary", Some(started)),
            PidIdentity::Other
        );
    }

    #[test]
    fn an_exited_process_is_gone() {
        #[cfg(windows)]
        let mut command = {
            let mut command = std::process::Command::new("cmd");
            command.args(["/d", "/c", "exit", "0"]);
            command
        };
        #[cfg(not(windows))]
        let mut command = std::process::Command::new("true");
        let mut child = command.spawn().expect("spawn a short process");
        let pid = child.id();
        child.wait().expect("wait for it");
        // The handle keeps an exited process queryable on Windows.
        drop(child);

        assert_eq!(pid_identity(pid, "anything", None), PidIdentity::Gone);
    }

    const SLEEPER_ENV: &str = "MESH_LLM_SYSTEM_TEST_SLEEPER";

    /// Holds the copy started by the next test running; returns at once
    /// otherwise.
    #[test]
    fn sleeper() {
        if std::env::var_os(SLEEPER_ENV).is_some() {
            std::thread::sleep(std::time::Duration::from_secs(30));
        }
    }

    // The kernel names a process after the file it was started from, and
    // that name has to survive the file being renamed or deleted while the
    // process runs, as when a release replaces the binary under a node.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn a_running_process_keeps_its_identity_when_its_executable_is_renamed_or_deleted() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let executable = dir.path().join("mesh-sleeper");
        std::fs::copy(std::env::current_exe().unwrap(), &executable).expect("copy the test binary");
        let mut child = std::process::Command::new(&executable)
            .args(["--exact", "pid_identity_tests::sleeper", "--nocapture"])
            .env(SLEEPER_ENV, "1")
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("start the copy");
        let pid = child.id();
        let started = process_started_at_unix(pid)
            .expect("read the start time")
            .expect("the child is running");

        let mut identities = vec![pid_identity(pid, "mesh-sleeper", Some(started))];
        let renamed = dir.path().join("renamed");
        std::fs::rename(&executable, &renamed).expect("rename the executable");
        identities.push(pid_identity(pid, "mesh-sleeper", Some(started)));
        std::fs::remove_file(&renamed).expect("delete the executable");
        identities.push(pid_identity(pid, "mesh-sleeper", Some(started)));
        let liveness = process_liveness(pid);

        child.kill().expect("stop the child");
        child.wait().expect("reap the child");
        assert_eq!(identities, [PidIdentity::Recorded; 3]);
        assert_eq!(liveness, Liveness::Alive);
    }

    #[cfg(windows)]
    #[test]
    fn an_exited_process_another_handle_still_references_is_gone() {
        let mut child = std::process::Command::new("cmd")
            .args(["/d", "/c", "exit", "0"])
            .spawn()
            .expect("spawn a short process");
        child.wait().expect("wait for it");
        // `child` still holds its handle, so the process object outlives the
        // process, as it does while a parent keeps an exited child's handle.

        assert_eq!(
            process_comm(child.id()).expect("query the exited process"),
            None
        );
        assert_eq!(pid_identity(child.id(), "cmd", None), PidIdentity::Gone);
    }

    // A normal session is refused even the least query on the services of
    // other accounts; an elevated one may query them all, leaving nothing to
    // check.
    #[cfg(windows)]
    #[test]
    fn a_process_of_another_account_is_another_process() {
        let Some(pid) = (4..65_536)
            .step_by(4)
            .find(|&pid| platform::process_query_denied(pid))
        else {
            eprintln!("no process refuses this session's queries; nothing to check");
            return;
        };

        assert_eq!(pid_identity(pid, "mesh-llm", None), PidIdentity::Other);
    }

    // 259 is also STILL_ACTIVE, the code a running process reports, so only
    // waiting on the handle tells this exit apart from a running process.
    #[cfg(windows)]
    #[test]
    fn a_process_that_exited_with_code_259_is_gone() {
        let mut child = std::process::Command::new("cmd")
            .args(["/d", "/c", "exit", "259"])
            .spawn()
            .expect("spawn a short process");
        assert_eq!(child.wait().expect("wait for it").code(), Some(259));

        assert_eq!(
            process_comm(child.id()).expect("query the exited process"),
            None
        );
    }
}

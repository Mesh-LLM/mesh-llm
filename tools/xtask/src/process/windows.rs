use super::Failure;
use std::io::{self, Read};
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus};
use std::thread;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{ERROR_BROKEN_PIPE, HANDLE};
use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, GenerateConsoleCtrlEvent};
use windows_sys::Win32::System::Pipes::PeekNamedPipe;
use windows_sys::Win32::System::Threading::{CREATE_NEW_PROCESS_GROUP, CREATE_SUSPENDED};

#[path = "windows_job.rs"]
mod job;
#[path = "windows_resume.rs"]
mod resume;

pub(super) trait Pipe: AsRawHandle {}
impl Pipe for ChildStdout {}
impl Pipe for ChildStderr {}
pub(super) fn prepare_pipe(_: &impl Pipe) -> io::Result<()> {
    Ok(())
}

pub(super) fn pending_bytes(pipe: &impl Pipe) -> io::Result<usize> {
    let mut available = 0;
    // SAFETY: live borrowed pipe and writable count, with no data buffer.
    if unsafe {
        PeekNamedPipe(
            pipe.as_raw_handle(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            &mut available,
            std::ptr::null_mut(),
        )
    } == 0
    {
        let error = io::Error::last_os_error();
        if error
            .raw_os_error()
            .and_then(|code| u32::try_from(code).ok())
            == Some(ERROR_BROKEN_PIPE)
        {
            return Ok(0);
        }
        return Err(error);
    }
    usize::try_from(available).map_err(io::Error::other)
}

pub(super) fn read_pipe(pipe: &mut (impl Read + Pipe), buffer: &mut [u8]) -> io::Result<usize> {
    let mut available = 0;
    // SAFETY: the pipe handle stays borrowed, and available is writable. No
    // other reader consumes this pipe between PeekNamedPipe and read.
    if unsafe {
        PeekNamedPipe(
            pipe.as_raw_handle(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            &mut available,
            std::ptr::null_mut(),
        )
    } == 0
    {
        let error = io::Error::last_os_error();
        if error
            .raw_os_error()
            .and_then(|code| u32::try_from(code).ok())
            == Some(ERROR_BROKEN_PIPE)
        {
            return Ok(0);
        }
        return Err(error);
    }
    let count = usize::try_from(available)
        .unwrap_or(usize::MAX)
        .min(buffer.len());
    if count == 0 {
        return Err(io::ErrorKind::WouldBlock.into());
    }
    pipe.read(&mut buffer[..count])
}

pub(super) struct OwnedChild {
    child: Child,
    job: job::Job,
    armed: bool,
    assigned: bool,
}

impl OwnedChild {
    pub(super) fn spawn(command: &mut Command) -> Result<Self, Failure> {
        let job = job::Job::new()?;
        command.creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP);
        let child = command
            .spawn()
            .map_err(|error| Failure::io("spawn", error))?;
        let mut owned = Self {
            child,
            job,
            armed: true,
            assigned: false,
        };
        owned.job.assign(owned.child.as_raw_handle())?;
        owned.assigned = true;
        resume::resume(owned.child.id())?;
        Ok(owned)
    }

    pub(super) fn id(&self) -> u32 {
        self.child.id()
    }
    pub(super) fn stdout(&mut self) -> Option<ChildStdout> {
        self.child.stdout.take()
    }
    pub(super) fn stderr(&mut self) -> Option<ChildStderr> {
        self.child.stderr.take()
    }
    pub(super) fn exited(&mut self) -> Result<bool, Failure> {
        self.child
            .try_wait()
            .map(|status| status.is_some())
            .map_err(|error| Failure::io("observe exit", error))
    }
    pub(super) fn active(&mut self) -> Result<bool, Failure> {
        self.job.active()
    }
    pub(super) fn graceful(&mut self) -> Result<(), Failure> {
        // SAFETY: CREATE_NEW_PROCESS_GROUP made this owned PID the group ID.
        // The retained process handle prevents PID reuse until the last signal.
        if unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, self.child.id()) } == 0 {
            return Err(Failure::io("console break", io::Error::last_os_error()));
        }
        Ok(())
    }
    pub(super) fn force(&mut self) -> Result<(), Failure> {
        self.job.terminate()
    }
    pub(super) fn reap(&mut self) -> Result<Option<ExitStatus>, Failure> {
        let status = self
            .child
            .try_wait()
            .map_err(|error| Failure::io("reap", error))?;
        if status.is_some() {
            self.armed = false;
        }
        Ok(status)
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let result = (|| {
            if self.assigned {
                self.force()?;
            } else {
                self.child
                    .kill()
                    .map_err(|error| Failure::io("stop suspended child", error))?;
            }
            let until = Instant::now() + Duration::from_secs(2);
            while self.active()? || self.reap()?.is_none() {
                if Instant::now() >= until {
                    return Err(Failure::CleanupDeadline);
                }
                thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        })();
        if let Err(error) = result {
            eprintln!("process emergency cleanup failed: {error}");
        }
    }
}

pub(super) fn win_result(success: i32, operation: &'static str) -> Result<(), Failure> {
    if success == 0 {
        Err(Failure::io(operation, io::Error::last_os_error()))
    } else {
        Ok(())
    }
}

pub(super) fn handle(raw: HANDLE) -> Result<std::os::windows::io::OwnedHandle, Failure> {
    use std::os::windows::io::FromRawHandle;
    if raw.is_null() || raw == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
        return Err(Failure::io("create handle", io::Error::last_os_error()));
    }
    // SAFETY: caller transfers a freshly created, non-null owned kernel handle.
    Ok(unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(raw) })
}

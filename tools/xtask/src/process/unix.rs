use super::Failure;
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
#[path = "unix_macos.rs"]
mod membership;

#[cfg(target_os = "macos")]
#[path = "macos_capture.rs"]
mod capture;

#[cfg(all(test, target_os = "macos"))]
pub(super) fn test_pipe() -> io::Result<(std::os::fd::OwnedFd, std::os::fd::OwnedFd)> {
    capture::test_pipe()
}
#[cfg(target_os = "linux")]
#[path = "unix_linux.rs"]
mod membership;

pub(super) trait Pipe: AsRawFd {}
impl Pipe for ChildStdout {}
impl Pipe for ChildStderr {}

pub(super) fn prepare_pipe(pipe: &impl Pipe) -> io::Result<()> {
    // SAFETY: the borrowed descriptor remains open for both fcntl calls.
    let flags = unsafe { libc::fcntl(pipe.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: F_SETFL consumes an integer flag word, not a pointer.
    if unsafe { libc::fcntl(pipe.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(super) fn read_pipe(pipe: &mut (impl Read + Pipe), buffer: &mut [u8]) -> io::Result<usize> {
    pipe.read(buffer)
}

pub(super) fn pending_bytes(pipe: &impl Pipe) -> io::Result<usize> {
    let mut bytes: libc::c_int = 0;
    // SAFETY: FIONREAD writes one int through a live pipe's borrowed descriptor.
    if unsafe { libc::ioctl(pipe.as_raw_fd(), libc::FIONREAD, &mut bytes) } < 0 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(bytes).map_err(io::Error::other)
}

pub(super) struct OwnedChild {
    child: Child,
    group: libc::pid_t,
    armed: bool,
}

impl OwnedChild {
    pub(super) fn spawn(command: &mut Command) -> Result<Self, Failure> {
        command.process_group(0);
        #[cfg(target_os = "macos")]
        let child = capture::spawn(command);
        #[cfg(not(target_os = "macos"))]
        let child = command.spawn();
        let child = child.map_err(|error| Failure::io("spawn", error))?;
        let group = libc::pid_t::try_from(child.id())
            .map_err(|_| Failure::InvalidSpec("PID out of range"))?;
        Ok(Self {
            child,
            group,
            armed: true,
        })
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
        // SAFETY: siginfo_t is a C POD; zeroing initializes its no-event state.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: info is writable and sized correctly. WNOWAIT preserves our
        // exclusive child wait ownership and reserves the group leader's PID.
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                self.child.id(),
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result < 0 {
            return Err(Failure::io("observe exit", io::Error::last_os_error()));
        }
        // SAFETY: waitid initialized the SIGCHLD union for this wait operation.
        Ok(unsafe { info.si_pid() } != 0)
    }

    pub(super) fn active(&mut self) -> Result<bool, Failure> {
        membership::group_active(self.group)
    }
    pub(super) fn graceful(&mut self) -> Result<(), Failure> {
        self.signal(libc::SIGTERM)
    }
    pub(super) fn force(&mut self) -> Result<(), Failure> {
        self.signal(libc::SIGKILL)
    }

    fn signal(&self, signal: libc::c_int) -> Result<(), Failure> {
        if !self.armed {
            return Ok(());
        }
        // SAFETY: positive group belongs to the still-unreaped child; negative
        // PID targets only that group, never the supervisor or a recycled PGID.
        if unsafe { libc::kill(-self.group, signal) } < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(Failure::io("signal group", error));
            }
        }
        Ok(())
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
            self.force()?;
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

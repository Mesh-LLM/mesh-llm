use crate::support::Case;
use std::fs::File;
use std::io::{self, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, ExitStatus, Stdio};
use std::time::{Duration, Instant};

pub struct Terminal<'a> {
    pub child: Child,
    master: File,
    case: &'a Case,
    completed: bool,
    disarmed: bool,
}

impl<'a> Terminal<'a> {
    pub fn start(case: &'a Case) -> Self {
        Self::start_command(case, "client-readiness")
    }

    pub fn start_command(case: &'a Case, name: &str) -> Self {
        let (master, slave) = pair().unwrap();
        let mut arguments = case.arguments();
        arguments[5] = "60".into();
        let stdout = File::create(case.native.join("cli.stdout")).unwrap();
        let stderr = File::create(case.native.join("cli.stderr")).unwrap();
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"));
        command
            .current_dir(crate::support::repository())
            .args(["automation", name])
            .args(arguments)
            .stdin(Stdio::from(slave))
            .stdout(stdout)
            .stderr(stderr);
        Self::start_prepared(case, command, master)
    }

    pub fn start_prepared(
        case: &'a Case,
        mut command: std::process::Command,
        master: File,
    ) -> Self {
        // SAFETY: the fork child calls only async-signal-safe terminal syscalls before exec.
        unsafe {
            command.pre_exec(|| {
                #[cfg(target_os = "linux")]
                let request = libc::TIOCSCTTY;
                #[cfg(not(target_os = "linux"))]
                let request = libc::TIOCSCTTY.into();
                if libc::setsid() < 0 || libc::ioctl(0, request, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
                if libc::tcsetpgrp(0, libc::getpgrp()) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        Self {
            child: command.spawn().unwrap(),
            master,
            case,
            completed: false,
            disarmed: false,
        }
    }

    pub fn wait_file(&mut self, name: &str) {
        let until = Instant::now() + Duration::from_secs(5);
        while !self.case.native.join(name).is_file() {
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "CLI exited before {name}"
            );
            assert!(Instant::now() < until, "fixture handshake timeout: {name}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    pub fn ctrl_c(&mut self) {
        let pid = i32::try_from(self.child.id()).unwrap();
        // SAFETY: the master is open and this query verifies the terminal's actual foreground group.
        assert_eq!(unsafe { libc::tcgetpgrp(self.master.as_raw_fd()) }, pid);
        self.master.write_all(&[3]).unwrap();
    }

    pub fn terminate(&self) {
        let pid = i32::try_from(self.child.id()).unwrap();
        // SAFETY: this is the owned, unreaped CLI PID, not a process-name lookup.
        assert_eq!(unsafe { libc::kill(pid, libc::SIGTERM) }, 0);
    }

    pub fn finish(&mut self) -> ExitStatus {
        let until = Instant::now() + Duration::from_secs(12);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                self.completed = true;
                return status;
            }
            assert!(
                Instant::now() < until,
                "CLI interruption exceeded outer deadline"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    pub fn disarm(&mut self) {
        eprintln!(
            "interruption cleanup verified: cli={}, client={}, state_entries=0, diagnostics={}",
            self.child.id(),
            self.case.audit().pid,
            std::fs::read_to_string(self.case.native.join("cli.stderr")).unwrap()
        );
        self.disarmed = true;
    }
}

impl Drop for Terminal<'_> {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        if let Ok(bytes) = std::fs::read(self.case.native.join("audit.json"))
            && let Ok(audit) = serde_json::from_slice::<crate::protocol::Audit>(&bytes)
        {
            let group = i32::try_from(audit.pid).unwrap();
            // SAFETY: the outer guard targets only this fixture's recorded owned group.
            if unsafe { libc::kill(-group, libc::SIGKILL) } < 0
                && io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
            {
                eprintln!("outer fixture cleanup: {}", io::Error::last_os_error());
            }
            let until = Instant::now() + Duration::from_secs(2);
            loop {
                // SAFETY: signal zero only observes the fixture-recorded group during bounded teardown.
                if unsafe { libc::kill(-group, 0) } < 0 {
                    break;
                }
                if Instant::now() >= until {
                    eprintln!("outer fixture group {group} still present at cleanup deadline");
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        if !self.completed {
            if let Err(error) = self.child.kill() {
                eprintln!("outer CLI cleanup: {error}");
            }
            let until = Instant::now() + Duration::from_secs(2);
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < until => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    result => {
                        eprintln!("outer CLI reap deadline/failure: {result:?}");
                        break;
                    }
                }
            }
        }
    }
}

pub(crate) fn pair() -> io::Result<(File, File)> {
    let mut master = -1;
    let mut slave = -1;
    // SAFETY: both out pointers are writable; null optional arguments request terminal defaults.
    if unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } < 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful openpty transfers two distinct open descriptors to this owner.
    let master = unsafe { File::from_raw_fd(master) };
    // SAFETY: the slave descriptor has not yet been wrapped or closed.
    let slave = unsafe { File::from_raw_fd(slave) };
    for file in [&master, &slave] {
        // SAFETY: each descriptor is open and F_SETFD takes an integer flag argument.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok((master, slave))
}

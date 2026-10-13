use crate::protocol::Behavior;
use crate::support::{Case, ready};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::CommandExt;

#[derive(Clone, Copy)]
enum Mask {
    Normal,
    Blocked(libc::c_int),
    Pending(libc::c_int),
}

#[test]
fn migration_lifecycle_blocked_sigint_rejects_before_effects() {
    assert_rejected(Mask::Blocked(libc::SIGINT));
}

#[test]
fn migration_lifecycle_blocked_sigterm_rejects_before_effects() {
    assert_rejected(Mask::Blocked(libc::SIGTERM));
}

#[test]
fn migration_lifecycle_blocked_pending_sigint_rejects_before_effects() {
    assert_rejected(Mask::Pending(libc::SIGINT));
}

#[test]
fn migration_lifecycle_blocked_pending_sigterm_rejects_before_effects() {
    assert_rejected(Mask::Pending(libc::SIGTERM));
}

#[test]
fn migration_lifecycle_blocked_normal_mask_control_succeeds() {
    let case = Case::new(Behavior::Clean, vec![ready()]);

    let output = run(&case, Mask::Normal);

    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let port = case.audit().arguments[3].parse::<u16>().unwrap();
    assert_eq!(
        output.stdout,
        format!("client readiness observed on port {port}\n").as_bytes()
    );
    assert!(output.stderr.is_empty());
    assert!(case.native.join("handler").is_file());
    case.assert_removed();
}

fn assert_rejected(mask: Mask) {
    let case = Case::new(Behavior::Clean, vec![ready()]);
    let before = std::fs::metadata(&case.state).unwrap();

    let output = run(&case, mask);

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("blocked"),
        "{output:?}"
    );
    assert!(!case.native.join("audit.json").exists());
    assert!(!case.native.join("startup.armed").exists());
    assert_eq!(std::fs::read_dir(&case.state).unwrap().count(), 0);
    let after = std::fs::metadata(&case.state).unwrap();
    assert_eq!(
        (
            before.mtime(),
            before.mtime_nsec(),
            before.ctime(),
            before.ctime_nsec()
        ),
        (
            after.mtime(),
            after.mtime_nsec(),
            after.ctime(),
            after.ctime_nsec()
        ),
        "state parent changed despite admission rejection"
    );
}

fn run(case: &Case, mask: Mask) -> std::process::Output {
    let mut command = case.command();
    // SAFETY: the fork child performs only signal-set, mask, raise and alarm syscalls before exec.
    unsafe {
        command.pre_exec(move || {
            let mut signals = std::mem::MaybeUninit::<libc::sigset_t>::zeroed();
            if libc::sigemptyset(signals.as_mut_ptr()) < 0 {
                return Err(io::Error::last_os_error());
            }
            let mut signals = signals.assume_init();
            match mask {
                Mask::Normal => (),
                Mask::Blocked(signal) | Mask::Pending(signal) => {
                    if libc::sigaddset(&mut signals, signal) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
            }
            if libc::sigprocmask(libc::SIG_SETMASK, &signals, std::ptr::null_mut()) < 0 {
                return Err(io::Error::last_os_error());
            }
            match mask {
                Mask::Pending(signal) => {
                    if libc::raise(signal) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Mask::Normal | Mask::Blocked(_) => (),
            }
            libc::alarm(12);
            Ok(())
        });
    }
    command.output().unwrap()
}

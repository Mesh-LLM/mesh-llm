use super::Failure;
use std::io;

pub(super) fn group_active(group: libc::pid_t) -> Result<bool, Failure> {
    const PROC_PGRP_ONLY: u32 = 2;
    let mut members = [0_i32; 4096];
    let size =
        i32::try_from(std::mem::size_of_val(&members)).map_err(|_| Failure::EnumerationLimit)?;
    let group = u32::try_from(group).map_err(|_| Failure::EnumerationLimit)?;
    // SAFETY: __error returns this thread's errno slot; clearing it lets a
    // zero-byte result be distinguished from a reported failure.
    unsafe { *libc::__error() = 0 };
    // SAFETY: members is an aligned initialized PID array, and size is its exact
    // writable byte length. libproc writes at most size bytes.
    let bytes =
        unsafe { libc::proc_listpids(PROC_PGRP_ONLY, group, members.as_mut_ptr().cast(), size) };
    if bytes < 0 {
        return Err(Failure::io("list group", io::Error::last_os_error()));
    }
    if bytes == 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(0) {
            return Ok(false);
        }
        return Err(Failure::io("list group", error));
    }
    if bytes == size {
        return Err(Failure::EnumerationLimit);
    }
    let count =
        usize::try_from(bytes).map_err(|_| Failure::EnumerationLimit)? / std::mem::size_of::<i32>();
    for pid in &members[..count] {
        if *pid == 0 {
            continue;
        }
        // SAFETY: proc_bsdshortinfo is a C POD whose integer/byte fields admit zero.
        let mut info: libc::proc_bsdshortinfo = unsafe { std::mem::zeroed() };
        let size =
            i32::try_from(std::mem::size_of_val(&info)).map_err(|_| Failure::EnumerationLimit)?;
        // Short BSD info exposes the process/group/status fields needed here
        // and is exempt from the full-info same-user check. That check returns
        // EPERM for a group member whose effective UID differs from ours: on
        // macOS `/bin/ps` is setuid root, and mesh-llm spawns it inside its own
        // process group every few seconds (local instance scanner), so the full
        // flavor failed clean shutdown whenever a poll overlapped that child.
        // Include zombies (arg=1) so an unreaped exiting child is observed as
        // inactive. Errors other than ESRCH still fail closed.
        // SAFETY: the short BSD-info flavor matches the writable info buffer.
        let read = unsafe {
            libc::proc_pidinfo(
                *pid,
                libc::PROC_PIDT_SHORTBSDINFO,
                1,
                (&mut info as *mut libc::proc_bsdshortinfo).cast(),
                size,
            )
        };
        if read == 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                continue;
            }
            return Err(Failure::io("inspect group member", error));
        }
        if read != size {
            return Err(Failure::EnumerationLimit);
        }
        if info.pbsi_pgid == group && member_is_live(info.pbsi_status, info.pbsi_flags) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// PROC_FLAG_INEXIT from <sys/proc_info.h>: the process is working its way
/// through exit(). libc does not export it.
const PROC_FLAG_INEXIT: u32 = 0x4;

/// A member still needs to be stopped only while it is neither a zombie nor
/// already exiting. An exiting member can still report `SRUN` before it turns
/// into a zombie, but it is no longer a signal target (`kill` on the group
/// returns `EPERM`), and it finishes exiting without our help.
fn member_is_live(status: u32, flags: u32) -> bool {
    status != libc::SZOMB && flags & PROC_FLAG_INEXIT == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exiting_or_zombie_members_are_not_live() {
        assert!(member_is_live(2, 0x10));
        assert!(!member_is_live(libc::SZOMB, 0x10));
        assert!(!member_is_live(2, 0x0140_0094));
    }
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    /// Effective UID of `pid` through the same short flavor the supervisor uses.
    fn effective_uid(pid: i32) -> u32 {
        // SAFETY: proc_bsdshortinfo is a C POD whose integer/byte fields admit zero.
        let mut info: libc::proc_bsdshortinfo = unsafe { std::mem::zeroed() };
        let size = i32::try_from(std::mem::size_of_val(&info)).unwrap();
        // SAFETY: the short BSD-info flavor matches the writable info buffer.
        let read = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDT_SHORTBSDINFO,
                1,
                (&mut info as *mut libc::proc_bsdshortinfo).cast(),
                size,
            )
        };
        assert_eq!(read, size, "{}", io::Error::last_os_error());
        info.pbsi_uid
    }

    /// A setuid-root member (`top`, like the `ps` mesh-llm spawns) must be
    /// inspectable: the full BSD-info flavor answers EPERM for it and failed
    /// the Laya smoke's clean-shutdown gate.
    #[test]
    fn macos_group_membership_inspects_setuid_member_without_eperm() {
        // SAFETY: geteuid has no preconditions.
        if unsafe { libc::geteuid() } == 0 {
            return; // root passes the same-user check; nothing to prove here
        }
        let mut child = Command::new("/usr/bin/top")
            .args(["-l", "0", "-s", "1", "-n", "0"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .unwrap();
        let group = i32::try_from(child.id()).unwrap();
        // Prove the precondition: the member runs with a different effective UID.
        let deadline = Instant::now() + Duration::from_secs(5);
        while effective_uid(group) != 0 {
            assert!(Instant::now() < deadline, "top did not become setuid root");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(group_active(group).unwrap());
        child.kill().unwrap();
        child.wait().unwrap();
    }

    /// Keep the leader unreaped, as the supervisor does, throughout shutdown.
    #[test]
    fn macos_group_membership_observes_live_then_unreaped_exit() {
        let mut child = Command::new("/bin/sleep")
            .arg("30")
            .process_group(0)
            .spawn()
            .unwrap();
        let group = i32::try_from(child.id()).unwrap();
        assert!(group_active(group).unwrap());
        child.kill().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while group_active(group).unwrap() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        child.wait().unwrap();
    }
}

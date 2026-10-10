use super::Failure;
use std::io;

pub(super) fn group_active(group: libc::pid_t) -> Result<bool, Failure> {
    const PROC_PGRP_ONLY: u32 = 2;
    let mut members = [0_i32; 4096];
    let size =
        i32::try_from(std::mem::size_of_val(&members)).map_err(|_| Failure::EnumerationLimit)?;
    let group = u32::try_from(group).map_err(|_| Failure::EnumerationLimit)?;
    // SAFETY: members is an aligned initialized PID array, and size is its exact
    // writable byte length. libproc writes at most size bytes.
    let bytes =
        unsafe { libc::proc_listpids(PROC_PGRP_ONLY, group, members.as_mut_ptr().cast(), size) };
    if bytes <= 0 {
        return Err(Failure::io("list group", io::Error::last_os_error()));
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
        // SAFETY: proc_bsdinfo is a C POD whose integer/byte fields admit zero.
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size =
            i32::try_from(std::mem::size_of_val(&info)).map_err(|_| Failure::EnumerationLimit)?;
        // SAFETY: the BSD-info flavor matches the aligned, writable info buffer.
        let read = unsafe {
            libc::proc_pidinfo(
                *pid,
                libc::PROC_PIDTBSDINFO,
                0,
                (&mut info as *mut libc::proc_bsdinfo).cast(),
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
        if info.pbi_pgid == group && info.pbi_status != libc::SZOMB {
            return Ok(true);
        }
    }
    Ok(false)
}

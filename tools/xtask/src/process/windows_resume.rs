use super::{Failure, handle, win_result};
use std::io;
use std::os::windows::io::AsRawHandle;
use windows_sys::Win32::Foundation::ERROR_NO_MORE_FILES;
use windows_sys::Win32::System::Diagnostics::ToolHelp::*;
use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

pub(super) fn resume(pid: u32) -> Result<(), Failure> {
    // SAFETY: snapshot returns a new owned handle and does not borrow memory.
    let snapshot = handle(unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) })?;
    let mut entry = THREADENTRY32 {
        dwSize: u32::try_from(std::mem::size_of::<THREADENTRY32>())
            .map_err(|_| Failure::EnumerationLimit)?,
        ..Default::default()
    };
    // SAFETY: entry is a correctly sized writable THREADENTRY32.
    win_result(
        unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) },
        "enumerate suspended thread",
    )?;
    for _ in 0..131072 {
        if entry.th32OwnerProcessID == pid {
            // SAFETY: the child is suspended and retains the PID/thread identity.
            let thread =
                handle(unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) })?;
            // SAFETY: handle has THREAD_SUSPEND_RESUME access to our child thread.
            if unsafe { ResumeThread(thread.as_raw_handle()) } == u32::MAX {
                return Err(Failure::io("resume child", io::Error::last_os_error()));
            }
            return Ok(());
        }
        entry.dwSize = u32::try_from(std::mem::size_of::<THREADENTRY32>())
            .map_err(|_| Failure::EnumerationLimit)?;
        // SAFETY: entry remains writable, initialized, and correctly sized.
        if unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) } == 0 {
            let error = io::Error::last_os_error();
            if error
                .raw_os_error()
                .and_then(|code| u32::try_from(code).ok())
                == Some(ERROR_NO_MORE_FILES)
            {
                break;
            }
            return Err(Failure::io("enumerate thread", error));
        }
    }
    Err(Failure::EnumerationLimit)
}

use super::{Failure, handle, win_result};
use std::os::windows::io::{AsRawHandle, OwnedHandle};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::JobObjects::*;

pub(super) struct Job(OwnedHandle);

impl Job {
    pub(super) fn new() -> Result<Self, Failure> {
        // SAFETY: null attributes produce a non-inheritable unnamed job handle.
        let owned = handle(unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) })?;
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: typed info matches the selected class and lives for the call.
        win_result(
            unsafe {
                SetInformationJobObject(
                    owned.as_raw_handle(),
                    JobObjectExtendedLimitInformation,
                    (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    u32::try_from(std::mem::size_of_val(&info))
                        .map_err(|_| Failure::EnumerationLimit)?,
                )
            },
            "set job limits",
        )?;
        Ok(Self(owned))
    }

    pub(super) fn assign(&self, child: HANDLE) -> Result<(), Failure> {
        // SAFETY: child is suspended and both process/job handles remain owned.
        win_result(
            unsafe { AssignProcessToJobObject(self.0.as_raw_handle(), child) },
            "assign job",
        )
    }

    pub(super) fn active(&self) -> Result<bool, Failure> {
        let mut info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
        // SAFETY: the writable accounting structure matches the selected class.
        win_result(
            unsafe {
                QueryInformationJobObject(
                    self.0.as_raw_handle(),
                    JobObjectBasicAccountingInformation,
                    (&mut info as *mut JOBOBJECT_BASIC_ACCOUNTING_INFORMATION).cast(),
                    u32::try_from(std::mem::size_of_val(&info))
                        .map_err(|_| Failure::EnumerationLimit)?,
                    std::ptr::null_mut(),
                )
            },
            "query job",
        )?;
        Ok(info.ActiveProcesses != 0)
    }

    pub(super) fn terminate(&self) -> Result<(), Failure> {
        // SAFETY: this handle owns only the child tree assigned by this module.
        win_result(
            unsafe { TerminateJobObject(self.0.as_raw_handle(), 1) },
            "terminate job",
        )
    }
}

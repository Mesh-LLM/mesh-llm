use super::{Failure, handle, win_result};
use crate::ProcessMember;
use std::os::windows::io::{AsRawHandle, OwnedHandle};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::JobObjects::*;
use windows_sys::Win32::System::Threading::{
    OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};

// The variable-length JobObjectBasicProcessIdList header followed by PID slots.
#[repr(C)]
struct ProcessIds {
    assigned: u32,
    count: u32,
    ids: [usize; 4096],
}

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

    pub(super) fn members(&self) -> Result<Vec<ProcessMember>, Failure> {
        let mut info = Box::new(ProcessIds {
            assigned: 0,
            count: 0,
            ids: [0; 4096],
        });
        // SAFETY: repr(C) matches the selected variable-length header, with
        // aligned writable PID slots and the exact allocated buffer size.
        win_result(
            unsafe {
                QueryInformationJobObject(
                    self.0.as_raw_handle(),
                    JobObjectBasicProcessIdList,
                    (&mut *info as *mut ProcessIds).cast(),
                    u32::try_from(std::mem::size_of::<ProcessIds>())
                        .map_err(|_| Failure::EnumerationLimit)?,
                    std::ptr::null_mut(),
                )
            },
            "enumerate pre-force job",
        )?;
        let count = usize::try_from(info.count).map_err(|_| Failure::EnumerationLimit)?;
        let ids = info.ids.get(..count).ok_or(Failure::EnumerationLimit)?;
        ids.iter()
            .map(|&pid| {
                let pid = u32::try_from(pid).map_err(|_| Failure::EnumerationLimit)?;
                Ok(ProcessMember {
                    pid,
                    image: image_name(pid),
                })
            })
            .collect()
    }

    pub(super) fn terminate(&self) -> Result<(), Failure> {
        // SAFETY: this handle owns only the child tree assigned by this module.
        win_result(
            unsafe { TerminateJobObject(self.0.as_raw_handle(), 1) },
            "terminate job",
        )
    }
}

fn image_name(pid: u32) -> Result<String, Failure> {
    // SAFETY: OpenProcess returns a new owned handle with query-only access.
    let process = handle(unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) })?;
    let mut buffer = vec![0u16; 32768];
    let mut length = u32::try_from(buffer.len()).map_err(|_| Failure::EnumerationLimit)?;
    // SAFETY: live query handle and writable UTF-16 buffer of length elements.
    win_result(
        unsafe {
            QueryFullProcessImageNameW(
                process.as_raw_handle(),
                PROCESS_NAME_WIN32,
                buffer.as_mut_ptr(),
                &mut length,
            )
        },
        "query pre-force process image",
    )?;
    let length = usize::try_from(length).map_err(|_| Failure::EnumerationLimit)?;
    let name = String::from_utf16_lossy(buffer.get(..length).ok_or(Failure::EnumerationLimit)?);
    Ok(name.rsplit('\\').next().unwrap_or(&name).to_owned())
}

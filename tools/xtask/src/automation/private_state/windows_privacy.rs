use super::Error;
use std::ffi::c_void;
use std::io;
use std::mem::size_of_val;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::Path;
use std::ptr::null_mut;
use windows_sys::Win32::Foundation::LocalFree;
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER, TokenUser,
};
use windows_sys::Win32::Storage::FileSystem::CreateDirectoryW;
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/windows_privacy.rs"]
mod tests;

pub(super) fn create(path: &Path) -> Result<(), Error> {
    let sid = current_user_sid()?;
    let descriptor_text = format!("D:P(A;OICI;FA;;;{sid})");
    let descriptor_text: Vec<u16> = descriptor_text.encode_utf16().chain(Some(0)).collect();
    let mut descriptor = null_mut();
    // SAFETY: terminated SDDL and writable allocation output; the API allocates the descriptor.
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            descriptor_text.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            null_mut(),
        )
    } == 0
    {
        return Err(Error::io("create private DACL", io::Error::last_os_error()));
    }
    let descriptor = LocalAllocation(descriptor);
    let mut attributes = SECURITY_ATTRIBUTES {
        nLength: 0,
        lpSecurityDescriptor: descriptor.0,
        bInheritHandle: 0,
    };
    attributes.nLength = u32::try_from(size_of_val(&attributes))
        .map_err(|_| Error::Invalid("security attributes exceed Windows size"))?;
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: the path, attributes and descriptor remain live through atomic directory creation.
    if unsafe { CreateDirectoryW(path.as_ptr(), &attributes) } == 0 {
        return Err(Error::io(
            "create private directory",
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

fn current_user_sid() -> Result<String, Error> {
    // SAFETY: GetCurrentProcess returns the calling process pseudo-handle without ownership.
    let process = unsafe { GetCurrentProcess() };
    let mut token = null_mut();
    // SAFETY: requests query-only access and writes one owned token handle on success.
    if unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) } == 0 {
        return Err(Error::io("query current user", io::Error::last_os_error()));
    }
    // SAFETY: successful OpenProcessToken transferred exactly one owned non-null handle.
    let token = unsafe { OwnedHandle::from_raw_handle(token) };
    let mut storage = [0_usize; 64];
    let capacity = u32::try_from(size_of_val(&storage))
        .map_err(|_| Error::Invalid("token storage exceeds Windows size"))?;
    let mut written = 0;
    // SAFETY: aligned writable storage bounds the TOKEN_USER and SID written by the API.
    if unsafe {
        GetTokenInformation(
            token.as_raw_handle(),
            TokenUser,
            storage.as_mut_ptr().cast(),
            capacity,
            &mut written,
        )
    } == 0
    {
        return Err(Error::io("read current user", io::Error::last_os_error()));
    }
    // SAFETY: successful TokenUser query initializes the aligned TOKEN_USER header in storage.
    let sid = unsafe { (*storage.as_ptr().cast::<TOKEN_USER>()).User.Sid };
    let mut text = null_mut();
    // SAFETY: the token's initialized SID stays in scope; output receives allocated terminated UTF-16.
    if unsafe { ConvertSidToStringSidW(sid, &mut text) } == 0 {
        return Err(Error::io("format current user", io::Error::last_os_error()));
    }
    let _allocation = LocalAllocation(text.cast());
    let mut encoded = Vec::with_capacity(184);
    for offset in 0..184 {
        // SAFETY: the API guarantees a terminated SID string; reading stops at its first NUL.
        let unit = unsafe { *text.add(offset) };
        if unit == 0 {
            return String::from_utf16(&encoded)
                .map_err(|_| Error::Invalid("Windows returned an invalid SID string"));
        }
        encoded.push(unit);
    }
    Err(Error::Invalid("Windows SID string exceeds its bound"))
}

struct LocalAllocation(*mut c_void);

impl Drop for LocalAllocation {
    fn drop(&mut self) {
        // SAFETY: each successful conversion transfers one LocalAlloc allocation into this owner.
        if !unsafe { LocalFree(self.0) }.is_null() {
            eprintln!("private-state security allocation release failed");
        }
    }
}

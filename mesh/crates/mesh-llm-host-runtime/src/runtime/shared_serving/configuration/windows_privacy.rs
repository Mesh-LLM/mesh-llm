//! Private shared-serving state on Windows. Install a protected, current-user
//! DACL at creation; never repair or truncate an unsafe existing object.

use anyhow::{Context, Result, ensure};
use std::ffi::c_void;
use std::fs::File;
use std::mem::{size_of, zeroed};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::Path;
use std::ptr::null_mut;
use windows_sys::Win32::Foundation::{
    ERROR_ALREADY_EXISTS, ERROR_PATH_NOT_FOUND, GENERIC_READ, GENERIC_WRITE, GetLastError,
    INVALID_HANDLE_VALUE, LocalFree,
};
use windows_sys::Win32::Security::Authorization::{GetSecurityInfo, SE_FILE_OBJECT};
use windows_sys::Win32::Security::{
    ACCESS_ALLOWED_ACE, ACL, ACL_REVISION, AddAccessAllowedAceEx, CONTAINER_INHERIT_ACE,
    DACL_SECURITY_INFORMATION, EqualSid, GetAce, GetLengthSid, GetSecurityDescriptorControl,
    GetTokenInformation, InitializeAcl, InitializeSecurityDescriptor, OBJECT_INHERIT_ACE,
    OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID, SE_DACL_PROTECTED, SECURITY_ATTRIBUTES,
    SECURITY_DESCRIPTOR, SetSecurityDescriptorControl, SetSecurityDescriptorDacl,
    SetSecurityDescriptorOwner, TOKEN_QUERY, TOKEN_USER, TokenUser,
};
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, CreateDirectoryW, CreateFileW, FILE_ALL_ACCESS,
    FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS,
    FILE_FLAG_OPEN_REPARSE_POINT, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, FILE_SHARE_WRITE,
    GetFileInformationByHandle, OPEN_ALWAYS, OPEN_EXISTING, READ_CONTROL,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

pub(super) fn private_directory(path: &Path) -> Result<()> {
    // Absolute paths make a missing relative parent unambiguous, including
    // a single-component state path. Existing parents need not be private.
    let path = std::path::absolute(path)?;
    with_private_security(true, |attributes, sid| {
        create_directory(&path, attributes)?;
        let (file, _) = open(&path, true, attributes)?;
        verify(&file, sid, true)
    })
    .with_context(|| {
        format!(
            "shared state directory must have a private Windows DACL: {}",
            path.display()
        )
    })
}

fn create_directory(path: &Path, attributes: &SECURITY_ATTRIBUTES) -> Result<()> {
    let wide = wide_path(path)?;
    // SAFETY: the path is NUL-terminated and the security descriptor and its
    // SID/ACL storage remain alive throughout this call and any recursion.
    if unsafe { CreateDirectoryW(wide.as_ptr(), attributes) } != 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error().map(|code| code as u32) {
        Some(ERROR_ALREADY_EXISTS) => Ok(()),
        Some(ERROR_PATH_NOT_FOUND) => {
            let parent = path.parent().context("state directory has no parent")?;
            create_directory(parent, attributes)?;
            if unsafe { CreateDirectoryW(wide.as_ptr(), attributes) } != 0 {
                Ok(())
            } else {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() == Some(ERROR_ALREADY_EXISTS as i32) {
                    Ok(())
                } else {
                    Err(error.into())
                }
            }
        }
        _ => Err(error.into()),
    }
}

/// The boolean distinguishes a new file from an existing empty/corrupt key.
pub(super) fn private_file(path: &Path) -> Result<(File, bool)> {
    with_private_security(false, |attributes, sid| {
        let (file, created) = open(path, false, attributes)?;
        verify(&file, sid, false)?;
        Ok((file, created))
    })
    .with_context(|| {
        format!(
            "shared state file must have a private Windows DACL: {}",
            path.display()
        )
    })
}

fn open(path: &Path, directory: bool, attributes: &SECURITY_ATTRIBUTES) -> Result<(File, bool)> {
    let wide = wide_path(path)?;
    let access = if directory {
        FILE_READ_ATTRIBUTES
    } else {
        GENERIC_READ | GENERIC_WRITE
    };
    // Do not follow a reparse point, truncate an existing file, or permit its
    // replacement while the returned handle is used to read/write secrets.
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            access | READ_CONTROL,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            attributes,
            if directory {
                OPEN_EXISTING
            } else {
                OPEN_ALWAYS
            },
            FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(std::io::Error::last_os_error().into());
    }
    // OPEN_ALWAYS documents zero for creation, ERROR_ALREADY_EXISTS otherwise.
    let created = !directory && unsafe { GetLastError() } != ERROR_ALREADY_EXISTS;
    // SAFETY: CreateFileW returned a new owned, valid handle.
    Ok((unsafe { File::from_raw_handle(handle) }, created))
}

fn verify(file: &File, sid: PSID, directory: bool) -> Result<()> {
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { zeroed() };
    ensure!(
        unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } != 0,
        "read state file attributes"
    );
    ensure!(
        info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT == 0,
        "shared state must not be a reparse point"
    );
    ensure!(
        (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0) == directory,
        "shared state has the wrong file type"
    );
    ensure!(
        directory || info.nNumberOfLinks == 1,
        "shared state file must not have other hard links"
    );
    let mut owner = null_mut();
    let mut dacl = null_mut();
    let mut descriptor = null_mut();
    let status = unsafe {
        GetSecurityInfo(
            file.as_raw_handle(),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            null_mut(),
            &mut dacl,
            null_mut(),
            &mut descriptor,
        )
    };
    let _descriptor = SecurityDescriptor(descriptor);
    ensure!(
        status == 0 && !owner.is_null() && !dacl.is_null() && !descriptor.is_null(),
        "read private state DACL"
    );
    ensure!(
        unsafe { EqualSid(owner, sid) } != 0,
        "shared state must be owned by the current user"
    );
    let mut control = 0;
    let mut revision = 0;
    ensure!(
        unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) } != 0
            && control & SE_DACL_PROTECTED != 0,
        "shared state DACL must not inherit access from its parent"
    );
    ensure!(
        unsafe { (*dacl).AceCount } == 1,
        "shared state must grant access only to its owner"
    );
    let mut ace = null_mut();
    ensure!(
        unsafe { GetAce(dacl, 0, &mut ace) } != 0 && !ace.is_null(),
        "read private state ACE"
    );
    let allowed = ace.cast::<ACCESS_ALLOWED_ACE>();
    // The OS returned a valid descriptor; inspect only a conventional allow
    // ACE before reading its mask/SID. Unknown or conditional ACEs fail closed.
    ensure!(
        unsafe { (*allowed).Header.AceType } == 0,
        "shared state requires an owner-only allow ACE"
    );
    ensure!(
        unsafe { (*allowed).Header.AceFlags as u32 } == inheritance(directory)
            && unsafe { (*allowed).Mask } == FILE_ALL_ACCESS
            && unsafe {
                EqualSid(
                    (&(*allowed).SidStart as *const u32)
                        .cast_mut()
                        .cast::<c_void>(),
                    sid,
                )
            } != 0,
        "shared state requires full access for its owner only"
    );
    Ok(())
}

fn with_private_security<T>(
    directory: bool,
    f: impl FnOnce(&SECURITY_ATTRIBUTES, PSID) -> Result<T>,
) -> Result<T> {
    with_current_user_sid(|sid| {
        let bytes = size_of::<ACL>() + size_of::<ACCESS_ALLOWED_ACE>() - size_of::<u32>()
            + unsafe { GetLengthSid(sid) as usize };
        let mut storage = vec![0_u64; bytes.div_ceil(size_of::<u64>())];
        let acl = storage.as_mut_ptr().cast::<ACL>();
        let mut descriptor: SECURITY_DESCRIPTOR = unsafe { zeroed() };
        let descriptor_ptr = (&mut descriptor as *mut SECURITY_DESCRIPTOR).cast::<c_void>();
        // SAFETY: all buffers are aligned, sized for the OS structures and
        // remain alive until the callback returns; no pointers escape it.
        ensure!(
            unsafe {
                InitializeAcl(acl, bytes as u32, ACL_REVISION) != 0
                    && AddAccessAllowedAceEx(
                        acl,
                        ACL_REVISION,
                        inheritance(directory),
                        FILE_ALL_ACCESS,
                        sid,
                    ) != 0
                    && InitializeSecurityDescriptor(descriptor_ptr, 1) != 0
                    && SetSecurityDescriptorOwner(descriptor_ptr, sid, 0) != 0
                    && SetSecurityDescriptorDacl(descriptor_ptr, 1, acl, 0) != 0
                    && SetSecurityDescriptorControl(
                        descriptor_ptr,
                        SE_DACL_PROTECTED,
                        SE_DACL_PROTECTED,
                    ) != 0
            },
            "construct private Windows DACL"
        );
        let attributes = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor_ptr,
            bInheritHandle: 0,
        };
        f(&attributes, sid)
    })
}

fn with_current_user_sid<T>(f: impl FnOnce(PSID) -> Result<T>) -> Result<T> {
    let mut token = null_mut();
    ensure!(
        unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } != 0,
        "open current user token"
    );
    let _token = unsafe { OwnedHandle::from_raw_handle(token) };
    let mut bytes = 0;
    unsafe {
        GetTokenInformation(token, TokenUser, null_mut(), 0, &mut bytes);
    }
    ensure!(bytes > 0, "read current user token size");
    let mut storage = vec![0_usize; (bytes as usize).div_ceil(size_of::<usize>())];
    ensure!(
        unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                storage.as_mut_ptr().cast::<c_void>(),
                bytes,
                &mut bytes,
            )
        } != 0,
        "read current user token"
    );
    let sid = unsafe { (*storage.as_ptr().cast::<TOKEN_USER>()).User.Sid };
    ensure!(!sid.is_null(), "current user token has no SID");
    f(sid)
}

fn inheritance(directory: bool) -> u32 {
    if directory {
        OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
    } else {
        0
    }
}

fn wide_path(path: &Path) -> Result<Vec<u16>> {
    let mut wide: Vec<_> = path.as_os_str().encode_wide().collect();
    ensure!(!wide.contains(&0), "state path contains NUL");
    wide.push(0);
    Ok(wide)
}

struct SecurityDescriptor(PSECURITY_DESCRIPTOR);

impl Drop for SecurityDescriptor {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                LocalFree(self.0);
            }
        }
    }
}

#[cfg(test)]
mod tests;

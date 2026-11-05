//! The ACL of a file that gets replaced. The temporary file inherits its
//! directory's default ACL (Linux) or inheritable entries (macOS), which can
//! give an account access the old file never gave, so the old file's ACL, or
//! none, is set on it instead.

use std::fs;
use std::io;
use std::path::Path;

#[cfg(target_os = "linux")]
const ACCESS: &std::ffi::CStr = c"system.posix_acl_access";

/// A file's POSIX access ACL as the kernel stores it, or `None` without one.
#[cfg(target_os = "linux")]
pub(super) struct Acl(Option<Vec<u8>>);

#[cfg(target_os = "linux")]
impl Acl {
    pub(super) fn of(path: &Path) -> io::Result<Self> {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(path.as_os_str().as_bytes())?;
        loop {
            // SAFETY: both strings are NUL-terminated; a null buffer of size 0
            // asks for the value's size.
            let size =
                unsafe { libc::getxattr(path.as_ptr(), ACCESS.as_ptr(), std::ptr::null_mut(), 0) };
            if size < 0 {
                return Self::none_or(io::Error::last_os_error());
            }
            let mut value = vec![0_u8; size as usize];
            // SAFETY: both strings are NUL-terminated and `value` is valid for
            // its length.
            let read = unsafe {
                libc::getxattr(
                    path.as_ptr(),
                    ACCESS.as_ptr(),
                    value.as_mut_ptr().cast(),
                    value.len(),
                )
            };
            if read >= 0 {
                value.truncate(read as usize);
                return Ok(Self(Some(value)));
            }
            let error = io::Error::last_os_error();
            // The ACL grew between the two calls.
            if error.raw_os_error() != Some(libc::ERANGE) {
                return Self::none_or(error);
            }
        }
    }

    /// No ACL on the file, or none on its filesystem.
    fn none_or(error: io::Error) -> io::Result<Self> {
        match error.raw_os_error() {
            Some(libc::ENODATA | libc::EOPNOTSUPP) => Ok(Self(None)),
            _ => Err(error),
        }
    }

    pub(super) fn apply(&self, file: &fs::File) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        let fd = file.as_raw_fd();
        match &self.0 {
            Some(value) => {
                // SAFETY: the name is NUL-terminated and `value` is valid for
                // its length.
                let result = unsafe {
                    libc::fsetxattr(fd, ACCESS.as_ptr(), value.as_ptr().cast(), value.len(), 0)
                };
                if result != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            None => {
                // SAFETY: the name is NUL-terminated.
                if unsafe { libc::fremovexattr(fd, ACCESS.as_ptr()) } != 0 {
                    Self::none_or(io::Error::last_os_error())?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod ffi {
    use std::ffi::{c_char, c_int, c_void};

    pub(super) const ACL_TYPE_EXTENDED: c_int = 0x0000_0100;

    unsafe extern "C" {
        pub(super) fn acl_get_file(path: *const c_char, kind: c_int) -> *mut c_void;
        pub(super) fn acl_set_fd_np(fd: c_int, acl: *mut c_void, kind: c_int) -> c_int;
        pub(super) fn acl_init(count: c_int) -> *mut c_void;
        pub(super) fn acl_free(object: *mut c_void) -> c_int;
    }
}

/// A file's extended ACL; an empty one when the file has none.
#[cfg(target_os = "macos")]
pub(super) struct Acl {
    handle: *mut std::ffi::c_void,
    empty: bool,
}

#[cfg(target_os = "macos")]
impl Acl {
    pub(super) fn of(path: &Path) -> io::Result<Self> {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(path.as_os_str().as_bytes())?;
        // SAFETY: the path is NUL-terminated.
        let handle = unsafe { ffi::acl_get_file(path.as_ptr(), ffi::ACL_TYPE_EXTENDED) };
        if !handle.is_null() {
            return Ok(Self {
                handle,
                empty: false,
            });
        }
        let error = io::Error::last_os_error();
        // ENOENT: the file exists (its mode was just read) but has no ACL.
        if !matches!(
            error.raw_os_error(),
            Some(libc::ENOENT | libc::ENOTSUP | libc::EOPNOTSUPP)
        ) {
            return Err(error);
        }
        // SAFETY: acl_init has no preconditions.
        let handle = unsafe { ffi::acl_init(0) };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            handle,
            empty: true,
        })
    }

    pub(super) fn apply(&self, file: &fs::File) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        // SAFETY: `handle` is a live ACL from acl_get_file or acl_init.
        let result =
            unsafe { ffi::acl_set_fd_np(file.as_raw_fd(), self.handle, ffi::ACL_TYPE_EXTENDED) };
        if result == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if self.empty && matches!(error.raw_os_error(), Some(libc::ENOTSUP | libc::EOPNOTSUPP)) {
            return Ok(());
        }
        Err(error)
    }
}

#[cfg(target_os = "macos")]
impl Drop for Acl {
    fn drop(&mut self) {
        // SAFETY: `handle` came from acl_get_file or acl_init and is freed once.
        unsafe { ffi::acl_free(self.handle) };
    }
}

/// Other Unix systems: the mode alone.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) struct Acl;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
impl Acl {
    pub(super) fn of(_path: &Path) -> io::Result<Self> {
        Ok(Self)
    }

    pub(super) fn apply(&self, _file: &fs::File) -> io::Result<()> {
        Ok(())
    }
}

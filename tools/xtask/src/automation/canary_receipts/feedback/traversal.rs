use super::super::super::Error;
use super::super::contract_error;
use std::{
    ffi::{CStr, CString, OsString},
    fs::{File, OpenOptions},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::{OsStrExt, OsStringExt},
            fs::OpenOptionsExt,
        },
    },
    path::Path,
};

pub(super) fn open_root(path: &Path) -> Result<File, Error> {
    // Open each component from an owned descriptor; a replaced ancestor cannot
    // redirect the snapshot through a symlink between inspection and reading.
    use std::os::unix::fs::MetadataExt;
    use std::path::Component;
    let expected = std::fs::symlink_metadata(path)?;
    if !expected.is_dir() {
        return Err(contract_error("feedback root must be a real directory"));
    }
    // Normal machine ancestors such as macOS /var may be symlinks. Resolve
    // those once, then pin the actual root inode during descriptor admission.
    let absolute = path.canonicalize()?;
    let mut directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open("/")?;
    for component in absolute.components() {
        match component {
            Component::RootDir => (),
            Component::Normal(name) => {
                let next = open_child(&directory, name)?;
                if !next.metadata()?.is_dir() {
                    return Err(contract_error("feedback root ancestor must be a directory"));
                }
                directory = next;
            }
            _ => {
                return Err(contract_error(
                    "feedback root must not contain dot components",
                ));
            }
        }
    }
    let observed = directory.metadata()?;
    if (observed.dev(), observed.ino()) != (expected.dev(), expected.ino()) {
        return Err(contract_error("feedback root changed during admission"));
    }
    Ok(directory)
}

pub(super) fn open_child(parent: &File, name: &std::ffi::OsStr) -> Result<File, Error> {
    let name =
        CString::new(name.as_bytes()).map_err(|_| contract_error("NUL feedback member name"))?;
    // SAFETY: parent is an owned directory descriptor and name is one terminated
    // child component. NOFOLLOW prevents links; NONBLOCK prevents FIFO stalls.
    let descriptor = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: openat returned a new descriptor whose ownership transfers to File.
    Ok(unsafe { File::from_raw_fd(descriptor) })
}

struct Directory(*mut libc::DIR);
impl Drop for Directory {
    fn drop(&mut self) {
        // SAFETY: fdopendir produced the unique live handle owned by this value.
        unsafe {
            libc::closedir(self.0);
        }
    }
}

pub(super) fn names(directory: &File) -> Result<Vec<OsString>, Error> {
    // Open "." instead of dup: duplicated descriptors share a directory offset.
    let descriptor = open_child(directory, std::ffi::OsStr::new("."))?;
    use std::os::fd::IntoRawFd;
    let descriptor = descriptor.into_raw_fd();
    // SAFETY: the new directory descriptor transfers to fdopendir on success.
    let pointer = unsafe { libc::fdopendir(descriptor) };
    if pointer.is_null() {
        let error = std::io::Error::last_os_error();
        // SAFETY: failed fdopendir leaves the transferred descriptor unowned.
        unsafe {
            libc::close(descriptor);
        }
        return Err(error.into());
    }
    let directory = Directory(pointer);
    let mut names = Vec::new();
    loop {
        // SAFETY: errno belongs to this thread; readdir's entry lives until the
        // next call. Copying its terminated name preserves no borrowed pointer.
        let name = unsafe {
            *errno() = 0;
            let entry = libc::readdir(directory.0);
            if entry.is_null() {
                if *errno() != 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
                break;
            }
            CStr::from_ptr((*entry).d_name.as_ptr()).to_bytes().to_vec()
        };
        if name != b"." && name != b".." {
            names.push(OsString::from_vec(name));
            if names.len() > super::MAXIMUM_FILES {
                return Err(contract_error("feedback directory entry count exceeded"));
            }
        }
    }
    names.sort();
    Ok(names)
}

unsafe fn errno() -> *mut libc::c_int {
    #[cfg(target_os = "macos")]
    {
        unsafe { libc::__error() }
    }
    #[cfg(target_os = "linux")]
    {
        unsafe { libc::__errno_location() }
    }
}

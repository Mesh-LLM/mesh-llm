use super::super::process;
use crate::command::DynResult;
use std::{
    fs::File,
    path::Path,
    time::{Duration, Instant},
};

const ROOT: &str = "/Library/Application Support/MeshLLM/locks";
const NAME: &str = "mesh-canary-family-host.lock";

pub(super) fn acquire(deadline: Instant) -> DynResult<(File, bool, Duration)> {
    let started = Instant::now();
    let file = open(Path::new(ROOT), 0)?;
    let mut contended = false;
    loop {
        process::check()?;
        if Instant::now() >= deadline {
            return Err("family host lock wait exhausted certification budget".into());
        }
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: descriptor is an open validated regular lock file.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                return Ok((file, contended, started.elapsed()));
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::WouldBlock {
                return Err(error.into());
            }
        }
        #[cfg(not(unix))]
        {
            return Err("family host lock requires Unix".into());
        }
        contended = true;
        std::thread::sleep(Duration::from_millis(100));
    }
}
#[cfg(unix)]
pub(super) fn open(root: &Path, uid: u32) -> DynResult<File> {
    use std::{
        ffi::CString,
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::{ffi::OsStrExt, fs::MetadataExt},
        },
    };
    let path = CString::new(root.as_os_str().as_bytes())?;
    // SAFETY: terminated path; new fd is transferred immediately to File ownership.
    let fd = unsafe {
        libc::open(
            path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: successful open returned a new uniquely owned descriptor.
    let directory = unsafe { File::from_raw_fd(fd) };
    let metadata = directory.metadata()?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o022 != 0 {
        return Err("family lock directory owner/mode invalid".into());
    }
    let name = CString::new(NAME)?;
    // SAFETY: held directory descriptor and terminated leaf; do not create/replace the stable lock.
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDWR | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: successful openat returned a new uniquely owned descriptor.
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.uid() != uid
        || metadata.mode() & 0o666 != 0o666
    {
        return Err("family host lock owner/type/link/mode invalid".into());
    }
    Ok(file)
}
#[cfg(not(unix))]
fn open(_: &Path, _: u32) -> DynResult<File> {
    Err("family host lock requires Unix".into())
}

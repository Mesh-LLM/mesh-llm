//! macOS 13.3 has atomic O_CLOEXEC open but no pipe2. Unlinked private FIFOs
//! avoid pipe()+fcntl()'s inheritance window even across unrelated spawn calls.

use std::ffi::{CStr, CString};
use std::fs;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdout, Command, Stdio};

pub(super) fn spawn(command: &mut Command) -> io::Result<Child> {
    let directory = Directory::new()?;
    let (stdout, stdout_writer) = pipe(&directory.0.join("stdout"))?;
    let (stderr, stderr_writer) = pipe(&directory.0.join("stderr"))?;
    directory.close()?;
    command
        .stdout(Stdio::from(stdout_writer))
        .stderr(Stdio::from(stderr_writer));
    let child = command.spawn();
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = child?;
    child.stdout = Some(ChildStdout::from(stdout));
    child.stderr = Some(ChildStderr::from(stderr));
    Ok(child)
}

#[cfg(test)]
pub(super) fn test_pipe() -> io::Result<(OwnedFd, OwnedFd)> {
    let directory = Directory::new()?;
    let pipe = pipe(&directory.0.join("capture"))?;
    directory.close()?;
    Ok(pipe)
}

fn pipe(path: &Path) -> io::Result<(OwnedFd, OwnedFd)> {
    let name = CString::new(path.as_os_str().as_bytes())?;
    // SAFETY: name is NUL-terminated; mkfifo creates only this private pathname.
    if unsafe { libc::mkfifo(name.as_ptr(), 0o600) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let reader = open(&name, libc::O_RDONLY | libc::O_NONBLOCK)?;
    let writer = open(&name, libc::O_WRONLY | libc::O_NONBLOCK)?;
    // SAFETY: writer owns a live descriptor; F_GETFL takes no pointer argument.
    let flags = unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: integer flags restore blocking child writes without changing FD_CLOEXEC.
    if unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_SETFL, flags & !libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    fs::remove_file(path)?;
    Ok((reader, writer))
}

fn open(name: &CStr, flags: libc::c_int) -> io::Result<OwnedFd> {
    // SAFETY: name is a valid C string; O_CLOEXEC is set atomically by open.
    let fd = unsafe { libc::open(name.as_ptr(), flags | libc::O_CLOEXEC | libc::O_NOFOLLOW) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: open returned a fresh owned descriptor, transferred exactly once.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

struct Directory(PathBuf);

impl Directory {
    fn new() -> io::Result<Self> {
        let path = std::env::temp_dir().join("mesh-process-capture.XXXXXX");
        let mut name = CString::new(path.as_os_str().as_bytes())?.into_bytes_with_nul();
        // SAFETY: the writable NUL-terminated template ends in six X bytes.
        if unsafe { libc::mkdtemp(name.as_mut_ptr().cast()) }.is_null() {
            return Err(io::Error::last_os_error());
        }
        let path = CStr::from_bytes_until_nul(&name).map_err(io::Error::other)?;
        Ok(Self(
            std::ffi::OsString::from_vec(path.to_bytes().to_vec()).into(),
        ))
    }

    fn close(mut self) -> io::Result<()> {
        fs::remove_dir(&self.0)?;
        self.0.clear();
        Ok(())
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        if !self.0.as_os_str().is_empty()
            && let Err(error) = fs::remove_dir_all(&self.0)
        {
            eprintln!("private capture directory cleanup failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_process_macos_capture_opens_cloexec_and_unlinks_before_spawn() {
        let directory = Directory::new().unwrap();
        let path = directory.0.join("capture");
        let (reader, writer) = pipe(&path).unwrap();
        assert!(!path.exists());
        for fd in [&reader, &writer] {
            // SAFETY: F_GETFD reads flags of a live borrowed descriptor.
            let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) };
            assert_ne!(flags, -1);
            assert_ne!(flags & libc::FD_CLOEXEC, 0);
        }
        directory.close().unwrap();
    }
}

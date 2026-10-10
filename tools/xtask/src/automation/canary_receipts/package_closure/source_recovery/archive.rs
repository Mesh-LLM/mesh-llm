//! Diagnostic links are archived literally; no source traversal follows a link.
use super::git::Git;
use crate::command::DynResult;
use flate2::{Compression, write::GzEncoder};
use std::{
    ffi::{CString, OsStr},
    fs::{File, OpenOptions},
    io::{self, Read},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt},
        },
    },
    path::{Component, Path},
};

const MAX_FILES: usize = 65536;
const MAX_BYTES: u64 = 1024 * 1024 * 1024;

pub(super) struct Root {
    file: File,
}
impl Root {
    pub(super) fn open(path: &Path) -> DynResult<Self> {
        if !path.is_absolute() {
            return Err("recovery directory must be absolute".into());
        }
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open("/")?;
        for component in path.components() {
            match component {
                Component::RootDir => (),
                Component::Normal(name) => {
                    file = open_at(
                        &file,
                        &CString::new(name.as_bytes())?,
                        libc::O_RDONLY | libc::O_DIRECTORY,
                    )?
                }
                _ => return Err("unsafe recovery directory component".into()),
            }
        }
        Ok(Self { file })
    }
    pub(super) fn unchanged(&self, path: &Path) -> DynResult<()> {
        let current = Self::open(path)?;
        let before = self.file.metadata()?;
        let after = current.file.metadata()?;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return Err("recovery directory custody changed".into());
        }
        Ok(())
    }
    fn parent(&self, relative: &Path) -> DynResult<(File, CString)> {
        let mut components = relative.components().peekable();
        let mut directory = self.file.try_clone()?;
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                return Err("unsafe recovery member path".into());
            };
            let name = CString::new(name.as_bytes())?;
            if components.peek().is_none() {
                return Ok((directory, name));
            }
            directory = open_at(&directory, &name, libc::O_RDONLY | libc::O_DIRECTORY)?;
        }
        Err("empty recovery member path".into())
    }
    pub(super) fn file(&self, relative: &Path) -> DynResult<File> {
        let (parent, name) = self.parent(relative)?;
        let file = open_at(&parent, &name, libc::O_RDONLY | libc::O_NONBLOCK)?;
        if !file.metadata()?.is_file() {
            return Err("recovery member is not regular".into());
        }
        Ok(file)
    }
    pub(super) fn create_file(&self, relative: &Path) -> DynResult<File> {
        let (parent, name) = self.parent(relative)?;
        Ok(open_at(
            &parent,
            &name,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?)
    }
    pub(super) fn publish(&self, source: &OsStr, destination: &OsStr) -> DynResult<()> {
        let source = CString::new(source.as_bytes())?;
        let destination = CString::new(destination.as_bytes())?;
        #[cfg(target_os = "linux")]
        let status = unsafe {
            libc::renameat2(
                self.file.as_raw_fd(),
                source.as_ptr(),
                self.file.as_raw_fd(),
                destination.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        #[cfg(target_os = "macos")]
        let status = {
            unsafe extern "C" {
                fn renameatx_np(
                    fromfd: libc::c_int,
                    from: *const libc::c_char,
                    tofd: libc::c_int,
                    to: *const libc::c_char,
                    flags: libc::c_uint,
                ) -> libc::c_int;
            }
            unsafe {
                renameatx_np(
                    self.file.as_raw_fd(),
                    source.as_ptr(),
                    self.file.as_raw_fd(),
                    destination.as_ptr(),
                    0x4,
                )
            }
        };
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = (source, destination);
            return Err("atomic diagnostic recovery publication requires macOS or Linux".into());
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        if status != 0 {
            return Err(io::Error::last_os_error().into());
        }
        Ok(())
    }
    pub(super) fn nested(&self, relative: &Path) -> DynResult<Option<Self>> {
        let (parent, name) = match self.parent(relative) {
            Ok(value) => value,
            Err(error)
                if error
                    .downcast_ref::<io::Error>()
                    .is_some_and(|e| e.kind() == io::ErrorKind::NotFound) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        match open_at(&parent, &name, libc::O_RDONLY | libc::O_DIRECTORY) {
            Ok(file) => Ok(Some(Self { file })),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
}
fn open_at(parent: &File, name: &CString, flags: i32) -> io::Result<File> {
    // Owned parent descriptor and a single component exclude ancestor/leaf links.
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}
fn stat(parent: &File, name: &CString) -> io::Result<libc::stat> {
    let mut result = std::mem::MaybeUninit::uninit();
    if unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            name.as_ptr(),
            result.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { result.assume_init() })
}
fn link(parent: &File, name: &CString) -> DynResult<Vec<u8>> {
    let mut bytes = vec![0; 65536];
    let count = unsafe {
        libc::readlinkat(
            parent.as_raw_fd(),
            name.as_ptr(),
            bytes.as_mut_ptr().cast(),
            bytes.len(),
        )
    };
    if count < 0 {
        return Err(io::Error::last_os_error().into());
    }
    let count = usize::try_from(count)?;
    if count == bytes.len() {
        return Err("recovery symlink target exceeds 64 KiB".into());
    }
    bytes.truncate(count);
    Ok(bytes)
}
struct BoundRead<'a> {
    file: &'a mut File,
    git: &'a Git,
    remaining: &'a mut u64,
}
impl Read for BoundRead<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.git
            .remaining()
            .map_err(|e| io::Error::other(e.to_string()))?;
        let count = self.file.read(bytes)?;
        *self.remaining = self
            .remaining
            .checked_sub(count as u64)
            .ok_or_else(|| io::Error::other("recovery untracked bytes exceed 1 GiB"))?;
        Ok(count)
    }
}
pub(super) fn write(root: &Root, names: &[&[u8]], output: File, git: &Git) -> DynResult<usize> {
    if names.len() > MAX_FILES {
        return Err("recovery untracked members exceed 65536".into());
    }
    let mut builder = tar::Builder::new(GzEncoder::new(output, Compression::default()));
    let mut remaining = MAX_BYTES;
    let mut captured = 0;
    for name in names {
        git.remaining()?;
        let path = Path::new(OsStr::from_bytes(name));
        let (parent, leaf) = root.parent(path)?;
        let before = match stat(&parent, &leaf) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err("untracked source disappeared during recovery".into());
            }
            Err(error) => return Err(error.into()),
        };
        let mut header = tar::Header::new_gnu();
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        if before.st_mode & libc::S_IFMT == libc::S_IFLNK {
            let target = link(&parent, &leaf)?;
            remaining = remaining
                .checked_sub(target.len() as u64)
                .ok_or("recovery untracked bytes exceed 1 GiB")?;
            header.set_entry_type(tar::EntryType::Symlink);
            header.set_size(0);
            header.set_mode(0o777);
            builder.append_link(&mut header, path, Path::new(OsStr::from_bytes(&target)))?;
        } else if before.st_mode & libc::S_IFMT == libc::S_IFREG {
            let mut source = open_at(&parent, &leaf, libc::O_RDONLY | libc::O_NONBLOCK)?;
            let metadata = source.metadata()?;
            if metadata.ino() != before.st_ino
                || metadata.dev() != device_number(before.st_dev)
                || metadata.len() > remaining
            {
                return Err("untracked source changed or exceeds recovery budget".into());
            }
            header.set_size(metadata.len());
            header.set_mode(metadata.mode() & 0o777);
            builder.append_data(
                &mut header,
                path,
                BoundRead {
                    file: &mut source,
                    git,
                    remaining: &mut remaining,
                },
            )?;
            let after = source.metadata()?;
            if after.len() != metadata.len()
                || after.mtime() != metadata.mtime()
                || after.mtime_nsec() != metadata.mtime_nsec()
            {
                return Err("untracked source changed while capturing".into());
            }
        } else {
            return Err("recovery refuses nonregular untracked source".into());
        }
        let after = stat(&parent, &leaf)?;
        if before.st_ino != after.st_ino
            || before.st_dev != after.st_dev
            || before.st_size != after.st_size
            || before.st_mtime != after.st_mtime
            || before.st_mtime_nsec != after.st_mtime_nsec
            || before.st_ctime != after.st_ctime
            || before.st_ctime_nsec != after.st_ctime_nsec
        {
            return Err("untracked recovery source was replaced".into());
        }
        captured += 1;
    }
    builder.into_inner()?.finish()?.sync_all()?;
    Ok(captured)
}

#[cfg(target_os = "linux")]
fn device_number(value: libc::dev_t) -> u64 {
    value
}

#[cfg(not(target_os = "linux"))]
fn device_number(value: libc::dev_t) -> u64 {
    value as u64
}

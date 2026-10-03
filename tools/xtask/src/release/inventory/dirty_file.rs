//! Stream admitted working-tree regular files; hash symlink targets without following them.
use super::{
    observation::Observation,
    provenance::{Error, Result},
};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::Read,
    path::{Component, Path},
};
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Content {
    pub(crate) kind: &'static str,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}
fn io() -> Error {
    Error("release inventory source file unavailable or changed".into())
}
pub(crate) fn relative(path: &str) -> Result<&Path> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(Error(
            "Git untracked path must be a contained relative pathname".into(),
        ));
    }
    Ok(path)
}
pub(crate) fn content(root: &Path, path: &str, observation: &Observation) -> Result<Content> {
    content_checked(root, path, &mut || observation.check())
}
pub(crate) fn content_checked(
    root: &Path,
    path: &str,
    check: &mut impl FnMut() -> Result<()>,
) -> Result<Content> {
    check()?;
    let entry = platform::Entry::open(root, relative(path)?)?;
    let before = entry.identity()?;
    let (kind, bytes, hash) = if before.kind == platform::Kind::Symlink {
        check()?;
        let target = entry.target()?;
        check()?;
        (
            "symlink",
            target.len() as u64,
            Sha256::digest(&target).to_vec(),
        )
    } else if before.kind == platform::Kind::Regular {
        let mut file = entry.file()?;
        if platform::file_identity(&file)? != before {
            return Err(io());
        }
        let mut hash = Sha256::new();
        let mut total = 0u64;
        let mut chunk = vec![0u8; 1024 * 1024];
        loop {
            check()?;
            let count = file.read(&mut chunk).map_err(|_| io())?;
            if count == 0 {
                break;
            }
            total = total.checked_add(count as u64).ok_or_else(io)?;
            if total > before.bytes {
                return Err(Error("untracked source grew during bounded hashing".into()));
            }
            hash.update(&chunk[..count]);
        }
        check()?;
        if total != before.bytes || platform::file_identity(&file)? != before {
            return Err(io());
        }
        ("regular", total, hash.finalize().to_vec())
    } else {
        return Err(Error(
            "untracked evidence must be a regular file or symlink; special files refuse".into(),
        ));
    };
    check()?;
    if entry.identity()? != before {
        return Err(io());
    }
    Ok(Content {
        kind,
        bytes,
        sha256: hex::encode(hash),
    })
}
/// Bounded previous report bytes use the same regular-file/path identity admission.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ReportIdentity(platform::Identity);
impl ReportIdentity {
    /// Atomic replacement ownership, separate from mutable timestamps/content.
    pub(crate) fn same_file(&self, other: &Self) -> bool {
        platform::same_file(&self.0, &other.0)
    }
}
/// Identity of the exact owned file returned by atomic publication.
pub(crate) fn report_identity(file: &File) -> Result<ReportIdentity> {
    platform::file_identity(file).map(ReportIdentity)
}
pub(crate) fn report_state(
    root: &Path,
    path: &str,
    observation: &Observation,
    maximum: u64,
) -> Result<(Vec<u8>, ReportIdentity)> {
    observation.check()?;
    let entry = platform::Entry::open(root, relative(path)?)?;
    let before = entry.identity()?;
    if before.kind != platform::Kind::Regular || before.bytes > maximum {
        return Err(Error(
            "previous report must be a bounded regular file".into(),
        ));
    }
    let mut file = entry.file()?;
    if platform::file_identity(&file)? != before {
        return Err(io());
    }
    let mut bytes = Vec::new();
    let mut chunk = vec![0u8; 65536];
    loop {
        observation.check()?;
        let count = file.read(&mut chunk).map_err(|_| io())?;
        if count == 0 {
            break;
        }
        if count as u64 > before.bytes.saturating_sub(bytes.len() as u64) {
            return Err(io());
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    observation.check()?;
    if bytes.len() as u64 != before.bytes
        || platform::file_identity(&file)? != before
        || entry.identity()? != before
    {
        return Err(io());
    }
    Ok((bytes, ReportIdentity(before)))
}
#[cfg(unix)]
mod platform {
    use super::*;
    use std::{
        ffi::CString,
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::{ffi::OsStrExt, fs::OpenOptionsExt},
        },
    };
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum Kind {
        Regular,
        Symlink,
        Other,
    }
    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct Identity {
        pub kind: Kind,
        pub bytes: u64,
        dev: u64,
        ino: u64,
        mtime: (i64, i64),
        ctime: (i64, i64),
    }
    pub(super) struct Entry {
        parent: File,
        name: CString,
    }
    fn descriptor(parent: i32, name: &CString, flags: i32) -> Result<File> {
        // SAFETY: name is terminated, descriptor is borrowed, successful returned descriptor is owned once.
        let fd = unsafe {
            libc::openat(
                parent,
                name.as_ptr(),
                flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(io());
        }
        // SAFETY: openat returned a new owned descriptor.
        Ok(unsafe { File::from_raw_fd(fd) })
    }
    impl Entry {
        pub(super) fn open(root: &Path, relative: &Path) -> Result<Self> {
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC);
            let mut parent = options.open(root).map_err(|_| io())?;
            let mut components = relative.components().peekable();
            while let Some(component) = components.next() {
                let name = CString::new(component.as_os_str().as_bytes()).map_err(|_| io())?;
                if components.peek().is_none() {
                    return Ok(Self { parent, name });
                }
                parent = descriptor(
                    parent.as_raw_fd(),
                    &name,
                    libc::O_RDONLY | libc::O_DIRECTORY,
                )?;
            }
            Err(io())
        }
        pub(super) fn identity(&self) -> Result<Identity> {
            let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
            // SAFETY: parent is owned, name terminated and stat has sufficient writable storage.
            if unsafe {
                libc::fstatat(
                    self.parent.as_raw_fd(),
                    self.name.as_ptr(),
                    stat.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            } < 0
            {
                return Err(io());
            }
            // SAFETY: successful fstatat initializes the stat object.
            Ok(identity(&unsafe { stat.assume_init() }))
        }
        pub(super) fn file(&self) -> Result<File> {
            descriptor(
                self.parent.as_raw_fd(),
                &self.name,
                libc::O_RDONLY | libc::O_NONBLOCK,
            )
        }
        pub(super) fn target(&self) -> Result<Vec<u8>> {
            let mut bytes = vec![0u8; 65536];
            // SAFETY: name/parent are valid; readlinkat writes at most bytes.len() bytes, without termination.
            let count = unsafe {
                libc::readlinkat(
                    self.parent.as_raw_fd(),
                    self.name.as_ptr(),
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                )
            };
            if count < 0 || count as usize == bytes.len() {
                return Err(io());
            }
            bytes.truncate(count as usize);
            Ok(bytes)
        }
    }
    #[allow(clippy::unnecessary_cast)] // libc scalar widths vary between supported Unix hosts.
    fn identity(stat: &libc::stat) -> Identity {
        let kind = match stat.st_mode & libc::S_IFMT {
            libc::S_IFREG => Kind::Regular,
            libc::S_IFLNK => Kind::Symlink,
            _ => Kind::Other,
        };
        let (mtime, ctime) = (
            (stat.st_mtime as i64, stat.st_mtime_nsec as i64),
            (stat.st_ctime as i64, stat.st_ctime_nsec as i64),
        );
        Identity {
            kind,
            bytes: stat.st_size as u64,
            dev: stat.st_dev as u64,
            ino: stat.st_ino as u64,
            mtime,
            ctime,
        }
    }
    pub(super) fn same_file(a: &Identity, b: &Identity) -> bool {
        a.dev == b.dev && a.ino == b.ino
    }
    pub(super) fn file_identity(file: &File) -> Result<Identity> {
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: file owns a live descriptor and stat storage is writable.
        if unsafe { libc::fstat(file.as_raw_fd(), stat.as_mut_ptr()) } < 0 {
            return Err(io());
        }
        // SAFETY: successful fstat initialized stat.
        Ok(identity(&unsafe { stat.assume_init() }))
    }
}
#[cfg(windows)]
mod platform {
    use super::*;
    use std::fs;
    use std::{
        ffi::OsString,
        os::windows::{
            ffi::OsStringExt,
            fs::{MetadataExt, OpenOptionsExt},
            io::AsRawHandle,
        },
        path::PathBuf,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS,
        FILE_FLAG_OPEN_REPARSE_POINT, GetFileInformationByHandle, GetFinalPathNameByHandleW,
    };
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum Kind {
        Regular,
        Symlink,
        Other,
    }
    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct Identity {
        pub kind: Kind,
        pub bytes: u64,
        volume: u32,
        index: (u32, u32),
        modified: u64,
    }
    pub(super) struct Entry {
        root: PathBuf,
        path: PathBuf,
        opened: File,
    }
    impl Entry {
        pub(super) fn open(root: &Path, relative: &Path) -> Result<Self> {
            let root = root.canonicalize().map_err(|_| io())?;
            let path = root.join(relative);
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS);
            let opened = options.open(&path).map_err(|_| io())?;
            if !final_path(&opened)?.starts_with(&root) {
                return Err(io());
            }
            Ok(Self { root, path, opened })
        }
        pub(super) fn identity(&self) -> Result<Identity> {
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS);
            let current = options.open(&self.path).map_err(|_| io())?;
            if !final_path(&current)?.starts_with(&self.root) {
                return Err(io());
            }
            file_identity(&current)
        }
        pub(super) fn file(&self) -> Result<File> {
            self.opened.try_clone().map_err(|_| io())
        }
        pub(super) fn target(&self) -> Result<Vec<u8>> {
            let target = fs::read_link(&self.path).map_err(|_| io())?;
            Ok(target.as_os_str().as_encoded_bytes().to_vec())
        }
    }
    fn final_path(file: &File) -> Result<PathBuf> {
        let mut buffer = vec![0u16; 32768];
        // SAFETY: handle is live and buffer correctly sized; zero flags request the normalized DOS volume path.
        let count = unsafe {
            GetFinalPathNameByHandleW(
                file.as_raw_handle().cast(),
                buffer.as_mut_ptr(),
                buffer.len() as u32,
                0,
            )
        };
        if count == 0 || count as usize >= buffer.len() {
            return Err(io());
        }
        buffer.truncate(count as usize);
        Ok(PathBuf::from(OsString::from_wide(&buffer)))
    }
    pub(super) fn same_file(a: &Identity, b: &Identity) -> bool {
        a.volume == b.volume && a.index == b.index
    }
    pub(super) fn file_identity(file: &File) -> Result<Identity> {
        let metadata = file.metadata().map_err(|_| io())?;
        let mut information = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::uninit();
        // SAFETY: handle is borrowed and information points to correctly sized storage.
        if unsafe {
            GetFileInformationByHandle(file.as_raw_handle().cast(), information.as_mut_ptr())
        } == 0
        {
            return Err(io());
        }
        // SAFETY: successful call initialized the structure.
        let information = unsafe { information.assume_init() };
        let kind = if metadata.file_type().is_symlink() {
            Kind::Symlink
        } else if metadata.is_file()
            && metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0
        {
            Kind::Regular
        } else {
            Kind::Other
        };
        Ok(Identity {
            kind,
            bytes: metadata.len(),
            volume: information.dwVolumeSerialNumber,
            index: (information.nFileIndexHigh, information.nFileIndexLow),
            modified: metadata.last_write_time(),
        })
    }
}

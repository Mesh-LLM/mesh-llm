//! Descriptor-anchored replacement of one admitted SWE-ReX module source.
use anyhow::{Result, bail};
use std::{
    ffi::{CString, OsStr},
    fs::{File, Metadata, Permissions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, PermissionsExt},
        },
    },
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAX_SOURCE_BYTES: u64 = 1024 * 1024;
static STAGE_ID: AtomicU64 = AtomicU64::new(0);

pub(super) struct AdmittedSource {
    parent: File,
    leaf: CString,
    metadata: Metadata,
    path: PathBuf,
    root: PathBuf,
    pub(super) bytes: Vec<u8>,
}

fn open_at(parent: &File, name: &OsStr, flags: i32, mode: libc::mode_t) -> Result<File> {
    let name = CString::new(name.as_bytes())?;
    // mode_t is u16 on macOS; C variadic arguments require integer promotion.
    #[allow(clippy::useless_conversion)]
    let promoted_mode = libc::c_uint::from(mode);
    // SAFETY: the live parent descriptor and NUL-terminated leaf are valid for this call.
    let fd = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags, promoted_mode) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: openat returned a newly owned descriptor.
    Ok(unsafe { File::from_raw_fd(fd) })
}

fn source_parent(path: &Path, root: &Path) -> Result<File> {
    let clean = |value: &Path| {
        value.is_absolute()
            && value
                .components()
                .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
    };
    if !clean(path)
        || !clean(root)
        || root == Path::new("/")
        || path == root
        || !path.starts_with(root)
    {
        bail!("SWE-ReX module source must be inside the explicit absolute environment root");
    }
    let mut directory = File::open("/")?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("missing module parent"))?;
    for part in parent.components() {
        if let Component::Normal(name) = part {
            directory = open_at(
                &directory,
                name,
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW,
                0,
            )?;
        }
    }
    Ok(directory)
}

fn read_source(parent: &File, leaf: &CString) -> Result<(Metadata, Vec<u8>)> {
    let file = open_at(
        parent,
        OsStr::from_bytes(leaf.as_bytes()),
        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        0,
    )?;
    let before = file.metadata()?;
    if !before.is_file() || before.len() > MAX_SOURCE_BYTES {
        bail!("SWE-ReX module source must be a bounded regular file");
    }
    let mut bytes = Vec::new();
    (&file).take(MAX_SOURCE_BYTES + 1).read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    if bytes.len() as u64 != before.len() || !same_metadata(&before, &after) {
        bail!("SWE-ReX module source changed during read");
    }
    Ok((before, bytes))
}

fn same_metadata(first: &Metadata, second: &Metadata) -> bool {
    first.dev() == second.dev()
        && first.ino() == second.ino()
        && first.len() == second.len()
        && first.mode() == second.mode()
        && first.mtime() == second.mtime()
        && first.mtime_nsec() == second.mtime_nsec()
        && first.ctime() == second.ctime()
        && first.ctime_nsec() == second.ctime_nsec()
}

impl AdmittedSource {
    pub(super) fn open(path: &Path, root: &Path) -> Result<Self> {
        let parent = source_parent(path, root)?;
        let leaf = CString::new(
            path.file_name()
                .ok_or_else(|| anyhow::anyhow!("missing module filename"))?
                .as_bytes(),
        )?;
        let (metadata, bytes) = read_source(&parent, &leaf)?;
        Ok(Self {
            parent,
            leaf,
            metadata,
            path: path.to_owned(),
            root: root.to_owned(),
            bytes,
        })
    }

    fn backup_leaf(&self) -> Result<CString> {
        let mut name = self.leaf.as_bytes().to_vec();
        name.extend_from_slice(b".bak");
        Ok(CString::new(name)?)
    }

    pub(super) fn backup_bytes(&self) -> Result<Option<Vec<u8>>> {
        match read_source(&self.parent, &self.backup_leaf()?) {
            Ok((_, bytes)) => Ok(Some(bytes)),
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn require_named_parent(&self) -> Result<()> {
        let named = source_parent(&self.path, &self.root)?.metadata()?;
        let anchored = self.parent.metadata()?;
        if named.dev() != anchored.dev() || named.ino() != anchored.ino() {
            bail!("SWE-ReX module parent changed before mutation");
        }
        Ok(())
    }

    /// Preserve the original bytes once, without replacing an existing backup.
    pub(super) fn preserve_backup(&self) -> Result<()> {
        self.require_named_parent()?;
        if let Some(bytes) = self.backup_bytes()? {
            if bytes != self.bytes {
                bail!("SWE-ReX original backup differs from admitted source");
            }
            return Ok(());
        }
        let stage = Stage::new(&self.parent)?;
        let mut output = &stage.file;
        output.write_all(&self.bytes)?;
        output.set_permissions(Permissions::from_mode(self.metadata.mode() & 0o7777))?;
        output.sync_all()?;
        let (metadata, bytes) = read_source(&self.parent, &self.leaf)?;
        if !same_metadata(&self.metadata, &metadata)
            || bytes != self.bytes
            || !stage.owns_leaf()
            || read_source(&self.parent, &stage.name)?.1 != bytes
        {
            bail!("SWE-ReX original changed before backup");
        }
        let backup = self.backup_leaf()?;
        // SAFETY: fixed leaves and live parent descriptors; linkat never replaces a backup.
        let result = unsafe {
            libc::linkat(
                self.parent.as_raw_fd(),
                stage.name.as_ptr(),
                self.parent.as_raw_fd(),
                backup.as_ptr(),
                0,
            )
        };
        if result != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        self.parent.sync_all()?;
        if self.backup_bytes()?.as_deref() != Some(self.bytes.as_slice()) {
            bail!("SWE-ReX original backup changed after publication");
        }
        Ok(())
    }

    pub(super) fn replace(&self, bytes: &[u8]) -> Result<()> {
        if u64::try_from(bytes.len())? > MAX_SOURCE_BYTES {
            bail!("rewritten SWE-ReX module source exceeds the size bound");
        }
        let stage = Stage::new(&self.parent)?;
        let mut output = &stage.file;
        output.write_all(bytes)?;
        output.set_permissions(Permissions::from_mode(self.metadata.mode() & 0o7777))?;
        output.sync_all()?;
        let (current, current_bytes) = read_source(&self.parent, &self.leaf)?;
        if !same_metadata(&self.metadata, &current) || self.bytes != current_bytes {
            bail!("SWE-ReX module source changed before replacement");
        }
        self.require_named_parent()?;
        if !stage.owns_leaf() || read_source(&self.parent, &stage.name)?.1 != bytes {
            bail!("SWE-ReX staged source changed before replacement");
        }
        // SAFETY: names are valid leaves anchored to the same live directory descriptor.
        let result = unsafe {
            libc::renameat(
                self.parent.as_raw_fd(),
                stage.name.as_ptr(),
                self.parent.as_raw_fd(),
                self.leaf.as_ptr(),
            )
        };
        if result != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        self.parent.sync_all()?;
        Ok(())
    }
}

struct Stage<'a> {
    parent: &'a File,
    name: CString,
    file: File,
}

impl<'a> Stage<'a> {
    fn new(parent: &'a File) -> Result<Self> {
        let id = STAGE_ID.fetch_add(1, Ordering::Relaxed);
        let name = CString::new(format!(".swerex-source-{}-{id}.tmp", std::process::id()))?;
        let file = open_at(
            parent,
            OsStr::from_bytes(name.as_bytes()),
            libc::O_WRONLY | libc::O_CLOEXEC | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW,
            0o600,
        )?;
        // The cleanup guard exists only after exclusive creation succeeds.
        Ok(Self { parent, name, file })
    }

    fn owns_leaf(&self) -> bool {
        let Ok(file) = open_at(
            self.parent,
            OsStr::from_bytes(self.name.as_bytes()),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            0,
        ) else {
            return false;
        };
        let (Ok(expected), Ok(current)) = (self.file.metadata(), file.metadata()) else {
            return false;
        };
        current.is_file() && same_metadata(&expected, &current)
    }
}

impl Drop for Stage<'_> {
    fn drop(&mut self) {
        // SAFETY: best-effort removal of our stage leaf under its live directory descriptor.
        if self.owns_leaf() {
            unsafe {
                libc::unlinkat(self.parent.as_raw_fd(), self.name.as_ptr(), 0);
            }
        }
    }
}

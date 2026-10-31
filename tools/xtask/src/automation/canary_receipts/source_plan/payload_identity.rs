//! Local verification receipts for immutable model payloads without blob-path identity.
use crate::automation::canary_receipts::Digest;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    fs::{self, File, Metadata, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

#[derive(Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FileState {
    size: u64,
    modified: (u64, u32),
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed: (i64, i64),
}

impl FileState {
    fn read(metadata: &Metadata) -> DynResult<Self> {
        if !metadata.is_file() {
            return Err("model payload is not a regular file".into());
        }
        let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?;
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            size: metadata.len(),
            modified: (modified.as_secs(), modified.subsec_nanos()),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: u8,
    path: PathBuf,
    digest: Digest,
    state: FileState,
}

pub(super) fn verify(path: &Path, expected: &Digest, size: u64) -> DynResult<()> {
    verify_with(path, expected, size, receipt_directory().as_deref(), hash)
}

fn verify_with(
    path: &Path,
    expected: &Digest,
    size: u64,
    directory: Option<&Path>,
    hash: impl FnOnce(&mut File, u64) -> DynResult<Digest>,
) -> DynResult<()> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    let before = FileState::read(&file.metadata()?)?;
    if before.size != size {
        return Err("model payload differs from pinned size".into());
    }
    let receipt_path = directory
        .and_then(|dir| private_directory(dir).ok())
        .map(|dir| {
            dir.join(format!(
                "{}.json",
                Digest::of_bytes(path.as_os_str().as_encoded_bytes()).as_str()
            ))
        });
    let cached = receipt_path.as_ref().is_some_and(|receipt| {
        read_receipt(receipt).is_some_and(|r| {
            r.schema == 1 && r.path == path && r.digest == *expected && r.state == before
        })
    });
    if !cached && hash(&mut file, size)? != *expected {
        return Err(format!(
            "model payload SHA-256 differs from pinned blob identity: {}",
            path.display()
        )
        .into());
    }
    if FileState::read(&file.metadata()?)? != before
        || FileState::read(&fs::metadata(path)?)? != before
    {
        return Err("model payload changed during integrity verification".into());
    }
    if !cached && let Some(receipt_path) = receipt_path {
        // A receipt is only an accelerator. Read-only local state must not reject valid models.
        let _ = write_receipt(
            &receipt_path,
            &Receipt {
                schema: 1,
                path: path.into(),
                digest: expected.clone(),
                state: before,
            },
        );
    }
    Ok(())
}

fn hash(file: &mut File, expected_size: u64) -> DynResult<Digest> {
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut total = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or("model payload size overflow")?;
        if total > expected_size {
            return Err("model payload grew during integrity verification".into());
        }
        digest.update(&buffer[..count]);
    }
    if total != expected_size {
        return Err("model payload shrank during integrity verification".into());
    }
    Ok(hex::encode(digest.finalize()).try_into()?)
}

fn read_receipt(path: &Path) -> Option<Receipt> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > 8192 {
        return None;
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(8193).read_to_end(&mut bytes).ok()?;
    if bytes.len() > 8192 {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

fn write_receipt(path: &Path, receipt: &Receipt) -> DynResult<()> {
    let mut staged =
        tempfile::NamedTempFile::new_in(path.parent().ok_or("receipt has no parent")?)?;
    serde_json::to_writer(staged.as_file_mut(), receipt)?;
    staged.persist(path)?;
    Ok(())
}

#[cfg(unix)]
fn private_directory(path: &Path) -> DynResult<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)?;
    let metadata = fs::symlink_metadata(path)?;
    // SAFETY: geteuid has no arguments or memory preconditions.
    let uid = unsafe { libc::geteuid() };
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o077 != 0 {
        return Err(
            "model verification receipts require an account-owned private directory".into(),
        );
    }
    let canonical = path.canonicalize()?;
    // A private leaf is insufficient when another account can replace a parent.
    for ancestor in canonical.ancestors() {
        let metadata = fs::symlink_metadata(ancestor)?;
        let trusted_owner = metadata.uid() == uid || metadata.uid() == 0;
        let sticky = metadata.mode() & 0o1000 != 0;
        if !metadata.is_dir() || !trusted_owner || (metadata.mode() & 0o022 != 0 && !sticky) {
            return Err("model verification receipt ancestry is not account-trusted".into());
        }
    }
    Ok(canonical)
}

#[cfg(not(unix))]
fn private_directory(_: &Path) -> DynResult<PathBuf> {
    Err("model verification receipts require Unix file change identity".into())
}

fn receipt_directory() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        let home = PathBuf::from(std::env::var_os("HOME")?);
        if !home.is_absolute() {
            return None;
        }
        let root = if cfg!(target_os = "macos") {
            home.join("Library/Caches")
        } else {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .filter(|p| p.is_absolute())
                .unwrap_or_else(|| home.join(".cache"))
        };
        Some(root.join("mesh-llm/canary-payload-integrity-v1"))
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[cfg(test)]
#[path = "payload_identity_tests.rs"]
mod tests;

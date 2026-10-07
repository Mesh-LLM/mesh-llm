//! Descriptor-owned evidence snapshots. Logs are streamed, never buffered.
use super::super::{Digest, Error, ErrorKind};
use super::contract_error;
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Write},
    path::Path,
};

pub(super) const MAXIMUM_FILES: usize = 16_384;
pub(super) const MAXIMUM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAXIMUM_LOG: u64 = 256 * 1024 * 1024;
#[cfg(any(target_os = "macos", target_os = "linux"))]
const MAXIMUM_DEPTH: usize = 32;

pub(in super::super) struct Snapshot {
    directory: tempfile::TempDir,
    pub(super) files: BTreeMap<String, Digest>,
    pub(super) directories: BTreeSet<String>,
    pub(super) bytes: u64,
}

impl Snapshot {
    pub(in super::super) fn files(&self) -> &BTreeMap<String, Digest> {
        &self.files
    }
    pub(in super::super) fn directories(&self) -> &BTreeSet<String> {
        &self.directories
    }
    pub(in super::super) fn capture(root: &Path) -> Result<Self, Error> {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let source = open_directory(root)?;
            let mut snapshot = Self {
                directory: tempfile::Builder::new()
                    .prefix("canary-feedback-evidence-")
                    .tempdir()?,
                files: BTreeMap::new(),
                directories: BTreeSet::new(),
                bytes: 0,
            };
            snapshot.walk(&source, Path::new(""), 0)?;
            Ok(snapshot)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = root;
            Err(contract_error(
                "safe feedback snapshots require macOS or Linux",
            ))
        }
    }

    pub(in super::super) fn root(&self) -> &Path {
        self.directory.path()
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn walk(&mut self, source: &File, relative: &Path, depth: usize) -> Result<(), Error> {
        if depth > MAXIMUM_DEPTH {
            return Err(limit_error("feedback directory depth exceeded"));
        }
        for name in traversal::names(source)? {
            let path = relative.join(&name);
            let key = safe_relative(&path)?;
            let mut file = traversal::open_child(source, &name)?;
            let metadata = file.metadata()?;
            if metadata.is_dir() {
                if self.directories.len() + self.files.len() >= MAXIMUM_FILES {
                    return Err(limit_error("feedback entry count exceeded"));
                }
                self.directories.insert(key);
                fs::create_dir(self.root().join(&path))?;
                self.walk(&file, &path, depth + 1)?;
            } else if metadata.is_file() {
                if self.files.len() + self.directories.len() >= MAXIMUM_FILES {
                    return Err(limit_error("feedback file count exceeded"));
                }
                let limit = file_limit(&path);
                if metadata.len() > limit {
                    return Err(limit_error("feedback evidence file exceeds limit"));
                }
                let destination = self.root().join(&path);
                let (digest, bytes) = copy_file(&mut file, &destination, limit)?;
                self.bytes = self
                    .bytes
                    .checked_add(bytes)
                    .ok_or_else(|| limit_error("feedback byte count overflow"))?;
                if self.bytes > MAXIMUM_BYTES {
                    return Err(limit_error("feedback total evidence bytes exceeded"));
                }
                self.files.insert(key, digest);
            } else {
                return Err(contract_error(
                    "feedback evidence must contain only regular files and directories",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn read_metadata(&self, path: &Path) -> Result<Vec<u8>, Error> {
        super::super::storage::read_bounded(
            &self.root().join(path),
            super::super::storage::RECEIPT_LIMIT,
        )
    }

    pub(super) fn copy_into(&self, destination: &Path) -> Result<(), Error> {
        let checked = Self::capture(self.root())?;
        if checked.files != self.files || checked.directories != self.directories {
            return Err(contract_error("admitted feedback snapshot changed"));
        }
        for directory in &checked.directories {
            fs::create_dir_all(destination.join(directory))?;
        }
        for (path, digest) in &checked.files {
            let mut source = File::open(checked.root().join(path))?;
            let (observed, _) = copy_file(
                &mut source,
                &destination.join(path),
                file_limit(Path::new(path)),
            )?;
            if &observed != digest {
                return Err(contract_error("feedback snapshot changed during export"));
            }
        }
        Ok(())
    }
}

fn file_limit(path: &Path) -> u64 {
    match path.file_name().and_then(|name| name.to_str()) {
        Some("receipt.json" | "feedback.json" | "memory-admission.json") => {
            super::super::storage::RECEIPT_LIMIT
        }
        Some("results.jsonl") => super::super::storage::RESULTS_LIMIT,
        _ => MAXIMUM_LOG,
    }
}

fn copy_file(source: &mut File, destination: &Path, limit: u64) -> Result<(Digest, u64), Error> {
    let before = source.metadata()?;
    let mut output = File::create_new(destination)?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(
                u64::try_from(count).map_err(|_| limit_error("feedback byte count overflow"))?,
            )
            .ok_or_else(|| limit_error("feedback byte count overflow"))?;
        if bytes > limit {
            return Err(limit_error("feedback evidence file exceeds limit"));
        }
        digest.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
    }
    output.flush()?;
    output.sync_all()?;
    let after = source.metadata()?;
    if bytes != before.len()
        || before.len() != after.len()
        || before.modified()? != after.modified()?
    {
        return Err(contract_error("feedback source changed during capture"));
    }
    let digest = Digest::try_from(hex::encode(digest.finalize())).map_err(contract_error)?;
    Ok((digest, bytes))
}

pub(super) fn safe_relative(path: &Path) -> Result<String, Error> {
    use std::path::Component;
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(contract_error("unsafe feedback evidence member path"));
    }
    let text = path
        .to_str()
        .ok_or_else(|| contract_error("feedback evidence path must be UTF-8"))?;
    if text.is_empty()
        || text.contains(['\\', '\0', '\r', '\n'])
        || text.split('/').any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(contract_error("unsafe feedback evidence member path"));
    }
    Ok(text.to_owned())
}

fn limit_error(message: &str) -> Error {
    Error::new(ErrorKind::InputLimit, message)
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[path = "traversal.rs"]
mod traversal;

pub(super) fn open_directory(root: &Path) -> Result<File, Error> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        traversal::open_root(root)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = root;
        Err(contract_error(
            "safe feedback directory admission requires macOS or Linux",
        ))
    }
}

pub(super) fn same_directory(expected: &File, path: &Path) -> Result<(), Error> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        use std::os::unix::fs::MetadataExt;
        let actual = open_directory(path)?.metadata()?;
        let expected = expected.metadata()?;
        if (actual.dev(), actual.ino()) != (expected.dev(), expected.ino()) {
            return Err(contract_error("feedback publication parent changed"));
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (expected, path);
        Err(contract_error(
            "safe feedback directory admission requires macOS or Linux",
        ))
    }
}

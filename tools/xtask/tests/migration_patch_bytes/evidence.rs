use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn save(path: &Path, bytes: &[u8]) {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}

pub fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn inventory(root: &Path) -> BTreeMap<PathBuf, String> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let kind = entry.file_type().unwrap();
            assert!(!kind.is_symlink());
            if kind.is_dir() {
                pending.push(path);
            } else {
                assert!(kind.is_file());
                files.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    digest(&fs::read(path).unwrap()),
                );
            }
        }
    }
    files
}

pub struct Scratch {
    pub path: PathBuf,
    _temporary: Option<tempfile::TempDir>,
}

impl Scratch {
    pub fn new(name: &str) -> Self {
        match std::env::var_os("PATCH_BYTE_EVIDENCE") {
            Some(parent) => {
                let path = PathBuf::from(parent).canonicalize().unwrap().join(name);
                fs::create_dir(&path).unwrap();
                Self {
                    path,
                    _temporary: None,
                }
            }
            None => {
                let temporary = tempfile::Builder::new()
                    .prefix("patch-byte-")
                    .tempdir()
                    .unwrap();
                Self {
                    path: temporary.path().canonicalize().unwrap(),
                    _temporary: Some(temporary),
                }
            }
        }
    }
}

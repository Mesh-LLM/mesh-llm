#[path = "cleanup_ownership.rs"]
mod validation;
use std::{
    fs,
    io::{self, Write},
    path::Path,
};

pub(crate) fn register(root: &tempfile::TempDir) {
    let location = root.path().canonicalize().unwrap();
    let mut marker = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(location.join(".runner-cleanup-fixture-owner"))
        .unwrap();
    marker
        .write_all(location.as_os_str().as_encoded_bytes())
        .unwrap();
}

pub(crate) fn check(root: &tempfile::TempDir, path: &Path) -> io::Result<()> {
    validation::validate(&root.path().canonicalize()?, path)
}

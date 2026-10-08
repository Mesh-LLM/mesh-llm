use super::*;
use sha2::{Digest, Sha256};

#[test]
fn retained_cell_reader_requires_valid_json_within_its_actual_byte_budget() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cell.json");
    std::fs::write(&path, b"{}\n").unwrap();
    let (value, bytes) = read_cell(&path, 3).unwrap();
    assert!(value.is_object());
    assert_eq!(bytes, 3);
    assert!(read_cell(&path, 2).is_err());
    std::fs::write(&path, b"{").unwrap();
    assert!(read_cell(&path, 3).is_err());
    assert!(read_cell(directory.path(), 3).is_err());
}

#[test]
fn binary_pin_checks_actual_bytes_and_preserves_supplied_source_provenance() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.bin");
    std::fs::write(&path, b"pinned binary").unwrap();
    let mut binary = Binary {
        path,
        sha256: hex::encode(Sha256::digest(b"pinned binary")),
        supplied_commit: "a".repeat(40),
    };
    binary.verify().unwrap();
    assert_eq!(binary.supplied_commit, "a".repeat(40));
    std::fs::write(&binary.path, b"mutated binary").unwrap();
    assert!(binary.verify().is_err());
    binary.supplied_commit = "short".into();
    assert!(binary.verify().is_err());
}

#[test]
fn complete_comparison_budget_reserves_forced_eof_drain_after_grace_and_force() {
    assert_eq!(comparison_budget(10, 4).unwrap(), 920);
    assert_eq!(comparison_budget(86180, 1).unwrap(), 86400);
    assert!(comparison_budget(u64::MAX, 2).is_err());
}

#[cfg(unix)]
#[test]
fn binary_and_retained_cell_refuse_owned_fifo_before_blocking_open() {
    use std::os::unix::ffi::OsStrExt as _;
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("unopened.fifo");
    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let mut binary = Binary {
        path: path.clone(),
        sha256: "a".repeat(64),
        supplied_commit: "b".repeat(40),
    };
    assert!(
        binary
            .verify()
            .unwrap_err()
            .to_string()
            .contains("regular file before hashing")
    );
    assert!(
        read_cell(&path, 100)
            .unwrap_err()
            .to_string()
            .contains("before opening")
    );
    let regular = root.path().join("regular.json");
    std::fs::write(&regular, b"{}").unwrap();
    let link = root.path().join("link.json");
    std::os::unix::fs::symlink(&regular, &link).unwrap();
    assert!(read_cell(&link, 100).is_err());
    assert_eq!(
        regular_binary(&link).unwrap(),
        regular.canonicalize().unwrap()
    );
    root.close().unwrap();
}

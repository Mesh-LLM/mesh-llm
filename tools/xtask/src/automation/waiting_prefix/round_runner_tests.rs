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

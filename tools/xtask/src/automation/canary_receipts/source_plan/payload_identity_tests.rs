use super::*;
#[cfg(unix)]
use std::cell::Cell;

#[cfg(unix)]
#[test]
fn unchanged_files_reuse_receipts_but_replacements_and_preserved_mtime_edits_rehash() {
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("model.gguf");
    let receipts = state.path().join("receipts");
    fs::write(&path, b"good").unwrap();
    let expected = Digest::of_bytes(b"good");
    let modified = fs::metadata(&path).unwrap().modified().unwrap();
    let calls = Cell::new(0);
    let checked_hash = |file: &mut File, size| {
        calls.set(calls.get() + 1);
        hash(file, size)
    };
    verify_with(&path, &expected, 4, Some(&receipts), checked_hash).unwrap();
    verify_with(&path, &expected, 4, Some(&receipts), checked_hash).unwrap();
    assert_eq!(calls.get(), 1, "warm admission must not reread the payload");

    let replacement = state.path().join("replacement");
    fs::write(&replacement, b"good").unwrap();
    File::options()
        .write(true)
        .open(&replacement)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    fs::rename(replacement, &path).unwrap();
    verify_with(&path, &expected, 4, Some(&receipts), checked_hash).unwrap();
    assert_eq!(
        calls.get(),
        2,
        "replacement invalidates the old inode receipt"
    );

    fs::write(&path, b"evil").unwrap();
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    assert!(verify_with(&path, &expected, 4, Some(&receipts), checked_hash).is_err());
    assert_eq!(
        calls.get(),
        3,
        "ctime invalidates same-size, preserved-mtime changes"
    );
}

#[test]
fn invalid_or_unavailable_receipts_cannot_admit_a_wrong_digest_or_size() {
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("model.gguf");
    let receipts = state.path().join("receipts");
    fs::write(&path, b"good").unwrap();
    let expected = Digest::of_bytes(b"good");
    verify_with(&path, &expected, 4, Some(&receipts), hash).unwrap();
    if receipts.is_dir() {
        for receipt in fs::read_dir(&receipts).unwrap() {
            fs::write(receipt.unwrap().path(), b"invalid receipt").unwrap();
        }
    }
    verify_with(&path, &expected, 4, Some(&receipts), hash).unwrap();
    assert!(verify_with(&path, &Digest::of_bytes(b"evil"), 4, Some(&receipts), hash).is_err());
    assert!(verify_with(&path, &expected, 5, Some(&receipts), hash).is_err());
    verify_with(&path, &expected, 4, None, hash).unwrap();
    assert!(verify_with(&path, &Digest::of_bytes(b"evil"), 4, None, hash).is_err());
}

#[test]
fn payload_changes_during_hashing_are_not_recorded_as_verified() {
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("model.gguf");
    let receipts = state.path().join("receipts");
    fs::write(&path, b"good").unwrap();
    let result = verify_with(
        &path,
        &Digest::of_bytes(b"good"),
        4,
        Some(&receipts),
        |file, size| {
            let digest = hash(file, size)?;
            fs::write(&path, b"changed")?;
            Ok(digest)
        },
    );
    assert!(result.unwrap_err().to_string().contains("changed during"));
    assert!(!receipts.is_dir() || fs::read_dir(receipts).unwrap().next().is_none());
}

#[cfg(unix)]
#[test]
fn shared_or_symlinked_receipt_directories_never_supply_cache_authority() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("model.gguf");
    fs::write(&path, b"good").unwrap();
    let shared = state.path().join("shared");
    fs::create_dir(&shared).unwrap();
    fs::set_permissions(&shared, fs::Permissions::from_mode(0o777)).unwrap();
    let linked = state.path().join("linked");
    symlink(&shared, &linked).unwrap();
    let private_leaf = shared.join("private-leaf");
    for directory in [&shared, &linked, &private_leaf] {
        verify_with(&path, &Digest::of_bytes(b"good"), 4, Some(directory), hash).unwrap();
    }
    assert!(fs::read_dir(private_leaf).unwrap().next().is_none());
}

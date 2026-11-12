use super::*;
#[cfg(unix)]
use std::cell::Cell;

/// A clock far enough ahead that freshly written fixtures count as settled.
fn settled_clock() -> SystemTime {
    SystemTime::now() + SETTLE_WINDOW * 30
}

/// Production records only states older than [`SETTLE_WINDOW`], so a later
/// edit always gets a later ctime. Tests that record fresh files through
/// [`settled_clock`] restore that ordering by waiting for the filesystem's
/// change clock to move past the recorded state.
#[cfg(unix)]
fn wait_for_later_change_time(path: &Path) {
    let recorded = FileState::read(&fs::metadata(path).unwrap()).unwrap();
    let probe = path.with_extension("probe");
    loop {
        // A new inode takes the current change time even where truncation would not.
        let _ = fs::remove_file(&probe);
        fs::write(&probe, b"probe").unwrap();
        if FileState::read(&fs::metadata(&probe).unwrap())
            .unwrap()
            .changed
            > recorded.changed
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    fs::remove_file(probe).unwrap();
}

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
    let verify = |clock| verify_with(&path, &expected, 4, Some(&receipts), clock, checked_hash);
    verify(settled_clock()).unwrap();
    verify(settled_clock()).unwrap();
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
    verify(settled_clock()).unwrap();
    assert_eq!(
        calls.get(),
        2,
        "replacement invalidates the old inode receipt"
    );

    wait_for_later_change_time(&path);
    fs::write(&path, b"evil").unwrap();
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    assert!(verify(settled_clock()).is_err());
    assert_eq!(
        calls.get(),
        3,
        "ctime invalidates same-size, preserved-mtime changes"
    );
}

#[cfg(unix)]
#[test]
fn recently_changed_files_are_rehashed_until_their_timestamps_settle() {
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
    // The clock reads the file's own change time, as on a coarse-ctime kernel
    // where an immediate same-size edit can keep every stat field.
    let unsettled = modified + SETTLE_WINDOW / 2;
    let verify = || {
        verify_with(
            &path,
            &expected,
            4,
            Some(&receipts),
            unsettled,
            checked_hash,
        )
    };
    verify().unwrap();
    verify().unwrap();
    assert_eq!(calls.get(), 2, "an unsettled state must not be recorded");
    assert!(!receipts.is_dir() || fs::read_dir(&receipts).unwrap().next().is_none());

    fs::write(&path, b"evil").unwrap();
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    assert!(verify().is_err());
    assert_eq!(calls.get(), 3);
}

#[test]
fn invalid_or_unavailable_receipts_cannot_admit_a_wrong_digest_or_size() {
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("model.gguf");
    let receipts = state.path().join("receipts");
    fs::write(&path, b"good").unwrap();
    let expected = Digest::of_bytes(b"good");
    verify_with(&path, &expected, 4, Some(&receipts), settled_clock(), hash).unwrap();
    if receipts.is_dir() {
        for receipt in fs::read_dir(&receipts).unwrap() {
            fs::write(receipt.unwrap().path(), b"invalid receipt").unwrap();
        }
    }
    let clock = settled_clock();
    verify_with(&path, &expected, 4, Some(&receipts), clock, hash).unwrap();
    let evil = Digest::of_bytes(b"evil");
    assert!(verify_with(&path, &evil, 4, Some(&receipts), clock, hash).is_err());
    assert!(verify_with(&path, &expected, 5, Some(&receipts), clock, hash).is_err());
    verify_with(&path, &expected, 4, None, clock, hash).unwrap();
    assert!(verify_with(&path, &evil, 4, None, clock, hash).is_err());
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
        settled_clock(),
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
        let good = Digest::of_bytes(b"good");
        verify_with(&path, &good, 4, Some(directory), settled_clock(), hash).unwrap();
    }
    assert!(fs::read_dir(private_leaf).unwrap().next().is_none());
}

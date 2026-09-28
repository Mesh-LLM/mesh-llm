use super::*;
use tempfile::TempDir;

#[test]
fn first_sighting_is_fresh_and_records_a_baseline() {
    let dir = TempDir::new().expect("tempdir");
    assert_eq!(record(dir.path(), "0.77.0"), VersionTransition::Fresh);
    assert!(dir.path().join(VERSION_FILE).exists());
}

#[test]
fn an_unchanged_version_reports_unchanged() {
    let dir = TempDir::new().expect("tempdir");
    record(dir.path(), "0.77.0");
    assert_eq!(record(dir.path(), "0.77.0"), VersionTransition::Unchanged);
}

#[test]
fn an_upgrade_reports_the_version_it_came_from() {
    let dir = TempDir::new().expect("tempdir");
    record(dir.path(), "0.76.2");
    assert_eq!(
        record(dir.path(), "0.77.0"),
        VersionTransition::Changed {
            from: "0.76.2".to_owned()
        }
    );
}

#[test]
fn an_upgrade_is_reported_once_not_on_every_later_run() {
    let dir = TempDir::new().expect("tempdir");
    record(dir.path(), "0.76.2");
    record(dir.path(), "0.77.0");
    assert_eq!(record(dir.path(), "0.77.0"), VersionTransition::Unchanged);
}

/// A downgrade is a real event too — a rollback after a bad release is
/// exactly the thing worth seeing, so it is not special-cased away.
#[test]
fn a_downgrade_reports_like_any_other_change() {
    let dir = TempDir::new().expect("tempdir");
    record(dir.path(), "0.77.0");
    assert_eq!(
        record(dir.path(), "0.76.2"),
        VersionTransition::Changed {
            from: "0.77.0".to_owned()
        }
    );
}

#[test]
fn surrounding_whitespace_is_not_a_version_change() {
    let dir = TempDir::new().expect("tempdir");
    fs::write(dir.path().join(VERSION_FILE), "  0.77.0  \n").expect("seed");
    assert_eq!(record(dir.path(), "0.77.0"), VersionTransition::Unchanged);
}

#[test]
fn an_empty_record_is_treated_as_no_record() {
    let dir = TempDir::new().expect("tempdir");
    fs::write(dir.path().join(VERSION_FILE), "\n").expect("seed");
    assert_eq!(record(dir.path(), "0.77.0"), VersionTransition::Fresh);
}

#[test]
fn creates_the_state_directory_when_missing() {
    let dir = TempDir::new().expect("tempdir");
    let nested = dir.path().join("missing").join("state");
    assert_eq!(record(&nested, "0.77.0"), VersionTransition::Fresh);
    assert!(nested.join(VERSION_FILE).exists());
}

/// A state path that cannot be written must not look like an upgrade on every
/// run. It reports `Fresh`, which callers treat as "nothing to report".
///
/// A regular file where the state directory belongs blocks the write for every
/// user, root included, so this covers the contract in CI as well as on a
/// workstation.
#[test]
fn an_unwritable_state_path_never_reports_a_change() {
    let dir = TempDir::new().expect("tempdir");
    let blocked = dir.path().join("blocked");
    fs::write(&blocked, "not a directory").expect("seed");

    assert_eq!(record(&blocked, "0.76.2"), VersionTransition::Fresh);
    assert_eq!(record(&blocked, "0.77.0"), VersionTransition::Fresh);
}

/// An unwritable directory must not look like an upgrade on every run. It
/// reports `Fresh`, which callers treat as "nothing to report".
#[cfg(unix)]
#[test]
fn an_unwritable_directory_never_reports_a_change() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().expect("tempdir");
    let locked = dir.path().join("locked");
    fs::create_dir(&locked).expect("mkdir");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o500)).expect("chmod");

    if fs::write(locked.join("probe"), "").is_ok() {
        // Root ignores the mode bits, so the directory is writable after all
        // and there is nothing unwritable about it to assert on here.
        let _ = fs::remove_file(locked.join("probe"));
        let _ = fs::set_permissions(&locked, fs::Permissions::from_mode(0o700));
        return;
    }

    assert_eq!(record(&locked, "0.76.2"), VersionTransition::Fresh);
    assert_eq!(record(&locked, "0.77.0"), VersionTransition::Fresh);

    // Restore so the tempdir can clean itself up.
    let _ = fs::set_permissions(&locked, fs::Permissions::from_mode(0o700));
}

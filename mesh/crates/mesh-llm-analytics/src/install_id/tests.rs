use super::*;
use tempfile::TempDir;

#[test]
fn generates_then_reuses_a_stable_identifier() {
    let dir = TempDir::new().expect("tempdir");
    let first = load_or_create(dir.path()).expect("create");
    assert!(first.is_first_run());
    assert!(Uuid::parse_str(first.as_str()).is_ok());

    let second = load_or_create(dir.path()).expect("reload");
    assert!(!second.is_first_run());
    assert_eq!(first.as_str(), second.as_str());
}

#[test]
fn replaces_a_corrupted_identifier_instead_of_failing() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join(INSTALL_ID_FILE);
    fs::write(&path, "not-a-uuid").expect("seed");

    let recovered = load_or_create(dir.path()).expect("recover");
    assert!(recovered.is_first_run());
    assert!(Uuid::parse_str(recovered.as_str()).is_ok());
}

#[test]
fn creates_the_state_directory_when_missing() {
    let dir = TempDir::new().expect("tempdir");
    let nested = dir.path().join("missing").join("state");
    let created = load_or_create(&nested).expect("create nested");
    assert!(nested.join(INSTALL_ID_FILE).exists());
    assert!(created.is_first_run());
}

#[test]
fn identifiers_are_unrelated_across_installs() {
    let first_dir = TempDir::new().expect("tempdir");
    let second_dir = TempDir::new().expect("tempdir");
    let first = load_or_create(first_dir.path()).expect("first");
    let second = load_or_create(second_dir.path()).expect("second");
    assert_ne!(first.as_str(), second.as_str());
}

#[cfg(unix)]
#[test]
fn identifier_file_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().expect("tempdir");
    load_or_create(dir.path()).expect("create");
    let mode = fs::metadata(dir.path().join(INSTALL_ID_FILE))
        .expect("metadata")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600);
}

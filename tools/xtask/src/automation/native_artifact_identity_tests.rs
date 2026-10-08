use super::*;

#[test]
fn native_identity_requires_complete_nonempty_bytes_and_rejects_mutation() {
    let directory = tempfile::tempdir().unwrap();
    assert!(verify(directory.path(), &"a".repeat(64)).is_err());
    let file = directory.path().join("runtime.bin");
    std::fs::write(&file, b"native fixture").unwrap();
    let expected = "1b07b22d6970b884a350d667ba62c9ee63ba15ba9816eff304eda67ce9f0fdc6";
    assert_eq!(verify(directory.path(), expected).unwrap(), expected);
    std::fs::write(&file, b"changed fixture").unwrap();
    assert!(verify(directory.path(), expected).is_err());
    assert!(verify(directory.path(), "invalid").is_err());
}

#[cfg(unix)]
#[test]
fn native_identity_allows_internal_file_links_but_refuses_escaping_or_directory_links() {
    use std::os::unix::fs::symlink;
    let outside = tempfile::tempdir().unwrap();
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("outside.bin"), b"outside").unwrap();
    let link = directory.path().join("linked");
    symlink(outside.path(), &link).unwrap();
    assert!(complete_tree(directory.path()).is_err());
    std::fs::remove_file(&link).unwrap();
    symlink(outside.path().join("outside.bin"), &link).unwrap();
    assert!(complete_tree(directory.path()).is_err());
    std::fs::remove_file(&link).unwrap();
    let native = directory.path().join("native.bin");
    std::fs::write(&native, b"inside").unwrap();
    symlink(&native, &link).unwrap();
    complete_tree(directory.path()).unwrap();
}

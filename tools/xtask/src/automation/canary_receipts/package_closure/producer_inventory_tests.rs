use super::super::{fixture_scope, process};
use super::inventory;
use std::{fs, os::unix::fs::symlink};

#[test]
fn source_inventory_rejects_parent_symlink_escape_without_reading_external_bytes() {
    if fixture_scope::isolated(
        module_path!(),
        "source_inventory_rejects_parent_symlink_escape_without_reading_external_bytes",
    ) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("source");
    fs::create_dir_all(root.join("src")).unwrap();
    process::text(&root, &["init", "--quiet"]).unwrap();
    fs::write(root.join("src/owned.rs"), b"owned source\n").unwrap();
    process::text(&root, &["add", "src/owned.rs"]).unwrap();
    assert!(inventory(&root).is_ok());
    let external = directory.path().join("external");
    fs::create_dir(&external).unwrap();
    fs::write(external.join("owned.rs"), b"external source\n").unwrap();
    fs::remove_dir_all(root.join("src")).unwrap();
    symlink(&external, root.join("src")).unwrap();
    let error = inventory(&root).err().unwrap().to_string();
    assert!(
        error.contains("parent escapes selected checkout"),
        "{error}"
    );
}

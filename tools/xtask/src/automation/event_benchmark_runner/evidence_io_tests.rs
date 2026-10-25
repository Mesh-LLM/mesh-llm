use super::*;
use serde_json::{Value, json};

#[test]
fn regular_inputs_are_bounded_and_invalid_json_is_not_repaired() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("input.json");
    std::fs::write(&path, b"{\"ok\":true}").unwrap();
    assert_eq!(read::<Value>(&path, 11).unwrap(), json!({"ok":true}));
    assert!(read::<Value>(&path, 10).is_err());
    std::fs::write(&path, [0xff]).unwrap();
    assert!(read::<Value>(&path, INPUT_BYTES).is_err());
    assert!(read::<Value>(directory.path(), INPUT_BYTES).is_err());
}

#[test]
fn publication_is_newline_terminated_and_never_replaces_existing_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("output.json");
    publish(&path, &json!({"schema_version":1}), RECEIPT_BYTES).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert_eq!(before.last(), Some(&b'\n'));
    assert!(publish(&path, &json!({"schema_version":2}), RECEIPT_BYTES).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn oversized_output_and_relative_paths_leave_no_artifacts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("output.json");
    assert!(publish(&path, &json!({"large":"x".repeat(100)}), 32).is_err());
    assert!(!path.exists());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    assert!(publish(Path::new("relative.json"), &json!({}), MANIFEST_BYTES).is_err());
    assert!(read::<Value>(Path::new("relative.json"), INPUT_BYTES).is_err());
}

#[cfg(unix)]
#[test]
fn symlink_input_and_existing_output_refuse_without_touching_the_target() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("target.json");
    let alias = directory.path().join("alias.json");
    std::fs::write(&target, b"{}").unwrap();
    std::os::unix::fs::symlink(&target, &alias).unwrap();
    assert!(read::<Value>(&alias, INPUT_BYTES).is_err());
    assert!(publish(&alias, &json!({"changed":true}), RECEIPT_BYTES).is_err());
    assert_eq!(std::fs::read(target).unwrap(), b"{}");
}

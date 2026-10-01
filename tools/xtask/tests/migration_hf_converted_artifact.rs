#[path = "../src/automation/hf_converted_artifact/mod.rs"]
mod hf_converted_artifact;

use std::fs;
use std::path::Path;

use hf_converted_artifact::{converted_artifact_dir, validate_converted_artifact};
use tempfile::TempDir;

fn write_manifest(artifact_dir: &Path, manifest: &str) {
    fs::write(artifact_dir.join("README.md"), "beta").unwrap();
    fs::write(artifact_dir.join("skippy-convert-manifest.json"), manifest).unwrap();
}

#[test]
fn accepts_declared_numbered_shards_when_all_files_exist() {
    let temp = TempDir::new().unwrap();
    let artifact_dir = temp.path().join("target/BF16");
    fs::create_dir_all(&artifact_dir).unwrap();
    write_manifest(
        &artifact_dir,
        r#"{"expected_splits":2,"output_basename":"Inkling-BF16"}"#,
    );
    fs::write(
        artifact_dir.join("Inkling-BF16-00001-of-00002.gguf"),
        b"one",
    )
    .unwrap();
    fs::write(
        artifact_dir.join("Inkling-BF16-00002-of-00002.gguf"),
        b"two",
    )
    .unwrap();

    let selected = converted_artifact_dir(temp.path(), Path::new("BF16"), true).unwrap();

    assert_eq!(selected, artifact_dir);
}

#[test]
fn accepts_declared_single_shard_when_unsplit_file_exists() {
    let temp = TempDir::new().unwrap();
    write_manifest(
        temp.path(),
        r#"{"expected_splits":1,"output_basename":"Inkling-BF16"}"#,
    );
    fs::write(temp.path().join("Inkling-BF16.gguf"), b"one").unwrap();

    let result = validate_converted_artifact(temp.path());

    assert!(result.is_ok());
}

#[test]
fn selects_unchecked_path_when_upload_only_is_false() {
    let temp = TempDir::new().unwrap();

    let selected = converted_artifact_dir(temp.path(), Path::new("custom"), false).unwrap();

    assert_eq!(selected, temp.path().join("target/custom"));
}

#[test]
fn rejects_missing_card_and_manifest_before_parsing() {
    let temp = TempDir::new().unwrap();

    let error = validate_converted_artifact(temp.path()).unwrap_err();

    assert_eq!(
        error.to_string(),
        format!(
            "complete converted artifact not found: {}; missing README.md, skippy-convert-manifest.json",
            temp.path().display()
        )
    );
}

#[test]
fn rejects_missing_card_before_invalid_manifest() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("skippy-convert-manifest.json"), "not json").unwrap();

    let error = validate_converted_artifact(temp.path()).unwrap_err();

    assert!(error.to_string().contains("; missing README.md"));
}

#[test]
fn rejects_missing_directory_before_reading_manifest() {
    let temp = TempDir::new().unwrap();
    let artifact_dir = temp.path().join("target/BF16");

    let error = converted_artifact_dir(temp.path(), Path::new("BF16"), true).unwrap_err();

    assert_eq!(
        error.to_string(),
        format!(
            "complete converted artifact not found: {}; missing README.md, skippy-convert-manifest.json",
            artifact_dir.display()
        )
    );
}

#[test]
fn rejects_invalid_split_count_before_invalid_basename() {
    let temp = TempDir::new().unwrap();
    write_manifest(temp.path(), r#"{"expected_splits":0,"output_basename":""}"#);

    let error = validate_converted_artifact(temp.path()).unwrap_err();

    assert!(error.to_string().starts_with("invalid expected_splits in "));
}

#[test]
fn rejects_false_split_count_as_invalid() {
    let temp = TempDir::new().unwrap();
    write_manifest(
        temp.path(),
        r#"{"expected_splits":false,"output_basename":"Inkling-BF16"}"#,
    );

    let error = validate_converted_artifact(temp.path()).unwrap_err();

    assert!(error.to_string().starts_with("invalid expected_splits in "));
}

#[test]
fn accepts_true_split_count_as_single_file_like_python_int() {
    let temp = TempDir::new().unwrap();
    write_manifest(
        temp.path(),
        r#"{"expected_splits":true,"output_basename":"Inkling-BF16"}"#,
    );
    fs::write(temp.path().join("Inkling-BF16.gguf"), b"one").unwrap();

    let result = validate_converted_artifact(temp.path());

    assert!(result.is_ok());
}

#[test]
fn rejects_empty_basename_before_missing_shards() {
    let temp = TempDir::new().unwrap();
    write_manifest(temp.path(), r#"{"expected_splits":2,"output_basename":""}"#);

    let error = validate_converted_artifact(temp.path()).unwrap_err();

    assert!(error.to_string().starts_with("invalid output_basename in "));
}

#[test]
fn rejects_incomplete_shards_in_declared_order() {
    let temp = TempDir::new().unwrap();
    write_manifest(
        temp.path(),
        r#"{"expected_splits":3,"output_basename":"Inkling-BF16"}"#,
    );
    fs::write(temp.path().join("Inkling-BF16-00002-of-00003.gguf"), b"two").unwrap();

    let error = validate_converted_artifact(temp.path()).unwrap_err();

    assert_eq!(
        error.to_string(),
        "converted artifact is incomplete: missing Inkling-BF16-00001-of-00003.gguf, Inkling-BF16-00003-of-00003.gguf"
    );
}

#[test]
fn rejects_wrong_single_shard_name() {
    let temp = TempDir::new().unwrap();
    write_manifest(
        temp.path(),
        r#"{"expected_splits":1,"output_basename":"Inkling-BF16"}"#,
    );
    fs::write(temp.path().join("Inkling-BF16-00001-of-00001.gguf"), b"one").unwrap();

    let error = validate_converted_artifact(temp.path()).unwrap_err();

    assert_eq!(
        error.to_string(),
        "converted artifact is incomplete: missing Inkling-BF16.gguf"
    );
}

#[test]
fn rejects_directory_in_place_of_shard() {
    let temp = TempDir::new().unwrap();
    write_manifest(
        temp.path(),
        r#"{"expected_splits":1,"output_basename":"Inkling-BF16"}"#,
    );
    fs::create_dir(temp.path().join("Inkling-BF16.gguf")).unwrap();

    let error = validate_converted_artifact(temp.path()).unwrap_err();

    assert_eq!(
        error.to_string(),
        "converted artifact is incomplete: missing Inkling-BF16.gguf"
    );
}

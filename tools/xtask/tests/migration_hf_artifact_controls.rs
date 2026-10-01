#[path = "../src/automation/hf_converted_artifact/mod.rs"]
mod hf_converted_artifact;

use hf_converted_artifact::{ArtifactError, converted_artifact_dir, validate_converted_artifact};
use std::{fs, path::Path};
use tempfile::TempDir;

fn artifact(manifest: &[u8]) -> TempDir {
    let directory = TempDir::new().unwrap();
    fs::write(directory.path().join("README.md"), b"beta").unwrap();
    fs::write(
        directory.path().join("skippy-convert-manifest.json"),
        manifest,
    )
    .unwrap();
    fs::write(directory.path().join("Inkling-BF16.gguf"), b"one").unwrap();
    directory
}

#[test]
fn accepts_last_valid_count_when_duplicate_keys_are_present() {
    let given =
        artifact(br#"{"expected_splits":0,"expected_splits":1,"output_basename":"Inkling-BF16"}"#);

    let when = validate_converted_artifact(given.path());

    assert!(when.is_ok(), "{when:?}");
}

#[test]
fn rejects_last_invalid_count_when_duplicate_keys_are_present() {
    let given =
        artifact(br#"{"expected_splits":1,"expected_splits":0,"output_basename":"Inkling-BF16"}"#);

    let when = validate_converted_artifact(given.path());

    assert!(matches!(when, Err(ArtifactError::InvalidSplitCount(_))));
}

#[test]
fn rejects_float_when_its_value_is_integral() {
    let given = artifact(br#"{"expected_splits":1.0,"output_basename":"Inkling-BF16"}"#);

    let when = validate_converted_artifact(given.path());

    assert!(matches!(when, Err(ArtifactError::InvalidSplitCount(_))));
}

#[test]
fn reports_missing_card_when_manifest_is_invalid() {
    let given = TempDir::new().unwrap();
    fs::write(
        given.path().join("skippy-convert-manifest.json"),
        b"not json",
    )
    .unwrap();

    let when = validate_converted_artifact(given.path());

    assert!(
        matches!(when, Err(ArtifactError::MissingArtifact { details, .. }) if details == "; missing README.md")
    );
}

#[test]
fn reports_count_when_basename_is_also_invalid() {
    let given = artifact(br#"{"expected_splits":0,"output_basename":""}"#);

    let when = validate_converted_artifact(given.path());

    assert!(matches!(when, Err(ArtifactError::InvalidSplitCount(_))));
}

#[test]
fn reports_basename_when_count_is_valid() {
    let given = artifact(br#"{"expected_splits":2,"output_basename":""}"#);

    let when = validate_converted_artifact(given.path());

    assert!(matches!(when, Err(ArtifactError::InvalidBasename(_))));
}

#[test]
fn reports_middle_shard_when_other_shards_exist() {
    let given = artifact(br#"{"expected_splits":3,"output_basename":"Inkling-BF16"}"#);
    for index in [1, 3] {
        fs::write(
            given
                .path()
                .join(format!("Inkling-BF16-{index:05}-of-00003.gguf")),
            b"one",
        )
        .unwrap();
    }

    let when = validate_converted_artifact(given.path());

    assert!(
        matches!(when, Err(ArtifactError::MissingShards(names)) if names == "Inkling-BF16-00002-of-00003.gguf")
    );
}

#[test]
fn reports_parse_error_when_manifest_has_invalid_utf8() {
    let given = artifact(&[0xff]);

    let when = validate_converted_artifact(given.path());

    assert!(matches!(when, Err(ArtifactError::ParseManifest { .. })));
}

#[test]
fn reports_count_when_root_is_array() {
    let given = artifact(b"[]");

    let when = validate_converted_artifact(given.path());

    assert!(matches!(when, Err(ArtifactError::InvalidSplitCount(_))));
}

#[test]
fn reports_count_when_integer_exceeds_u64() {
    let given = artifact(br#"{"expected_splits":18446744073709551616,"output_basename":""}"#);

    let when = validate_converted_artifact(given.path());

    assert!(matches!(when, Err(ArtifactError::InvalidSplitCount(_))));
}

#[test]
fn selects_file_api_path_when_upload_preflight_succeeds() {
    let given = artifact(br#"{"expected_splits":1,"output_basename":"Inkling-BF16"}"#);

    let when = converted_artifact_dir(given.path(), given.path(), true);

    assert_eq!(when.unwrap(), given.path());
}

#[test]
fn rejects_file_api_path_when_upload_artifact_is_absent() {
    let given = TempDir::new().unwrap();

    let when = converted_artifact_dir(given.path(), Path::new("BF16"), true);

    assert!(matches!(when, Err(ArtifactError::MissingArtifact { .. })));
}

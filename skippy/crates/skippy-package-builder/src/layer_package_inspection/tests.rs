use super::*;
use clap::Parser as _;
use serde_json::json;
use std::{fs, path::Path};
#[path = "fixtures.rs"]
mod fixtures;
use fixtures::{fixture, save};
fn dispatch(root: &Path, extras: &[&str]) -> (Result<()>, Vec<u8>) {
    let mut argv = vec![
        "tool".to_owned(),
        "inspect-layer-package".into(),
        "--package".into(),
        root.to_str().unwrap().into(),
    ];
    argv.extend(extras.iter().map(|v| (*v).into()));
    let args = crate::cli::Args::try_parse_from(argv).unwrap();
    assert!(!args.command.requires_download_preparation());
    let mut bytes = Vec::new();
    let result = crate::run_with_output(args, &mut bytes);
    (result, bytes)
}
#[test]
fn full_native_cli_dispatch_checks_all_declared_artifacts_and_truthful_geometry() {
    let temp = tempfile::tempdir().unwrap();
    fixture(temp.path());
    let (result, bytes) = dispatch(
        temp.path(),
        &[
            "--expected-layer-count",
            "2",
            "--expected-activation-width",
            "4096",
        ],
    );
    result.unwrap();
    let report: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(report["artifact_count"], 6);
    assert_eq!(report["activation_width"], 4096);
    assert_eq!(report["artifact_sha256_verified"], false);
    assert_eq!(report["inference_qualified"], false);
    assert!(bytes.ends_with(b"}\n"));
}
#[test]
fn missing_or_truncated_shared_layer_and_projector_refuse_before_machine_output() {
    for path in [
        "metadata.gguf",
        "embeddings.gguf",
        "output.gguf",
        "layers/0.gguf",
        "layers/1.gguf",
        "projector.gguf",
    ] {
        for missing in [true, false] {
            let temp = tempfile::tempdir().unwrap();
            fixture(temp.path());
            if missing {
                fs::remove_file(temp.path().join(path)).unwrap();
            } else {
                fs::write(temp.path().join(path), b"x").unwrap();
            }
            let (result, bytes) = dispatch(temp.path(), &[]);
            assert!(result.is_err(), "{path}");
            assert!(bytes.is_empty());
        }
    }
}
#[test]
fn duplicate_missing_layers_unsafe_paths_and_schema_refuse_without_output() {
    for mutation in 0..4 {
        let temp = tempfile::tempdir().unwrap();
        let mut manifest = fixture(temp.path());
        match mutation {
            0 => manifest["layers"][1]["layer_index"] = json!(0),
            1 => {
                manifest["layers"].as_array_mut().unwrap().pop();
            }
            2 => manifest["shared"]["metadata"]["path"] = json!("../escape.gguf"),
            _ => manifest["schema_version"] = json!(2),
        }
        save(temp.path(), &manifest);
        let (result, bytes) = dispatch(temp.path(), &[]);
        assert!(result.is_err());
        assert!(bytes.is_empty());
    }
}
#[test]
fn expected_geometry_mismatch_and_bad_width_refuse_before_output() {
    for extras in [
        vec!["--expected-layer-count", "3"],
        vec!["--expected-activation-width", "2048"],
        vec!["--expected-layer-count", "0"],
    ] {
        let temp = tempfile::tempdir().unwrap();
        fixture(temp.path());
        let (result, bytes) = dispatch(temp.path(), &extras);
        assert!(result.is_err());
        assert!(bytes.is_empty());
    }
    let absent = tempfile::tempdir().unwrap();
    let mut manifest = fixture(absent.path());
    manifest.as_object_mut().unwrap().remove("activation_width");
    save(absent.path(), &manifest);
    let (result, bytes) = dispatch(absent.path(), &[]);
    assert!(result.is_err());
    assert!(bytes.is_empty());
    for width in [
        Value::Null,
        json!(0),
        json!("4096"),
        json!(4096.5),
        json!(true),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let mut manifest = fixture(temp.path());
        manifest["activation_width"] = width;
        save(temp.path(), &manifest);
        let (result, bytes) = dispatch(temp.path(), &[]);
        assert!(result.is_err());
        assert!(bytes.is_empty());
    }
}
#[cfg(unix)]
#[test]
fn normal_hf_snapshot_blob_symlinks_are_supported_and_missing_blob_refuses() {
    let temp = tempfile::tempdir().unwrap();
    let snapshot = temp.path().join("snapshot");
    fs::create_dir(&snapshot).unwrap();
    fixture(&snapshot);
    let blobs = temp.path().join("blobs");
    fs::create_dir(&blobs).unwrap();
    let blob = blobs.join("metadata");
    fs::rename(snapshot.join("metadata.gguf"), &blob).unwrap();
    std::os::unix::fs::symlink("../blobs/metadata", snapshot.join("metadata.gguf")).unwrap();
    dispatch(&snapshot, &[]).0.unwrap();
    fs::remove_file(blob).unwrap();
    let (result, bytes) = dispatch(&snapshot, &[]);
    assert!(result.is_err());
    assert!(bytes.is_empty());
}
#[test]
fn local_inspection_refuses_writer_failures_after_custody() {
    struct Refused;
    impl std::io::Write for Refused {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let temp = tempfile::tempdir().unwrap();
    fixture(temp.path());
    let args = crate::cli::Args::try_parse_from([
        "tool",
        "inspect-layer-package",
        "--package",
        temp.path().to_str().unwrap(),
    ])
    .unwrap();
    assert!(crate::run_with_output(args, &mut Refused).is_err());
}

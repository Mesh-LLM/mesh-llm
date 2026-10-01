use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

type TestResult = Result<(), Box<dyn Error>>;

fn run(args: &[&str]) -> Result<Output, Box<dyn Error>> {
    Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .output()?)
}

fn complete_artifact() -> Result<(TempDir, PathBuf), Box<dyn Error>> {
    let temp = TempDir::new()?;
    let artifact_dir = temp.path().join("target/BF16");
    fs::create_dir_all(&artifact_dir)?;
    fs::write(artifact_dir.join("README.md"), b"beta")?;
    fs::write(
        artifact_dir.join("skippy-convert-manifest.json"),
        br#"{"expected_splits":2,"output_basename":"Inkling-BF16"}"#,
    )?;
    fs::write(
        artifact_dir.join("Inkling-BF16-00001-of-00002.gguf"),
        b"one",
    )?;
    fs::write(
        artifact_dir.join("Inkling-BF16-00002-of-00002.gguf"),
        b"two",
    )?;
    Ok((temp, artifact_dir))
}

fn files(root: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>, Box<dyn Error>> {
    fs::read_dir(root)?
        .map(|entry| {
            let path = entry?.path();
            Ok((
                path.file_name().ok_or("fixture entry has no name")?.into(),
                fs::read(path)?,
            ))
        })
        .collect()
}

#[test]
fn migration_hf_artifact_cli_preflight_accepts_complete_shards_without_mutation() -> TestResult {
    // Given: a local two-shard artifact and its byte snapshot.
    let (_temp, artifact_dir) = complete_artifact()?;
    let before = files(&artifact_dir)?;
    let artifact_arg = artifact_dir.to_str().ok_or("non-UTF8 fixture")?;

    // When: xtask preflights the explicit local artifact directory.
    let output = run(&[
        "hf-converted-artifact",
        "preflight",
        "--artifact-dir",
        artifact_arg,
    ])?;

    // Then: it reports success and leaves all fixture bytes unchanged.
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(
        String::from_utf8(output.stdout)?,
        format!(
            "converted artifact preflight passed: {}\n",
            artifact_dir.display()
        )
    );
    assert_eq!(files(&artifact_dir)?, before);
    Ok(())
}

#[test]
fn migration_hf_artifact_cli_preflight_rejects_missing_shard_without_mutation() -> TestResult {
    // Given: a two-shard manifest with only its first shard present.
    let (_temp, artifact_dir) = complete_artifact()?;
    fs::remove_file(artifact_dir.join("Inkling-BF16-00002-of-00002.gguf"))?;
    let before = files(&artifact_dir)?;
    let artifact_arg = artifact_dir.to_str().ok_or("non-UTF8 fixture")?;

    // When: xtask preflights the incomplete local directory.
    let output = run(&[
        "hf-converted-artifact",
        "preflight",
        "--artifact-dir",
        artifact_arg,
    ])?;

    // Then: it reports the missing second shard and leaves the artifact unchanged.
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("converted artifact is incomplete: missing Inkling-BF16-00002-of-00002.gguf")
    );
    assert_eq!(files(&artifact_dir)?, before);
    Ok(())
}

#[test]
fn migration_hf_artifact_cli_preflight_reports_both_metadata_files_in_order() -> TestResult {
    // Given: an empty local artifact directory.
    let temp = TempDir::new()?;
    let artifact_dir = temp.path().join("artifact");
    fs::create_dir(&artifact_dir)?;
    let before = files(&artifact_dir)?;
    let artifact_arg = artifact_dir.to_str().ok_or("non-UTF8 fixture")?;

    // When: xtask preflights the directory without either required metadata file.
    let output = run(&[
        "hf-converted-artifact",
        "preflight",
        "--artifact-dir",
        artifact_arg,
    ])?;

    // Then: README precedes the manifest in the error and no files are created.
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("; missing README.md, skippy-convert-manifest.json")
    );
    assert_eq!(files(&artifact_dir)?, before);
    Ok(())
}

#[test]
fn migration_hf_artifact_cli_preflight_requires_manifest_when_card_exists() -> TestResult {
    // Given: an artifact directory containing its README but no manifest.
    let temp = TempDir::new()?;
    let artifact_dir = temp.path().join("artifact");
    fs::create_dir(&artifact_dir)?;
    fs::write(artifact_dir.join("README.md"), b"beta")?;
    let before = files(&artifact_dir)?;
    let artifact_arg = artifact_dir.to_str().ok_or("non-UTF8 fixture")?;

    // When: xtask preflights that local directory.
    let output = run(&[
        "hf-converted-artifact",
        "preflight",
        "--artifact-dir",
        artifact_arg,
    ])?;

    // Then: it reports the missing manifest and leaves the files unchanged.
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("; missing skippy-convert-manifest.json")
    );
    assert_eq!(files(&artifact_dir)?, before);
    Ok(())
}

#[test]
fn migration_hf_artifact_cli_preflight_preserves_metadata_rejection_order() -> TestResult {
    // Given: an invalid manifest in a directory with no README card.
    let temp = TempDir::new()?;
    let artifact_dir = temp.path().join("artifact");
    fs::create_dir(&artifact_dir)?;
    fs::write(
        artifact_dir.join("skippy-convert-manifest.json"),
        b"not json",
    )?;
    let before = files(&artifact_dir)?;
    let artifact_arg = artifact_dir.to_str().ok_or("non-UTF8 fixture")?;

    // When: xtask preflights that local directory.
    let output = run(&[
        "hf-converted-artifact",
        "preflight",
        "--artifact-dir",
        artifact_arg,
    ])?;

    // Then: the missing README wins over manifest parsing, without mutation.
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("; missing README.md"));
    assert!(!stderr.contains("cannot parse converted artifact manifest"));
    assert_eq!(files(&artifact_dir)?, before);
    Ok(())
}

#[test]
fn migration_hf_artifact_cli_preflight_help_needs_no_artifact_directory() -> TestResult {
    // Given: no local artifact path.
    // When: help is requested for preflight.
    let output = run(&["hf-converted-artifact", "preflight", "--help"])?;

    // Then: usage is printed successfully without inspecting a directory.
    assert_eq!(output.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&output.stdout).contains("--artifact-dir <directory>"));
    assert!(output.stderr.is_empty());
    Ok(())
}

#[test]
fn migration_hf_artifact_cli_preflight_rejects_upload_flags() -> TestResult {
    // Given: a complete local artifact and no upload action in the CLI contract.
    let (_temp, artifact_dir) = complete_artifact()?;
    let before = files(&artifact_dir)?;
    let artifact_arg = artifact_dir.to_str().ok_or("non-UTF8 fixture")?;

    // When: an upload-only flag is supplied to the local preflight command.
    let output = run(&[
        "hf-converted-artifact",
        "preflight",
        "--artifact-dir",
        artifact_arg,
        "--upload-only",
    ])?;

    // Then: argument parsing rejects it before validation and does not mutate files.
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unrecognized arguments"));
    assert_eq!(files(&artifact_dir)?, before);
    Ok(())
}

#[test]
fn migration_hf_artifact_cli_preflight_checks_count_before_basename() -> TestResult {
    // Given: both manifest fields are invalid and the artifact has no shards.
    let temp = TempDir::new()?;
    let artifact_dir = temp.path().join("artifact");
    fs::create_dir(&artifact_dir)?;
    fs::write(artifact_dir.join("README.md"), b"beta")?;
    fs::write(
        artifact_dir.join("skippy-convert-manifest.json"),
        br#"{"expected_splits":0,"output_basename":""}"#,
    )?;
    let before = files(&artifact_dir)?;
    let artifact_arg = artifact_dir.to_str().ok_or("non-UTF8 fixture")?;

    // When: xtask preflights the directory with both invalid manifest fields.
    let output = run(&[
        "hf-converted-artifact",
        "preflight",
        "--artifact-dir",
        artifact_arg,
    ])?;

    // Then: invalid count is the first reported error and no files are changed.
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid expected_splits"));
    assert!(!stderr.contains("invalid output_basename"));
    assert!(!stderr.contains("converted artifact is incomplete"));
    assert_eq!(files(&artifact_dir)?, before);
    Ok(())
}

#[test]
fn migration_hf_artifact_cli_preflight_checks_basename_before_shards() -> TestResult {
    // Given: a valid split count, empty basename, and no shard files.
    let temp = TempDir::new()?;
    let artifact_dir = temp.path().join("artifact");
    fs::create_dir(&artifact_dir)?;
    fs::write(artifact_dir.join("README.md"), b"beta")?;
    fs::write(
        artifact_dir.join("skippy-convert-manifest.json"),
        br#"{"expected_splits":2,"output_basename":""}"#,
    )?;
    let before = files(&artifact_dir)?;
    let artifact_arg = artifact_dir.to_str().ok_or("non-UTF8 fixture")?;

    // When: xtask preflights the directory with an empty basename.
    let output = run(&[
        "hf-converted-artifact",
        "preflight",
        "--artifact-dir",
        artifact_arg,
    ])?;

    // Then: basename is rejected before shard lookup and no files are changed.
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid output_basename"));
    assert!(!stderr.contains("converted artifact is incomplete"));
    assert_eq!(files(&artifact_dir)?, before);
    Ok(())
}

use super::support::{Fixture, read};
use std::{ffi::OsString, fs};

#[test]
fn relative_paths_keep_invocation_directory_when_rust_child_uses_repository_directory()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let cwd = fixture.root.path().join("invocation");
    fs::create_dir(&cwd)?;
    let mut arguments: Vec<_> = fixture
        .command("granite-3.1-2b", None)
        .get_args()
        .map(OsString::from)
        .collect();
    for (option, value) in [
        ("--matrix", "../matrix.json"),
        ("--dataset-file", "../data.parquet"),
        ("--model-file", "../model.gguf"),
    ] {
        let index = arguments
            .windows(2)
            .position(|pair| pair[0] == option)
            .ok_or("fixture option")?;
        arguments[index + 1] = value.into();
    }
    let output = fixture.execute_at(arguments, &cwd)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(cwd.join("result/run.json").is_file());
    assert!(!fixture.root.path().join("result").exists());
    let argv = read(&fixture.argv)?;
    let expected = fixture.root.path().join("data.parquet").canonicalize()?;
    assert!(
        argv.lines()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|pair| pair == ["--dataset-file", expected.to_str().unwrap_or_default()])
    );
    Ok(())
}

#[test]
fn json_alias_reloads_exported_matrix_before_child() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let mut arguments: Vec<_> = fixture
        .command("granite-3.1-2b", None)
        .get_args()
        .map(OsString::from)
        .collect();
    let option = arguments
        .windows(2)
        .position(|pair| pair[0] == "--json-output")
        .ok_or("JSON option")?;
    arguments[option + 1] = "matrix.json".into();
    let output = fixture.execute(arguments)?;
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(output.stderr)?,
        "matrix replay block is missing\n"
    );
    assert!(output.stdout.is_empty());
    assert!(!fixture.marker.exists());
    assert!(read(&fixture.root.path().join("matrix.json"))?.contains("\"mode\": \"all\""));
    assert!(fixture.env.is_file());
    Ok(())
}

#[test]
fn env_alias_reloads_appended_matrix_before_child() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let mut arguments: Vec<_> = fixture
        .command("granite-3.1-2b", None)
        .get_args()
        .map(OsString::from)
        .collect();
    let option = arguments
        .windows(2)
        .position(|pair| pair[0] == "--github-env")
        .ok_or("env option")?;
    arguments[option + 1] = "matrix.json".into();
    let output = fixture.execute(arguments)?;
    assert_eq!(output.status.code(), Some(1));
    assert!(!output.stderr.is_empty());
    assert!(output.stdout.is_empty());
    assert!(!fixture.marker.exists());
    assert!(read(&fixture.root.path().join("matrix.json"))?.contains("AGENTIC_REPLAY_MODE=all\n"));
    assert!(fixture.json.is_file());
    Ok(())
}

#[test]
fn env_failure_keeps_completed_json_before_selection() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    fs::create_dir(&fixture.env)?;
    let output = fixture.run("missing-family", "exit", 0, None)?;
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8(output.stderr)?.contains("replay matrix export: GitHub env"));
    assert!(output.stdout.is_empty());
    assert!(!fixture.marker.exists());
    assert!(read(&fixture.json)?.contains("\"mode\": \"all\""));
    assert!(fixture.env.is_dir());
    Ok(())
}

#[test]
fn missing_ref_keeps_exports_without_starting_child() -> Result<(), Box<dyn std::error::Error>> {
    missing_dependency("--ref")
}

#[test]
fn missing_output_keeps_exports_without_starting_child() -> Result<(), Box<dyn std::error::Error>> {
    missing_dependency("--output")
}

#[test]
fn missing_model_file_keeps_exports_without_starting_reader()
-> Result<(), Box<dyn std::error::Error>> {
    missing_dependency("--model-file")
}

#[test]
fn mismatched_local_model_digest_keeps_exports_without_starting_reader()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    fs::write(&fixture.model, b"different model bytes")?;
    let output = fixture.run("granite-3.1-2b", "exit", 0, None)?;
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8(output.stderr)?.contains("model SHA-256 mismatch"));
    assert!(fixture.json.is_file());
    assert!(fixture.env.is_file());
    assert!(!fixture.marker.exists());
    assert!(!fixture.root.path().join("result").exists());
    Ok(())
}

fn missing_dependency(option: &str) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let mut arguments: Vec<_> = fixture
        .command("granite-3.1-2b", None)
        .get_args()
        .map(OsString::from)
        .collect();
    let position = arguments
        .windows(2)
        .position(|pair| pair[0] == option)
        .ok_or("dependency option")?;
    arguments.drain(position..position + 2);
    let output = fixture.execute(arguments)?;
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8(output.stderr)?.contains(&format!("--run-family needs {option}")));
    assert!(output.stdout.is_empty());
    assert!(!fixture.marker.exists());
    assert!(fixture.json.is_file());
    assert!(fixture.env.is_file());
    Ok(())
}

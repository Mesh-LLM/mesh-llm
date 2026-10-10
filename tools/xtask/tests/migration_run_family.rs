#![cfg(unix)]

#[path = "migration_run_family/history_consumer.rs"]
mod history_consumer;

#[path = "migration_run_family/lifecycle.rs"]
mod lifecycle;
#[path = "migration_run_family/reader.rs"]
mod reader;
#[path = "migration_run_family/sequencing.rs"]
mod sequencing;
#[path = "migration_run_family/support.rs"]
mod support;

use self::support::{Fixture, MODEL_URI, SHELL_LINE, read};
use std::ffi::OsString;

#[test]
fn run_family_uses_rust_execution_and_native_reader_boundary()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    assert!(!fixture.root.path().join("evals/agentic-replay.py").exists());
    let output = fixture.run("granite-3.1-2b", "exit", 0, None)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(fixture.marker.is_file());
    assert!(String::from_utf8(output.stdout)?.ends_with(SHELL_LINE));
    let run: serde_json::Value =
        serde_json::from_str(&read(&fixture.root.path().join("result/run.json"))?)?;
    assert_eq!(run["gates"]["evaluated"], false);
    assert!(run["gates"]["passed"].is_null());
    assert!(
        read(&fixture.root.path().join("result/summary/REPORT.md"))?
            .contains("Overall: **NOT EVALUATED**")
    );
    assert_eq!(run["config"]["model"], MODEL_URI);
    assert_eq!(
        run["config"]["model_file"],
        serde_json::to_value(fixture.model.canonicalize()?)?
    );
    let matrix: serde_json::Value =
        serde_json::from_str(&read(&fixture.root.path().join("matrix.json"))?)?;
    assert_eq!(run["config"]["model_sha256"], matrix["models"][0]["sha256"]);
    assert_eq!(
        run["inputs"]["dataset_file_sha256"],
        matrix["replay"]["dataset_sha256"]
    );
    assert_eq!(run["inputs"]["kind"], "thoughtworks");
    assert_eq!(
        run["config"]["required_frameworks"],
        serde_json::json!(["swe-agent", "mini-swe-agent", "openhands"])
    );
    assert_eq!(run["context_preflight"]["main"]["passed"], true);
    assert_eq!(run["results"].as_array().ok_or("results")?.len(), 1);
    assert!(
        fixture
            .root
            .path()
            .join("result/summary/comparison.json")
            .is_file()
    );
    assert!(
        fixture
            .root
            .path()
            .join("result/inputs/captured-trajectories.json")
            .is_file()
    );
    assert_eq!(
        read(&fixture.root.path().join("build-calls.txt"))?,
        "release-host-build\nrelease-runtime-build\nmetal\n"
    );
    let argv = read(&fixture.argv)?;
    let argv: Vec<_> = argv.lines().collect();
    assert_eq!(argv[0], "cohorts");
    assert!(!argv.contains(&"run"));
    assert!(!argv.contains(&"--model"));
    assert!(
        argv.windows(2)
            .any(|pair| pair == ["--sessions-per-cohort", "3"])
    );
    assert!(
        argv.windows(2)
            .any(|pair| pair == ["--framework", "openhands"])
    );
    assert!(read(&fixture.json)?.contains("\"mode\": \"all\""));
    assert!(read(&fixture.env)?.starts_with("AGENTIC_REPLAY_MODE=all\n"));
    history_consumer::verify(&fixture)?;
    Ok(())
}

#[test]
fn run_family_reader_failure_keeps_exports_and_never_prints_success_shell_line()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let output = fixture.run("granite-3.1-2b", "exit", 23, None)?;
    assert_eq!(output.status.code(), Some(1));
    assert!(fixture.marker.is_file());
    assert!(fixture.json.is_file());
    assert!(fixture.env.is_file());
    assert!(!String::from_utf8(output.stdout)?.ends_with(SHELL_LINE));
    assert!(String::from_utf8(output.stderr)?.contains("trajectory reader failed"));
    assert!(!fixture.root.path().join("build-calls.txt").exists());
    assert!(!fixture.root.path().join("result/run.json").exists());
    Ok(())
}
#[test]
fn run_family_exports_before_selection_rejection_without_starting_child()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let output = fixture.run("missing-family", "exit", 0, None)?;
    assert_eq!(output.status.code(), Some(1));
    assert!(!fixture.marker.exists());
    assert!(fixture.json.is_file());
    assert!(fixture.env.is_file());
    assert!(String::from_utf8(output.stderr)?.contains("family is absent from replay matrix"));
    Ok(())
}

#[test]
fn run_family_empty_value_preserves_legacy_export_only_behavior()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let arguments = [
        "--repo-root",
        fixture.root.path().to_str().ok_or("repo root path")?,
        "automation",
        "replay-matrix",
        "run-family",
        "--matrix",
        "matrix.json",
        "--run-family=",
        "--json-output",
        fixture.json.to_str().ok_or("JSON output path")?,
        "--github-env",
        fixture.env.to_str().ok_or("GitHub env path")?,
        "--print-shell",
    ]
    .map(OsString::from);
    let output = fixture.execute(arguments)?;
    assert_eq!(output.status.code(), Some(0));
    assert!(!fixture.marker.exists());
    assert!(fixture.json.is_file());
    assert!(fixture.env.is_file());
    assert!(String::from_utf8(output.stdout)?.starts_with(SHELL_LINE));
    assert!(output.stderr.is_empty());
    Ok(())
}

#[test]
fn run_family_late_dependency_rejection_keeps_exports_and_skips_child()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let output = fixture.run_without_dataset()?;
    assert_eq!(output.status.code(), Some(2));
    assert!(!fixture.marker.exists());
    assert!(fixture.json.is_file());
    assert!(fixture.env.is_file());
    assert!(String::from_utf8(output.stderr)?.contains("--run-family needs --dataset-file"));
    Ok(())
}

#[test]
fn run_family_export_failure_precedes_selection_and_child_spawn()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new()?;
    let absent = fixture.root.path().join("absent/params.json");
    let mut arguments: Vec<_> = fixture
        .command("missing-family", None)
        .get_args()
        .map(OsString::from)
        .collect();
    let json_output = arguments
        .windows(2)
        .position(|pair| pair[0] == "--json-output")
        .ok_or("JSON output option")?;
    arguments[json_output + 1] = absent.into_os_string();
    let output = fixture.execute(arguments)?;
    assert_eq!(output.status.code(), Some(1));
    assert!(!fixture.marker.exists());
    assert!(!fixture.env.exists());
    assert!(String::from_utf8(output.stderr)?.contains("replay matrix export: JSON"));
    Ok(())
}

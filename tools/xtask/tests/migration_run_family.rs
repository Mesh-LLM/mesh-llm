#![cfg(unix)]

#[path = "migration_run_family/support.rs"]
mod support;

#[path = "migration_run_family/sequencing.rs"]
mod sequencing;

use self::support::{Fixture, read};
use std::{ffi::OsString, fs, path::PathBuf};

#[test]
fn run_family_executes_only_fixture_for_child_exit_zero_and_twenty_three()
-> Result<(), Box<dyn std::error::Error>> {
    for (child_status, xtask_status) in [(0, 0), (23, 1)] {
        let fixture = Fixture::new()?;
        let output = fixture.run("granite-3.1-2b", "exit", child_status, None)?;
        assert_eq!(output.status.code(), Some(xtask_status));
        assert!(fixture.marker.is_file());
        let stdout = String::from_utf8(output.stdout)?;
        if child_status == 0 {
            assert!(stdout.contains("fixture-child\n"));
            assert!(
                stdout.ends_with(
                    "all\t16\t2\t131072\t32768\t32768\t131072\t5\t2\t4\t2048\t1,2,4,8\n"
                )
            );
        } else {
            assert!(stdout.contains("fixture-child\n"));
            assert!(!stdout.contains("all\t16\t"));
            assert!(String::from_utf8(output.stderr)?.contains("status=Some(23)"));
        }
        assert!(read(&fixture.json)?.contains("\"mode\": \"all\""));
        assert!(read(&fixture.env)?.starts_with("AGENTIC_REPLAY_MODE=all\n"));
        let argv = read(&fixture.argv)?;
        let argv: Vec<_> = argv.lines().collect();
        assert_eq!(
            argv[0],
            fixture
                .root
                .path()
                .join("evals/agentic-replay.py")
                .canonicalize()?
                .to_str()
                .ok_or("script path")?
        );
        let expected = [
            "run",
            "--model",
            "bartowski/granite-3.1-2b-instruct-GGUF@e47b8b46c04cede00f9e19d5a846551b14b2efce/granite-3.1-2b-instruct-Q4_K_M.gguf",
            "--backend",
            "metal",
            "--replay-mode",
            "all",
            "--expected-model-sha256",
            "774269c82fde2720ea18dcf457fb5bd028fe096139a0735f4ad59c0a270cfc9c",
            "--dataset-file",
            "data.parquet",
            "--output",
            "result",
            "--ref",
            "main=HEAD",
            "--sessions-per-concurrency",
            "16",
            "--minimum-worker-waves",
            "2",
            "--minimum-context-tokens",
            "131072",
            "--minimum-session-prompt-tokens",
            "32768",
            "--min-isl",
            "32768",
            "--max-isl",
            "131072",
            "--min-turns",
            "5",
            "--passes",
            "2",
            "--warmup-turns",
            "4",
            "--max-output-tokens",
            "2048",
            "--concurrency",
            "1",
            "--concurrency",
            "2",
            "--concurrency",
            "4",
            "--concurrency",
            "8",
        ];
        assert_eq!(&argv[1..], expected);
    }
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
    assert!(String::from_utf8(output.stderr)?.contains("FamilyCardinality"));
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
    assert!(
        String::from_utf8(output.stdout)?
            .starts_with("all\t16\t2\t131072\t32768\t32768\t131072\t5\t2\t4\t2048\t1,2,4,8\n")
    );
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

#[test]
fn run_family_timeout_and_cancellation_clean_the_owned_fixture_tree()
-> Result<(), Box<dyn std::error::Error>> {
    for mode in ["hang", "cancel"] {
        let fixture = Fixture::new()?;
        let output = fixture.run("granite-3.1-2b", mode, 0, (mode == "hang").then_some(1))?;
        assert_eq!(output.status.code(), Some(1), "{mode}");
        assert!(fixture.marker.is_file(), "{mode}");
        let stderr = String::from_utf8(output.stderr)?;
        assert!(
            stderr.contains("cleanup=Cleanup { complete: true"),
            "{stderr}"
        );
        if mode == "hang" {
            assert!(stderr.contains("outcome=Deadline"), "{stderr}");
        } else {
            assert!(fixture.cleanup_marker_exists(), "{mode}");
            assert!(stderr.contains("run-family cancelled"), "{stderr}");
        }
    }
    Ok(())
}

#[test]
fn run_family_rust_child_fixture() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("REPLAY_RUST_CHILD").is_none() {
        return Ok(());
    }
    let marker = PathBuf::from(std::env::var_os("REPLAY_FIXTURE_MARKER").ok_or("marker path")?);
    let json = PathBuf::from(std::env::var_os("REPLAY_JSON").ok_or("json path")?);
    let env = PathBuf::from(std::env::var_os("REPLAY_ENV").ok_or("env path")?);
    let mode = std::env::var("REPLAY_FIXTURE_MODE")?;
    let exit = std::env::var("REPLAY_FIXTURE_EXIT")?.parse::<i32>()?;
    assert!(json.is_file());
    assert!(env.is_file());
    fs::write(marker, b"started")?;
    match mode.as_str() {
        "exit" => {
            use std::io::Write as _;
            std::io::stdout().write_all(b"fixture-child\n")?;
            std::process::exit(exit);
        }
        "hang" => loop {
            std::thread::sleep(std::time::Duration::from_millis(20));
        },
        "cancel" => {
            let parent = std::env::var("REPLAY_SUPERVISOR_PID")?.parse::<u32>()?;
            let cleanup =
                PathBuf::from(std::env::var_os("REPLAY_CLEANUP_MARKER").ok_or("cleanup path")?);
            let status = std::process::Command::new("/bin/kill")
                .args(["-TERM", &parent.to_string()])
                .status()?;
            if !status.success() {
                return Err(format!("fixture cancellation command failed: {status}").into());
            }
            fs::write(cleanup, b"fixture cancelling")?;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        _ => Err("unknown fixture mode".into()),
    }
}

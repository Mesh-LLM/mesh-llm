use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[derive(Deserialize)]
struct LegacyCases {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    report: String,
    args: Vec<String>,
    code: i32,
    stdout: String,
    stderr: String,
    #[serde(default, rename = "rust_stderr")]
    _previous_rust_stderr: Option<String>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn overlay() -> serde_json::Value {
    serde_json::from_slice(
        &fs::read(root().join(
            "tools/xtask/tests/fixtures/migration/rewriter-report/typed-contract-overlay.json",
        ))
        .unwrap(),
    )
    .unwrap()
}

fn command(binary: &str) -> Command {
    let mut command = Command::new("gtimeout");
    command.args(["-k", "1", "5", binary]);
    command
}

fn retain(case: &Case, output: &Output) {
    if let Some(evidence) = std::env::var_os("REPORT_FROZEN_EVIDENCE") {
        let directory = PathBuf::from(evidence).join(&case.name);
        fs::create_dir(&directory).unwrap();
        for (name, bytes) in [
            ("stdout", output.stdout.clone()),
            ("stderr", output.stderr.clone()),
            (
                "status",
                format!("{:?}\n", output.status.code()).into_bytes(),
            ),
            ("argv.json", serde_json::to_vec(&case.args).unwrap()),
        ] {
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join(name))
                .unwrap()
                .write_all(&bytes)
                .unwrap();
        }
    }
}

#[test]
fn migration_native_fixtures_rewriter_report_matches_frozen_legacy_cli() {
    let root = root();
    let directory = root.join("tools/xtask/tests/fixtures/migration/rewriter-report");
    let frozen: LegacyCases = serde_json::from_slice(
        &fs::read(directory.join("legacy-cli.json")).expect("fixture manifest"),
    )
    .expect("typed fixture manifest");
    let binary = env!("CARGO_BIN_EXE_xtask");
    let mut accepted = 0;
    let mut rejected = 0;
    for case in frozen.cases {
        let path = directory.join(&case.report);
        let output = command(binary)
            .current_dir(&root)
            .args(["automation", "rewriter-report", "--report"])
            .arg(path)
            .args(&case.args)
            .output()
            .expect("xtask CLI");
        retain(&case, &output);
        assert_eq!(
            output.status.code(),
            Some(case.code),
            "{}: exit status",
            case.name
        );
        assert_eq!(
            output.stdout,
            case.stdout.as_bytes(),
            "{}: stdout",
            case.name
        );
        let expected_stderr = if case.name == "malformed" {
            let overlay = overlay();
            assert_eq!(
                fs::read(directory.join(&case.report)).unwrap(),
                overlay["malformed"]["input"].as_str().unwrap().as_bytes()
            );
            overlay["malformed"]["stderr"].as_str().unwrap().to_owned()
        } else {
            case.stderr
        };
        assert_eq!(
            output.stderr,
            expected_stderr.as_bytes(),
            "{}: stderr",
            case.name
        );
        if case.code == 0 {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    assert!(
        accepted > 0 && rejected > 0,
        "positive and negative CLI cases required"
    );
}

#[test]
fn migration_native_fixtures_rewriter_report_invalid_arguments_return_usage_error() {
    let binary = env!("CARGO_BIN_EXE_xtask");
    let report = root().join("tools/xtask/tests/fixtures/migration/rewriter-report/ready.json");
    for (flag, value) in [
        ("--mode", "other"),
        ("--patch-check", "skipped"),
        ("--patch-drift-gate", "skip"),
        ("--compile-result", "unknown"),
        ("--graph-verify-result", "unknown"),
    ] {
        let output = command(binary)
            .args(["automation", "rewriter-report", "--report"])
            .arg(&report)
            .args([flag, value])
            .output()
            .expect("xtask CLI");
        assert_eq!(output.status.code(), Some(2), "{flag}");
        assert!(output.stdout.is_empty(), "{flag}");
        assert!(output.stderr.starts_with(b"usage: "), "{flag}");
    }
}

#[test]
fn migration_native_fixtures_rewriter_report_rejects_wrong_types_and_missing_file() {
    let binary = env!("CARGO_BIN_EXE_xtask");
    let directory = root().join("tools/xtask/tests/fixtures/migration/rewriter-report");
    let wrong = command(binary)
        .args(["automation", "rewriter-report", "--report"])
        .arg(directory.join("bad-types.json"))
        .output()
        .expect("xtask CLI");
    assert_eq!(wrong.status.code(), Some(1));
    assert!(wrong.stdout.is_empty());
    assert_eq!(
        wrong.stderr,
        overlay()["bad-types"]["stderr"]
            .as_str()
            .unwrap()
            .as_bytes()
    );

    let missing = command(binary)
        .args(["automation", "rewriter-report", "--report"])
        .arg(directory.join("absent.json"))
        .output()
        .expect("xtask CLI");
    assert_eq!(missing.status.code(), Some(2));
    assert!(missing.stdout.is_empty());
    assert!(missing.stderr.starts_with(b"error: cannot load report: "));
}

#[test]
fn migration_native_fixtures_rewriter_report_preserves_legacy_decisions() {
    let binary = env!("CARGO_BIN_EXE_xtask");
    let directory = root().join("tools/xtask/tests/fixtures/migration/rewriter-report");
    #[derive(Deserialize)]
    struct Decision {
        name: String,
        mode: String,
        code: i32,
        input: serde_json::Value,
    }
    let cases: Vec<Decision> = serde_json::from_slice(
        &fs::read(directory.join("correction-cases.json")).expect("correction cases"),
    )
    .expect("valid correction cases");
    let temporary =
        std::env::temp_dir().join(format!("xtask-rewriter-report-{}.json", std::process::id()));
    for case in cases {
        fs::write(
            &temporary,
            serde_json::to_vec(&case.input).expect("encode report"),
        )
        .expect("write report");
        let output = command(binary)
            .args(["automation", "rewriter-report", "--report"])
            .arg(&temporary)
            .args(["--mode", &case.mode])
            .output()
            .expect("xtask CLI");
        assert_eq!(
            output.status.code(),
            Some(if case.name == "narrow-idempotence" {
                case.code
            } else {
                1
            }),
            "{}: stdout={} stderr={}",
            case.name,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let contract = overlay();
        let expected = if case.name == "narrow-idempotence" {
            ""
        } else {
            contract["correction-dispositions"][&case.name]
                .as_str()
                .expect("named typed-contract disposition")
        };
        assert_eq!(output.stderr, expected.as_bytes(), "{}", case.name);
    }
    fs::remove_file(&temporary).expect("remove report");
    let ordered = command(binary)
        .args(["automation", "rewriter-report", "--report"])
        .arg(directory.join("ordered-constructors.json"))
        .output()
        .expect("xtask CLI");
    assert_eq!(
        ordered.status.code(),
        Some(1),
        "object constructors are outside the typed contract"
    );
    assert_eq!(
        ordered.stderr,
        overlay()["ordered-constructors"]
            .as_str()
            .unwrap()
            .as_bytes()
    );
    let report = root().join("tools/xtask/tests/fixtures/migration/rewriter-report/ready.json");
    for args in [
        vec!["--help"],
        vec!["-h"],
        vec!["--mode=idempotence"],
        vec!["--mode", "idempotence"],
    ] {
        let output = command(binary)
            .args(["automation", "rewriter-report", "--report"])
            .arg(&report)
            .args(&args)
            .output()
            .expect("xtask CLI");
        assert_eq!(output.status.code(), Some(0), "{args:?}");
    }
}

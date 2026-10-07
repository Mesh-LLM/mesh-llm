//! Planned row identity gates apply to every independently verified artifact.
use crate::support::{TestResult, Tool, cases, execute};
use flate2::{Compression, write::GzEncoder};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};
const TARGET: &str = "x86_64-unknown-linux-gnu";
fn package(root: &Path, id: &str) -> PathBuf {
    let directory = root.join(id);
    fs::create_dir_all(directory.join("lib")).unwrap();
    let bytes = b"bounded runtime fixture";
    fs::write(directory.join("lib/runtime.bin"), bytes).unwrap();
    let digest = hex::encode(Sha256::digest(bytes));
    let manifest = json!({"schema_version":2,"runtime":{"id":id,"release_version":"0.75.0","skippy_abi":"0.1.32","platform":{"os":"linux","arch":"x86_64","target":TARGET},"backend":{"kind":"cpu"},"libraries":["lib/runtime.bin"],"files":{"lib/runtime.bin":digest}},"build":{"primary_library":"lib/runtime.bin","library_sha256":digest}});
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    directory
}
fn archive(root: &Path, directory: &Path) -> PathBuf {
    let name = directory.file_name().unwrap().to_str().unwrap();
    let archive = root.join(format!("{name}.tar.gz"));
    let gzip = GzEncoder::new(fs::File::create(&archive).unwrap(), Compression::default());
    let mut builder = tar::Builder::new(gzip);
    builder.append_dir_all(name, directory).unwrap();
    builder.into_inner().unwrap().finish().unwrap();
    let digest = hex::encode(Sha256::digest(fs::read(&archive).unwrap()));
    fs::write(
        root.join(format!("{name}.tar.gz.sha256")),
        format!("{digest}  {name}.tar.gz\n"),
    )
    .unwrap();
    archive
}
fn run(flags: &[&str], artifacts: &[&Path]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.args(["native", "verify-runtime-package"]);
    command.args(flags);
    for artifact in artifacts {
        command.arg(artifact);
    }
    command.output().unwrap()
}
fn mutate(directory: &Path, operation: impl FnOnce(&mut Value)) {
    let path = directory.join("manifest.json");
    let mut value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    operation(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn exact_row_matches_directory_and_archive_with_existing_integrity_checks() {
    let root = tempfile::tempdir().unwrap();
    let directory = package(root.path(), "linux-cpu");
    let archive = archive(root.path(), &directory);
    let output = run(
        &[
            "--portable",
            "--expected-backend",
            "cpu",
            "--expected-target",
            TARGET,
        ],
        &[&directory, &archive],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .matches("verified portable native runtime artifact")
            .count(),
        2
    );
    fs::write(directory.join("lib/runtime.bin"), b"corrupted library").unwrap();
    let refused = run(
        &[
            "--portable",
            "--expected-backend",
            "cpu",
            "--expected-target",
            TARGET,
        ],
        &[&directory],
    );
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("checksum mismatch"));
}

#[test]
fn row_mismatches_reject_directory_archive_and_later_inputs() {
    for (flag, expected, error) in [
        ("--expected-backend", "cuda", "backend"),
        ("--expected-target", "x86_64-linux-android", "target"),
    ] {
        let root = tempfile::tempdir().unwrap();
        let directory = package(root.path(), "linux-cpu");
        let archive = archive(root.path(), &directory);
        for artifact in [&directory, &archive] {
            let output = run(&["--portable", flag, expected], &[artifact]);
            assert!(!output.status.success());
            assert!(String::from_utf8_lossy(&output.stderr).contains(&format!(
                "cached runtime {error} does not match the planned row"
            )));
        }
    }
    let root = tempfile::tempdir().unwrap();
    let first = package(root.path(), "first");
    let second = package(root.path(), "second");
    mutate(&second, |manifest| {
        manifest["runtime"]["platform"]["target"] = json!("x86_64-linux-android")
    });
    let archived = archive(root.path(), &second);
    for artifact in [&second, &archived] {
        let output = run(
            &[
                "--portable",
                "--expected-backend",
                "cpu",
                "--expected-target",
                TARGET,
            ],
            &[&first, artifact],
        );
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8_lossy(&output.stdout)
                .matches("verified portable native runtime artifact")
                .count(),
            1
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("target does not match"));
    }
}

#[test]
fn planned_row_flags_require_values_and_refuse_duplicates_before_artifact_io() {
    for flags in [
        vec!["--expected-backend"],
        vec!["--expected-target"],
        vec!["--expected-backend", ""],
        vec!["--expected-target", "--portable"],
        vec!["--expected-backend", "cpu", "--expected-backend", "cpu"],
        vec!["--expected-target", TARGET, "--expected-target", TARGET],
    ] {
        let output = run(&flags, &[]);
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("requires a non-empty value") || error.contains("duplicate argument")
        );
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn expected_rows_do_not_supply_missing_or_invalid_manifest_fields() {
    for field in ["backend", "target", "schema"] {
        let root = tempfile::tempdir().unwrap();
        let directory = package(root.path(), "linux-cpu");
        mutate(&directory, |manifest| match field {
            "backend" => {
                manifest["runtime"]
                    .as_object_mut()
                    .unwrap()
                    .remove("backend");
            }
            "target" => {
                manifest["runtime"]["platform"]
                    .as_object_mut()
                    .unwrap()
                    .remove("target");
            }
            "schema" => manifest["schema_version"] = json!(1),
            _ => unreachable!(),
        });
        let output = run(
            &[
                "--portable",
                "--expected-backend",
                "cpu",
                "--expected-target",
                TARGET,
            ],
            &[&directory],
        );
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn matching_native_row_preserves_platform_probes_and_mismatch_stops_before_probes() -> TestResult {
    let mut case = cases(Tool::RuntimePackage, "happy")?
        .into_iter()
        .find(|case| case["name"] == "macos-valid-rpath")
        .ok_or("native probe fixture absent")?;
    let original = execute(Tool::RuntimePackage, &case)?;
    assert_eq!(original.code, Some(0));
    assert!(!original.calls.is_empty());
    case["args"] = json!([
        "--expected-backend",
        "cpu",
        "--expected-target",
        "x86_64-apple-darwin",
        "meshllm-native-runtime-darwin-x86_64-cpu"
    ]);
    let matched = execute(Tool::RuntimePackage, &case)?;
    assert_eq!(matched, original);
    case["args"][1] = json!("metal");
    let mismatch = execute(Tool::RuntimePackage, &case)?;
    assert_eq!(mismatch.code, Some(1));
    assert!(mismatch.calls.is_empty());
    assert!(
        mismatch
            .stderr
            .contains("backend does not match the planned row")
    );
    Ok(())
}

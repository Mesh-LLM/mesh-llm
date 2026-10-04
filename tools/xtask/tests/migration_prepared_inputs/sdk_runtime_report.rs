//! The SDK/product consumer of `mesh-llm runtime list --available --json`
//! (`migration_lifecycle::sdk_json_consumer`): select exactly one prepared adjacent
//! runtime or fail without building one.

use crate::support::{Case, Legacy, Scratch, TestResult, assert_output, snapshot};
use serde_json::{Value, json};
use std::path::Path;

const SELECT: Legacy = Legacy::Heredoc("scripts/ci-prepare-native-runtime.sh", 2);
const ABI: &str = "1.2.3";
fn rejected(output: &std::process::Output, diagnostic: &str) {
    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stdout.is_empty(),
        "rejection must not select an artifact"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(diagnostic),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn row(id: &str, backend: &str, supported: bool) -> Value {
    json!({"id": id, "backend": backend, "supported": supported})
}

/// A bundle root with one prepared runtime directory per `(dir, id, abi)`.
fn bundle(scratch: &Scratch, runtimes: &[(&str, &str, Value)]) -> TestResult {
    for (dir, id, abi) in runtimes {
        let manifest = json!({"runtime": {"id": id, "skippy_abi": abi}});
        let relative = if dir.is_empty() {
            String::new()
        } else {
            format!("{dir}/")
        };
        scratch.write(
            &format!("bundle/{relative}manifest.json"),
            &serde_json::to_vec(&manifest)?,
        )?;
    }
    if runtimes.is_empty() {
        std::fs::create_dir_all(scratch.join("bundle"))?;
    }
    Ok(())
}

fn select(
    scratch: &Scratch,
    backend: &str,
    report: &[u8],
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let report_path = scratch.write("report.json", report)?;
    let root = scratch.join("bundle");
    let args = [
        root.to_str().ok_or("path")?,
        backend,
        report_path.to_str().ok_or("path")?,
        ABI,
    ];
    Case::same(&["sdk-runtime-select"], &args, SELECT).run(scratch.path())
}

fn select_json(
    scratch: &Scratch,
    backend: &str,
    report: &Value,
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    select(scratch, backend, &serde_json::to_vec(report)?)
}

fn selected(dir: &Path) -> String {
    format!("{}\n", dir.display())
}

#[test]
fn migration_prepared_inputs_sdk_runtime_accepts_catalog_and_flat_reports() -> TestResult {
    let rows = json!([row("linux-cpu", "cpu", true)]);
    for report in [json!({"catalogs": {}, "runtimes": rows}), rows.clone()] {
        let scratch = Scratch::new("sdk-runtime")?;
        bundle(&scratch, &[("linux-cpu", "linux-cpu", json!(ABI))])?;
        scratch.write("report.json", &serde_json::to_vec(&report)?)?;
        let before = snapshot(scratch.path())?;
        let output = select_json(&scratch, "cpu", &report)?;
        assert_output(&output, 0, &selected(&scratch.join("bundle/linux-cpu")), "");
        assert_eq!(
            snapshot(scratch.path())?,
            before,
            "consumer must not build or modify runtimes"
        );
    }
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_runtime_prefers_aliased_backend() -> TestResult {
    for (alias, backend) in [("hip", "rocm"), ("cuda-blackwell", "cuda")] {
        let scratch = Scratch::new("sdk-runtime-alias")?;
        bundle(
            &scratch,
            &[
                ("cpu", "linux-cpu", json!(ABI)),
                ("accelerator", "linux-accelerator", json!(ABI)),
            ],
        )?;
        let report = json!([
            row("linux-cpu", "cpu", true),
            row("linux-accelerator", backend, true)
        ]);
        let output = select_json(&scratch, alias, &report)?;
        assert_output(
            &output,
            0,
            &selected(&scratch.join("bundle/accelerator")),
            "",
        );
    }
    let scratch = Scratch::new("sdk-runtime-sole-accelerator")?;
    bundle(&scratch, &[("", "linux-vulkan", json!(ABI))])?;
    let report = json!([row("linux-vulkan", "vulkan", true)]);
    assert_output(
        &select_json(&scratch, "cpu", &report)?,
        0,
        &selected(&scratch.join("bundle")),
        "",
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_runtime_rejects_malformed_reports() -> TestResult {
    for report in [
        json!({"runtimes": {}}),
        json!({"catalogs": []}),
        json!("runtimes"),
        json!([row("linux-cpu", "cpu", true), "linux-cpu"]),
        json!([{"id":null,"backend":"cpu","supported":true}]),
        json!([{"id":"linux-cpu","backend":7,"supported":true}]),
        json!([{"id":"linux-cpu","backend":"cpu","supported":1}]),
        json!([{"id":"linux-cpu","backend":"cpu"}]),
        // Unsupported rows still belong to the producer's typed report.
        json!([row("linux-cpu","cpu",true),{"id":null,"backend":"cpu","supported":false}]),
    ] {
        let scratch = Scratch::new("sdk-runtime-malformed")?;
        bundle(&scratch, &[("linux-cpu", "linux-cpu", json!(ABI))])?;
        rejected(
            &select_json(&scratch, "cpu", &report)?,
            "containing typed runtime rows",
        );
    }
    let scratch = Scratch::new("sdk-runtime-json")?;
    bundle(&scratch, &[("linux-cpu", "linux-cpu", json!(ABI))])?;
    let report_path = scratch.write("report.json", b"{\"runtimes\": [")?;
    let root = scratch.join("bundle");
    let args = [
        root.to_str().ok_or("path")?,
        "cpu",
        report_path.to_str().ok_or("path")?,
        ABI,
    ];
    let output = Case::same(&["sdk-runtime-select"], &args, SELECT)
        .status_only()
        .run(scratch.path())?;
    rejected(&output, "containing typed runtime rows");
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_runtime_requires_exactly_one_supported_row() -> TestResult {
    let prefix =
        "expected exactly one compatible adjacent native runtime (preferred backend cpu); found";
    let two = json!([
        row("linux-vulkan", "vulkan", true),
        row("linux-rocm", "rocm", true),
    ]);
    for (report, found) in [
        (json!([row("linux-cpu", "cpu", false)]), "none"),
        (two, "linux-vulkan:vulkan, linux-rocm:rocm"),
        (
            json!([row("cpu-one", "cpu", true), row("cpu-two", "cpu", true)]),
            "cpu-one:cpu, cpu-two:cpu",
        ),
    ] {
        let scratch = Scratch::new("sdk-runtime-count")?;
        bundle(&scratch, &[("linux-cpu", "linux-cpu", json!(ABI))])?;
        let expected = format!("{prefix} {found}\n");
        rejected(&select_json(&scratch, "cpu", &report)?, expected.trim_end());
    }
    let scratch = Scratch::new("sdk-runtime-id")?;
    bundle(&scratch, &[])?;
    let output = select_json(&scratch, "cpu", &json!([row(" \t", "cpu", true)]))?;
    rejected(&output, "require nonempty id and backend strings");
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_runtime_rejects_abi_drift_and_missing_directory() -> TestResult {
    let report = json!([row("linux-cpu", "cpu", true)]);
    for (abi, message) in [
        (
            json!("9.9.9"),
            format!("has Skippy ABI 9.9.9, expected {ABI}"),
        ),
        (Value::Null, "native runtime manifest is invalid".into()),
    ] {
        let scratch = Scratch::new("sdk-runtime-abi")?;
        bundle(&scratch, &[("linux-cpu", "linux-cpu", abi)])?;
        rejected(&select_json(&scratch, "cpu", &report)?, &message);
    }
    let scratch = Scratch::new("sdk-runtime-none")?;
    bundle(&scratch, &[("other", "linux-other", json!(ABI))])?;
    rejected(
        &select_json(&scratch, "cpu", &report)?,
        "expected one adjacent artifact directory for runtime linux-cpu; found 0",
    );
    let scratch = Scratch::new("sdk-runtime-two")?;
    bundle(
        &scratch,
        &[
            ("", "linux-cpu", json!(ABI)),
            ("b", "linux-cpu", json!(ABI)),
        ],
    )?;
    rejected(
        &select_json(&scratch, "cpu", &report)?,
        "expected one adjacent artifact directory for runtime linux-cpu; found 2",
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn migration_prepared_inputs_sdk_runtime_rejects_symlink_escape() -> TestResult {
    let scratch = Scratch::new("sdk-runtime-link")?;
    bundle(&scratch, &[])?;
    let manifest = json!({"runtime": {"id": "linux-cpu", "skippy_abi": ABI}});
    scratch.write("outside/manifest.json", &serde_json::to_vec(&manifest)?)?;
    crate::support::symlink(&scratch.join("outside"), &scratch.join("bundle/linked"))?;
    let expected = format!(
        "selected native runtime escapes adjacent bundle root: {}\n",
        scratch.join("outside").display()
    );
    let output = select_json(&scratch, "cpu", &json!([row("linux-cpu", "cpu", true)]))?;
    rejected(&output, expected.trim_end());
    Ok(())
}

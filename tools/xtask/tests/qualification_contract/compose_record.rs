//! `product compose --qualification-receipts`: the host record binds into the
//! v2 manifest, and recomposition auto-discovers the bundled record.
#![cfg(unix)]

use super::support::{self, stderr};
use serde_json::{Value, json};
use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;

fn bundle_fixture(bundle: &Path) {
    std::fs::create_dir_all(bundle.join("runtime")).unwrap();
    let host = bundle.join("mesh-llm");
    std::fs::write(
        &host,
        "#!/bin/sh\n[ \"$*\" = \"--log-format json --print-build-contract\" ] || exit 9\nprintf '%s\\n' '{\"schema_version\":1,\"product_version\":\"2.0.0\",\"runtime_release\":\"1.0.0\",\"skippy_abi\":\"7\"}'\n",
    )
    .unwrap();
    std::fs::set_permissions(&host, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::write(
        bundle.join("runtime/manifest.json"),
        br#"{"schema_version":2,"runtime":{"id":"runtime-fixture","release_version":"1.0.0","skippy_abi":"7","backend":{"kind":"cpu"}}}"#,
    )
    .unwrap();
}

fn record() -> Value {
    json!({
        "schema_version": 1,
        "source_sha": support::SOURCE_SHA,
        "plan_digest": support::PLAN_DIGEST,
        "platform": "linux",
        "required_row": "linux-cpu",
        "receipts": [{
            "row_id": "linux-cpu",
            "status": "qualified",
            "receipt_sha256": "c".repeat(64),
        }],
    })
}

fn compose(bundle: &Path, extra: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "product",
            "compose",
            "--bundle",
            bundle.to_str().unwrap(),
            "--host",
            bundle.join("mesh-llm").to_str().unwrap(),
            "--runtime",
            bundle.join("runtime").to_str().unwrap(),
            "--version",
            "v2.0.0",
            "--backend",
            "cpu",
        ])
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn explicit_and_discovered_records_bind_identically() {
    let temp = tempfile::tempdir().unwrap();
    let bundle = temp.path().join("bundle");
    bundle_fixture(&bundle);
    support::write_json(
        &bundle.join("skippy-qualification-receipts.json"),
        &record(),
    );
    let explicit = compose(
        &bundle,
        &[
            "--qualification-receipts",
            bundle
                .join("skippy-qualification-receipts.json")
                .to_str()
                .unwrap(),
        ],
    );
    assert!(explicit.status.success(), "{}", stderr(&explicit));
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(bundle.join("product-manifest.json")).unwrap())
            .unwrap();
    let qualification = &manifest["skippy_qualification"];
    assert_eq!(qualification["schema_version"], 1);
    assert_eq!(
        qualification["record_path"],
        "skippy-qualification-receipts.json"
    );
    assert_eq!(qualification["platform"], "linux");
    assert_eq!(qualification["receipts"][0]["row_id"], "linux-cpu");

    // Recomposition without the flag auto-discovers the bundled record.
    std::fs::remove_file(bundle.join("product-manifest.json")).unwrap();
    let discovered = compose(&bundle, &[]);
    assert!(discovered.status.success(), "{}", stderr(&discovered));
    let remanifested: Value =
        serde_json::from_slice(&std::fs::read(bundle.join("product-manifest.json")).unwrap())
            .unwrap();
    assert_eq!(remanifested, manifest);

    // `--check` validates against the discovered record.
    let check = compose(&bundle, &["--check"]);
    assert!(check.status.success(), "{}", stderr(&check));
}

#[test]
fn an_unqualified_required_row_record_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let bundle = temp.path().join("bundle");
    bundle_fixture(&bundle);
    let mut broken = record();
    broken["receipts"][0]["status"] = json!("hardware-unavailable");
    support::write_json(&bundle.join("skippy-qualification-receipts.json"), &broken);
    let result = compose(&bundle, &[]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("not qualified"),
        "{}",
        stderr(&result)
    );
    assert!(!bundle.join("product-manifest.json").exists());
}

#[test]
fn a_qualified_accelerator_cannot_replace_the_mandatory_cpu_row() {
    let temp = tempfile::tempdir().unwrap();
    let bundle = temp.path().join("bundle");
    bundle_fixture(&bundle);
    let mut broken = record();
    broken["required_row"] = json!("linux-cuda");
    broken["receipts"][0]["row_id"] = json!("linux-cuda");
    support::write_json(&bundle.join("skippy-qualification-receipts.json"), &broken);
    let result = compose(&bundle, &[]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("mandatory row"),
        "{}",
        stderr(&result)
    );
    assert!(!bundle.join("product-manifest.json").exists());
}

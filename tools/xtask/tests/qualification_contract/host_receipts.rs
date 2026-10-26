//! `qualify host-receipts`: the host-side provenance record.
use super::support::{self, Checkout, stderr, write_json, xtool};
use serde_json::{Value, json};
use std::path::Path;

fn receipt(row_id: &str, status: &str) -> Value {
    json!({
        "schema_version": 1,
        "source_sha": support::SOURCE_SHA,
        "plan_digest": support::PLAN_DIGEST,
        "row_id": row_id,
        "status": status,
    })
}

fn host_receipts(root: &Path, receipts: &Path, platform: &str) -> std::process::Output {
    xtool(
        root,
        &[
            "qualify",
            "host-receipts",
            "--receipts",
            receipts.to_str().unwrap(),
            "--source-sha",
            support::SOURCE_SHA,
            "--plan-digest",
            support::PLAN_DIGEST,
            "--platform",
            platform,
            "--output",
            root.join("skippy-qualification-receipts.json")
                .to_str()
                .unwrap(),
        ],
    )
}

#[test]
fn accepts_a_qualified_required_row_alongside_an_unavailable_accelerator() {
    let checkout = Checkout::new();
    let receipts = checkout.root().join("receipts/nested");
    std::fs::create_dir_all(&receipts).unwrap();
    write_json(
        &receipts.join("linux-cpu.json"),
        &receipt("linux-cpu", "qualified"),
    );
    write_json(
        &receipts.join("linux-cuda.json"),
        &receipt("linux-cuda", "hardware-unavailable"),
    );
    let result = host_receipts(checkout.root(), &checkout.root().join("receipts"), "linux");
    assert!(result.status.success(), "{}", stderr(&result));
    let record: Value = serde_json::from_slice(
        &std::fs::read(checkout.root().join("skippy-qualification-receipts.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["platform"], "linux");
    assert_eq!(record["required_row"], "linux-cpu");
    let rows: Vec<&str> = record["receipts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["row_id"].as_str().unwrap())
        .collect();
    assert_eq!(rows, ["linux-cpu", "linux-cuda"]);
    assert!(
        record["receipts"][0]["receipt_sha256"]
            .as_str()
            .unwrap()
            .len()
            == 64
    );
}

#[test]
fn rejects_an_empty_receipt_set() {
    let checkout = Checkout::new();
    let receipts = checkout.root().join("empty");
    std::fs::create_dir_all(&receipts).unwrap();
    let result = host_receipts(checkout.root(), &receipts, "linux");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("no standalone qualification receipts"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn rejects_foreign_source_plan_and_platform() {
    for (label, row_id, foreign) in [
        ("foreign source", "linux-cpu", "source_sha"),
        ("foreign plan", "linux-cpu", "plan_digest"),
    ] {
        let checkout = Checkout::new();
        let receipts = checkout.root().join("receipts");
        std::fs::create_dir_all(&receipts).unwrap();
        let mut value = receipt(row_id, "qualified");
        value[foreign] = json!(if foreign == "source_sha" {
            "f".repeat(40)
        } else {
            "f".repeat(64)
        });
        write_json(&receipts.join("r.json"), &value);
        let result = host_receipts(checkout.root(), &receipts, "linux");
        assert!(!result.status.success(), "{label}");
    }
    // A receipt from another platform is foreign to the host platform.
    let checkout = Checkout::new();
    let receipts = checkout.root().join("receipts");
    std::fs::create_dir_all(&receipts).unwrap();
    write_json(&receipts.join("a.json"), &receipt("linux-cpu", "qualified"));
    write_json(
        &receipts.join("b.json"),
        &receipt("windows-cpu", "qualified"),
    );
    let result = host_receipts(checkout.root(), &receipts, "linux");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("another platform"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn rejects_an_unregistered_row_and_unknown_status() {
    let checkout = Checkout::new();
    let receipts = checkout.root().join("receipts");
    std::fs::create_dir_all(&receipts).unwrap();
    write_json(&receipts.join("a.json"), &receipt("linux-cpu", "qualified"));
    write_json(
        &receipts.join("b.json"),
        &receipt("linux-mystery", "qualified"),
    );
    let result = host_receipts(checkout.root(), &receipts, "linux");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("unknown core row"),
        "{}",
        stderr(&result)
    );

    let checkout = Checkout::new();
    let receipts = checkout.root().join("receipts");
    std::fs::create_dir_all(&receipts).unwrap();
    write_json(&receipts.join("a.json"), &receipt("linux-cpu", "qualified"));
    write_json(&receipts.join("b.json"), &receipt("linux-cuda", "green"));
    let result = host_receipts(checkout.root(), &receipts, "linux");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("invalid qualification status"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn rejects_duplicate_rows() {
    let checkout = Checkout::new();
    let receipts = checkout.root().join("receipts");
    std::fs::create_dir_all(&receipts).unwrap();
    write_json(&receipts.join("a.json"), &receipt("linux-cpu", "qualified"));
    write_json(&receipts.join("b.json"), &receipt("linux-cpu", "qualified"));
    let result = host_receipts(checkout.root(), &receipts, "linux");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("duplicate receipt"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn requires_a_qualified_mandatory_row() {
    // Missing mandatory row.
    let checkout = Checkout::new();
    let receipts = checkout.root().join("receipts");
    std::fs::create_dir_all(&receipts).unwrap();
    write_json(
        &receipts.join("a.json"),
        &receipt("linux-cuda", "qualified"),
    );
    let result = host_receipts(checkout.root(), &receipts, "linux");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("produced no receipt"),
        "{}",
        stderr(&result)
    );
    // Unqualified mandatory row.
    let checkout = Checkout::new();
    let receipts = checkout.root().join("receipts");
    std::fs::create_dir_all(&receipts).unwrap();
    write_json(
        &receipts.join("a.json"),
        &receipt("macos-metal", "hardware-unavailable"),
    );
    let result = host_receipts(checkout.root(), &receipts, "macos");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("is not qualified"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn rejects_an_unknown_platform() {
    let checkout = Checkout::new();
    let receipts = checkout.root().join("receipts");
    std::fs::create_dir_all(&receipts).unwrap();
    write_json(&receipts.join("a.json"), &receipt("linux-cpu", "qualified"));
    let result = host_receipts(checkout.root(), &receipts, "plan9");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("unknown platform"),
        "{}",
        stderr(&result)
    );
}

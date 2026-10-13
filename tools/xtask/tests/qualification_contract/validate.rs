//! `qualify assemble` + `qualify validate`: the receipt happy path and the
//! fail-closed negative battery.

use super::support::{
    self, Checkout, Product, stderr, write_availability, write_evidence, write_hardware,
    write_json, xtool,
};
use serde_json::Value;
use std::path::Path;

struct Row {
    checkout: Checkout,
    product: Product,
    staging: tempfile::TempDir,
    row_id: String,
}

impl Row {
    /// A validated row fixture: catalog, registry, product, availability,
    /// evidence, and (for available rows) hardware.
    fn new(row_id: &str, state: &str) -> Self {
        let checkout = Checkout::new();
        checkout.write_full_catalog();
        checkout.write_registry();
        let product = Product::new(row_id);
        let staging = tempfile::tempdir().unwrap();
        write_availability(staging.path(), row_id, state);
        write_evidence(
            &staging.path().join("evidence"),
            row_id,
            &product.manifest_digest(),
        );
        write_hardware(staging.path(), row_id);
        Self {
            checkout,
            product,
            staging,
            row_id: row_id.to_owned(),
        }
    }

    fn assemble(&self) -> std::process::Output {
        let root = self.checkout.root();
        let receipt = self.staging.path().join("receipt.json");
        let mut args = vec![
            "qualify".to_owned(),
            "assemble".to_owned(),
            "--product-dir".to_owned(),
            self.product.dir.to_string_lossy().into_owned(),
            "--availability".to_owned(),
            self.staging
                .path()
                .join("availability.json")
                .to_string_lossy()
                .into_owned(),
            "--evidence-dir".to_owned(),
            self.staging
                .path()
                .join("evidence")
                .to_string_lossy()
                .into_owned(),
            "--source-sha".to_owned(),
            support::SOURCE_SHA.to_owned(),
            "--plan-digest".to_owned(),
            support::PLAN_DIGEST.to_owned(),
            "--row-id".to_owned(),
            self.row_id.clone(),
            "--output".to_owned(),
            receipt.to_string_lossy().into_owned(),
        ];
        let availability: Value = serde_json::from_slice(
            &std::fs::read(self.staging.path().join("availability.json")).unwrap(),
        )
        .unwrap();
        if availability["state"] == "available" {
            args.extend([
                "--hardware".to_owned(),
                self.staging
                    .path()
                    .join("hardware.json")
                    .to_string_lossy()
                    .into_owned(),
            ]);
        }
        xtool(root, &args.iter().map(String::as_str).collect::<Vec<_>>())
    }

    fn validate(&self) -> std::process::Output {
        let root = self.checkout.root();
        xtool(
            root,
            &[
                "qualify",
                "validate",
                "--receipt",
                self.staging.path().join("receipt.json").to_str().unwrap(),
                "--product-manifest",
                self.product.manifest_path().to_str().unwrap(),
                "--product-dir",
                self.product.dir.to_str().unwrap(),
                "--availability",
                self.staging
                    .path()
                    .join("availability.json")
                    .to_str()
                    .unwrap(),
                "--evidence-dir",
                self.staging.path().join("evidence").to_str().unwrap(),
                "--source-sha",
                support::SOURCE_SHA,
                "--plan-digest",
                support::PLAN_DIGEST,
                "--row-id",
                &self.row_id,
            ],
        )
    }

    fn receipt(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.staging.path().join("receipt.json")).unwrap())
            .unwrap()
    }

    fn write_receipt(&self, receipt: &Value) {
        write_json(&self.staging.path().join("receipt.json"), receipt);
    }

    fn rewrite_availability(&self, state: &str) {
        let (path, _) = write_availability(self.staging.path(), &self.row_id, state);
        // Keep the receipt's availability digest bound to the new bytes.
        if self.staging.path().join("receipt.json").is_file() {
            let mut receipt = self.receipt();
            receipt["availability_sha256"] = Value::String(support::file_sha256(&path));
            self.write_receipt(&receipt);
        }
    }
}

#[test]
fn qualified_row_assembles_and_validates() {
    let row = Row::new("linux-cpu", "available");
    let assembled = row.assemble();
    assert!(assembled.status.success(), "{}", stderr(&assembled));
    let digest = String::from_utf8_lossy(&assembled.stdout);
    assert!(
        digest.trim().chars().all(|c| c.is_ascii_hexdigit()),
        "{digest}"
    );
    let validated = row.validate();
    assert!(validated.status.success(), "{}", stderr(&validated));
    assert_eq!(
        String::from_utf8_lossy(&validated.stdout).trim(),
        digest.trim()
    );
}

#[test]
fn unavailable_accelerator_row_assembles_with_packaging_only() {
    let row = Row::new("windows-cuda", "hardware-unavailable");
    let assembled = row.assemble();
    assert!(assembled.status.success(), "{}", stderr(&assembled));
    let validated = row.validate();
    assert!(validated.status.success(), "{}", stderr(&validated));
    let receipt = row.receipt();
    assert_eq!(receipt["status"], "hardware-unavailable");
    assert_eq!(
        receipt["suites"]["dense"],
        serde_json::json!({"status": "not-executed"})
    );
    assert_eq!(receipt["suites"]["packaging-runtime"]["status"], "passed");
}

#[test]
fn tampered_product_bytes_fail_even_when_manifest_is_unchanged() {
    let row = Row::new("linux-cpu", "available");
    assert!(row.assemble().status.success());
    std::fs::write(row.product.dir.join("skippy"), b"tampered\n").unwrap();
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("CLI bytes differ"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn changed_import_report_fails_even_when_manifest_is_unchanged() {
    let row = Row::new("linux-cpu", "available");
    assert!(row.assemble().status.success());
    std::fs::write(row.product.dir.join("host-imports.json"), b"{\"x\":1}\n").unwrap();
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("import report bytes differ"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn changed_runtime_bytes_fail_even_when_manifest_is_unchanged() {
    let row = Row::new("linux-cpu", "available");
    assert!(row.assemble().status.success());
    std::fs::write(
        row.product
            .dir
            .join("native-runtimes/runtime-1/lib/libskippy.so"),
        b"tampered\n",
    )
    .unwrap();
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("runtime bytes differ"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn duplicate_json_key_cannot_hide_a_failed_suite() {
    let row = Row::new("linux-cpu", "available");
    assert!(row.assemble().status.success());
    let receipt = std::fs::read_to_string(row.staging.path().join("receipt.json")).unwrap();
    let duplicated = receipt.replacen(
        "\"status\": \"qualified\"",
        "\"status\": \"hardware-unavailable\", \"status\": \"qualified\"",
        1,
    );
    std::fs::write(row.staging.path().join("receipt.json"), duplicated).unwrap();
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("duplicate JSON key"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn failed_evidence_cannot_be_labeled_as_passed_in_receipt() {
    let row = Row::new("linux-cpu", "available");
    assert!(row.assemble().status.success());
    let evidence_path = row.staging.path().join("evidence/packaging-runtime.json");
    let mut evidence: Value =
        serde_json::from_slice(&std::fs::read(&evidence_path).unwrap()).unwrap();
    evidence["status"] = Value::String("failed".into());
    write_json(&evidence_path, &evidence);
    // The stale receipt digest fails first...
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("evidence digest differs"),
        "{}",
        stderr(&result)
    );
    // ...and a receipt rebound to the failed bytes is still rejected.
    let mut receipt = row.receipt();
    receipt["suites"]["packaging-runtime"]["evidence_sha256"] =
        Value::String(support::file_sha256(&evidence_path));
    row.write_receipt(&receipt);
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("evidence did not pass"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn escaping_product_path_is_rejected() {
    let row = Row::new("linux-cpu", "available");
    assert!(row.assemble().status.success());
    let manifest_path = row.product.manifest_path();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["cli"]["path"] = Value::String("../escape".into());
    write_json(&manifest_path, &manifest);
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("escapes bundle"),
        "{}",
        stderr(&result)
    );
}

#[cfg(unix)]
#[test]
fn symlinked_runtime_parent_is_rejected() {
    let row = Row::new("linux-cpu", "available");
    assert!(row.assemble().status.success());
    let runtime = row.product.dir.join("native-runtimes/runtime-1");
    let target = row.staging.path().join("runtime-elsewhere");
    std::fs::rename(&runtime, &target).unwrap();
    std::os::unix::fs::symlink(&target, &runtime).unwrap();
    let result = row.validate();
    assert!(!result.status.success());
    assert!(stderr(&result).contains("symlink"), "{}", stderr(&result));
}

#[test]
fn unavailable_row_cannot_claim_execution() {
    let row = Row::new("linux-cuda", "available");
    assert!(row.assemble().status.success());
    // Re-label the row unavailable while leaving executed suites in place.
    row.rewrite_availability("hardware-unavailable");
    let mut receipt = row.receipt();
    receipt["status"] = Value::String("hardware-unavailable".into());
    receipt["hardware"] = Value::Null;
    row.write_receipt(&receipt);
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("claims execution on unavailable hardware"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn available_row_cannot_claim_unavailable() {
    let row = Row::new("linux-cpu", "available");
    assert!(row.assemble().status.success());
    let mut receipt = row.receipt();
    receipt["status"] = Value::String("hardware-unavailable".into());
    receipt["hardware"] = Value::Null;
    row.write_receipt(&receipt);
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("available row must be qualified"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn metal_row_cannot_be_downgraded_to_unavailable() {
    let row = Row::new("macos-metal", "hardware-unavailable");
    let assembled = row.assemble();
    assert!(!assembled.status.success());
    assert!(
        stderr(&assembled).contains("required available row cannot be hardware-unavailable"),
        "{}",
        stderr(&assembled)
    );
}

#[test]
fn cpu_fallback_cannot_satisfy_a_gpu_row() {
    let row = Row::new("linux-cuda", "available");
    assert!(row.assemble().status.success());
    let mut receipt = row.receipt();
    receipt["hardware"]["device"] = Value::String("CPU".into());
    receipt["hardware"]["selected_device"] = Value::String("CUDA0".into());
    row.write_receipt(&receipt);
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("CPU device"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn software_renderer_cannot_qualify_a_gpu_row() {
    let row = Row::new("linux-vulkan", "available");
    assert!(row.assemble().status.success());
    let mut receipt = row.receipt();
    receipt["hardware"]["device"] = Value::String("llvmpipe (LLVM 17)".into());
    row.write_receipt(&receipt);
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("software renderer"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn metal_downgrade_of_selected_device_is_rejected() {
    let row = Row::new("macos-metal", "available");
    assert!(row.assemble().status.success());
    let mut receipt = row.receipt();
    receipt["hardware"]["selected_device"] = Value::String("CPU".into());
    row.write_receipt(&receipt);
    let result = row.validate();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("requested backend device"),
        "{}",
        stderr(&result)
    );
}

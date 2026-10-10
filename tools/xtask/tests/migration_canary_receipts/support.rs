use crate::canary_receipts::{
    Digest, Family, ReceiptContext, SourceFamilyPlan, TestPackageInputs, WorkflowRun,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

pub const PLAN: &[u8] = include_bytes!("fixtures/plan.json");
pub const IDENTITY: &[u8] = include_bytes!("fixtures/identity.json");
pub const DENSE: &[u8] = include_bytes!("fixtures/dense.jsonl");
pub const HYBRID: &[u8] = include_bytes!("fixtures/hybrid.jsonl");
pub const PRETTY: &[u8] = include_bytes!("fixtures/pretty.jsonl");
pub const BATTERY: &[u8] = include_bytes!("fixtures/battery.jsonl");
pub const EMBEDDING: &[u8] = include_bytes!("fixtures/embedding.jsonl");
pub const PROJECTOR: &[u8] = include_bytes!("fixtures/projector.jsonl");

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub struct Fixture(pub PathBuf);

impl Fixture {
    pub fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "canary-receipts-{}-{stamp}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }

    pub fn complete() -> Self {
        let fixture = Self::new();
        fixture.put("dense", "dense", "2", "success");
        fixture.put("hybrid", "hybrid", "2", "success");
        fixture
    }

    pub fn put(&self, directory: &str, family: &str, attempt: &str, outcome: &str) {
        let root = self.0.join(directory);
        fs::create_dir(&root).unwrap();
        let results = if family == "hybrid" { HYBRID } else { DENSE };
        fs::write(root.join("results.jsonl"), results).unwrap();
        let receipt = json!({
            "candidate": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "family": family,
            "identity_sha256": Digest::of_bytes(IDENTITY).as_str(), "outcome": outcome,
            "pass_id": "repair-1", "results_sha256": Digest::of_bytes(results).as_str(),
            "run_attempt": attempt, "run_id": "123", "runner": "fixture",
        });
        fs::write(
            root.join("receipt.json"),
            serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
    }

    pub fn edit(&self, directory: &str, field: &str, value: Value) {
        let path = self.0.join(directory).join("receipt.json");
        let mut receipt: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        receipt[field] = value;
        fs::write(path, serde_json::to_vec(&receipt).unwrap()).unwrap();
    }

    pub fn results(&self, directory: &str) -> PathBuf {
        self.0.join(directory).join("results.jsonl")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test-owned evidence only");
    }
}

pub fn context() -> ReceiptContext {
    context_with(PLAN, "4")
}

pub fn context_with(plan: &[u8], current: &str) -> ReceiptContext {
    ReceiptContext::from_test_package(
        TestPackageInputs {
            identity: serde_json::from_slice(IDENTITY).unwrap(),
            identity_sha256: Digest::of_bytes(IDENTITY),
            plan: SourceFamilyPlan::parse(plan).unwrap(),
        },
        WorkflowRun {
            run_id: "123".to_owned(),
            run_attempt: current.to_owned().try_into().unwrap(),
        },
    )
    .unwrap()
}

pub fn family(value: &str) -> Family {
    value.to_owned().try_into().unwrap()
}

pub fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

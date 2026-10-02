use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    blob: PathBuf,
}
fn gguf() -> Vec<u8> {
    fn string(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(3u64.to_le_bytes());
    string(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    string(&mut b, "llama");
    for (key, value) in [
        ("llama.block_count", 16u64),
        ("llama.embedding_length", 2048),
    ] {
        string(&mut b, key);
        b.extend(10u32.to_le_bytes());
        b.extend(value.to_le_bytes());
    }
    b
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        fs::create_dir_all(root.join("ci/llama-canary")).unwrap();
        fs::create_dir_all(root.join("tools/xtask")).unwrap();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::write(root.join("tools/xtask/Cargo.toml"), "[package]\n").unwrap();
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source = fs::read_to_string(repo.join("scripts/llama-canary-agent-repair.sh")).unwrap();
        let selection = source
            .split("# Legacy workload automation selection begins.\n")
            .nth(1)
            .unwrap()
            .split("# Legacy workload automation selection ends.")
            .next()
            .unwrap();
        let helpers = source
            .split("repair_family_plan_step() {")
            .nth(1)
            .unwrap()
            .split("\nremaining_verification_seconds()")
            .next()
            .unwrap();
        fs::write(root.join("adapter.sh"),format!("set -euo pipefail\n{selection}\nrepair_family_plan_step() {{{helpers}\nrun_verification_logged() {{ local label=\"$1\" log=\"$2\"; shift 2; printf '%s\\n' \"$label\" >> \"$log\"; \"$@\"; }}\n")).unwrap();
        fs::copy(env!("CARGO_BIN_EXE_xtask"), root.join("controller")).unwrap();
        let mut manifest: Json = serde_json::from_slice(
            &fs::read(repo.join("ci/llama-canary/family-certified.json")).unwrap(),
        )
        .unwrap();
        let mut row = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["family"] == "llama")
            .unwrap()
            .clone();
        let bytes = gguf();
        let digest = hex::encode(Sha256::digest(&bytes));
        let revision = "a".repeat(40);
        row["artifact"] = json!({"repo":"fixture/repair","revision":revision,"files":["nested/model.gguf"],"selector":"Q4_K_M","file_integrity":{"nested/model.gguf":{"size_bytes":bytes.len(),"blob_id":digest}}});
        let mut second = row.clone();
        second["family"] = json!("llama-repair-fixture");
        manifest["models"] = json!([row, second]);
        fs::write(
            root.join("ci/llama-canary/family-certified.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let cache = root.join("selected cache/hub/models--fixture--repair");
        let blob = cache.join("blobs").join(&digest);
        let snapshot = cache
            .join("snapshots")
            .join(revision)
            .join("nested/model.gguf");
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        fs::write(&blob, bytes).unwrap();
        std::os::unix::fs::symlink(&blob, &snapshot).unwrap();
        Self {
            _temp: temp,
            root,
            blob,
        }
    }
    fn run(&self, body: &str) -> process::RawProcessReport {
        let environment: BTreeMap<_, _> = [
            ("PATH", std::env::var("PATH").unwrap()),
            ("HARNESS_MODE", "repair".into()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                self.root.join("controller").display().to_string(),
            ),
            ("ROOT", self.root.display().to_string()),
            ("TRUSTED_ROOT", self.root.display().to_string()),
            (
                "PLAN_PATH",
                self.root.join("plan.json").display().to_string(),
            ),
            (
                "HF_CACHE",
                self.root.join("selected cache/hub").display().to_string(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                environment,
                arguments: vec![
                    Value::Public("-c".into()),
                    Value::Public(format!("source ./adapter.sh\n{body}").into()),
                ],
            },
            &Limits {
                execution: Duration::from_secs(15),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(
            result.process.failure.is_none() && result.process.cleanup.complete,
            "{:?}",
            result.process
        );
        result
    }
    fn plan(&self) -> Json {
        serde_json::from_slice(&fs::read(self.root.join("plan.json")).unwrap()).unwrap()
    }
}
#[test]
fn actual_repair_cache_caller_uses_selected_root_full_plan_256_and_certificate_plan_one() {
    let f = Fixture::new();
    let report = f.run("check_family_cache");
    assert!(report.process.success(), "{:?}", report.process);
    assert_eq!(f.plan()["shards"].as_array().unwrap().len(), 2);
    assert_eq!(f.plan()["selected_family_count"], 2);
    let report = f.run("repair_family_plan 1 certificate.log");
    assert!(report.process.success(), "{:?}", report.process);
    assert_eq!(f.plan()["shards"].as_array().unwrap().len(), 1);
    assert_eq!(
        fs::read_to_string(f.root.join("certificate.log"))
            .unwrap()
            .lines()
            .count(),
        3
    );
}
#[test]
fn actual_repair_cache_caller_rejects_corrupt_explicit_cache_and_changed_frozen_owner() {
    let corrupt = Fixture::new();
    let mut bytes = fs::read(&corrupt.blob).unwrap();
    bytes[0] = b'X';
    fs::write(&corrupt.blob, bytes).unwrap();
    assert!(!corrupt.run("check_family_cache").process.success());
    let wrong = Fixture::new();
    assert!(
        !wrong
            .run("HF_CACHE=\"$ROOT/absent-cache\"\ncheck_family_cache")
            .process
            .success()
    );
    let changed = Fixture::new();
    let report =
        changed.run("printf changed >> \"$repair_workload_controller\"\ncheck_family_cache");
    assert!(!report.process.success());
    assert!(!changed.root.join("plan.json").exists());
}
#[test]
fn actual_verify_branch_keeps_legacy_planner_and_never_uses_repair_owner() {
    let f = Fixture::new();
    fs::create_dir(f.root.join("bin")).unwrap();
    let python = f.root.join("bin/python3");
    fs::write(
        &python,
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> legacy.args\nexit 0\n",
    )
    .unwrap();
    fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();
    let report = f.run("HARNESS_MODE=verify\nPATH=\"$ROOT/bin:$PATH\"\ncheck_family_cache");
    assert!(report.process.success());
    let log = fs::read_to_string(f.root.join("legacy.args")).unwrap();
    assert_eq!(log.lines().count(), 2);
    assert!(log.contains("--shard-count 256 --check-cache"));
    assert!(log.contains("--verify-plan"));
    assert!(!f.root.join("plan.json").exists());
}

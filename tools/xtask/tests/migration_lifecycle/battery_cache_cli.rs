use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

fn run(root: &Path, args: &[String]) -> process::ProcessReport {
    let spec = ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: args.iter().map(|v| Value::Public(v.into())).collect(),
        cwd: root.into(),
        environment: BTreeMap::new(),
    };
    supervise(spec)
}
fn supervise(spec: ProcessSpec) -> process::ProcessReport {
    let limits = Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let result = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        result.cleanup.complete && !result.stdout.truncated && !result.stderr.truncated,
        "{result:?}"
    );
    result
}
fn gguf(layers: u64) -> Vec<u8> {
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(3u64.to_le_bytes());
    string(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    string(&mut b, "llama");
    for (key, value) in [
        ("llama.block_count", layers),
        ("llama.embedding_length", 2048),
    ] {
        string(&mut b, key);
        b.extend(10u32.to_le_bytes());
        b.extend(value.to_le_bytes());
    }
    b
}
struct Fixture {
    temp: tempfile::TempDir,
    manifest: PathBuf,
    plan: PathBuf,
    hub: PathBuf,
    blob: PathBuf,
    snapshot: PathBuf,
}
impl Fixture {
    fn root(&self) -> &Path {
        self.temp.path()
    }
    fn new(layers: u64) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::create_dir_all(root.join("tools/xtask")).unwrap();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::write(root.join("tools/xtask/Cargo.toml"), "[package]\n").unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut manifest: Json = serde_json::from_slice(
            &fs::read(source.join("ci/llama-canary/family-certified.json")).unwrap(),
        )
        .unwrap();
        let row = manifest["models"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|r| r["family"] == "llama")
            .unwrap();
        let bytes = gguf(layers);
        let digest = hex::encode(Sha256::digest(&bytes));
        let revision = "a".repeat(40);
        row["artifact"] = json!({"repo":"fixture/selected","revision":revision,"files":["nested/model.gguf"],"selector":"Q4_K_M","file_integrity":{"nested/model.gguf":{"size_bytes":bytes.len(),"blob_id":digest}}});
        let hub = root.join("selected model cache/hub");
        let repo = hub.join("models--fixture--selected");
        let blob = repo.join("blobs").join(&digest);
        let snapshot = repo
            .join("snapshots")
            .join(&revision)
            .join("nested/model.gguf");
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        fs::write(&blob, bytes).unwrap();
        std::os::unix::fs::symlink(&blob, &snapshot).unwrap();
        let path = root.join("manifest.json");
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let plan = root.join("plan.json");
        let fixture = Self {
            temp,
            manifest: path,
            plan,
            hub,
            blob,
            snapshot,
        };
        fixture.plan();
        fixture
    }
    fn plan(&self) {
        let args = vec![
            "--repo-root".into(),
            self.root().display().to_string(),
            "ci".into(),
            "family-plan".into(),
            "--manifest".into(),
            self.manifest.display().to_string(),
            "--families".into(),
            "llama".into(),
            "--output".into(),
            self.plan.display().to_string(),
        ];
        let result = run(self.root(), &args);
        assert!(result.success(), "{result:?}");
    }
    fn cache(&self, path: &Path) -> process::ProcessReport {
        run(
            self.root(),
            &[
                "automation".into(),
                "family-battery-policy".into(),
                "--cache".into(),
                self.root().display().to_string(),
                self.manifest.display().to_string(),
                self.plan.display().to_string(),
                path.display().to_string(),
            ],
        )
    }
}
#[test]
fn actual_current_planner_filters_one_shard_and_cache_uses_explicit_selected_root() {
    let f = Fixture::new(16);
    let bytes = fs::read(&f.plan).unwrap();
    let plan: Json = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(plan["selected_family_count"], 1);
    assert_eq!(plan["shards"].as_array().unwrap().len(), 1);
    assert_eq!(plan["selected_models"][0]["family"], "llama");
    assert!(f.cache(&f.hub).success());
    assert!(f.cache(f.hub.parent().unwrap()).success());
    fs::create_dir_all(f.root().join("controller-home/hub")).unwrap();
    assert!(!f.cache(&f.root().join("controller-home")).success());
    assert_eq!(fs::read(&f.plan).unwrap(), bytes);
}
#[test]
fn actual_cache_rejects_same_size_corruption_regular_snapshot_and_metadata_mismatch() {
    let f = Fixture::new(16);
    let original = fs::read(&f.blob).unwrap();
    let mut corrupt = original.clone();
    corrupt[0] = b'X';
    fs::write(&f.blob, &corrupt).unwrap();
    assert!(!f.cache(&f.hub).success());
    fs::write(&f.blob, &original).unwrap();
    fs::remove_file(&f.snapshot).unwrap();
    fs::write(&f.snapshot, &original).unwrap();
    assert!(!f.cache(&f.hub).success());
    let mismatch = Fixture::new(15);
    assert!(!mismatch.cache(&mismatch.hub).success());
    let f = Fixture::new(16);
    let mut plan: Json = serde_json::from_slice(&fs::read(&f.plan).unwrap()).unwrap();
    plan["selected_models"] = json!([]);
    fs::write(&f.plan, serde_json::to_vec(&plan).unwrap()).unwrap();
    assert!(!f.cache(&f.hub).success());
}
#[test]
fn actual_dimension_projection_reads_selected_file_and_rejects_missing_metadata() {
    let f = Fixture::new(16);
    let args = [
        "automation".into(),
        "family-battery-policy".into(),
        "--inspect-gguf".into(),
        f.snapshot.display().to_string(),
    ];
    let result = run(f.root(), &args);
    assert!(result.success(), "{result:?}");
    let dimensions: Json = serde_json::from_slice(&result.stdout.bytes_retained).unwrap();
    assert_eq!(dimensions["layer_count"], 16);
    assert_eq!(dimensions["activation_width"], 2048);
    assert_eq!(dimensions["mtp_layers"], 0);
    fs::write(&f.blob, b"not gguf").unwrap();
    assert!(!run(f.root(), &args).success());
}

#[test]
fn actual_battery_prepare_uses_current_owner_and_preserves_supplied_plan_bytes() {
    let f = Fixture::new(16);
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let caller =
        fs::read_to_string(source.join("skippy/scripts/skippy-family-battery.sh")).unwrap();
    let function = caller
        .split("prepare_policy_plan() {")
        .nth(1)
        .unwrap()
        .split("\nprepare_policy_plan\n")
        .next()
        .unwrap();
    let shell = f.root().join("prepare.sh");
    fs::write(&shell,format!("set -euo pipefail\nautomation=(\"$OWNER\")\nPOLICY_PLAN=\"$SUPPLIED\"\nPOLICY_PLAN_COPY=\"$COPY\"\nSHARD_INDEX=\"\"\nFAMILY_FILTER=llama\nprepare_policy_plan() {{{function}\nprepare_policy_plan\nprintf admitted > \"$ROOT/downstream\"\n")).unwrap();
    fs::create_dir_all(f.root().join("scripts")).unwrap();
    fs::write(
        f.root().join("scripts/plan-family-battery.py"),
        "forbidden historical planner\n",
    )
    .unwrap();
    let before = fs::read(&f.plan).unwrap();
    for supplied in ["", f.plan.to_str().unwrap()] {
        for admitted in [true, false] {
            let destination = f.root().join("caller-plan.json");
            for path in [&destination, &f.root().join("downstream")] {
                if path.exists() {
                    fs::remove_file(path).unwrap();
                }
            }
            let cache = if admitted {
                f.hub.clone()
            } else {
                f.root().join("wrong-cache")
            };
            let spec = ProcessSpec {
                executable: "/bin/bash".into(),
                arguments: vec![Value::Public(shell.clone().into())],
                cwd: f.root().into(),
                environment: BTreeMap::from([
                    (
                        "OWNER".into(),
                        Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
                    ),
                    ("ROOT".into(), Value::Public(f.root().into())),
                    ("MANIFEST".into(), Value::Public(f.manifest.clone().into())),
                    ("HF_CACHE".into(), Value::Public(cache.into())),
                    ("SUPPLIED".into(), Value::Public(supplied.into())),
                    ("COPY".into(), Value::Public(destination.clone().into())),
                ]),
            };
            let result = supervise(spec);
            assert_eq!(result.success(), admitted, "{result:?}");
            assert_eq!(fs::read(&destination).unwrap(), before);
            assert_eq!(f.root().join("downstream").exists(), admitted);
            assert_eq!(fs::read(&f.plan).unwrap(), before);
        }
    }
}

#[test]
fn actual_cache_requires_the_declared_file_in_the_exact_pinned_snapshot() {
    let fixture = Fixture::new(16);
    let plan = fs::read(&fixture.plan).expect("plan");
    assert!(fixture.cache(&fixture.hub).success());
    fs::remove_file(&fixture.snapshot).expect("remove declared snapshot file");
    assert!(
        fixture.blob.is_file(),
        "blob existence alone cannot admit a snapshot"
    );
    assert!(!fixture.cache(&fixture.hub).success());
    assert_eq!(fs::read(&fixture.plan).expect("unchanged plan"), plan);
}

fn descriptor_fixture(dimensions: &[u64], kind: u32) -> Fixture {
    let f = Fixture::new(16);
    let mut bytes = gguf(16);
    bytes[8..16].copy_from_slice(&1_u64.to_le_bytes());
    bytes.extend(6_u64.to_le_bytes());
    bytes.extend(b"weight");
    bytes.extend((dimensions.len() as u32).to_le_bytes());
    for dimension in dimensions {
        bytes.extend(dimension.to_le_bytes());
    }
    bytes.extend(kind.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    let digest = hex::encode(Sha256::digest(&bytes));
    let new_blob = f.blob.parent().unwrap().join(&digest);
    fs::write(&new_blob, &bytes).unwrap();
    fs::remove_file(&f.snapshot).unwrap();
    std::os::unix::fs::symlink(&new_blob, &f.snapshot).unwrap();
    let mut manifest: Json = serde_json::from_slice(&fs::read(&f.manifest).unwrap()).unwrap();
    let model = manifest["models"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|model| model["family"] == "llama")
        .unwrap();
    model["artifact"]["file_integrity"]["nested/model.gguf"] =
        json!({"size_bytes":bytes.len(),"blob_id":digest});
    model["resources"]["estimated_model_bytes"] = json!(35);
    fs::write(&f.manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let constants = f.root().join(".deps/llama.cpp/gguf-py/gguf/constants.py");
    fs::create_dir_all(constants.parent().unwrap()).unwrap();
    fs::write(constants, "class GGMLQuantizationType(IntEnum):\n    Q4_0 = 2\n\nQK_K = 256\nGGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {\n    GGMLQuantizationType.Q4_0: (32, 2 + 16),\n}\nraise RuntimeError('must never execute layout data')\n").unwrap();
    f.plan();
    f
}
fn descriptor_cache(f: &Fixture) -> process::ProcessReport {
    run(
        f.root(),
        &[
            "automation".into(),
            "family-battery-policy".into(),
            "--cache-descriptors".into(),
            f.root().display().to_string(),
            f.manifest.display().to_string(),
            f.plan.display().to_string(),
            f.hub.display().to_string(),
        ],
    )
}
#[test]
fn actual_descriptor_admission_accepts_conservative_estimate_without_executing_constants() {
    let f = descriptor_fixture(&[64], 2); // Two Q4_0 blocks = 36 bytes; estimate = 35.
    let before = fs::read(&f.plan).unwrap();
    let report = descriptor_cache(&f);
    assert!(report.success(), "{report:?}");
    assert_eq!(fs::read(&f.plan).unwrap(), before);
    let plan: Json = serde_json::from_slice(&before).unwrap();
    assert_eq!(
        plan["selected_models"][0]["resources"]["estimated_model_bytes"],
        35
    );
    assert!(
        plan["selected_models"][0]["artifact"]["file_integrity"]["nested/model.gguf"]["size_bytes"]
            .as_u64()
            .unwrap()
            > 36
    );
}
#[test]
fn actual_descriptor_admission_refuses_pinned_malformed_tensors_and_preserves_plan() {
    for (dimensions, kind, diagnostic) in [
        (vec![], 2, "rank"),
        (vec![0], 2, "positive"),
        (vec![33], 2, "unaligned"),
        (vec![64], 999, "unknown GGML tensor type"),
        (vec![64, u64::MAX], 2, "overflow"),
    ] {
        let f = descriptor_fixture(&dimensions, kind);
        // Identity, size and trunk metadata pass: refusal must be caused by descriptors.
        assert!(f.cache(&f.hub).success());
        let before = fs::read(&f.plan).unwrap();
        let report = descriptor_cache(&f);
        assert!(!report.success(), "{report:?}");
        let stderr = String::from_utf8_lossy(&report.stderr.bytes_retained);
        assert!(
            stderr.contains("weight") && stderr.contains(diagnostic),
            "{report:?}"
        );
        assert_eq!(fs::read(&f.plan).unwrap(), before);
    }
}
#[test]
fn actual_descriptor_admission_refuses_missing_or_executable_layout_table() {
    let f = descriptor_fixture(&[64], 2);
    let constants = f.root().join(".deps/llama.cpp/gguf-py/gguf/constants.py");
    let before = fs::read(&f.plan).unwrap();
    fs::write(&constants, "class GGMLQuantizationType(IntEnum):\n    Q4_0 = 2\n\nQK_K = 256\nGGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {\n    GGMLQuantizationType.Q4_0: (32, __import__('os').system('false')),\n}\n").unwrap();
    assert!(!descriptor_cache(&f).success());
    fs::remove_file(constants).unwrap();
    assert!(!descriptor_cache(&f).success());
    assert_eq!(fs::read(&f.plan).unwrap(), before);
}

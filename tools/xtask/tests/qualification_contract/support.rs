//! Shared fixtures for the qualification contract tests: a scratch checkout
//! (catalog + registry), a composed standalone product, and a complete passing
//! evidence set, with real SHA-256 digests so validators read honest bytes.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub(crate) const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef01234567";
pub(crate) const PLAN_DIGEST: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
pub(crate) const DENSE_HASH: &str =
    "1111111111111111111111111111111111111111111111111111111111111111";
pub(crate) const HYBRID_HASH: &str =
    "2222222222222222222222222222222222222222222222222222222222222222";
pub(crate) const MOE_HASH: &str =
    "3333333333333333333333333333333333333333333333333333333333333333";
pub(crate) const LAYA_HASH: &str =
    "4444444444444444444444444444444444444444444444444444444444444444";
pub(crate) const DECISION_HASH: &str =
    "5555555555555555555555555555555555555555555555555555555555555555";

pub(crate) fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(crate) fn file_sha256(path: &Path) -> String {
    sha256(&std::fs::read(path).unwrap())
}

/// The qualification tree digest: len+path+file sha256 in sorted relative
/// POSIX order.
pub(crate) fn tree_sha256(root: &Path) -> String {
    let mut files = Vec::new();
    collect(root, root, &mut files);
    files.sort();
    let mut digest = Sha256::new();
    for relative in files {
        let bytes = relative.as_bytes();
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
        let file = hex::decode(file_sha256(&root.join(&relative))).unwrap();
        digest.update(file);
    }
    hex::encode(digest.finalize())
}

fn collect(root: &Path, dir: &Path, files: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let relative = path
            .strip_prefix(root)
            .unwrap()
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        if path.is_dir() {
            collect(root, &path, files);
        } else {
            files.push(relative);
        }
    }
}

pub(crate) fn write_json(path: &Path, value: &Value) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut text = serde_json::to_string_pretty(value).unwrap();
    text.push('\n');
    std::fs::write(path, text).unwrap();
}

/// A scratch checkout with the nine-row catalog and the qualification model
/// registry.
pub(crate) struct Checkout(pub(crate) tempfile::TempDir);

impl Checkout {
    pub(crate) fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        std::fs::create_dir_all(root.join("tools/xtask")).unwrap();
        std::fs::write(root.join("tools/xtask/Cargo.toml"), "[package]\n").unwrap();
        std::fs::create_dir_all(root.join("ci/model-artifacts")).unwrap();
        Self(temp)
    }

    pub(crate) fn root(&self) -> &Path {
        self.0.path()
    }

    /// Write the runtime catalog; pass fewer than nine rows to drift it.
    pub(crate) fn write_catalog(&self, rows: &[&str]) {
        let rows: Vec<Value> = rows
            .iter()
            .map(|id| {
                let (platform, backend) = id.split_once('-').unwrap();
                let target = match platform {
                    "linux" => "x86_64-unknown-linux-gnu",
                    "macos" => "aarch64-apple-darwin",
                    _ => "x86_64-pc-windows-msvc",
                };
                json!({"id": id, "platform": platform, "backend": backend, "target": target})
            })
            .collect();
        write_json(
            &self.root().join("ci/slices.yml"),
            &json!({"runtime_rows": rows}),
        );
    }

    pub(crate) fn write_full_catalog(&self) {
        self.write_catalog(&[
            "linux-cpu",
            "linux-cuda",
            "linux-rocm",
            "linux-vulkan",
            "macos-metal",
            "windows-cpu",
            "windows-cuda",
            "windows-rocm",
            "windows-vulkan",
        ]);
    }

    pub(crate) fn write_registry(&self) {
        let artifact = |id: &str, hash: &str, tags: &[&str]| {
            json!({
                "id": id,
                "capability_tags": tags,
                "artifact": {"files": [{"sha256": hash}]},
            })
        };
        write_json(
            &self.root().join("ci/model-artifacts/registry.json"),
            &json!({"artifacts": [
                artifact("dense-model", DENSE_HASH, &["dense"]),
                artifact("hybrid-model", HYBRID_HASH, &["hybrid"]),
                artifact("moe-model", MOE_HASH, &["moe"]),
                artifact("family-laya-multilingual", LAYA_HASH, &["system-one"]),
                artifact("decision-model", DECISION_HASH, &["decision"]),
            ]}),
        );
    }
}

pub(crate) const SUITE_CASES: [(&str, &[&str]); 7] = [
    (
        "packaging-runtime",
        &[
            "archive-integrity",
            "imports",
            "abi",
            "discovery",
            "version",
            "no-driver",
        ],
    ),
    (
        "dense",
        &[
            "load",
            "prefill-decode",
            "stream",
            "continuation",
            "restart",
        ],
    ),
    (
        "recurrent",
        &["prefill-decode", "state-preservation", "restart"],
    ),
    (
        "moe",
        &[
            "expert-execution",
            "repeated-restore",
            "suffix-continuation",
        ],
    ),
    (
        "kv-cache",
        &[
            "dense-prefix-hit",
            "recurrent-prefix-hit",
            "suffix-continuation",
            "divergent-prefix",
            "isolation",
        ],
    ),
    (
        "system-one",
        &[
            "laya-goldens",
            "reader-contract",
            "negative-cases",
            "lifecycle",
        ],
    ),
    (
        "decisions",
        &[
            "endpoint-equivalence",
            "probability-contract",
            "negative-cases",
            "lifecycle",
        ],
    ),
];

fn suite_models(suite: &str) -> Value {
    match suite {
        "packaging-runtime" => json!({}),
        "dense" => json!({"dense-model": DENSE_HASH}),
        "recurrent" => json!({"hybrid-model": HYBRID_HASH}),
        "moe" => json!({"moe-model": MOE_HASH}),
        "kv-cache" => json!({"dense-model": DENSE_HASH, "hybrid-model": HYBRID_HASH}),
        "system-one" => json!({"family-laya-multilingual": LAYA_HASH}),
        _ => json!({"decision-model": DECISION_HASH}),
    }
}

/// A composed standalone product directory for one row backend.
pub(crate) struct Product {
    pub(crate) dir: PathBuf,
    _temp: tempfile::TempDir,
}

impl Product {
    pub(crate) fn new(row_id: &str) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("product");
        std::fs::create_dir_all(dir.join("native-runtimes/runtime-1/lib")).unwrap();
        std::fs::write(dir.join("skippy"), b"fake-skippy-binary\n").unwrap();
        std::fs::write(dir.join("build-contract.json"), b"{\"schema_version\":1}\n").unwrap();
        std::fs::write(dir.join("host-imports.json"), b"{}\n").unwrap();
        std::fs::write(
            dir.join("native-runtimes/runtime-1/manifest.json"),
            b"{\"schema_version\":2}\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("native-runtimes/runtime-1/lib/libskippy.so"),
            b"lib\n",
        )
        .unwrap();
        let (platform, backend) = row_id.split_once('-').unwrap();
        let target = match platform {
            "linux" => "x86_64-unknown-linux-gnu",
            "macos" => "aarch64-apple-darwin",
            _ => "x86_64-pc-windows-msvc",
        };
        let runtime_dir = dir.join("native-runtimes/runtime-1");
        let manifest = json!({
            "schema_version": 1,
            "contract": "skippy-product-v1",
            "source_sha": SOURCE_SHA,
            "target": target,
            "backend": backend,
            "cli": {
                "path": "skippy",
                "sha256": file_sha256(&dir.join("skippy")),
                "build_contract_sha256": file_sha256(&dir.join("build-contract.json")),
                "host_imports_sha256": file_sha256(&dir.join("host-imports.json")),
            },
            "runtime": {
                "id": "runtime-1",
                "path": "native-runtimes/runtime-1",
                "sha256": tree_sha256(&runtime_dir),
                "manifest_sha256": file_sha256(&runtime_dir.join("manifest.json")),
            },
        });
        write_json(&dir.join("product-manifest.json"), &manifest);
        Self { dir, _temp: temp }
    }

    pub(crate) fn manifest_path(&self) -> PathBuf {
        self.dir.join("product-manifest.json")
    }

    pub(crate) fn manifest_digest(&self) -> String {
        file_sha256(&self.manifest_path())
    }
}

/// Write an availability record file and return its digest.
pub(crate) fn write_availability(dir: &Path, row_id: &str, state: &str) -> (PathBuf, String) {
    let mut value = json!({
        "schema_version": 1,
        "source_sha": SOURCE_SHA,
        "plan_digest": PLAN_DIGEST,
        "row_id": row_id,
        "policy_source": "protected-ci",
        "state": state,
    });
    if state == "available" {
        value["runner"] = json!(match row_id {
            "linux-cpu" => "linux-hosted",
            "macos-metal" => "macos-hosted",
            "windows-cpu" => "windows-hosted",
            "linux-rocm" => "gpu-amd",
            _ => "gpu-nvidia",
        });
    } else {
        value["reason"] = json!("no approved ephemeral runner");
    }
    let path = dir.join("availability.json");
    write_json(&path, &value);
    let digest = file_sha256(&path);
    (path, digest)
}

/// Write the complete passing seven-suite evidence set for a row.
pub(crate) fn write_evidence(dir: &Path, row_id: &str, product_digest: &str) {
    std::fs::create_dir_all(dir).unwrap();
    for (suite, cases) in SUITE_CASES {
        let mut sorted: Vec<&str> = cases.to_vec();
        sorted.sort_unstable();
        write_json(
            &dir.join(format!("{suite}.json")),
            &json!({
                "schema_version": 1,
                "status": "passed",
                "source_sha": SOURCE_SHA,
                "row_id": row_id,
                "suite": suite,
                "product_manifest_sha256": product_digest,
                "executed_cases": sorted,
                "models": suite_models(suite),
                "observations": {},
            }),
        );
    }
}

/// A hardware record valid for the row's backend.
pub(crate) fn write_hardware(dir: &Path, row_id: &str) -> PathBuf {
    let (platform, backend) = row_id.split_once('-').unwrap();
    let _ = platform;
    let (device, selected, extra) = match backend {
        "cpu" => ("CPU", "CPU", json!({"offloaded_layers": 0})),
        "metal" => (
            "MTL0",
            "MTL0",
            json!({"driver": "macOS 15.0", "runtime_version": "Metal 3.1", "device_architecture": "arm64"}),
        ),
        "cuda" => (
            "NVIDIA Test GPU",
            "CUDA0",
            json!({"driver": "575.0", "runtime_version": "CUDA 13", "device_architecture": "sm_86"}),
        ),
        "rocm" => (
            "AMD Test GPU",
            "ROCm0",
            json!({"driver": "6.3", "runtime_version": "6.3", "device_architecture": "gfx1100"}),
        ),
        _ => (
            "Vulkan Test GPU",
            "Vulkan0",
            json!({"driver": "1.3", "runtime_version": "1.3", "device_architecture": "gpu"}),
        ),
    };
    let runner = match row_id {
        "linux-cpu" => "linux-hosted",
        "macos-metal" => "macos-hosted",
        "windows-cpu" => "windows-hosted",
        "linux-rocm" => "gpu-amd",
        _ => "gpu-nvidia",
    };
    let mut value = json!({
        "actual_backend": backend,
        "runner": runner,
        "device": device,
        "selected_device": selected,
    });
    value
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let path = dir.join("hardware.json");
    write_json(&path, &value);
    path
}

/// Run one xtool subcommand against the fixture checkout.
pub(crate) fn xtool(root: &Path, args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args(args)
        .output()
        .unwrap()
}

pub(crate) fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

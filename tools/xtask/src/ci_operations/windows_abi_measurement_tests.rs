use super::super::windows_cache_consumers::{Context, policy};
use super::*;
use std::collections::BTreeMap;

const EPOCH: &str = "github-win22-20261115-X64";

fn repository() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn parsed_row(backend: &str, condition: &str) -> BTreeMap<String, String> {
    row(&repository(), backend, condition, "1")
        .unwrap()
        .lines()
        .map(|line| {
            let (key, value) = line.split_once('=').unwrap();
            (key.to_owned(), value.to_owned())
        })
        .collect()
}

fn toolkit(backend: &str) -> (&'static str, &'static str, &'static str) {
    match backend {
        "cuda" => ("12.6.3", "", ""),
        "vulkan" => ("", "1.4.328.1", ""),
        "rocm" => (
            "",
            "",
            "AMD-Software-PRO-Edition-25.Q3-WinSvr2022-For-HIP.exe",
        ),
        _ => ("", "", ""),
    }
}

fn row_key(backend: &str, condition: &str, hash: &str) -> String {
    let row = parsed_row(backend, condition);
    let (cuda, vulkan, rocm) = toolkit(backend);
    identity(&Inputs {
        backend: &row["backend"],
        architectures: &row["architecture_set"],
        epoch: EPOCH,
        input_hash: hash,
        cuda,
        vulkan,
        rocm,
    })
    .unwrap()
    .key
}

#[test]
fn cold_and_exact_warm_compute_the_same_registered_key_per_backend() {
    let hash = "a".repeat(64);
    let registered = policy(
        &repository(),
        &Context {
            epoch: EPOCH.into(),
            input_hash: hash.clone(),
            variables: BTreeMap::new(),
            max_bytes: 20 * 1024 * 1024 * 1024,
            history: 2,
        },
    )
    .unwrap();
    for (backend, family) in [
        ("cpu", "cpu--cpu"),
        ("cuda", "cuda-75-cuda-12.6.3-Jimver-v0.2.36"),
        (
            "rocm",
            "rocm-gfx1100-rocm-AMD-Software-PRO-Edition-25.Q3-WinSvr2022-For-HIP.exe",
        ),
        ("vulkan", "vulkan--vulkan-1.4.328.1-jakoch-v1.5.2"),
    ] {
        let cold = row_key(backend, "cold", &hash);
        let warm = row_key(backend, "exact-warm", &hash);
        assert_eq!(
            cold,
            format!("mesh-llm-windows-2022-skippy-abi-{family}-{EPOCH}-{hash}")
        );
        assert_eq!(cold, warm, "{backend} arms must share one identity");
        assert!(
            registered.current_keys.contains(&cold),
            "{backend} diagnostic must not introduce a cache family"
        );
    }
}

#[test]
fn rows_and_samples_are_closed() {
    let root = repository();
    assert!(row(&root, "metal", "cold", "1").is_err());
    assert!(row(&root, "cpu", "warm", "1").is_err());
    for sample in ["0", "6", "x", ""] {
        assert!(row(&root, "cpu", "cold", sample).is_err(), "{sample}");
    }
    let rocm = parsed_row("rocm", "cold");
    assert_eq!(
        rocm["build_dir"],
        ".deps/llama-build/windows/build-stage-abi-rocm"
    );
    assert_eq!(rocm["architecture_set"], "gfx1100");
    assert_eq!(rocm["target"], "x86_64-pc-windows-msvc");
}

fn tree(files: usize, stamp: Option<&str>) -> Tree {
    Tree {
        files,
        digest: "d".into(),
        stamp_sha256: stamp.map(str::to_owned),
    }
}

#[test]
fn admission_refuses_retrieval_in_cold_and_any_inexact_warm_state() {
    let key = "k";
    let empty = tree(0, None);
    let restored = tree(3, Some("s"));
    admit_sample(Condition::Cold, key, "", "", &empty).unwrap();
    admit_sample(Condition::ExactWarm, key, "true", key, &restored).unwrap();
    for (condition, hit, primary, observed, reason) in [
        (Condition::Cold, "false", "", &empty, "retrieval"),
        (Condition::Cold, "", key, &empty, "retrieval"),
        (Condition::Cold, "", "", &restored, "empty"),
        (Condition::ExactWarm, "false", key, &restored, "never cold"),
        (Condition::ExactWarm, "", "", &empty, "never cold"),
        (Condition::ExactWarm, "true", "other", &restored, "differs"),
        (Condition::ExactWarm, "true", key, &tree(3, None), "stamp"),
    ] {
        let error = admit_sample(condition, key, hit, primary, observed).unwrap_err();
        assert!(error.contains(reason), "{error}");
    }
}

#[test]
fn exact_warm_refuses_a_rebuild_and_both_arms_require_a_stamp() {
    let before = tree(3, Some("s"));
    complete_sample(Condition::ExactWarm, &before, &tree(3, Some("s"))).unwrap();
    complete_sample(Condition::Cold, &tree(0, None), &tree(9, Some("t"))).unwrap();
    let rebuilt = Tree {
        digest: "changed".into(),
        ..tree(3, Some("s"))
    };
    assert!(
        complete_sample(Condition::ExactWarm, &before, &rebuilt)
            .unwrap_err()
            .contains("rebuild")
    );
    assert!(complete_sample(Condition::ExactWarm, &before, &tree(3, Some("t"))).is_err());
    assert!(complete_sample(Condition::Cold, &tree(0, None), &tree(4, None)).is_err());
}

#[test]
fn snapshot_observes_rewritten_identical_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let stamp = directory.path().join(STAMP);
    fs::write(&stamp, "stamp\n").unwrap();
    fs::create_dir(directory.path().join("bin")).unwrap();
    fs::write(directory.path().join("bin/llama.dll"), "dll").unwrap();
    let before = snapshot(directory.path()).unwrap();
    assert_eq!(before.files, 2);
    assert_eq!(before, snapshot(directory.path()).unwrap());
    let file = fs::File::options().write(true).open(&stamp).unwrap();
    file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1))
        .unwrap();
    let after = snapshot(directory.path()).unwrap();
    assert_eq!(before.stamp_sha256, after.stamp_sha256);
    assert_ne!(before, after);
}

fn sample_env(
    condition: &'static str,
    build_dir: &Path,
    hit: &'static str,
    key: String,
) -> impl Fn(&str) -> String {
    let build_dir = build_dir.display().to_string();
    move |name: &str| {
        match name {
            "GITHUB_EVENT_NAME" => "workflow_dispatch",
            "MEASUREMENT_CONDITION" => condition,
            "MEASUREMENT_SAMPLE" => "1",
            "INPUT_TOOLCHAIN_EPOCH" | "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH" => EPOCH,
            "INPUT_BACKEND" => "cpu",
            "CACHE_INPUT_HASH" => return "b".repeat(64),
            "CACHE_HIT" => hit,
            "CACHE_PRIMARY_KEY" => return key.clone(),
            "BUILD_DIR" => return build_dir.clone(),
            _ => "",
        }
        .to_owned()
    }
}

fn mark_all(directory: &Path) {
    for (offset, phase) in PHASES.iter().enumerate() {
        let at = format!("2026-11-18T10:00:{:02}Z", offset * 5);
        mark(directory, phase, timestamp(&at).ok().flatten().unwrap()).unwrap();
    }
}

#[test]
fn exact_warm_receipt_binds_identity_phases_and_unchanged_tree() {
    let build = tempfile::tempdir().unwrap();
    fs::write(build.path().join(STAMP), "stamp\n").unwrap();
    let receipts = tempfile::tempdir().unwrap();
    let key = format!(
        "mesh-llm-windows-2022-skippy-abi-cpu--cpu-{EPOCH}-{}",
        "b".repeat(64)
    );
    let env = sample_env("exact-warm", build.path(), "true", key.clone());
    mark_all(receipts.path());
    admit(receipts.path(), &env).unwrap();
    finish(receipts.path(), &env).unwrap();
    let receipt: Value =
        serde_json::from_slice(&fs::read(receipts.path().join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["admitted"], true);
    assert_eq!(receipt["admission"]["key"], key.as_str());
    assert_eq!(receipt["admission"]["restored_key"], key.as_str());
    assert_eq!(receipt["seconds"]["lookup"], 5.0);
    assert_eq!(receipt["seconds"]["primary"], 15.0);
    assert_eq!(receipt["run"]["event"], "workflow_dispatch");

    fs::write(build.path().join(STAMP), "rebuilt\n").unwrap();
    assert!(
        finish(receipts.path(), &env)
            .unwrap_err()
            .contains("rebuild")
    );
    let refused: Value =
        serde_json::from_slice(&fs::read(receipts.path().join("receipt.json")).unwrap()).unwrap();
    assert_eq!(refused["admitted"], false);
}

#[test]
fn refused_admission_writes_evidence_but_no_receipt() {
    let build = tempfile::tempdir().unwrap();
    let receipts = tempfile::tempdir().unwrap();
    let env = sample_env("exact-warm", build.path(), "false", String::new());
    mark_all(receipts.path());
    assert!(
        admit(receipts.path(), &env)
            .unwrap_err()
            .contains("never cold")
    );
    let refusal: Value =
        serde_json::from_slice(&fs::read(receipts.path().join("refusal.json")).unwrap()).unwrap();
    assert_eq!(refusal["condition"], "exact-warm");
    assert!(finish(receipts.path(), &env).is_err());
    assert!(!receipts.path().join("receipt.json").exists());
}

#[test]
fn admission_requires_manual_dispatch_and_matching_epoch() {
    let build = tempfile::tempdir().unwrap();
    let receipts = tempfile::tempdir().unwrap();
    let base = sample_env("cold", build.path(), "", String::new());
    admit(receipts.path(), &base).unwrap();
    for (name, value) in [
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("GITHUB_EVENT_NAME", "push"),
        ("MESH_LLM_LLAMA_TOOLCHAIN_EPOCH", "github-win22-other-X64"),
        ("MEASUREMENT_CONDITION", "warm"),
        ("CACHE_INPUT_HASH", "short"),
    ] {
        let env = |key: &str| {
            if key == name {
                value.to_owned()
            } else {
                base(key)
            }
        };
        assert!(admit(receipts.path(), env).is_err(), "{name}={value}");
    }
}

#[test]
fn phases_are_closed_and_recorded_once() {
    let receipts = tempfile::tempdir().unwrap();
    let now = Instant::now();
    assert!(mark(receipts.path(), "queue", now).is_err());
    mark(receipts.path(), "lookup_start", now).unwrap();
    assert!(mark(receipts.path(), "lookup_start", now).is_err());
}

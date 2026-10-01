use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "migration_family_plan/arguments.rs"]
mod arguments;
#[path = "migration_family_plan/compatibility.rs"]
mod compatibility;
#[path = "migration_family_plan/decoder_isolation.rs"]
mod decoder_isolation;
#[path = "migration_family_plan/input_failures.rs"]
mod input_failures;
#[path = "migration_family_plan/numeric.rs"]
mod numeric;
#[path = "migration_family_plan/selection.rs"]
mod selection;
#[path = "migration_family_plan/strings.rs"]
mod strings;
#[path = "migration_family_plan/validation.rs"]
mod validation;
#[path = "migration_family_plan/verification.rs"]
mod verification;

static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
type ManifestChange = (&'static str, fn(&mut Value));

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("checkout")
}

fn fixture(name: &str, suffix: &str) -> Vec<u8> {
    fs::read(root().join(format!(
        "tools/xtask/tests/fixtures/family_evidence/{name}.{suffix}"
    )))
    .expect("frozen fixture")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root())
        .args([
            "--repo-root",
            root().to_str().expect("UTF-8 checkout"),
            "ci",
            "family-plan",
        ])
        .args(args)
        .output()
        .expect("xtask starts")
}

fn temp_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "task24-{label}-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

fn with_manifest(change: impl FnOnce(&mut Value)) -> PathBuf {
    let mut manifest: Value =
        serde_json::from_slice(&fixture("synthetic-manifest", "json")).expect("valid manifest");
    change(&mut manifest);
    let path = temp_path("manifest.json");
    fs::write(&path, serde_json::to_vec(&manifest).expect("JSON")).expect("write manifest");
    path
}

#[test]
fn complete_plan_matches_frozen_legacy_stdout_when_generated_from_selected_root() {
    for (name, args) in [
        ("real-1", vec!["--shard-count", "1"]),
        ("real-4", vec!["--shard-count", "4"]),
        ("real-256", vec!["--shard-count", "256"]),
        (
            "real-reversed",
            vec!["--families", "llama,qwen3-dense", "--shard-count", "1"],
        ),
        (
            "synthetic-equal",
            vec![
                "--manifest",
                "tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json",
                "--shard-count",
                "2",
            ],
        ),
        (
            "synthetic-uneven",
            vec![
                "--manifest",
                "tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json",
                "--shard-count",
                "3",
            ],
        ),
    ] {
        let output = run(&args);
        assert_eq!(
            output.status.code(),
            Some(0),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, fixture(name, "stdout"), "{name}");
        assert!(output.stderr.is_empty(), "{name}");
        let digest = hex::encode(Sha256::digest(&output.stdout));
        let expected = match name {
            "real-1" => "45ee6ee3461963e34936c12efe5870e612b4c3dacd94ebbb8e656899ffcb043d",
            "real-4" => "470cbde3fe82cda0e63134004063b440117754467953d5e0b6a1116343b4c9dc",
            "real-256" => "fa78863c0a0c09b81c012f41d2a03da9fb685ce4bcc7573bf41ac852127342c0",
            "real-reversed" => "72c7b7d03959dbe2f7528316d141da9674fcdb14c65d9ab425c42e5527f7bc93",
            "synthetic-equal" => "f404d1ff18cc164e412f4326e95aae1439de1c2acde8170d2ee831631803e8c0",
            "synthetic-uneven" => {
                "fe2bc7ab9b34354e03b4e2390d6643cf9d2d69fef080848add119f98b1f01908"
            }
            _ => unreachable!("closed fixture cases"),
        };
        assert_eq!(digest, expected, "{name} final plan byte digest");
        let plan: Value = serde_json::from_slice(&output.stdout).expect("plan JSON");
        let manifest_path = if name.starts_with("synthetic") {
            root().join("tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json")
        } else {
            root().join("ci/llama-canary/family-certified.json")
        };
        assert_eq!(
            plan["manifest_sha256"],
            hex::encode(Sha256::digest(
                fs::read(manifest_path).expect("manifest bytes")
            ))
        );
    }
}

#[test]
fn frozen_errors_keep_status_and_streams() {
    for (name, args) in [
        ("real-zero", vec!["--shard-count", "0"]),
        (
            "malformed-manifest",
            vec![
                "--manifest",
                "tools/xtask/tests/fixtures/family_evidence/malformed-manifest.json",
            ],
        ),
        (
            "tampered-plan",
            vec![
                "--verify-plan",
                "tools/xtask/tests/fixtures/family_evidence/tampered-plan.json",
            ],
        ),
    ] {
        let output = run(&args);
        assert_eq!(output.status.code(), Some(2), "{name}");
        assert_eq!(output.stdout, fixture(name, "stdout"), "{name}");
        assert_eq!(output.stderr, fixture(name, "stderr"), "{name}");
    }
}

#[test]
fn verify_frozen_plan_succeeds_silently_even_with_other_generation_count() {
    let output = run(&[
        "--verify-plan",
        "tools/xtask/tests/fixtures/family_evidence/real-4.stdout",
        "--shard-count",
        "0",
    ]);
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.is_empty() && output.stderr.is_empty());
}

#[test]
fn output_and_github_output_use_exact_written_plan_and_compact_matrix() {
    let plan_path = temp_path("plan.json");
    let github_path = temp_path("github.txt");
    fs::write(&github_path, b"existing=kept\n").expect("existing output");
    let output = run(&[
        "--families",
        "llama,qwen3-dense",
        "--output",
        plan_path.to_str().expect("UTF-8"),
        "--github-output",
        github_path.to_str().expect("UTF-8"),
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty() && output.stderr.is_empty());
    let bytes = fs::read(&plan_path).expect("plan written");
    assert_eq!(bytes, fixture("real-reversed", "stdout"));
    let plan: Value = serde_json::from_slice(&bytes).expect("plan JSON");
    let github = fs::read_to_string(&github_path).expect("github output");
    let appended = github
        .strip_prefix("existing=kept\n")
        .expect("append preserves existing output");
    assert!(appended.ends_with('\n'));
    let lines = appended.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 4);
    assert_eq!(
        lines[0],
        format!(
            "plan_path={}",
            plan_path.canonicalize().expect("written plan").display()
        )
    );
    assert_eq!(
        lines[1],
        format!(
            "manifest_sha256={}",
            plan["manifest_sha256"].as_str().expect("hash")
        )
    );
    assert_eq!(lines[2], "family_count=2");
    assert_eq!(
        lines[3],
        r#"matrix={"include":[{"id":"family-battery-01","shard_index":0,"families":"qwen3-dense,llama","estimated_work_bytes":1433358464}]}"#
    );
    fs::remove_file(plan_path).expect("cleanup plan");
    fs::remove_file(github_path).expect("cleanup output");
}

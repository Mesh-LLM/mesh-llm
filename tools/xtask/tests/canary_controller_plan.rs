use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};

#[cfg(unix)]
#[path = "canary_controller_metadata/mod.rs"]
mod metadata;
#[cfg(unix)]
#[path = "canary_controller_metadata/preflight.rs"]
mod preflight;

struct Fixture {
    directory: tempfile::TempDir,
    input: Value,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().expect("temporary fixture");
        let root = directory.path();
        let source = root.join("selected");
        let controller = root.join("controller");
        for checkout in [&source, &controller] {
            fs::create_dir_all(checkout).expect("checkout");
            git(checkout, &["init", "-q"]);
            git(
                checkout,
                &[
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "fixture",
                ],
            );
        }
        let manifest = source.join("ci/llama-canary/family-certified.json");
        fs::create_dir_all(manifest.parent().expect("parent")).expect("manifest directory");
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/family_evidence/synthetic-manifest.json"),
            &manifest,
        )
        .expect("manifest");
        fs::create_dir_all(source.join("scripts")).expect("scripts");
        fs::write(source.join("scripts/skippy-family-battery.sh"),
            "set -eu\ntest \"$PWD\" = \"$(cd \"$(dirname \"$0\")/..\" && pwd)\"\ntest \"$1\" = --skip-build\ntest \"$2\" = --dry-run\ntest \"$3\" = --plan\ntest -f \"$4\"\ntest -z \"${HF_CACHE:-}\"\ntest -z \"${SKIPPY_WORKLOAD_PRODUCER_MANIFEST:-}\"\n").expect("battery");
        let input = json!({"controller_root":controller,"source_root":source,
            "controller_revision":git(&controller, &["rev-parse", "HEAD"]),
            "selected_revision":git(&source, &["rev-parse", "HEAD"]),
            "manifest":"ci/llama-canary/family-certified.json","output":root.join("output"),
            "cache":{"mode":"not_checked"}});
        Self { directory, input }
    }

    fn run(&self, verb: &str) -> std::process::Output {
        let input = self.directory.path().join("input.json");
        fs::write(&input, serde_json::to_vec(&self.input).expect("input JSON")).expect("input");
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "canary-receipts", verb, "--input"])
            .arg(input)
            .output()
            .expect("controller command")
    }

    fn output(&self, name: &str) -> std::path::PathBuf {
        self.directory.path().join("output").join(name)
    }
}

fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .expect("git");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("git text")
        .trim()
        .to_owned()
}

#[test]
fn full_selected_roster_is_planned_when_controller_has_no_manifest_or_planner() {
    let fixture = Fixture::new();

    let result = fixture.run("source-plan");

    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let plan: Value = serde_json::from_slice(&fs::read(fixture.output("plan.json")).expect("plan"))
        .expect("JSON");
    let models = plan["selected_models"].as_array().expect("models");
    let rows = plan["github_matrix"]["include"].as_array().expect("matrix");
    let mut families = models
        .iter()
        .map(|model| model["family"].as_str().expect("family"))
        .collect::<Vec<_>>();
    families.sort_unstable();
    assert_eq!(families, ["alpha", "beta", "delta", "gamma", "zeta"]);
    assert_eq!(rows.len(), models.len());
    for row in rows {
        assert_eq!(
            models
                .iter()
                .filter(|model| model["family"] == row["families"])
                .count(),
            1
        );
    }
    assert_eq!(plan["manifest"], "ci/llama-canary/family-certified.json");
}

#[test]
fn wrong_selected_revision_fails_before_battery_or_output() {
    let mut fixture = Fixture::new();
    fixture.input["selected_revision"] = "a".repeat(40).into();

    let result = fixture.run("source-plan");

    assert!(!result.status.success());
    assert!(!fixture.output("plan.json").exists());
}

#[test]
fn battery_failure_cannot_publish_a_verified_plan_identity() {
    let fixture = Fixture::new();
    let source = fixture.directory.path().join("selected");
    fs::write(source.join("scripts/skippy-family-battery.sh"), "exit 37\n")
        .expect("rejecting consumer");

    let result = fixture.run("source-plan");

    assert!(!result.status.success());
    assert!(fixture.output("plan.json").exists());
    assert!(!fixture.output("source-plan.json").exists());
}

#[test]
fn identity_binds_exact_plan_bytes_when_preflight_passes() {
    let fixture = Fixture::new();

    let result = fixture.run("source-plan");

    assert!(result.status.success());
    let bytes = fs::read(fixture.output("plan.json")).expect("plan");
    let identity: Value =
        serde_json::from_slice(&fs::read(fixture.output("source-plan.json")).expect("identity"))
            .expect("JSON");
    assert_eq!(identity["plan_sha256"], hex::encode(Sha256::digest(bytes)));
    assert_eq!(
        identity["selected_revision"],
        fixture.input["selected_revision"]
    );
    assert_eq!(
        identity["controller_revision"],
        fixture.input["controller_revision"]
    );
    assert_eq!(identity["cache_admission"], "not_checked");
    assert_eq!(identity["gguf_admission"], "pending");
}

#[test]
fn verification_rejects_changed_matrix_even_if_identity_digest_is_rebound() {
    let fixture = Fixture::new();
    assert!(fixture.run("source-plan").status.success());
    let mut plan: Value =
        serde_json::from_slice(&fs::read(fixture.output("plan.json")).expect("plan"))
            .expect("JSON");
    plan["github_matrix"]["include"][0]["families"] = "foreign".into();
    let bytes = serde_json::to_vec(&plan).expect("JSON");
    fs::write(fixture.output("plan.json"), &bytes).expect("modified plan");
    let mut identity: Value =
        serde_json::from_slice(&fs::read(fixture.output("source-plan.json")).expect("identity"))
            .expect("JSON");
    identity["plan_sha256"] = hex::encode(Sha256::digest(bytes)).into();
    fs::write(
        fixture.output("source-plan.json"),
        serde_json::to_vec(&identity).expect("JSON"),
    )
    .expect("identity");

    let result = fixture.run("verify-source-plan");

    assert!(!result.status.success());
}

#[test]
fn verification_preserves_plan_bytes_when_identity_is_valid() {
    let fixture = Fixture::new();
    assert!(fixture.run("source-plan").status.success());
    let before = fs::read(fixture.output("plan.json")).expect("plan");

    let result = fixture.run("verify-source-plan");

    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::read(fixture.output("plan.json")).expect("plan"), before);
}

#[test]
fn changed_plan_bytes_fail_verification_without_rebinding_identity() {
    let fixture = Fixture::new();
    assert!(fixture.run("source-plan").status.success());
    let mut bytes = fs::read(fixture.output("plan.json")).expect("plan");
    bytes.push(b' ');
    fs::write(fixture.output("plan.json"), bytes).expect("changed plan");

    let result = fixture.run("verify-source-plan");

    assert!(!result.status.success());
}

#[test]
fn missing_cache_fails_before_selected_battery_is_run() {
    let mut fixture = Fixture::new();
    let cache = fixture.directory.path().join("cache");
    fs::create_dir_all(cache.join("hub")).expect("empty cache");
    fixture.input["cache"] = json!({"mode":"blob_identity","root":cache});

    let result = fixture.run("source-plan");

    assert!(!result.status.success());
    assert!(!fixture.output("plan.json").exists());
}

#[cfg(unix)]
#[test]
fn corrupt_content_addressed_blob_fails_even_when_name_and_size_match() {
    let mut fixture = Fixture::new();
    let cache = fixture.directory.path().join("cache");
    let repository = cache.join("hub/models--fixture--zeta");
    let snapshot = repository.join(format!("snapshots/{}", "a".repeat(40)));
    fs::create_dir_all(repository.join("blobs")).expect("blobs");
    fs::create_dir_all(&snapshot).expect("snapshot");
    let blob = repository.join("blobs").join("a".repeat(64));
    fs::write(&blob, b"x").expect("one byte blob");
    std::os::unix::fs::symlink(blob, snapshot.join("zeta.gguf")).expect("snapshot link");
    fixture.input["cache"] = json!({"mode":"blob_identity","root":cache});

    let result = fixture.run("source-plan");

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("SHA-256 mismatch"));
    assert!(!fixture.output("plan.json").exists());
}

#[test]
fn consumer_plan_mutation_cannot_publish_identity() {
    let fixture = Fixture::new();
    fs::write(
        fixture
            .directory
            .path()
            .join("selected/scripts/skippy-family-battery.sh"),
        "printf ' ' >> \"$4\"\n",
    )
    .expect("mutating battery");

    let result = fixture.run("source-plan");

    assert!(!result.status.success());
    assert!(!fixture.output("source-plan.json").exists());
}

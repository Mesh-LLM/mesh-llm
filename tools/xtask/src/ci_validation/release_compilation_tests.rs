use super::*;

const VALID: &str = "jobs:\n  metadata:\n    runs-on: ubuntu-24.04\n    steps:\n      - uses: mozilla-actions/sccache-action@pin\n      - uses: ./.github/actions/configure-sccache-gha\n      - run: cargo fmt --all --check\n";

#[test]
fn accepts_hosted_setup_before_cargo() {
    check(Path::new("."), VALID).unwrap();
}

#[test]
fn rejects_absent_late_swapped_conditional_or_suppressed_setup() {
    for source in [
        VALID.replace("      - uses: mozilla-actions/sccache-action@pin\n", ""),
        VALID.replace("      - uses: ./.github/actions/configure-sccache-gha\n", ""),
        VALID.replace("      - uses: mozilla-actions/sccache-action@pin\n", "      - uses: mozilla-actions/sccache-action@pin\n        if: false\n"),
        VALID.replace("      - uses: mozilla-actions/sccache-action@pin\n", "      - uses: mozilla-actions/sccache-action@pin\n        continue-on-error: true\n"),
        VALID.replace("    steps:", "    continue-on-error: true\n    steps:"),
        VALID.replace("jobs:", "continue-on-error: true\njobs:"),
        VALID.replace("      - uses: mozilla-actions/sccache-action@pin\n      - uses: ./.github/actions/configure-sccache-gha\n", "      - uses: ./.github/actions/configure-sccache-gha\n      - uses: mozilla-actions/sccache-action@pin\n"),
        VALID.replace("      - run: cargo fmt --all --check\n", "").replace("    steps:\n", "    steps:\n      - run: cargo fmt --all --check\n"),
    ] {
        assert!(check(Path::new("."), &source).is_err(), "{source}");
    }
}

#[test]
fn rejects_direct_cargo_in_composer() {
    let source =
        "steps:\n  - run: cargo build -p xtask\n  - uses: ./.github/actions/restore-automation\n";
    let job = workflow_yaml::parse(source).unwrap();
    assert!(
        check_composer(Path::new("."), &job)
            .unwrap_err()
            .to_string()
            .contains("composition-only")
    );
}

#[test]
fn rejects_bootstrap_hidden_in_local_action() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join(".github/actions/bootstrap")).unwrap();
    std::fs::write(
        root.path().join(".github/actions/bootstrap/action.yml"),
        "runs:\n  using: composite\n  steps:\n    - run: just automation-bootstrap\n",
    )
    .unwrap();
    std::fs::create_dir_all(root.path().join(".github/actions/restore-automation")).unwrap();
    std::fs::write(
        root.path()
            .join(".github/actions/restore-automation/action.yml"),
        "runs:\n  steps:\n    - run: true\n",
    )
    .unwrap();
    let source = "steps:\n  - uses: ./.github/actions/restore-automation\n  - uses: ./.github/actions/bootstrap\n";
    let job = workflow_yaml::parse(source).unwrap();
    assert!(
        check_composer(root.path(), &job)
            .unwrap_err()
            .to_string()
            .contains("reaches Cargo")
    );
}

#[test]
fn package_bypass_requires_effective_guards_on_package_step() {
    let source = "jobs:\n  compose_linux_cuda:\n    env:\n      MESH_RELEASE_HOST_PRESTAMPED: '1'\n      MESH_RELEASE_ATTESTATION_PREVERIFIED: '1'\n    steps:\n      - run: scripts/package-release.sh dist\n";
    let document = workflow_yaml::parse(source).unwrap();
    let job = document
        .get("jobs")
        .unwrap()
        .get("compose_linux_cuda")
        .unwrap();
    assert!(!cargo_step(job, &steps(job)[0]));
    let unguarded = source.replace(
        "MESH_RELEASE_HOST_PRESTAMPED: '1'",
        "MESH_RELEASE_HOST_PRESTAMPED: '0'",
    );
    let document = workflow_yaml::parse(&unguarded).unwrap();
    let job = document
        .get("jobs")
        .unwrap()
        .get("compose_linux_cuda")
        .unwrap();
    assert!(cargo_step(job, &steps(job)[0]));
}

#[test]
fn ignores_comments_and_names_but_scans_jobs_after_comments() {
    let source = format!(
        "{VALID}# publication\n  publish:\n    steps:\n      - name: cargo mentioned here\n        run: |\n          # cargo ignored here\n          scripts/release-version.sh tag\n"
    );
    assert!(
        check(Path::new("."), &source)
            .unwrap_err()
            .to_string()
            .contains("publish")
    );
}

#[test]
fn immutable_restore_exports_verified_bytes_and_rejects_tampering() {
    use sha2::{Digest, Sha256};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let action =
        std::fs::read_to_string(root.join(".github/actions/restore-automation/action.yml"))
            .unwrap();
    let document = workflow_yaml::parse(&action).unwrap();
    let run = steps(document.get("runs").unwrap())
        .iter()
        .find(|step| field(step, "name") == Some("Verify and export automation"))
        .and_then(|step| field(step, "run"))
        .unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let artifact = temporary.path().join("immutable-automation-restored");
    std::fs::create_dir(&artifact).unwrap();
    let bytes = b"#!/bin/sh\nexit 0\n";
    let source = b"bd5514dd8de8db64d42975d04de0b0ad0583c7ba\n";
    std::fs::write(artifact.join("xtask"), bytes).unwrap();
    std::fs::write(artifact.join("source.txt"), source).unwrap();
    std::fs::write(
        artifact.join("SHA256SUMS"),
        format!(
            "{}  xtask\n{}  source.txt\n",
            hex::encode(Sha256::digest(bytes)),
            hex::encode(Sha256::digest(source))
        ),
    )
    .unwrap();
    let environment = temporary.path().join("env");
    let execute = || {
        std::process::Command::new("bash")
            .args(["-c", run])
            .env("RUNNER_TEMP", temporary.path())
            .env("RUNNER_OS", "Linux")
            .env("AUTOMATION_ARTIFACT_ID", "")
            .env("AUTOMATION_BINARY_SHA256", "")
            .env(
                "AUTOMATION_SOURCE_SHA",
                "bd5514dd8de8db64d42975d04de0b0ad0583c7ba",
            )
            .env("GITHUB_ENV", &environment)
            .output()
            .unwrap()
    };
    let success = execute();
    assert!(
        success.status.success(),
        "{}",
        String::from_utf8_lossy(&success.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(&environment).unwrap(),
        format!("MESH_LLM_AUTOMATION_BIN={}/xtask\n", artifact.display())
    );
    std::fs::write(artifact.join("xtask"), b"changed").unwrap();
    assert!(!execute().status.success());
}

#![cfg(feature = "automation-contracts")]
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{Duration, SystemTime},
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
}
struct Fixture {
    directory: tempfile::TempDir,
    bin: PathBuf,
    main: String,
}
impl Fixture {
    fn new(mode: &str, active: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path();
        fs::create_dir_all(path.join("tools/xtask")).unwrap();
        fs::create_dir_all(path.join("ci/governance")).unwrap();
        fs::write(path.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::write(path.join("tools/xtask/Cargo.toml"), "[package]\n").unwrap();
        let activation = json!({"schema_version":1,"repository":"Mesh-LLM/mesh-llm","active":active,
            "repository_id":1,"reviewer_team_id":2,"administration_environment_id":3,
            "observer_app_id":4,"observer_installation_id":5,"publisher_app_id":4,"publisher_installation_id":5,
            "native_ruleset_ids":[8],"human_ids":{"micspiral":11,"ndizazzo":12,"i386":13}});
        fs::write(
            path.join("ci/governance/activation.json"),
            serde_json::to_vec(&activation).unwrap(),
        )
        .unwrap();
        let policy = fs::read_to_string(root().join("ci/governance/policy.json")).unwrap();
        fs::write(path.join("ci/governance/policy.json"), &policy).unwrap();
        git(path, &["init", "-q", "-b", "main"]);
        git(path, &["add", "."]);
        git(
            path,
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
        );
        let main = String::from_utf8(git(path, &["rev-parse", "HEAD"]).stdout)
            .unwrap()
            .trim()
            .to_owned();
        let seconds = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 600;
        let expires_at = plist::Date::from(SystemTime::UNIX_EPOCH + Duration::from_secs(seconds))
            .to_xml_format();
        fs::write(
            path.join(".governance-fixture.json"),
            serde_json::to_vec(
                &json!({"main_sha":main,"policy":policy,"expires_at":expires_at,"mode":mode}),
            )
            .unwrap(),
        )
        .unwrap();
        let bin = path.join("fixture-bin");
        fs::create_dir(&bin).unwrap();
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| root().join("target"));
        let example = target.join("debug/examples").join(if cfg!(windows) {
            "ci_governance_github_fixture.exe"
        } else {
            "ci_governance_github_fixture"
        });
        assert!(
            example.is_file(),
            "prebuild inert ci_governance_github_fixture example through Just"
        );
        for tool in ["gh", "curl"] {
            fs::copy(
                &example,
                bin.join(if cfg!(windows) {
                    format!("{tool}.exe")
                } else {
                    tool.to_owned()
                }),
            )
            .unwrap();
        }
        Self {
            directory,
            bin,
            main,
        }
    }
    fn run(&self, args: &[&str], key: bool, event: bool) -> Output {
        let mut paths = vec![self.bin.clone()];
        paths.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command
            .current_dir(self.directory.path())
            .arg("--repo-root")
            .arg(self.directory.path())
            .arg("ci-governance")
            .args(args)
            .env("PATH", std::env::join_paths(paths).unwrap())
            .env_remove("CI_GOVERNANCE_APP_PRIVATE_KEY")
            .env_remove("CI_GOVERNANCE_APP_JWT")
            .env_remove("CI_GOVERNANCE_OBSERVER_TOKEN");
        if key {
            command.env(
                "CI_GOVERNANCE_APP_PRIVATE_KEY",
                include_str!("ci_governance_fixtures/synthetic-app-test-key.pem"),
            );
        } else {
            command.env("CI_GOVERNANCE_APP_JWT", "fixture-jwt");
        }
        if event {
            let event_path = self.directory.path().join(".event.json");
            let fixture: Value = serde_json::from_slice(
                &fs::read(self.directory.path().join(".governance-fixture.json")).unwrap(),
            )
            .unwrap();
            let reconcile = fixture["mode"] == "reconcile";
            fs::write(&event_path,serde_json::to_vec(&json!({"action":"completed","repository":{"id":1,"full_name":"Mesh-LLM/mesh-llm"},"workflow_run":{"id":500},"ref":"refs/heads/main"})).unwrap()).unwrap();
            command.env("GITHUB_EVENT_PATH",event_path).env("GITHUB_EVENT_NAME",if reconcile {"push"}else{"workflow_run"})
                .env("GITHUB_REPOSITORY","Mesh-LLM/mesh-llm").env("GITHUB_REF","refs/heads/main")
                .env("GITHUB_SHA",&self.main).env("GITHUB_WORKFLOW_SHA",&self.main)
                .env("GITHUB_WORKFLOW_REF","Mesh-LLM/mesh-llm/.github/workflows/ci-governance-publisher.yml@refs/heads/main");
        }
        command.output().unwrap()
    }
    fn calls(&self) -> Vec<Value> {
        fs::read_to_string(self.directory.path().join(".governance-calls.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    fn advance_main(&mut self) {
        let policy_path = self.directory.path().join("ci/governance/policy.json");
        let mut policy = fs::read_to_string(&policy_path).unwrap();
        policy.push('\n');
        fs::write(&policy_path, &policy).unwrap();
        git(self.directory.path(), &["add", "ci/governance/policy.json"]);
        git(
            self.directory.path(),
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "new protected policy revision",
            ],
        );
        self.main = String::from_utf8(git(self.directory.path(), &["rev-parse", "HEAD"]).stdout)
            .unwrap()
            .trim()
            .to_owned();
        let path = self.directory.path().join(".governance-fixture.json");
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["main_sha"] = json!(self.main);
        value["policy"] = json!(policy);
        fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    }
}
fn git(root: &Path, args: &[&str]) -> Output {
    let output =
        Command::new(std::env::var_os("MIGRATION_TEST_GIT").unwrap_or_else(|| "git".into()))
            .current_dir(root)
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
fn assert_clean_output(output: &Output) {
    for bytes in [&output.stdout, &output.stderr] {
        let text = String::from_utf8_lossy(bytes);
        for forbidden in [
            "fixture-jwt",
            "fixture-observer",
            "fixture-publisher",
            "PRIVATE KEY",
        ] {
            assert!(!text.contains(forbidden), "credential material leaked");
        }
    }
}

#[test]
fn actual_publisher_creates_pending_before_evaluation_and_publishes_current_allow_or_deny() {
    for (mode, code, conclusion) in [("approved", 0, "success"), ("denied", 1, "failure")] {
        let fixture = Fixture::new(mode, true);
        let output = fixture.run(&["publish-pr", "2781"], false, false);
        assert_eq!(
            output.status.code(),
            Some(code),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_clean_output(&output);
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["conclusion"], conclusion);
        assert_eq!(report["native_enforcement_verified"], false);
        let calls = fixture.calls();
        let create = calls
            .iter()
            .position(|entry| {
                entry["value"]["method"] == "POST"
                    && entry["value"]["endpoint"] == "repos/Mesh-LLM/mesh-llm/check-runs"
            })
            .unwrap();
        let review = calls
            .iter()
            .position(|entry| {
                entry["value"]["body"]["query"]
                    .as_str()
                    .is_some_and(|query| query.contains("reviews("))
            })
            .unwrap();
        assert!(create < review);
        assert_eq!(
            calls
                .iter()
                .filter(|entry| entry["value"]["method"] == "PATCH")
                .count(),
            1
        );
        for call in calls
            .iter()
            .filter(|entry| entry["kind"] == "jwt-header-path")
        {
            assert!(
                !Path::new(call["value"].as_str().unwrap()).exists(),
                "JWT header tempfile retained"
            );
        }
    }
}

#[test]
fn real_cli_rejects_forged_check_readbacks_and_broader_minted_permissions_before_patch() {
    for mode in ["wrong-app", "wrong-head", "wrong-external", "wide-token"] {
        let fixture = Fixture::new(mode, true);
        let output = fixture.run(&["publish-pr", "2781"], false, false);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_clean_output(&output);
        assert!(
            !fixture
                .calls()
                .iter()
                .any(|entry| entry["value"]["method"] == "PATCH"),
            "{mode}"
        );
    }
}

#[test]
fn committed_inactive_activation_cannot_be_enabled_by_a_dirty_worktree_file() {
    let fixture = Fixture::new("approved", false);
    let path = fixture
        .directory
        .path()
        .join("ci/governance/activation.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value["active"] = json!(true);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = fixture.run(&["publish-pr", "2781"], false, false);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("inactive"));
    assert!(fixture.calls().is_empty());
}

#[test]
fn actual_openssl_factory_signs_synthetic_key_without_credential_argv_or_persistent_headers() {
    let fixture = Fixture::new("approved", true);
    let output = fixture.run(&["collect-pr", "2781"], true, false);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_clean_output(&output);
    let snapshot: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(snapshot["snapshot"]["head_sha"], "2".repeat(40));
    assert!(!fixture.calls().iter().any(|entry| {
        entry["value"]["endpoint"]
            .as_str()
            .is_some_and(|endpoint| endpoint.contains("check-runs"))
    }));
}

#[test]
fn documented_review_run_metadata_resolves_empty_association_but_refuses_changed_lock_target() {
    let fixture = Fixture::new("approved", true);
    let output = fixture.run(&["resolve-publisher-event"], false, true);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let target: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(target["pr_numbers"], json!([2781]));
    let fixture = Fixture::new("retarget", true);
    let output = fixture.run(&["publish-event", "--expected-pr", "2781"], false, true);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("target changed"));
    assert!(
        !fixture
            .calls()
            .iter()
            .any(|entry| entry["value"]["endpoint"] == "repos/Mesh-LLM/mesh-llm/check-runs")
    );
}

#[test]
fn main_push_reconciliation_refreshes_the_same_pr_head_under_new_protected_policy() {
    let mut fixture = Fixture::new("reconcile", true);
    assert!(
        fixture
            .run(&["publish-pr", "2781"], false, false)
            .status
            .success()
    );
    let before: Value = serde_json::from_slice(
        &fs::read(fixture.directory.path().join(".governance-check.json")).unwrap(),
    )
    .unwrap();
    fixture.advance_main();
    let resolved = fixture.run(&["resolve-publisher-event"], false, true);
    assert!(
        resolved.status.success(),
        "{}",
        String::from_utf8_lossy(&resolved.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&resolved.stdout).unwrap()["pr_numbers"],
        json!([2781])
    );
    let published = fixture.run(&["publish-event", "--expected-pr", "2781"], false, true);
    assert!(
        published.status.success(),
        "{}",
        String::from_utf8_lossy(&published.stderr)
    );
    let after: Value = serde_json::from_slice(
        &fs::read(fixture.directory.path().join(".governance-check.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(before["head_sha"], after["head_sha"]);
    assert_ne!(before["external_id"], after["external_id"]);
    assert!(
        after["external_id"]
            .as_str()
            .unwrap()
            .ends_with(&fixture.main)
    );
}

#[test]
fn workflow_contracts_preserve_main_source_scoped_secrets_and_independent_per_pr_work() {
    let publisher =
        fs::read_to_string(root().join(".github/workflows/ci-governance-publisher.yml")).unwrap();
    assert!(!publisher.contains("checks: write"));
    assert!(!publisher.contains("group: ci-governance-publisher\n"));
    assert!(publisher.contains("needs.resolve.outputs.repository_id"));
    assert!(publisher.contains("matrix.pr_number"));
    assert!(publisher.contains("pr_numbers"));
    assert!(publisher.contains("max-parallel: 4"));
    assert!(publisher.contains("fail-fast: false"));
    assert!(publisher.contains("branches: [main]"));
    assert!(publisher.contains("*/15 * * * *"));
    assert!(publisher.contains("publish-event --expected-pr"));
    assert!(publisher.contains("source-profile: protected-clean"));
    assert!(publisher.contains("./.github/actions/restore-automation"));
    assert!(!publisher.contains("ref: ${{ github.event.pull_request.head.sha }}"));
    let relay =
        fs::read_to_string(root().join(".github/workflows/ci-governance-review-signal.yml"))
            .unwrap();
    assert!(relay.contains("permissions: {}"));
    assert!(!relay.contains("checkout"));
    assert!(!relay.contains("secrets."));
    let admin =
        fs::read_to_string(root().join(".github/workflows/ci-governance-administration.yml"))
            .unwrap();
    assert!(admin.contains("workflow_dispatch:"));
    assert!(!admin.contains("pull_request:"));
    assert!(admin.contains("environment: ci-governance-administration"));
    assert!(admin.contains("inspect-administration-approval"));
    assert!(!admin.contains("ADMIN_TOKEN"));
}

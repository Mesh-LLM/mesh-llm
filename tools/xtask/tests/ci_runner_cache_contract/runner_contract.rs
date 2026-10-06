//! Actual retained plan/scan adapters; local execution does not prove hosted authority.
use crate::{
    support,
    workflow_yaml::{self, Node},
};
use serde_json::{Value, json};
use std::{fs, path::PathBuf, process::Command};
fn body(name: &str) -> String {
    let text =
        fs::read_to_string(support::root().join(".github/workflows/ci-runner-contract-slice.yml"))
            .unwrap();
    let workflow = workflow_yaml::parse(&text).unwrap();
    let Node::Seq(steps) = workflow
        .get("jobs")
        .unwrap()
        .get("runner_contract")
        .unwrap()
        .get("steps")
        .unwrap()
    else {
        panic!("steps")
    };
    steps
        .iter()
        .find(|step| step.get("name").and_then(Node::text) == Some(name))
        .unwrap()
        .get("run")
        .and_then(Node::text)
        .unwrap()
        .to_owned()
}
fn command(fixture: &support::Fixture, source: &str) -> Command {
    let mut command = Command::new("bash");
    let mut paths = vec![fixture.path().join("bin")];
    paths.extend(std::env::split_paths(
        &std::env::var_os("PATH").expect("fixture tools PATH"),
    ));
    command
        .env_clear()
        .env("PATH", std::env::join_paths(paths).unwrap())
        .current_dir(fixture.path())
        .args(["-c", source]);
    command
}
fn plan() -> Value {
    json!({"schema_version":1,"profile":"pr-targeted","budgets":{"total_max_workers":4},"required_slices":["linux","windows"],"cache_modes":{"linux":"restore-only","windows":"restore-only"},"runner_roles":{"linux":"hosted-linux","windows":"hosted-windows"}})
}
#[test]
fn authority_actual_runner_plan_rejects_wrong_profile_budget_roles_and_trusted_pr_cache() {
    let fixture = support::Fixture::new();
    let source = body("Validate plan ownership and cache boundary");
    let valid = plan();
    let mut rows = vec![(valid.clone(), true)];
    for path in [
        vec!["schema_version"],
        vec!["profile"],
        vec!["budgets", "total_max_workers"],
        vec!["required_slices"],
        vec!["runner_roles"],
        vec!["runner_roles", "linux"],
        vec!["cache_modes", "linux"],
    ] {
        let mut changed = valid.clone();
        let replacement = match path.as_slice() {
            ["schema_version"] => json!(2),
            ["profile"] => json!("main"),
            ["budgets", _] => json!(5),
            ["required_slices"] => json!([]),
            ["runner_roles"] => json!({"linux":"hosted-linux"}),
            ["runner_roles", _] => json!("DePoT-ordinary"),
            _ => json!("trusted-readwrite"),
        };
        let mut slot = &mut changed;
        for key in path {
            slot = &mut slot[key];
        }
        *slot = replacement;
        rows.push((changed, false));
    }
    for replacement in [Value::Null, json!(false), json!(""), json!(1)] {
        let mut changed = valid.clone();
        changed["runner_roles"]["linux"] = replacement;
        rows.push((changed, false));
    }
    for (document, expected) in rows {
        let mut child = command(&fixture, &source);
        child
            .env("PLAN_JSON", serde_json::to_string(&document).unwrap())
            .env("PROFILE", "pr-targeted")
            .env("TOTAL_MAX_WORKERS", "4");
        let output = fixture.run(child);
        assert_eq!(
            output.status.success(),
            expected,
            "{document}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fixture.0.close().unwrap();
}
fn scan_tree(fixture: &support::Fixture) {
    let workflows = fixture.path().join(".github/workflows");
    fs::create_dir_all(&workflows).unwrap();
    for name in [
        "main_fixture.yml",
        "pr_quality.yml",
        "pr_website.yml",
        "pr_linux.yml",
        "pr_macos.yml",
        "pr_windows.yml",
        "ci-control.yml",
        "ci-fixture-lane.yml",
        "ci-fixture-slice.yml",
        "ci-runner-contract-slice.yml",
    ] {
        fs::write(workflows.join(name), "on:\n  pull_request:\n").unwrap();
    }
    let action = fixture.path().join(".github/actions/select-ci-runners");
    fs::create_dir_all(&action).unwrap();
    fs::write(
        action.join("action.yml"),
        "pull_request|pull_request_target\ndepot_enabled=false\n",
    )
    .unwrap();
}
#[test]
fn authority_actual_runner_scan_refuses_missing_forbidden_sources_and_preserves_self_exclusion() {
    let source = body("Verify PR runner policy remains fail-closed");
    for mode in [
        "valid",
        "self-only",
        "forbidden",
        "missing",
        "selector-drift",
    ] {
        let fixture = support::Fixture::new();
        scan_tree(&fixture);
        match mode {
            "self-only" => fs::write(
                fixture
                    .path()
                    .join(".github/workflows/ci-runner-contract-slice.yml"),
                "on:\n  pull_request_target:\n",
            )
            .unwrap(),
            "forbidden" => fs::write(
                fixture.path().join(".github/workflows/pr_quality.yml"),
                "on:\n  pull_request_target:\n",
            )
            .unwrap(),
            "missing" => {
                fs::remove_file(fixture.path().join(".github/workflows/pr_linux.yml")).unwrap()
            }
            "selector-drift" => fs::write(
                fixture
                    .path()
                    .join(".github/actions/select-ci-runners/action.yml"),
                "depot_enabled=true\n",
            )
            .unwrap(),
            _ => (),
        }
        let output = fixture.run(command(&fixture, &source));
        assert_eq!(
            output.status.success(),
            ["valid", "self-only"].contains(&mode),
            "{mode}"
        );
        fixture.0.close().unwrap();
    }
}
fn grep_binary() -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|p| p.join("grep"))
        .find(|p| p.is_file())
        .expect("existing native grep")
}
#[test]
fn authority_actual_runner_scan_fails_closed_on_grep_io_error() {
    let fixture = support::Fixture::new();
    scan_tree(&fixture);
    let grep = grep_binary();
    let quoted = format!("'{}'", grep.to_str().unwrap().replace('\'', "'\\''"));
    fixture.executable("grep",&format!("if [[ $1 == -nE ]]; then printf '%s\\n' scan-io-error >&2; exit 2; fi\nexec {quoted} \"$@\""));
    let output = fixture.run(command(
        &fixture,
        &body("Verify PR runner policy remains fail-closed"),
    ));
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("scan-io-error"));
    fixture.0.close().unwrap();
}

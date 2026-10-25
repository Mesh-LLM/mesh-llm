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

#[test]
fn authority_actual_runner_scan_excludes_pr_maintenance_but_scans_every_validation_entry() {
    use std::collections::BTreeSet;
    let fixture = support::Fixture::new();
    scan_tree(&fixture);
    for name in [
        "pr_auto_assign.yml",
        "pr_cleanup.yml",
        "pr_unreviewed_fixture.yml",
    ] {
        fs::write(
            fixture.path().join(".github/workflows").join(name),
            "on:\n  pull_request_target:\n",
        )
        .unwrap();
    }
    let grep = grep_binary();
    let quoted = format!("'{}'", grep.to_str().unwrap().replace('\'', "'\\''"));
    fixture.executable(
        "grep",
        &format!(
            "if [[ $1 == -nE ]]; then printf '%s\\n' \"${{@:3}}\" > \"$SCAN_FIXTURE_ARGUMENTS\"; fi\nexec {quoted} \"$@\""
        ),
    );
    let mut child = command(
        &fixture,
        &body("Verify PR runner policy remains fail-closed"),
    );
    child.env(
        "SCAN_FIXTURE_ARGUMENTS",
        fixture.path().join("scan-arguments"),
    );
    let output = fixture.run(child);
    assert!(output.status.success(), "{output:?}");
    let arguments = fs::read_to_string(fixture.path().join("scan-arguments")).unwrap();
    let actual_pr = arguments
        .lines()
        .filter(|path| path.starts_with(".github/workflows/pr_"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_pr,
        BTreeSet::from([
            ".github/workflows/pr_quality.yml",
            ".github/workflows/pr_website.yml",
            ".github/workflows/pr_linux.yml",
            ".github/workflows/pr_macos.yml",
            ".github/workflows/pr_windows.yml",
        ])
    );
    assert!(
        arguments
            .lines()
            .any(|path| path == ".github/workflows/ci-control.yml")
    );
    assert!(
        !arguments
            .lines()
            .any(|path| path == ".github/workflows/ci-runner-contract-slice.yml")
    );
    fixture.0.close().expect("owned scanner fixture cleanup");
}

#[test]
fn authority_legacy_main_filename_is_reusable_only_without_event_authority() {
    let source = fs::read_to_string(support::root().join(".github/workflows/ci.yml")).unwrap();
    let document = workflow_yaml::parse(&source).unwrap();
    let triggers = document.get("on").unwrap().entries();
    assert_eq!(
        triggers
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        ["workflow_call"]
    );
    assert!(document.get("permissions").unwrap().entries().is_empty());
    let jobs = document.get("jobs").unwrap().entries();
    assert_eq!(
        jobs.iter().map(|(key, _)| key.as_str()).collect::<Vec<_>>(),
        ["compatibility"]
    );
    let job = &jobs[0].1;
    assert!(job.get("uses").is_none() && job.get("secrets").is_none());
    assert!(job.get("permissions").unwrap().entries().is_empty());
}

#[test]
fn authority_swift_main_remains_hosted_while_pr_placement_uses_bounded_selector() {
    let source =
        fs::read_to_string(support::root().join(".github/workflows/swift-sdk-artifact.yml"))
            .unwrap();
    let document = workflow_yaml::parse(&source).unwrap();
    let policy = document.get("jobs").unwrap().get("runner_policy").unwrap();
    let Node::Seq(steps) = policy.get("steps").unwrap() else {
        panic!("policy steps")
    };
    let calls = steps
        .iter()
        .filter(|step| {
            step.get("uses").and_then(Node::text) == Some("./.github/actions/select-ci-runners")
        })
        .collect::<Vec<_>>();
    let [call] = calls.as_slice() else {
        panic!("one central policy")
    };
    let inputs = call.get("with").unwrap();
    assert_eq!(
        inputs.get("depot_main_enabled").and_then(Node::text),
        Some("false")
    );
    assert_eq!(
        inputs.get("depot_pr_enabled").and_then(Node::text),
        Some("${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}")
    );
}

#[test]
fn authority_platform_slice_policy_outputs_and_runner_jobs_bind_the_same_central_choice() {
    for platform in ["macos", "windows"] {
        let output = format!("runner_{platform}");
        for component in ["host", "runtime", "product"] {
            let name = format!("ci-{platform}-{component}-slice.yml");
            let source =
                fs::read_to_string(support::root().join(".github/workflows").join(&name)).unwrap();
            let document = workflow_yaml::parse(&source).unwrap();
            let jobs = document.get("jobs").unwrap();
            let policy = jobs.get("runner_policy").unwrap();
            assert_eq!(
                policy
                    .get("outputs")
                    .unwrap()
                    .get(&output)
                    .and_then(Node::text),
                Some(format!("${{{{ steps.policy.outputs.{output} }}}}").as_str()),
                "{name}"
            );
            let expected = format!("${{{{ needs.runner_policy.outputs.{output} }}}}");
            let mut owned_runner = false;
            for (job_name, job) in jobs.entries() {
                if job_name == "runner_policy" || job.get("runs-on").is_none() {
                    continue;
                }
                assert_eq!(
                    job.get("runs-on").and_then(Node::text),
                    Some(expected.as_str()),
                    "{name}/{job_name}"
                );
                assert!(
                    job.get("needs").unwrap().list().contains(&"runner_policy"),
                    "{name}/{job_name}"
                );
                owned_runner = true;
            }
            assert!(owned_runner, "{name}: no platform execution owner");
        }
    }
}

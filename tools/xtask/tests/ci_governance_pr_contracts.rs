#![cfg(feature = "automation-contracts")]
#[path = "ci_governance_fixtures/support.rs"]
mod support;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs};
use support::{TestResult, denied, root, run, success, write_json};

const POLICY: &str = "ci/governance/policy.json";

fn snapshot() -> Value {
    let mut snapshot: Value =
        serde_json::from_str(include_str!("ci_governance_fixtures/pr-approved.json")).unwrap();
    snapshot["policy_sha256"] = json!(hex::encode(Sha256::digest(
        fs::read(root().join(POLICY)).unwrap()
    )));
    snapshot["collection"]["observed_at"] = json!(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    snapshot
}

fn validate(snapshot: &Value) -> std::process::Output {
    let directory = tempfile::tempdir().unwrap();
    let path = write_json(directory.path(), "PR snapshot.json", snapshot);
    run(&[
        "validate-pr",
        "--policy",
        POLICY,
        "--snapshot",
        path.to_str().unwrap(),
    ])
}

#[test]
fn one_independent_approval_passes_and_product_only_changes_need_no_extra_review() {
    let approved = success(validate(&snapshot()));
    assert_eq!(approved["policy_change"], true);
    assert_eq!(approved["independent_approvers"], json!(["micspiral"]));
    assert_eq!(
        approved["evidence_status"],
        "local-snapshot-validation-only"
    );
    for path in [
        "mesh/crates/mesh-llm-ui/src/component.tsx",
        "skippy/llama_cpp/patches/product-kernel.patch",
    ] {
        let mut product = snapshot();
        product["changes"][0]["path"] = json!(path);
        product["reviews"] = json!([]);
        product["team"] = Value::Null;
        let result = success(validate(&product));
        assert_eq!(result["policy_change"], false, "{path}");
        assert_eq!(result["independent_approvers"], json!([]));
    }
}

#[test]
fn missing_self_stale_dismissed_and_bot_sponsor_evidence_fail_on_the_real_cli() {
    for variant in 0..5 {
        let mut input = snapshot();
        match variant {
            0 => input["reviews"] = json!([]),
            1 => {
                input["author"] = input["reviews"][0]["reviewer"].clone();
                input["sponsor"] = input["author"].clone();
            }
            2 => input["reviews"][0]["commit_sha"] = input["base_sha"].clone(),
            3 => input["reviews"][0]["state"] = json!("dismissed"),
            _ => input["author"]["kind"] = json!("bot"),
        }
        denied(
            validate(&input),
            if variant == 4 {
                "authenticated sponsor"
            } else {
                "independent eligible human approval"
            },
        );
    }
}

#[test]
fn metadata_claims_cannot_replace_exact_policy_head_or_complete_collection() {
    for (pointer, value, reason) in [
        (
            "/policy_sha256",
            json!("0".repeat(64)),
            "protected-base policy",
        ),
        (
            "/current_head_sha",
            json!("3".repeat(40)),
            "exact current head",
        ),
        (
            "/current_protected_sha",
            json!("3".repeat(40)),
            "protected-base policy",
        ),
        ("/collection/files_complete", json!(false), "incomplete"),
        ("/collection/observed_at", json!(1), "stale"),
        ("/collection/observed_at", json!(u64::MAX), "future"),
        ("/collection/truncated", json!(true), "truncated"),
        ("/team/repository_permission", json!("read"), "write access"),
    ] {
        let mut input = snapshot();
        *input.pointer_mut(pointer).unwrap() = value;
        denied(validate(&input), reason);
    }
}

#[test]
fn checked_in_policy_and_terminal_codeowners_block_are_required_quality_contracts() -> TestResult {
    let result = success(run(&[
        "validate-ownership",
        "--policy",
        POLICY,
        "--codeowners",
        ".github/CODEOWNERS",
    ]));
    assert_eq!(result["reviewer_team"], "Mesh-LLM/ci-policy-reviewers");
    assert_eq!(result["evidence_status"], "local-ownership-validation-only");
    let patterns = result["protected_patterns"].as_array().ok_or("patterns")?;
    for pattern in ["/.github/", "/ci/", "/tools/", "/.agents/skills/manage-ci/"] {
        assert!(patterns.contains(&json!(pattern)), "missing {pattern}");
    }
    let directory = tempfile::tempdir()?;
    let overridden = directory.path().join("CODEOWNERS");
    let mut source = fs::read_to_string(root().join(".github/CODEOWNERS"))?;
    source.push_str("\n/.github/workflows/windows.yml @unlisted\n");
    fs::write(&overridden, source)?;
    denied(
        run(&[
            "validate-ownership",
            "--policy",
            POLICY,
            "--codeowners",
            overridden.to_str().ok_or("path")?,
        ]),
        "later overrides are forbidden",
    );
    Ok(())
}

#[test]
fn default_help_invalid_options_and_explicit_root_keep_exit_and_output_contracts() -> TestResult {
    for args in [&[][..], &["--help"][..]] {
        let output = run(args);
        assert_eq!(output.status.code(), Some(0));
        let text = String::from_utf8(output.stdout)?;
        for command in [
            "validate-pr",
            "validate-ownership",
            "check-drift",
            "validate-change",
        ] {
            assert!(text.contains(command));
        }
        assert!(text.contains("Local snapshot checks only"));
    }
    for (args, reason) in [
        (vec!["validate-pr"], "missing --policy"),
        (vec!["validate-pr", "--policy"], "flag/value pairs"),
        (
            vec!["validate-pr", "--policy", POLICY, "--policy", POLICY],
            "duplicate",
        ),
        (
            vec!["validate-pr", "--policy", POLICY, "--unknown", "value"],
            "required options",
        ),
        (
            vec!["execute", "--policy", POLICY],
            "unknown ci-governance command",
        ),
    ] {
        denied(run(&args), reason);
    }
    let outside = tempfile::tempdir()?;
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(outside.path())
        .arg("--repo-root")
        .arg(root())
        .args([
            "ci-governance",
            "validate-ownership",
            "--policy",
            POLICY,
            "--codeowners",
            ".github/CODEOWNERS",
        ])
        .output()?;
    success(output);
    Ok(())
}

#[test]
fn every_governance_target_is_in_both_exact_quality_rosters() -> TestResult {
    let targets: Vec<String> = serde_json::from_slice(&fs::read(
        root().join("ci/quality-rust-contract-targets.json"),
    )?)?;
    let recipe = fs::read_to_string(root().join("just/ci.just"))?;
    let body = recipe
        .split("\nci-automation-contracts:\n")
        .nth(1)
        .ok_or("normal recipe")?
        .split("\n# ")
        .next()
        .ok_or("recipe body")?;
    let selected: BTreeSet<_> = body.split_whitespace().collect();
    for entry in fs::read_dir(root().join("tools/xtask/tests"))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or("target name")?;
        if name.starts_with("ci_governance_") {
            assert_eq!(
                targets
                    .iter()
                    .filter(|target| target.as_str() == name)
                    .count(),
                1,
                "unregistered {name}"
            );
            assert!(
                selected.contains(name),
                "normal Quality recipe omits {name}"
            );
        }
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn live_inspection_cli_uses_fixed_gets_and_preserves_denials_as_unknown() -> TestResult {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    let temp = tempfile::tempdir()?;
    let bin = temp.path().join("bin");
    fs::create_dir(&bin)?;
    let gh = bin.join("gh");
    fs::write(
        &gh,
        r##"#!/bin/sh
set -eu
directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
method_ok=false
endpoint=
for arg in "$@"; do
    [ "$arg" != GET ] || method_ok=true
    case "$arg" in repos/Mesh-LLM/mesh-llm*|orgs/Mesh-LLM/teams/ci-policy-reviewers*) endpoint=$arg;; esac
done
[ "$method_ok" = true ] || exit 42
case "$endpoint" in
  'repos/Mesh-LLM/mesh-llm'|'repos/Mesh-LLM/mesh-llm/branches/main'|\
  'orgs/Mesh-LLM/teams/ci-policy-reviewers'|\
  'orgs/Mesh-LLM/teams/ci-policy-reviewers/repos/Mesh-LLM/mesh-llm'|\
  'repos/Mesh-LLM/mesh-llm/environments/ci-governance-administration'|\
  'repos/Mesh-LLM/mesh-llm/branches/main/protection'|\
  'repos/Mesh-LLM/mesh-llm/rulesets?per_page=100&page=1') ;;
  *) exit 43;;
esac
printf '%s\n' "$endpoint" >> "$directory/requests"
printf 'HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\n\r\n{"message":"fixture-private-error"}'
exit 1
"##,
    )?;
    fs::set_permissions(&gh, fs::Permissions::from_mode(0o700))?;
    let path = format!("{}:/usr/bin:/bin", bin.display());
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root())
        .args(["ci-governance", "inspect-live-controls"])
        .env("PATH", path)
        .env("MESH_CI_GOVERNANCE_AUDIT_TOKEN", "fixture-private-token")
        .output()?;
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(!stdout.contains("fixture-private-token"));
    assert!(!stdout.contains("fixture-private-error"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-private"));
    let report: Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["status"], "PARTIAL_UNVERIFIED");
    assert_eq!(report["drift_status"], "UNKNOWN");
    assert_eq!(report["activation_authorized"], false);
    for source in [
        "repository",
        "main_branch",
        "reviewer_team",
        "team_repository_access",
        "administration_environment",
        "legacy_branch_protection",
        "repository_rulesets_first_page",
    ] {
        assert_eq!(
            report["observations"][source]["reason"], "access_denied",
            "{source}"
        );
    }
    let requests = fs::read_to_string(bin.join("requests"))?;
    let lines = requests.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 7);
    assert_eq!(lines.iter().copied().collect::<BTreeSet<_>>().len(), 7);
    Ok(())
}

#[cfg(unix)]
#[test]
fn live_inspection_cli_reports_bounded_membership_without_asserting_authority() -> TestResult {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    let temp = tempfile::tempdir()?;
    let bin = temp.path().join("bin");
    fs::create_dir(&bin)?;
    let gh = bin.join("gh");
    fs::write(
        &gh,
        r##"#!/bin/sh
set -eu
directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
method_ok=false
endpoint=
for arg in "$@"; do
    [ "$arg" != GET ] || method_ok=true
    case "$arg" in repos/Mesh-LLM/mesh-llm*|orgs/Mesh-LLM/teams/ci-policy-reviewers*) endpoint=$arg;; esac
done
[ "$method_ok" = true ] || exit 42
printf '%s\n' "$endpoint" >> "$directory/requests"
case "$endpoint" in
  'repos/Mesh-LLM/mesh-llm') body='{"id":1,"full_name":"Mesh-LLM/mesh-llm","default_branch":"main"}';;
  'repos/Mesh-LLM/mesh-llm/branches/main') body='{"name":"main","protected":true,"commit":{"sha":"1111111111111111111111111111111111111111"}}';;
  'orgs/Mesh-LLM/teams/ci-policy-reviewers') body='{"id":2,"slug":"ci-policy-reviewers","privacy":"closed","organization":{"login":"Mesh-LLM"},"parent":null}';;
  'orgs/Mesh-LLM/teams/ci-policy-reviewers/repos/Mesh-LLM/mesh-llm') body='{"id":1,"full_name":"Mesh-LLM/mesh-llm","role_name":"write"}';;
  'repos/Mesh-LLM/mesh-llm/environments/ci-governance-administration') body='{"id":3,"name":"ci-governance-administration","protection_rules":[]}';;
  'repos/Mesh-LLM/mesh-llm/branches/main/protection')
    printf 'HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\n\r\n{"message":"fixture-private-error"}'
    exit 1;;
  'repos/Mesh-LLM/mesh-llm/rulesets?per_page=100&page=1') body='[]';;
  'orgs/Mesh-LLM/teams/ci-policy-reviewers/members?per_page=100&page=1') body='[{"id":11,"login":"micspiral","type":"User"},{"id":12,"login":"ndizazzo","type":"User"},{"id":13,"login":"i386","type":"User"}]';;
  'orgs/Mesh-LLM/teams/ci-policy-reviewers/members?per_page=100&page=2') body='[]';;
  *) exit 43;;
esac
printf 'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n%s' "$body"
"##,
    )?;
    fs::set_permissions(&gh, fs::Permissions::from_mode(0o700))?;
    let path = format!("{}:/usr/bin:/bin", bin.display());
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root())
        .args(["ci-governance", "inspect-live-controls"])
        .env("PATH", path)
        .env("MESH_CI_GOVERNANCE_AUDIT_TOKEN", "fixture-private-token")
        .output()?;
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(!stdout.contains("fixture-private"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-private"));
    let report: Value = serde_json::from_str(&stdout)?;
    assert_eq!(
        report["observations"]["reviewer_membership_names"]["fields"]["expected_human_login_set"]["value"],
        true
    );
    assert_eq!(
        report["observations"]["legacy_branch_protection"]["reason"],
        "access_denied"
    );
    assert_eq!(report["drift_status"], "UNKNOWN");
    assert_eq!(report["evidence_authenticated"], false);
    assert_eq!(report["execution_authorized"], false);
    let requests = fs::read_to_string(bin.join("requests"))?;
    assert_eq!(requests.lines().count(), 12);
    assert_eq!(
        requests
            .lines()
            .filter(|line| *line == "repos/Mesh-LLM/mesh-llm/branches/main/protection")
            .count(),
        1
    );
    assert_eq!(
        requests
            .lines()
            .filter(|line| *line
                == "orgs/Mesh-LLM/teams/ci-policy-reviewers/members?per_page=100&page=1")
            .count(),
        2
    );
    Ok(())
}

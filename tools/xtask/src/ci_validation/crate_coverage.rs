use crate::command::{DynResult, ensure_contains, ensure_set_eq};
use crate::repo_consistency::{script_workspace_members, workspace_package_names};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn check_ci_crate_test_coverage(
    linux_lane_workflow: &str,
    quality_workflow: &str,
) -> DynResult<()> {
    ensure_contains(
        linux_lane_workflow,
        "rust_tests_matrix: ${{ toJson(fromJson(inputs.lane_plan_json).matrices.rust_tests) }}",
        "Linux lane Rust test matrix input",
    )?;
    ensure_contains(
        linux_lane_workflow,
        "uses: ./.github/workflows/ci-rust-tests-slice.yml",
        "Linux lane shared Rust test slice",
    )?;
    ensure_contains(
        quality_workflow,
        "python3 -m unittest discover -s scripts/tests -p 'test_*.py'",
        "quality CI contract test suite",
    )?;

    Ok(())
}

pub(crate) fn check_ci_crate_test_coverage_files(repo_root: &Path) -> DynResult<()> {
    let linux_lane_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/ci-linux-lane.yml"))?;
    let quality_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/ci-quality-slice.yml"))?;
    check_ci_crate_test_coverage(&linux_lane_workflow, &quality_workflow)?;
    check_main_plan_test_coverage(repo_root)
}

fn check_main_plan_test_coverage(repo_root: &Path) -> DynResult<()> {
    let input = r#"{"profile":"main","event_name":"push","source_sha":"0000000000000000000000000000000000000000","base_sha":"","changed_files":[]}"#;
    let mut child = Command::new("python3")
        .current_dir(repo_root)
        .args(["scripts/plan-ci.py"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("failed to open planner stdin")?
        .write_all(input.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!(
            "main CI planner failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }

    let plan: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let batches = plan
        .get("matrices")
        .and_then(|matrices| matrices.get("rust_tests"))
        .ok_or("main CI plan is missing matrices.rust_tests")?;
    let mut actual = std::collections::BTreeSet::new();
    for crate_name in batches
        .as_array()
        .ok_or("main CI rust-test matrix must be an array")?
        .iter()
        .flat_map(|batch| batch["crates"].as_array().into_iter().flatten())
    {
        let crate_name = crate_name
            .as_str()
            .ok_or("main CI rust-test crate names must be strings")?;
        if !actual.insert(crate_name.to_owned()) {
            return Err(format!("main CI rust-test matrix duplicated crate `{crate_name}`").into());
        }
    }

    let expected = workspace_package_names(repo_root)?;
    ensure_set_eq(&expected, &actual, "main CI rust-test workspace coverage")
}

pub(crate) fn check_ci_script_workspace_members(repo_root: &Path) -> DynResult<()> {
    let expected = workspace_package_names(repo_root)?;
    let scripts = ["scripts/affected-crates.sh"];

    for script in scripts {
        let actual = script_workspace_members(repo_root, script)?;
        ensure_set_eq(&expected, &actual, &format!("{script} WORKSPACE_MEMBERS"))?;
    }

    Ok(())
}

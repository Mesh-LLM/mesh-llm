//! The `ci plan` argv and stdin contract shared with `scripts/plan-ci.py`.

use crate::support::{Run, Stage, TestResult, text};
use std::path::Path;

fn plan(args: &[&str], stdin: &[u8]) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let stage = Stage::new("surface")?;
    let path = stage.search_path()?;
    let run = Run {
        args,
        stdin,
        path: &path,
    };
    let ported = run.ported()?;
    Ok(ported)
}

#[test]
fn migration_ci_plan_rejects_non_json_stdin_with_legacy_status() -> TestResult {
    // Given: stdin that is not JSON.
    // When: the planner reads it.
    let output = plan(&[], b"not json")?;
    // Then: it fails with the legacy prefix and status 2, printing no plan.
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(text(&output.stdout), "");
    assert!(
        text(&output.stderr).starts_with("ERROR: unable to build CI plan: "),
        "{}",
        text(&output.stderr)
    );
    Ok(())
}

#[test]
fn migration_ci_plan_missing_manifest_names_the_unreadable_catalog() -> TestResult {
    // Given: a manifest root that has no `ci/` catalogs.
    let stage = Stage::new("missing-catalog")?;
    let empty = stage.path().join("empty");
    std::fs::create_dir_all(&empty)?;
    let empty_arg = empty.to_str().ok_or("non-UTF8 path")?;
    let path = stage.search_path()?;
    let sha = "a".repeat(40);
    let input = format!(
        r#"{{"profile":"pr-ready","event_name":"pull_request","source_sha":"{sha}","base_sha":"","changed_files":["docs/a.md"]}}"#
    );
    let run = Run {
        args: &["--manifest-root", empty_arg],
        stdin: input.as_bytes(),
        path: &path,
    };
    // When: the planner loads the catalogs.
    let ported = run.ported()?;
    // Then: the ownership catalog is reported exactly as the legacy OSError.
    let catalog = Path::new(empty_arg).join("ci/ownership.yml");
    assert!(text(&ported.stderr).contains(&catalog.display().to_string()));
    assert_eq!(ported.status.code(), Some(2));
    Ok(())
}

#[test]
fn migration_ci_plan_rejects_unknown_arguments_with_usage_status() -> TestResult {
    // Given/When: an argument the legacy parser does not define.
    let output = plan(&["--bogus"], b"{}")?;
    // Then: argparse's usage status is kept and nothing is planned.
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(text(&output.stdout), "");
    assert!(text(&output.stderr).contains("unrecognized arguments: --bogus"));
    Ok(())
}

#[cfg(unix)]
#[test]
fn source_catalog_root_does_not_move_workspace_metadata_or_affected_discovery() -> TestResult {
    use std::os::unix::fs::PermissionsExt;
    use std::{
        fs,
        io::Write,
        process::{Command, Stdio},
    };
    let stage = Stage::new("distinct-roots")?;
    let manifests = stage.manifest_root("default")?;
    let workspace = stage.path().join("protected");
    fs::create_dir_all(workspace.join("scripts"))?;
    fs::write(workspace.join("Cargo.toml"), "[workspace]\nmembers=[]\n")?;
    fs::create_dir_all(workspace.join("tools/xtask"))?;
    fs::write(
        workspace.join("tools/xtask/Cargo.toml"),
        "[package]\nname='xtask'\n",
    )?;
    fs::write(
        workspace.join("scripts/affected-crates.sh"),
        "#!/bin/sh\ntest \"$PWD\" = \"$EXPECTED_WORKSPACE\" || exit 99\nprintf '{\"affected\":[\"mesh-llm\"]}'\n",
    )?;
    let cargo = stage.path().join("bin/cargo");
    let metadata = crate::support::fixture_root().join("cargo-metadata.json");
    fs::write(
        &cargo,
        format!(
            "#!/bin/sh\ntest \"$PWD\" = \"$EXPECTED_WORKSPACE\" || exit 99\ncat '{}'\n",
            metadata.display()
        ),
    )?;
    fs::set_permissions(&cargo, fs::Permissions::from_mode(0o755))?;
    let input = r#"{"profile":"pr-ready","event_name":"pull_request","source_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","base_sha":"","changed_files":["CONTRIBUTING.md"]}"#;
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(&workspace)
        .args([
            "--repo-root",
            workspace.to_str().ok_or("workspace")?,
            "ci",
            "plan",
            "--manifest-root",
            manifests.to_str().ok_or("manifests")?,
        ])
        .env("PATH", stage.search_path()?)
        .env("EXPECTED_WORKSPACE", &workspace)
        .env("RUSTC_WRAPPER", "missing-wrapper")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("stdin")?
        .write_all(input.as_bytes())?;
    let output = child.wait_with_output()?;
    assert!(output.status.success(), "{}", text(&output.stderr));
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(plan["domains"], serde_json::json!(["docs"]));
    Ok(())
}

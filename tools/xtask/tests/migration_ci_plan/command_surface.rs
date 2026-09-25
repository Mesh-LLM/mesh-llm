//! The `ci plan` argv and stdin contract shared with `scripts/plan-ci.py`.

use crate::support::{Run, Stage, TestResult, assert_same_process, text};
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
    if let Some(legacy) = run.legacy()? {
        assert_eq!(legacy.status.code(), ported.status.code(), "status parity");
        assert_eq!(text(&legacy.stdout), text(&ported.stdout), "stdout parity");
    }
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
    if let Some(legacy) = run.legacy()? {
        assert_same_process(&legacy, &ported, "missing catalog");
    }
    // Then: the ownership catalog is reported exactly as the legacy OSError.
    let catalog = Path::new(empty_arg).join("ci/ownership.yml");
    assert_eq!(
        text(&ported.stderr),
        format!(
            "ERROR: unable to build CI plan: unable to load {0}: [Errno 2] No such file or directory: '{0}'\n",
            catalog.display()
        )
    );
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

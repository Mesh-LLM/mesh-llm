use crate::support::{Case, TestResult, Tool, check};

const BODY: &str = "## What's Changed\n* feat(ui): add search by @alice in https://github.com/o/r/pull/1\n* fix: repair cache by @bob in https://github.com/o/r/pull/2\n* Durable KV prefix cache: stable by @cat in https://github.com/o/r/pull/3\n\n## New Contributors\n* @cat made their first contribution\n\n**Full Changelog**: https://github.com/o/r/compare/v1...v2\n";
const PLAN: &str = r#"{"version":"1.0.0","date":"2026-09-10","sections":[{"title":"Added","prs":[1,3]},{"title":"Fixed","prs":[2]}]}"#;

fn case(plan: &str) -> Case {
    Case::new(&[
        "--body",
        "body.md",
        "--plan",
        "plan.json",
        "--out",
        "notes.md",
    ])
    .file("body.md", BODY)
    .file("plan.json", plan)
    .output("notes.md")
}

#[test]
fn migration_release_regroup_preserves_credits_and_generated_tail() -> TestResult {
    let result = check(Tool::Regroup, "credits_and_tail", &case(PLAN))?;
    let rendered = result["outputs"]["notes.md"]
        .as_str()
        .ok_or("missing notes")?;
    for credit in [
        "by @alice in https://github.com/o/r/pull/1",
        "by @bob in https://github.com/o/r/pull/2",
        "by @cat in https://github.com/o/r/pull/3",
    ] {
        assert_eq!(rendered.matches(credit).count(), 1);
    }
    assert!(rendered.ends_with("## New Contributors\n* @cat made their first contribution\n\n**Full Changelog**: https://github.com/o/r/compare/v1...v2\n"));
    assert!(rendered.contains("* Add search by @alice"));
    Ok(())
}

#[test]
fn migration_release_regroup_rejects_duplicate_and_missing_entries() -> TestResult {
    let plan = r#"{"sections":[{"title":"Added","prs":[1,1,3]},{"title":"Fixed","prs":[42]}]}"#;
    let result = check(Tool::Regroup, "duplicate_missing_unknown", &case(plan))?;
    assert_eq!(result["code"], 1);
    let error = result["stderr"].as_str().ok_or("missing error")?;
    for marker in [
        "assigned to more than one section",
        "missing from the plan",
        "not in the release body",
    ] {
        assert!(error.contains(marker), "{error}");
    }
    assert!(result["outputs"]["notes.md"].is_null());
    Ok(())
}

#[test]
fn migration_release_regroup_rejects_untrusted_agent_heading() -> TestResult {
    let plan = r#"{"version":"9.9.9","sections":[{"title":"Added","groups":[{"title":"<img src=x>","prs":[1,2,3]}]}]}"#;
    let result = check(Tool::Regroup, "agent_heading", &case(plan))?;
    assert_eq!(result["code"], 1);
    assert!(
        result["stderr"]
            .as_str()
            .is_some_and(|text| text.contains("not a plain heading"))
    );
    Ok(())
}

#[test]
fn migration_release_regroup_trusts_deterministic_metadata() -> TestResult {
    let agent = r#"{"version":"evil\n## heading","date":"1999-01-01","sections":[{"title":"Added","prs":[1,2,3]}]}"#;
    let case = Case::new(&[
        "--body",
        "body.md",
        "--plan",
        "agent.json",
        "--metadata-from",
        "trusted.json",
        "--out",
        "notes.md",
    ])
    .file("body.md", BODY)
    .file("agent.json", agent)
    .file("trusted.json", PLAN)
    .output("notes.md");
    let result = check(Tool::Regroup, "trusted_metadata", &case)?;
    let rendered = result["outputs"]["notes.md"]
        .as_str()
        .ok_or("missing notes")?;
    assert!(rendered.starts_with("## [1.0.0] - 2026-09-10\n"));
    Ok(())
}

#[test]
fn migration_release_regroup_rejects_duplicate_source_pr() -> TestResult {
    let duplicated = BODY.replace("/pull/2", "/pull/1");
    let case = case(PLAN).file("body.md", &duplicated);
    let result = check(Tool::Regroup, "duplicate_source", &case)?;
    assert_eq!(result["code"], 1);
    assert!(
        result["stderr"]
            .as_str()
            .is_some_and(|text| text.contains("PR #1 appears twice"))
    );
    Ok(())
}

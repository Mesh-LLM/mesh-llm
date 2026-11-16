//! `repository conventional-commits` parity with
//! `scripts/check-conventional-commit.py`.

use crate::support::{Invocation, Scratch, TestResult, assert_output, commit_all, git_init, text};
use std::path::Path;
use std::process::Output;

const TYPES: &str =
    "build, chore, ci, deps, docs, feat, fix, perf, refactor, revert, security, style, test";
const GUIDANCE: &str = "\nConventional Commits: https://www.conventionalcommits.org/en/v1.0.0/\n\
The type decides which release-notes section the change lands in.\n\
Add a 'Release-Notes: <Section>' trailer to override, or\n\
'BREAKING CHANGE: <what>' for a breaking change.\n\
\nCatch this at commit time instead of in CI:  just hooks-install\n\
Bypass once with --no-verify if you know the commit is not a release entry.\n";

fn rejected(subject: &str, problems: &[&str]) -> String {
    let mut report = format!("commit message rejected: {subject}\n");
    for problem in problems {
        report.push_str(&format!("  {problem}\n"));
    }
    report + GUIDANCE
}

fn check(cwd: &Path, args: &[&str]) -> Result<Output, Box<dyn std::error::Error>> {
    let mut ported = vec!["repository", "conventional-commits"];
    ported.extend_from_slice(args);
    Invocation {
        cwd,
        args: &ported,
        stdin: None,
        env: &[],
    }
    .run()
}

#[test]
fn migration_repository_commits_accepts_valid_subjects() -> TestResult {
    let scratch = Scratch::new("commits-valid")?;
    for subject in [
        "fix(skippy): restore recurrent prefix reuse",
        "feat: expose a tokenizer capability",
        "chore(deps)!: drop the old runtime",
        "ci: publish the SDK smoke cache policy",
        "fix(ui): use advertised capacity (#1746)",
        "fix: API names stay uppercase",
        "Merge branch 'main' into feature",
        "Revert \"fix: repair a thing\"",
        "fixup! fix: repair a thing",
        "v0.76.0: prepare release source",
    ] {
        // Given/When: a conventional, exempt or squash-suffixed subject.
        let output = check(scratch.path(), &["--message", subject])?;
        // Then: accepted silently.
        assert_output(&output, 0, "", "");
    }
    Ok(())
}

#[test]
fn migration_repository_commits_rejects_non_conventional_subject() -> TestResult {
    let scratch = Scratch::new("commits-invalid")?;
    // Given/When: a capitalised type without the grammar.
    let output = check(scratch.path(), &["--message", "Fix: Bad thing."])?;
    // Then: the grammar report with the sorted closed type set, exit 1.
    let received = format!("  received: {}", "Fix: Bad thing.");
    let types = format!("  types:    {TYPES}");
    let problems = [
        "subject is not Conventional Commits v1.0.0",
        "  expected: <type>(<optional scope>)<optional !>: <description>",
        &received,
        &types,
    ];
    assert_output(&output, 1, "", &rejected("Fix: Bad thing.", &problems));
    Ok(())
}

#[test]
fn migration_repository_commits_reports_every_subject_problem() -> TestResult {
    let scratch = Scratch::new("commits-problems")?;
    let long = format!("fix: {}", "x".repeat(100));
    let unknown = format!("unknown type 'task'; use one of: {TYPES}");
    let cases: [(&str, Vec<&str>); 4] = [
        (
            "fix: Repair the thing.",
            vec![
                "description must not end with a period",
                "description should start lowercase unless it is a proper noun",
            ],
        ),
        ("task: do a thing", vec![&unknown]),
        (&long, vec!["subject is 105 characters; keep it under 100"]),
        ("   ", vec!["empty commit subject"]),
    ];
    for (subject, problems) in cases {
        // When: the subject breaks one or more rules.
        let output = check(scratch.path(), &["--message", subject])?;
        // Then: every problem is reported in rule order.
        let shown = if subject.trim().is_empty() {
            ""
        } else {
            subject
        };
        assert_output(&output, 1, "", &rejected(shown, &problems));
    }
    Ok(())
}

const ATTRIBUTED: &str = "fix: ok\n# comment\n\nCo-Authored-By: Claude <noreply@anthropic.com>\nSigned-off-by: Bot [bot] <x@y.z>\nReviewed-by: Sol Luna <sol@meshllm.communities.buzz.xyz>\nTested-by: Solomon <real@example.com>\nHelped-by: x <dependabot[bot]@users.noreply.github.com>\n";

#[test]
fn migration_repository_commits_accepts_attribution_in_files_and_messages() -> TestResult {
    let scratch = Scratch::new("commits-file")?;
    scratch.write("msg.txt", ATTRIBUTED)?;
    for args in [&["msg.txt"][..], &["--message", ATTRIBUTED][..]] {
        let output = check(scratch.path(), args)?;
        assert_output(&output, 0, "", "");
    }
    // Attribution is accepted, but it cannot excuse an invalid subject.
    let invalid = ATTRIBUTED.replacen("fix: ok", "Bad subject", 1);
    assert_eq!(
        check(scratch.path(), &["--message", &invalid])?
            .status
            .code(),
        Some(1)
    );
    Ok(())
}

#[test]
fn migration_repository_commits_range_reports_each_bad_commit() -> TestResult {
    // Given: history with a valid commit, a bad subject and accepted attribution.
    let scratch = Scratch::new("commits-range")?;
    let repo = scratch.path().join("repo");
    git_init(&repo)?;
    let base = commit_all(&repo, "chore: base")?;
    commit_all(&repo, "feat: good change")?;
    commit_all(&repo, "Bad subject")?;
    commit_all(
        &repo,
        "fix: fine\n\nCo-authored-by: Codex <codex@example.com>",
    )?;
    let range = format!("{base}..HEAD");
    // When: the range is checked.
    let output = check(&repo, &["--range", &range])?;
    // Then: newest first, one report per bad commit.
    let received = "  received: Bad subject".to_owned();
    let types = format!("  types:    {TYPES}");
    let expected = rejected(
        "Bad subject",
        &[
            "subject is not Conventional Commits v1.0.0",
            "  expected: <type>(<optional scope>)<optional !>: <description>",
            &received,
            &types,
        ],
    );
    assert_output(&output, 1, "", &expected);
    Ok(())
}

#[test]
fn migration_repository_commits_range_with_missing_base_fails() -> TestResult {
    // Given: a repository without the requested base revision.
    let scratch = Scratch::new("commits-missing-base")?;
    let repo = scratch.path().join("repo");
    git_init(&repo)?;
    commit_all(&repo, "chore: base")?;
    // When: the range names an unknown base.
    let output = Invocation {
        cwd: &repo,
        args: &[
            "repository",
            "conventional-commits",
            "--range",
            "missing-base..HEAD",
        ],
        stdin: None,
        env: &[],
    }
    .run()?;
    // Then: git's failure is fatal (exit 1) and nothing is accepted.
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(text(&output.stdout), "");
    assert!(
        text(&output.stderr).contains("missing-base..HEAD"),
        "{}",
        text(&output.stderr)
    );
    Ok(())
}

#[test]
fn migration_repository_commits_requires_exactly_one_source() -> TestResult {
    let scratch = Scratch::new("commits-usage")?;
    for args in [&[][..], &["--message", "fix: a", "--range", "a..b"][..]] {
        let mut ported = vec!["repository", "conventional-commits"];
        ported.extend_from_slice(args);
        let output = Invocation {
            cwd: scratch.path(),
            args: &ported,
            stdin: None,
            env: &[],
        }
        .run()?;
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert_eq!(text(&output.stdout), "");
    }
    Ok(())
}

//! `release notes-base` against `scripts/select-release-notes-base.py`.

use crate::support::{Case, TestResult, Tool, check};

const TAGS: &str = "v0.9.9\nv1.0.0-rc.1\nv1.0.0\n 1.1.0\nv1.2.0\nv01.1.0\nv1.1.0\nnot-a-tag\n";

fn base(name: &str, args: &[&str], stdin: &[u8]) -> TestResult {
    let mut case = Case::new(args);
    case.stdin = stdin.to_vec();
    check(Tool::Base, name, &case)?;
    Ok(())
}

#[test]
fn migration_release_notes_base_selects_previous_stable_tag() -> TestResult {
    base("previous_stable", &["v1.2.0"], TAGS.as_bytes())?;
    base("prerelease_target", &[" 1.2.0-rc.1 "], TAGS.as_bytes())?;
    base("patch_target", &["v1.0.1"], TAGS.as_bytes())
}

#[test]
fn migration_release_notes_base_prints_nothing_without_a_candidate() -> TestResult {
    base("no_candidate", &["v0.9.9"], TAGS.as_bytes())?;
    base("empty_stdin", &["v1.0.0"], b"")?;
    base(
        "carriage_return_is_not_a_line",
        &["v2.0.0"],
        b"v1.0.0\rv1.1.0\n",
    )
}

#[test]
fn migration_release_notes_base_reports_usage_and_invalid_input() -> TestResult {
    base("no_arguments", &[], b"")?;
    base("two_arguments", &["v1.0.0", "v2.0.0"], b"")?;
    base("invalid_target", &["release-1"], TAGS.as_bytes())?;
    base("help_is_a_tag", &["-h"], b"")?;
    base("invalid_utf8_stdin", &["v2.0.0"], b"v1.0.0\n\xff\n")
}

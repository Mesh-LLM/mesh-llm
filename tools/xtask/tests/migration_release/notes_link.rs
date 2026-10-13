//! `release notes-link` against `scripts/release-notes-link.py`, mirroring
//! `LinkTest` in `scripts/tests/test_release_notes.py` through the command
//! line. `git log` and `gh` are stubs first on `PATH`; no network is used.

use crate::support::{Case, TestResult, Tool, check};

const ARGS: [&str; 10] = [
    "--body",
    "body.md",
    "--range",
    "v1.0.0..v1.0.1",
    "--repo",
    "o/r",
    "--out-body",
    "body.linked.md",
    "--out-links",
    "links.json",
];

fn entry(pr: u32, title: &str) -> String {
    format!("* {title} by @someone in https://github.com/o/r/pull/{pr}")
}

fn body(entries: &[String]) -> String {
    let mut lines = vec!["## What's Changed".to_owned()];
    lines.extend(entries.iter().cloned());
    lines.extend([
        "".to_owned(),
        "**Full Changelog**: compare".to_owned(),
        String::new(),
    ]);
    lines.join("\n")
}

/// `git log --format=%H%x1f%s%x1f%b%x1e` output for `(sha, subject, body)`.
fn log(commits: &[(&str, &str, &str)]) -> String {
    commits
        .iter()
        .map(|(sha, subject, text)| format!("{sha}\u{1f}{subject}\u{1f}{text}\u{1e}\n"))
        .collect()
}

fn details(title: &str, login: &str) -> String {
    format!("{{\"title\": \"{title}\", \"author\": {{\"login\": \"{login}\"}}}}\n")
}

fn case(body_text: &str, commits: &[(&str, &str, &str)]) -> Case {
    Case::new(&ARGS)
        .file("body.md", body_text)
        .file("git/out", &log(commits))
        .output("body.linked.md")
        .output("links.json")
}

fn link(name: &str, case: &Case) -> TestResult {
    check(Tool::Link, name, case)?;
    Ok(())
}

#[test]
fn migration_release_notes_link_resolves_pull_requests() -> TestResult {
    let published = body(&[entry(7, "fix: repair a thing")]);
    link(
        "suffix_is_authoritative",
        &case(&published, &[("aaa", "fix: repair a thing (#7)", "")]),
    )?;
    link(
        "suffixless_linked_through_api",
        &case(
            &published,
            &[(
                "aaa",
                "test(ci): guard selector drift",
                "Release-Notes: Internal\n",
            )],
        )
        .file("gh/pulls_aaa.out", "[1733]\n")
        .file(
            "gh/pr_1733.out",
            &details("test(ci): guard selector drift", "i386"),
        ),
    )?;
    link(
        "unplaced_commit_is_not_guessed",
        &case(&published, &[("aaa", "a direct push", "")]).file("gh/pulls_aaa.out", "[]\n"),
    )
}

#[test]
fn migration_release_notes_link_recovers_uncredited_entries() -> TestResult {
    let commits = [
        (
            "a",
            "fix(openai): reject malformed tool definitions (#1761)",
            "",
        ),
        ("b", "fix(skippy): link Apple OpenMP runtime (#1766)", ""),
        ("c", "fix: address review feedback (#1789)", ""),
        ("d", "fix: repair another (#11)", ""),
    ];
    link(
        "orphans_beneath_roll_up_and_trailing",
        &case(
            &body(&[
                entry(1673, "feat: a thing"),
                entry(1789, "fix(release): roll up"),
            ]),
            &commits,
        )
        .file(
            "gh/pr_1761.out",
            &details("fix(openai): reject malformed tool definitions", "i386"),
        )
        .file(
            "gh/pr_1766.out",
            "{\"title\": \"\", \"author\": {\"login\": \"i386\"}}\n",
        )
        .file("gh/pr_11.out", &details("fix: repair another", "someone")),
    )?;
    link(
        "body_without_entries_recovers_release",
        &case(
            "**Full Changelog**: compare\n",
            &[("a", "fix: install composed bundles (#1844)", "")],
        )
        .file(
            "gh/pr_1844.out",
            &details("fix: install composed bundles", "someone"),
        ),
    )?;
    link(
        "pull_request_without_author_is_skipped",
        &case(&body(&[]), &[("a", "fix: repair a thing (#10)", "")])
            .file("gh/pr_10.out", "{\"title\": \"fix\", \"author\": null}\n"),
    )?;
    link(
        "released_pull_request_credited_once",
        &case(
            &body(&[]),
            &[
                ("a", "fix: first commit (#10)", ""),
                ("b", "fix: second commit (#10)", ""),
            ],
        )
        .file("gh/pr_10.out", &details("fix: a thing", "someone")),
    )
}

#[test]
fn migration_release_notes_link_merges_duplicate_records() -> TestResult {
    let commits = [
        ("a", "fix(openai): repair a thing", "Release-Notes: Fixed\n"),
        (
            "b",
            "fix(openai): repair it again",
            "Security: CVE-1\nRelease-Notes: Security\nBREAKING CHANGE: api: gone\n",
        ),
        ("c", "fix: rolled up (#10)", ""),
    ];
    link(
        "duplicate_record_keeps_first_subject",
        &case(&body(&[entry(10, "fix: rolled up")]), &commits)
            .file("gh/pulls_a.out", "[1733]\n")
            .file("gh/pulls_b.out", "[1733, 5]\n")
            .file(
                "gh/pr_1733.out",
                &details("fix(openai): r\u{e9}pair \\u007f", "i386"),
            ),
    )
}

#[test]
fn migration_release_notes_link_bounds_api_calls() -> TestResult {
    let commits: Vec<(String, String)> = (0..5)
        .map(|n| (format!("s{n}"), format!("chore: a change {n}")))
        .collect();
    let borrowed: Vec<(&str, &str, &str)> = commits
        .iter()
        .map(|(sha, subject)| (sha.as_str(), subject.as_str(), ""))
        .collect();
    let mut limited = case(&body(&[]), &borrowed);
    limited
        .args
        .extend(["--api-budget".to_owned(), "2".to_owned()]);
    for n in 0..5 {
        let pr = n + 1;
        limited = limited.file(&format!("gh/pulls_s{n}.out"), &format!("[{pr}]\n"));
        limited = limited.file(&format!("gh/pr_{pr}.out"), &details("t", "u"));
    }
    let actual = check(Tool::Link, "api_budget_bounds_release", &limited)?;
    assert_eq!(actual["gh_argv"].as_array().unwrap().len(), 2);
    assert_eq!(actual["gh_argv"][0][0], "api");
    assert_eq!(actual["gh_argv"][1][0], "api");
    Ok(())
}

#[test]
fn migration_release_notes_link_survives_gh_failures() -> TestResult {
    let commits = [
        ("a", "fix: one", ""),
        ("b", "fix: two", ""),
        ("c", "fix: three", ""),
        ("d", "fix: four (#44)", ""),
    ];
    link(
        "gh_failures_are_best_effort",
        &case(&body(&[entry(1, "x")]), &commits)
            .file("gh/pulls_a.err", "HTTP 502\n")
            .file("gh/pulls_a.code", "1")
            .file("gh/pulls_b.out", "not json\n")
            .file("gh/pulls_c.out", "\u{0}\u{ff}")
            .file("gh/pr_44.out", ""),
    )?;
    let mut missing = case(
        &body(&[]),
        &[("a", "fix: one", ""), ("b", "fix: two (#3)", "")],
    );
    missing.with_gh = false;
    link("gh_missing_from_path", &missing)?;
    link(
        "gh_killed_by_signal",
        &case(&body(&[]), &[("a", "fix: one", "")]).file("gh/pulls_a.signal", "9"),
    )
}

#[test]
fn migration_release_notes_link_reports_argument_errors() -> TestResult {
    link("no_arguments", &Case::new(&[]))?;
    link("help", &Case::new(&["--body", "b", "-h"]))?;
    link("invalid_budget", &Case::new(&["--api-budget", "1.5"]))?;
    link("ambiguous_option", &Case::new(&["--out", "x"]))?;
    let mut extra = Case::new(&ARGS);
    extra.args.push("stray".to_owned());
    link("unrecognized_argument", &extra)
}

#[test]
fn migration_release_notes_link_fails_like_the_legacy_script() -> TestResult {
    let commits = [("a", "fix: one (#1)", "")];
    link("missing_body", &Case::new(&ARGS))?;
    link(
        "empty_body_and_range",
        &Case::new(&ARGS)
            .file("body.md", "")
            .output("body.linked.md")
            .output("links.json"),
    )?;
    let mut bad = case(&body(&[]), &commits);
    bad.files
        .push(("body.md".to_owned(), b"* x /pull/1\n\xff\n".to_vec()));
    link("undecodable_body", &bad)?;
    link(
        "git_failure",
        &case(&body(&[]), &commits)
            .file("git/code", "128")
            .file("git/err", "fatal\n"),
    )?;
    link(
        "git_killed",
        &case(&body(&[]), &commits).file("git/signal", "15"),
    )?;
    let mut root = case(&body(&[]), &commits);
    root.args
        .extend(["--repo-root".to_owned(), "missing-dir".to_owned()]);
    link("repo_root_missing", &root)?;
    let mut sub = case(&body(&[]), &commits).file("sub/keep", "");
    sub.args
        .extend(["--repo-root".to_owned(), "sub".to_owned()]);
    link("repo_root_is_git_cwd", &sub)?;
    let mut unwritable = case(&body(&[]), &commits).file("gh/pr_1.out", &details("t", "u"));
    unwritable.args[7] = "missing/out.md".to_owned();
    link("unwritable_out_body", &unwritable)?;
    link(
        "crlf_body_and_trailing_space_entry",
        &case(
            "## What's Changed\r\n* a by @x in https://github.com/o/r/pull/0005 \t\r\n\r\n## New Contributors\r\n* b in https://github.com/o/r/pull/9\r\n",
            &[("a", "fix: five (#5)", ""), ("b", "fix: nine (#9)", ""), ("c", "fix: six (#6)", "")],
        )
        .file("gh/pr_9.out", &details("fix: nine", "x"))
        .file("gh/pr_6.out", "{\"author\": {\"login\": \"12\"}, \"title\": \"3.5\"}\n"),
    )
}

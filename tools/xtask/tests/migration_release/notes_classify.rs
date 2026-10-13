//! `release notes-classify` against `scripts/release-notes-classify.py`,
//! mirroring `ClassifyTest` and `OverrideHardeningTest` in
//! `scripts/tests/test_release_notes.py` through the command line. `git log`
//! is a stub first on `PATH`; no network is used.

use crate::support::{Case, TestResult, Tool, check};

const ARGS: [&str; 10] = [
    "--body",
    "body.md",
    "--range",
    "v1.0.0..v1.0.1",
    "--version",
    "1.0.0",
    "--date",
    "2026-01-01",
    "--out",
    "plan.json",
];

fn body(prs: &[u32]) -> String {
    let mut text = String::from("## What's Changed\n");
    for pr in prs {
        text.push_str(&format!(
            "* change {pr} by @someone in https://github.com/o/r/pull/{pr}\n"
        ));
    }
    text.push_str("\n**Full Changelog**: compare\n");
    text
}

/// `git log --format=%s%x1f%b%x1e` output for `(subject, body)`.
fn log(commits: &[(&str, &str)]) -> String {
    commits
        .iter()
        .map(|(subject, text)| format!("{subject}\u{1f}{text}\u{1e}\n"))
        .collect()
}

fn case(prs: &[u32], commits: &[(&str, &str)]) -> Case {
    Case::new(&ARGS)
        .file("body.md", &body(prs))
        .file("git/out", &log(commits))
        .output("plan.json")
}

fn classify(name: &str, case: &Case) -> TestResult {
    check(Tool::Classify, name, case)?;
    Ok(())
}

#[test]
fn migration_release_notes_classify_maps_types_to_sections() -> TestResult {
    classify(
        "type_drives_the_section",
        &case(
            &[1, 2, 3, 4, 5],
            &[
                ("feat: add a thing (#1)", ""),
                ("fix: repair a thing (#2)", ""),
                ("perf: speed a thing up (#3)", ""),
                ("security: reject hostile input (#4)", ""),
                ("revert: undo a thing (#5)", ""),
            ],
        ),
    )?;
    classify(
        "breaking_change_is_changed_not_added",
        &case(
            &[1, 2],
            &[
                ("feat!: replace the subsystem (#1)", ""),
                (
                    "feat: replace it (#2)",
                    "BREAKING CHANGE: the old flag is gone\n",
                ),
            ],
        ),
    )?;
    classify(
        "tooling_scopes_are_internal_whatever_the_type",
        &case(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[
                ("fix(ci): make CUDA checks hermetic (#1)", ""),
                ("feat(bench): run a new suite (#2)", ""),
                ("fix(skippy): restore prefix reuse (#3)", ""),
                ("chore: bump a pin (#4)", ""),
                ("docs: explain it (#5)", ""),
                ("test: cover it (#6)", ""),
                ("feat(just): add a recipe (#7)", ""),
                ("fix(xtask): repair a port (#8)", ""),
            ],
        ),
    )
}

#[test]
fn migration_release_notes_classify_honours_trailers() -> TestResult {
    classify(
        "release_notes_trailer_overrides_the_type",
        &case(
            &[1, 2, 3, 4, 5, 6],
            &[
                (
                    "fix: redact provider health details (#1)",
                    "Release-Notes: Security\n",
                ),
                (
                    "fix: redact health details (#2)",
                    "Release-Notes: Secuirty\n",
                ),
                (
                    "Some non-conventional subject (#3)",
                    "Release-Notes: internal\n",
                ),
                (
                    "feat(ui): drop a flag (#4)",
                    "Deprecated: yes\nRemoved: yes\n",
                ),
                ("feat: remove a thing (#5)", "Removed: the old path\n"),
                ("fix: close a hole (#6)", "Security: CVE-1\n"),
            ],
        ),
    )
}

#[test]
fn migration_release_notes_classify_never_guesses() -> TestResult {
    classify(
        "non_conventional_and_missing_commits",
        &case(
            &[1, 2, 3, 7],
            &[
                ("Stop persisting KV cache state to disk (#1)", ""),
                ("wip: an unknown type (#2)", ""),
                ("fix(Bad): an invalid scope (#3)", ""),
                ("fix: no suffix", ""),
            ],
        ),
    )
}

#[test]
fn migration_release_notes_classify_subgroups_large_sections() -> TestResult {
    let subjects: Vec<String> = (1..=25)
        .map(|pr| {
            let scope = match pr {
                1..=10 => "(skippy)",
                11..=20 => "(ui)",
                21 => "(cli)",
                _ => "",
            };
            format!("fix{scope}: repair {pr} (#{pr})")
        })
        .collect();
    let commits: Vec<(&str, &str)> = subjects.iter().map(|s| (s.as_str(), "")).collect();
    let prs: Vec<u32> = (1..=25).rev().collect();
    classify("large_section_earns_subheadings", &case(&prs, &commits))?;
    let flat: Vec<(&str, &str)> = subjects.iter().map(|s| (s.as_str(), "")).take(21).collect();
    let prs: Vec<u32> = (1..=21).collect();
    let mut small = case(&prs, &flat);
    small.files[1].1 = log(&flat)
        .replace("(skippy)", "")
        .replace("(ui)", "")
        .into_bytes();
    classify("large_section_without_scopes_stays_flat", &small)
}

#[test]
fn migration_release_notes_classify_reads_links_and_body() -> TestResult {
    let mut linked = case(&[1, 2, 3], &[("fix: one (#1)", "")]).file(
        "links.json",
        "{\"2\": {\"subject\": \"feat(openai): link it\", \"trailers\": {}}, \
         \"1\": {\"subject\": \"feat: ignored\", \"trailers\": {}}, \
         \"03\": {\"subject\": \"ci: \\u00e9 tidy\", \"trailers\": {\"x\": \"y\"}}}\n",
    );
    linked
        .args
        .extend(["--links".to_owned(), "links.json".to_owned()]);
    classify("links_fill_missing_commits", &linked)?;
    let mut crlf = case(&[], &[("fix: five (#5)", "")]);
    crlf.files[0].1 = b"* a https://github.com/o/r/pull/0005 \t\r\n\r\n## New Contributors\r\n* b https://github.com/o/r/pull/9\r\n".to_vec();
    classify("crlf_body_and_tail", &crlf)?;
    classify("empty_body_is_an_error", &case(&[], &[]))
}

#[test]
fn migration_release_notes_classify_has_entries() -> TestResult {
    let has = |text: &str| Case::new(&["--body", "body.md", "--has-entries"]).file("body.md", text);
    classify("has_entries_true", &has(&body(&[7])))?;
    classify(
        "entries_after_the_release_tail_do_not_count",
        &has("**Full Changelog**: compare\n* x in https://github.com/o/r/pull/7\n"),
    )?;
    classify(
        "has_entries_missing_body",
        &Case::new(&["--body", "nope.md", "--has-entries"]),
    )
}

#[test]
fn migration_release_notes_classify_reports_argument_errors() -> TestResult {
    classify("no_arguments", &Case::new(&[]))?;
    classify("help", &Case::new(&["--body", "b", "-h"]))?;
    classify(
        "plan_options_required",
        &Case::new(&["--body", "b", "--ra", "r"]),
    )?;
    classify("ambiguous_option", &Case::new(&["--r", "x"]))?;
    classify("missing_value", &Case::new(&["--body"]))?;
    let mut extra = Case::new(&ARGS);
    extra.args.push("stray".to_owned());
    classify("unrecognized_argument", &extra)
}

#[test]
fn migration_release_notes_classify_fails_like_the_legacy_script() -> TestResult {
    let commits = [("fix: one (#1)", "")];
    classify("missing_body", &Case::new(&ARGS))?;
    let mut bad = case(&[1], &commits);
    bad.files[0].1 = b"* x /pull/1\n\xff\n".to_vec();
    classify("undecodable_body", &bad)?;
    classify(
        "git_failure",
        &case(&[1], &commits)
            .file("git/code", "128")
            .file("git/err", "fatal\n"),
    )?;
    classify("git_killed", &case(&[1], &commits).file("git/signal", "15"))?;
    let mut root = case(&[1], &commits);
    root.args
        .extend(["--repo-root".to_owned(), "missing-dir".to_owned()]);
    classify("repo_root_missing", &root)?;
    let mut unwritable = case(&[1], &commits);
    unwritable.args[9] = "missing/plan.json".to_owned();
    classify("unwritable_out", &unwritable)?;
    let links = |name: &str, text: &str| {
        let mut linked = case(&[1, 2], &commits).file("links.json", text);
        linked
            .args
            .extend(["--links".to_owned(), "links.json".to_owned()]);
        classify(name, &linked)
    };
    links("links_not_json", "{\"2\": \n")?;
    links("links_bad_key", "{\"x\": {}}\n")?;
    links("links_not_an_object", "[1]\n")?;
    links(
        "links_record_without_trailers",
        "{\"2\": {\"subject\": \"fix: a\"}}\n",
    )?;
    links("links_record_is_a_string", "{\"2\": \"fix: a\"}\n")?;
    links("links_record_is_null", "{\"2\": null}\n")?;
    links(
        "links_override_not_a_string",
        "{\"2\": {\"subject\": \"fix: a\", \"trailers\": {\"release-notes\": 5}}}\n",
    )?;
    links(
        "links_subject_not_a_string",
        "{\"2\": {\"subject\": 5, \"trailers\": []}}\n",
    )?;
    links(
        "links_subject_int_with_trailers",
        "{\"2\": {\"subject\": 5, \"trailers\": {}}}\n",
    )
}

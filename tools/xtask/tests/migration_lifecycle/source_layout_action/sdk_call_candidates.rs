//! Finite source-visible SDK adapter call grammar; unknown candidates require review.
use std::collections::BTreeSet;

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Calls {
    pub adapters: BTreeSet<String>,
    pub sourced_libraries: BTreeSet<String>,
}

fn script_name(word: &str) -> Option<&str> {
    let name = word.strip_prefix("scripts/")?;
    (name.ends_with(".sh")
        && name.len() > 3
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')))
    .then_some(word)
}

fn has_candidate(line: &str) -> bool {
    line.contains("scripts/") && line.contains(".sh")
}

fn reviewed_library(line: &str) -> Option<&'static str> {
    // Source is an import, not a direct adapter launch. This exact current form
    // has a separately recorded library owner; changed imports require review.
    (line == "source \"$REPO_ROOT/scripts/lib/automation.sh\"")
        .then_some("scripts/lib/automation.sh")
}

fn adapter(line: &str) -> Option<&str> {
    let mut words = line.split_ascii_whitespace();
    let first = words.next()?;
    let command = if first == "retry_transient" {
        words.next()?
    } else {
        first
    };
    let path = script_name(command)?;
    // Refuse a second candidate rather than silently accepting only the first.
    // Quoted arguments, substitutions and shell operators are never evaluated.
    (!words.any(has_candidate)).then_some(path)
}

pub(super) fn collect(source: &str) -> Result<Calls, String> {
    let mut calls = Calls::default();
    for (index, line) in source.lines().enumerate() {
        let line = line.trim_start();
        if line.starts_with('#') || !has_candidate(line) {
            continue;
        }
        if let Some(library) = reviewed_library(line) {
            calls.sourced_libraries.insert(library.into());
        } else if let Some(path) = adapter(line) {
            calls.adapters.insert(path.into());
        } else {
            return Err(format!(
                "line {}: unsupported source-visible SDK script candidate: {line}",
                index + 1
            ));
        }
    }
    Ok(calls)
}

#[test]
fn finite_sdk_calls_admit_bare_and_whitespace_retry_forms_inside_multiline_substitution() {
    let calls = collect(
        "# scripts/ignored.sh\nscripts/check-sdk-contract.sh\nretry_transient\t scripts/ci-sdk-fixture.sh \"$1\"\nnative=\"$(\n    scripts/ci-prepare-native-runtime.sh \\\n        \"$1\"\n)\"\nsource \"$REPO_ROOT/scripts/lib/automation.sh\"\n",
    )
    .unwrap();
    assert_eq!(
        calls.adapters,
        BTreeSet::from([
            "scripts/check-sdk-contract.sh".into(),
            "scripts/ci-sdk-fixture.sh".into(),
            "scripts/ci-prepare-native-runtime.sh".into(),
        ])
    );
    assert_eq!(
        calls.sourced_libraries,
        BTreeSet::from(["scripts/lib/automation.sh".into()])
    );
}

#[test]
fn one_unknown_sdk_candidate_refuses_the_census_even_when_other_calls_are_recognized() {
    for unknown in [
        "bash scripts/unknown.sh",
        "\"scripts/unknown.sh\"",
        "./scripts/unknown.sh",
        "env MODE=1 scripts/unknown.sh",
        "command scripts/unknown.sh",
        "value=\"$(scripts/unknown.sh)\"",
        "scripts/check-sdk-contract.sh; scripts/unknown.sh",
        "scripts/check-sdk-contract.sh && scripts/unknown.sh",
        "scripts/check-sdk-contract.sh \"scripts/unknown.sh\"",
        "source \"$REPO_ROOT/scripts/lib/other.sh\"",
        "echo scripts/unknown.sh",
    ] {
        let source = format!("scripts/check-sdk-contract.sh\n{unknown}\n");
        assert!(collect(&source).is_err(), "must refuse: {unknown}");
    }
}

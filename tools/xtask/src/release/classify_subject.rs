//! The `CONVENTIONAL` grammar `scripts/release-notes-classify.py` borrows
//! from `check-conventional-commit.py`: `SUBJECT_RE`, `TYPES`, plus its own
//! `INTERNAL_SCOPES` and the `str.title()` of a `Release-Notes` override.

pub(crate) const HYGIENE: &str = "Refactors, docs, and hygiene";

/// `CONVENTIONAL.TYPES`.
pub(crate) fn type_section(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "feat" => "Added",
        "fix" => "Fixed",
        "perf" | "revert" => "Changed",
        "security" => "Security",
        "refactor" | "style" | "test" | "build" | "deps" | "ci" | "chore" | "docs" => "Internal",
        _ => return None,
    })
}

/// `INTERNAL_SCOPES`.
pub(crate) fn internal_scope(scope: &str) -> Option<&'static str> {
    Some(match scope {
        "ci" | "release" => "CI and release engineering",
        "build" | "deps" | "xtask" => "Build and dependencies",
        "bench" => "Tests",
        "just" => HYGIENE,
        _ => return None,
    })
}

/// A `SUBJECT_RE` match of the authored subject.
pub(crate) struct Subject<'a> {
    pub(crate) kind: &'a str,
    pub(crate) scope: Option<&'a str>,
    pub(crate) breaking: bool,
}

/// `^([a-z]+)(?:\(([a-z0-9][a-z0-9._/-]*)\))?(!)?: (.+)$`, where `$` also
/// matches before one final newline.
pub(crate) fn parse_subject(subject: &str) -> Option<Subject<'_>> {
    let kind_end = subject
        .find(|ch: char| !ch.is_ascii_lowercase())
        .unwrap_or(subject.len());
    let (kind, mut rest) = subject.split_at(kind_end);
    if kind.is_empty() {
        return None;
    }
    let mut scope = None;
    if let Some(scoped) = rest.strip_prefix('(') {
        let scope_char = |ch: char| {
            ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '.' | '_' | '/' | '-')
        };
        let end = scoped.find(|ch| !scope_char(ch)).unwrap_or(scoped.len());
        let head_ok = scoped
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit());
        let closed = scoped[end..].strip_prefix(')')?;
        if !head_ok {
            return None;
        }
        scope = Some(&scoped[..end]);
        rest = closed;
    }
    let breaking = rest.starts_with('!');
    let description = rest.strip_prefix('!').unwrap_or(rest).strip_prefix(": ")?;
    let line = description.strip_suffix('\n').unwrap_or(description);
    (!line.is_empty() && !line.contains('\n')).then_some(Subject {
        kind,
        scope,
        breaking,
    })
}

/// `str.title()`: cased runs start upper and continue lower.
pub(crate) fn title(text: &str) -> String {
    let mut out = String::new();
    let mut previous_cased = false;
    for ch in text.chars() {
        if previous_cased {
            out.extend(ch.to_lowercase());
        } else {
            out.extend(ch.to_uppercase());
        }
        previous_cased = ch.is_lowercase() || ch.is_uppercase();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_supported_commit_type_classifies_with_its_consumed_section() {
        for (kind, section) in [
            ("feat", "Added"),
            ("fix", "Fixed"),
            ("perf", "Changed"),
            ("revert", "Changed"),
            ("security", "Security"),
            ("build", "Internal"),
            ("chore", "Internal"),
            ("ci", "Internal"),
            ("deps", "Internal"),
            ("docs", "Internal"),
            ("refactor", "Internal"),
            ("style", "Internal"),
            ("test", "Internal"),
        ] {
            let subject = format!("{kind}(a.b/c_d-e)!: change (#123)");
            let parsed = parse_subject(&subject).expect("supported grammar");
            assert_eq!(type_section(parsed.kind), Some(section));
            assert_eq!(parsed.scope, Some("a.b/c_d-e"));
            assert!(parsed.breaking);
        }
    }

    #[test]
    fn migration_release_classify_subject_grammar() {
        let found = parse_subject("feat(a.b/c-d)!: x").map(|s| (s.kind, s.scope, s.breaking));
        assert_eq!(found, Some(("feat", Some("a.b/c-d"), true)));
        assert!(parse_subject("fix(Bad): x").is_none());
        assert!(parse_subject("fix(-a): x").is_none());
        assert!(parse_subject("fix: ").is_none());
        assert!(parse_subject("fix: a\n").is_some());
        assert!(parse_subject("fix: a\nb").is_none());
        assert_eq!(title(" they're sECURITY "), " They'Re Security ");
        assert_eq!(type_section("wip"), None);
        assert_eq!(internal_scope("xtask"), Some("Build and dependencies"));
    }
}

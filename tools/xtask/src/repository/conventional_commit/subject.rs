//! Subject grammar from `scripts/check-conventional-commit.py`, hand-parsed
//! to the same acceptance as its `SUBJECT_RE`, `EXEMPT_RE` and
//! `TRAILING_PR_RE`.

use crate::repository::python_text::{is_decimal, is_space, is_upper, split_whitespace, strip};

/// The closed type set; each maps to one release-notes section.
pub(super) const TYPES: &[&str] = &[
    "build", "chore", "ci", "deps", "docs", "feat", "fix", "perf", "refactor", "revert",
    "security", "style", "test",
];

const MAX_SUBJECT: usize = 100;
const EXEMPT_PREFIXES: &[&str] = &["Merge ", "Revert \"", "fixup! ", "squash! ", "amend! "];
const RELEASE_SUFFIX: &str = ": prepare release source";

struct Parsed<'a> {
    kind: &'a str,
    description: &'a str,
}

/// Problems with one subject, in the legacy script's rule order.
pub(super) fn check_subject(subject: &str) -> Vec<String> {
    if strip(subject).is_empty() {
        return vec!["empty commit subject".to_owned()];
    }
    if is_exempt(subject) {
        return Vec::new();
    }
    let authored = without_pr_suffix(subject);
    let Some(parsed) = parse(authored) else {
        return vec![
            "subject is not Conventional Commits v1.0.0".to_owned(),
            "  expected: <type>(<optional scope>)<optional !>: <description>".to_owned(),
            format!("  received: {subject}"),
            format!("  types:    {}", TYPES.join(", ")),
        ];
    };
    let mut problems = Vec::new();
    if !TYPES.contains(&parsed.kind) {
        problems.push(format!(
            "unknown type '{}'; use one of: {}",
            parsed.kind,
            TYPES.join(", ")
        ));
    }
    if parsed.description.ends_with('.') {
        problems.push("description must not end with a period".to_owned());
    }
    if starts_with_capital_word(parsed.description) {
        problems.push("description should start lowercase unless it is a proper noun".to_owned());
    }
    let length = authored.chars().count();
    if length > MAX_SUBJECT {
        problems.push(format!(
            "subject is {length} characters; keep it under {MAX_SUBJECT}"
        ));
    }
    problems
}

/// `description[0].isupper() and not description.split()[0].isupper()`.
fn starts_with_capital_word(description: &str) -> bool {
    let first = description
        .chars()
        .next()
        .map(String::from)
        .unwrap_or_default();
    is_upper(&first) && !split_whitespace(description).next().is_some_and(is_upper)
}

fn is_exempt(subject: &str) -> bool {
    EXEMPT_PREFIXES
        .iter()
        .any(|prefix| subject.starts_with(prefix))
        || is_release_prep(subject)
}

/// `^v?\d+\.\d+\.\d+[^:]*: prepare release source$`.
fn is_release_prep(subject: &str) -> bool {
    let mut rest = subject.strip_prefix('v').unwrap_or(subject);
    for index in 0..3 {
        let digits = rest.len() - rest.trim_start_matches(is_decimal).len();
        if digits == 0 {
            return false;
        }
        rest = &rest[digits..];
        if index < 2 {
            let Some(tail) = rest.strip_prefix('.') else {
                return false;
            };
            rest = tail;
        }
    }
    rest.strip_suffix(RELEASE_SUFFIX)
        .is_some_and(|middle| !middle.contains(':'))
}

/// Removes GitHub's squash-merge `\s*(#1234)` suffix.
fn without_pr_suffix(subject: &str) -> &str {
    let Some(inner) = subject.strip_suffix(')') else {
        return subject;
    };
    let digits = inner.len() - inner.trim_end_matches(is_decimal).len();
    let Some(before) = inner[..inner.len() - digits].strip_suffix("(#") else {
        return subject;
    };
    if digits == 0 {
        return subject;
    }
    before.trim_end_matches(is_space)
}

/// `^([a-z]+)(?:\(([a-z0-9][a-z0-9._/-]*)\))?(!)?: (.+)$`.
fn parse(subject: &str) -> Option<Parsed<'_>> {
    let kind_len = subject.len()
        - subject
            .trim_start_matches(|ch: char| ch.is_ascii_lowercase())
            .len();
    if kind_len == 0 {
        return None;
    }
    let (kind, mut rest) = subject.split_at(kind_len);
    if let Some(scoped) = rest.strip_prefix('(') {
        let scope_len = scoped.len()
            - scoped
                .trim_start_matches(|ch: char| {
                    ch.is_ascii_lowercase()
                        || ch.is_ascii_digit()
                        || matches!(ch, '.' | '_' | '/' | '-')
                })
                .len();
        let scope_head = scoped.chars().next()?;
        if scope_len == 0 || !(scope_head.is_ascii_lowercase() || scope_head.is_ascii_digit()) {
            return None;
        }
        rest = scoped[scope_len..].strip_prefix(')')?;
    }
    rest = rest.strip_prefix('!').unwrap_or(rest);
    let description = rest.strip_prefix(": ")?;
    (!description.is_empty()).then_some(Parsed { kind, description })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_repository_commits_grammar_edges() {
        assert!(check_subject("fix(a.b/c-d_e)!: x").is_empty());
        assert!(!check_subject("fix(-a): x").is_empty());
        assert!(!check_subject("fix(): x").is_empty());
        assert!(check_subject("fix:  ").is_empty());
        assert!(!check_subject("fix: ").is_empty());
        assert!(check_subject("v1.2.3-rc.1: prepare release source").is_empty());
        assert!(!check_subject("v1.2: prepare release source").is_empty());
        assert!(!check_subject("1.2.3 a:b: prepare release source").is_empty());
        assert_eq!(without_pr_suffix("fix: a \t(#12)"), "fix: a");
        assert_eq!(without_pr_suffix("fix: a (#)"), "fix: a (#)");
    }

    #[test]
    fn migration_repository_commits_capital_rule_allows_acronyms() {
        assert!(check_subject("fix: API names stay uppercase").is_empty());
        assert_eq!(
            check_subject("fix: Api names"),
            ["description should start lowercase unless it is a proper noun"]
        );
    }
}

//! `read_commits` and `commit_links` of `scripts/release-notes-link.py`:
//! parsing `git log --format=%H%x1f%s%x1f%b%x1e` records into subjects and
//! trailers, the `(#N)` suffix, and the `links.json` records.

use crate::ci_operations::ci_metrics_value::Value;
use crate::repository::text::{is_space, splitlines, strip};

/// One commit of the range, oldest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Commit {
    pub(crate) sha: String,
    pub(crate) subject: String,
    /// Ordered like a Python dict: a repeated key keeps its first position.
    pub(crate) trailers: Vec<(String, String)>,
    /// `str(commit["pr"])`, `None` while unlinked.
    pub(crate) pr: Option<String>,
    pub(crate) linked_by_api: bool,
}

/// `git log` argv after the program name.
pub(crate) fn log_args(range: &str) -> Vec<String> {
    ["log", "--reverse", "--format=%H%x1f%s%x1f%b%x1e", range]
        .map(str::to_owned)
        .to_vec()
}

fn set(trailers: &mut Vec<(String, String)>, key: String, value: String) {
    match trailers.iter_mut().find(|(seen, _)| *seen == key) {
        Some(entry) => entry.1 = value,
        None => trailers.push((key, value)),
    }
}

/// `TRAILER_RE = ^(?P<key>[A-Za-z][A-Za-z -]*):\s*(?P<value>.+)$` on a
/// stripped line: the key and the untrimmed value.
fn trailer(line: &str) -> Option<(&str, &str)> {
    let (key, rest) = line.split_once(':')?;
    let valid_key = key
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_alphabetic())
        && key
            .chars()
            .all(|ch| ch.is_ascii_alphabetic() || ch == ' ' || ch == '-');
    if !valid_key || rest.contains('\n') {
        return None;
    }
    let value = rest.trim_start_matches(is_space);
    // `\s*` backtracks to leave `.+` one character when only spaces follow.
    let value = if value.is_empty() {
        rest.char_indices()
            .last()
            .map_or("", |(index, _)| &rest[index..])
    } else {
        value
    };
    (!value.is_empty()).then_some((key, value))
}

pub(crate) fn trailers(body: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for line in splitlines(body) {
        let stripped = strip(line);
        if let Some((key, value)) = trailer(stripped) {
            set(
                &mut found,
                strip(key).to_ascii_lowercase(),
                strip(value).to_owned(),
            );
        }
        if stripped.starts_with("BREAKING CHANGE:") {
            let after = line.split_once(':').map_or("", |(_, after)| after);
            set(
                &mut found,
                "breaking change".to_owned(),
                strip(after).to_owned(),
            );
        }
    }
    found
}

/// `read_commits` over the decoded, newline-translated `git log` stdout.
pub(crate) fn parse_log(stdout: &str) -> Vec<Commit> {
    stdout
        .split('\u{1e}')
        .filter(|record| !strip(record).is_empty())
        .map(|record| {
            let record = record.trim_matches('\n');
            let (sha, rest) = record.split_once('\u{1f}').unwrap_or((record, ""));
            let (subject, body) = rest.split_once('\u{1f}').unwrap_or((rest, ""));
            Commit {
                sha: strip(sha).to_owned(),
                subject: strip(subject).to_owned(),
                trailers: trailers(body),
                pr: None,
                linked_by_api: false,
            }
        })
        .collect()
}

/// `PR_SUFFIX_RE = \s*\(#(\d+)\)$`: the byte offset where the suffix
/// (with its leading whitespace) starts, and the digits.
pub(crate) fn pr_suffix(subject: &str) -> Option<(usize, &str)> {
    let inner = subject.strip_suffix(')')?;
    let start = inner
        .char_indices()
        .rev()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .last()
        .map(|(index, _)| index)?;
    let digits = &inner[start..];
    let open = inner[..start].strip_suffix("(#")?;
    let begin = open.trim_end_matches(is_space).len();
    Some((begin, digits))
}

/// `PR_SUFFIX_RE.sub("", subject).strip()`.
pub(crate) fn without_suffix(subject: &str) -> &str {
    strip(pr_suffix(subject).map_or(subject, |(begin, _)| &subject[..begin]))
}

/// `(pr, subject, trailers)` of one `links.json` record.
type Record = (String, String, Vec<(String, String)>);

/// `commit_links(commits)`: the first API-linked commit of each pull
/// request sets the subject; later ones only add missing trailers.
pub(crate) fn commit_links(commits: &[Commit]) -> Value {
    let mut links: Vec<Record> = Vec::new();
    for commit in commits.iter().filter(|commit| commit.linked_by_api) {
        let key = commit.pr.clone().unwrap_or_default();
        match links.iter_mut().find(|(pr, ..)| *pr == key) {
            None => links.push((key, commit.subject.clone(), commit.trailers.clone())),
            Some((_, _, trailers)) => {
                for (name, value) in &commit.trailers {
                    if !trailers.iter().any(|(seen, _)| seen == name) {
                        trailers.push((name.clone(), value.clone()));
                    }
                }
            }
        }
    }
    let record = |subject: String, trailers: Vec<(String, String)>| {
        let trailers = trailers
            .into_iter()
            .map(|(key, value)| (key, Value::Str(value)))
            .collect();
        Value::Object(vec![
            ("subject".to_owned(), Value::Str(subject)),
            ("trailers".to_owned(), Value::Object(trailers)),
        ])
    };
    Value::Object(
        links
            .into_iter()
            .map(|(pr, subject, trailers)| (pr, record(subject, trailers)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
        items
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn migration_release_link_parses_git_log_records() {
        let log = "aaa\u{1f}fix: one (#7)\u{1f}\u{1e}\nbbb\u{1f}  test(ci): two \u{1f}Release-Notes: Internal\nSigned-off-by: A B\nBREAKING CHANGE: x: y\nnot a: \n\u{1e}\n";
        let commits = parse_log(log);
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[0].sha, "aaa");
        assert_eq!(commits[1].subject, "test(ci): two");
        assert_eq!(
            commits[1].trailers,
            pairs(&[
                ("release-notes", "Internal"),
                ("signed-off-by", "A B"),
                ("breaking change", "x: y"),
            ])
        );
    }

    #[test]
    fn migration_release_link_reads_the_pr_suffix() {
        assert_eq!(pr_suffix("fix: a (#12)"), Some((6, "12")));
        assert_eq!(pr_suffix("fix: a(#0012)").map(|(_, pr)| pr), Some("0012"));
        assert_eq!(pr_suffix("fix: a (#12) x"), None);
        assert_eq!(pr_suffix("fix: a (#)"), None);
        assert_eq!(without_suffix("fix: a  (#12)"), "fix: a");
        assert_eq!(without_suffix("fix: a"), "fix: a");
    }

    #[test]
    fn migration_release_link_trailer_regex_backtracks() {
        assert_eq!(trailer("Key: value"), Some(("Key", "value")));
        assert_eq!(trailer("Key:x"), Some(("Key", "x")));
        assert_eq!(trailer("Key:"), None);
        assert_eq!(trailer("1Key: v"), None);
        assert_eq!(trailer("Ke_y: v"), None);
    }
}

//! The release body side of `scripts/release-notes-regroup.py`:
//! `parse_body`, `render_entry` and `normalize_subject`.

use crate::release::link_body::entry_pr;
use crate::repository::text::{is_space, splitlines, strip};
use std::collections::HashMap;

/// `DISPLAY_TYPES`, in the regex alternation's order.
const DISPLAY_TYPES: [&str; 20] = [
    "feat", "fix", "perf", "security", "revert", "refactor", "style", "test", "build", "deps",
    "ci", "chore", "docs", "task", "spec", "feature", "config", "runtime", "skippy", "bench",
];

/// The parsed body: rendered entries by pull request, their order, and the
/// lines from the release tail on.
pub(crate) struct Body {
    pub(crate) entries: HashMap<String, String>,
    pub(crate) order: Vec<String>,
    pub(crate) tail: Vec<String>,
}

/// `parse_body(text)`, or the `sys.exit` message it dies with.
pub(crate) fn parse_body(text: &str) -> Result<Body, String> {
    let lines = splitlines(text);
    let tail_at = lines
        .iter()
        .position(|line| {
            line.starts_with("## New Contributors") || line.starts_with("**Full Changelog**")
        })
        .unwrap_or(lines.len());
    let mut entries = HashMap::new();
    let mut order = Vec::new();
    for line in &lines[..tail_at] {
        let Some(pr) = entry_pr(line) else {
            continue;
        };
        if entries.contains_key(&pr) {
            return Err(format!("error: PR #{pr} appears twice in the source body"));
        }
        entries.insert(pr.clone(), render_entry(line));
        order.push(pr);
    }
    if entries.is_empty() {
        return Err("error: no '* ... /pull/<n>' entry lines found in the body".to_owned());
    }
    let tail = lines[tail_at..]
        .iter()
        .map(|line| (*line).to_owned())
        .collect();
    Ok(Body {
        entries,
        order,
        tail,
    })
}

/// `--list`: `<pr>\t<subject>` per entry, subject without its credit.
pub(crate) fn list_line(pr: &str, entry: &str) -> String {
    let rest: String = entry.chars().skip(2).collect();
    let subject = rest
        .rsplit_once(" by @")
        .map_or(rest.as_str(), |(head, _)| head);
    format!("{pr}\t{subject}\n")
}

/// `render_entry(line)`: the subject normalized, the credit untouched and
/// trailing whitespace dropped; a line `ENTRY_PARTS_RE` rejects is kept.
pub(crate) fn render_entry(line: &str) -> String {
    let Some(rest) = line.strip_prefix("* ") else {
        return line.to_owned();
    };
    // The lazy subject ends at the first ` by @` whose remainder matches.
    for (index, _) in rest.match_indices(" by @") {
        if let Some(length) = credit_length(&rest[index..]) {
            let credit = &rest[index..index + length];
            return format!("* {}{credit}", normalize_subject(&rest[..index]));
        }
    }
    line.to_owned()
}

/// The length of `(?P<credit> by @[^ ]+ in \S*/pull/\d+)` when `\s*$`
/// follows it.
fn credit_length(text: &str) -> Option<usize> {
    let author_at = " by @".len();
    let author_len = text[author_at..]
        .find(' ')
        .unwrap_or(text.len() - author_at);
    if author_len == 0 {
        return None;
    }
    let link_at = author_at + author_len + " in ".len();
    if !text[author_at + author_len..].starts_with(" in ") {
        return None;
    }
    let link = &text[link_at..];
    let link_len = link.find(is_space).unwrap_or(link.len());
    if !link[link_len..].chars().all(is_space) {
        return None;
    }
    let token = &link[..link_len];
    let digits = token.trim_end_matches(|ch: char| ch.is_ascii_digit());
    (digits.len() < token.len() && digits.ends_with("/pull/")).then_some(link_at + link_len)
}

/// `normalize_subject(subject)`: strip a known type prefix and upper-case
/// a lowercase first letter; any other subject is returned unchanged.
pub(crate) fn normalize_subject(subject: &str) -> String {
    let Some(rest) = DISPLAY_TYPES
        .iter()
        .find_map(|kind| strip_display_prefix(subject, kind))
    else {
        return subject.to_owned();
    };
    let stripped = strip(rest);
    let mut chars = stripped.chars();
    let Some(first) = chars.next() else {
        return subject.to_owned();
    };
    if is_lowercase_letter(first) {
        return first.to_uppercase().chain(chars).collect();
    }
    stripped.to_owned()
}

/// Python `ch.isalpha() and ch.islower()`: Lowercase minus the lowercase
/// characters outside general category `L*` (a mark, Nl numerals, So).
fn is_lowercase_letter(ch: char) -> bool {
    ch.is_lowercase()
        && !matches!(ch, '\u{345}' | '\u{2170}'..='\u{217f}' | '\u{24d0}'..='\u{24e9}')
}

/// `re.IGNORECASE` equality of a subject character with an ASCII pattern
/// letter, including the Unicode characters that fold onto it.
fn folds_to(ch: char, letter: u8) -> bool {
    ch.to_ascii_lowercase() == char::from(letter)
        || matches!(
            (letter, ch),
            (b'i', '\u{130}' | '\u{131}') | (b'k', '\u{212a}') | (b's', '\u{17f}')
        )
}

/// `^kind(?:\([^)]*\))?!?:\s+`: the text after the prefix, if it matches.
fn strip_display_prefix<'s>(subject: &'s str, kind: &str) -> Option<&'s str> {
    let mut chars = subject.char_indices();
    for letter in kind.bytes() {
        let (_, ch) = chars.next()?;
        if !folds_to(ch, letter) {
            return None;
        }
    }
    let mut at = chars.next().map_or(subject.len(), |(index, _)| index);
    if subject[at..].starts_with('(') {
        at += subject[at..].find(')')? + 1;
    }
    if subject[at..].starts_with('!') {
        at += 1;
    }
    let rest = subject[at..].strip_prefix(':')?;
    rest.starts_with(is_space).then_some(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(text: &str) -> String {
        let line = format!("* {text} by @x in https://github.com/o/r/pull/1");
        let rendered = render_entry(&line);
        rendered[2..].split(" by @").next().unwrap_or("").to_owned()
    }

    #[test]
    fn migration_release_regroup_normalizes_like_python() {
        let cases = [
            ("feat(skippy): load it", "Load it"),
            ("chore(deps)!: drop it", "Drop it"),
            ("CI: pin Depot", "Pin Depot"),
            ("feature: new", "New"),
            (
                "Durable KV prefix cache: agent",
                "Durable KV prefix cache: agent",
            ),
            ("skippy-quantize: compose", "skippy-quantize: compose"),
            ("fix:", "fix:"),
            ("fix(: x", "fix(: x"),
            ("fix:  ", "fix:  "),
            ("fix: ßig", "SSig"),
            ("ſecurity: x", "X"),
        ];
        for (original, expected) in cases {
            assert_eq!(subject(original), expected, "{original}");
        }
        assert_eq!(
            render_entry("* a by @b in x/pull/1  "),
            "* A by @b in x/pull/1".replace('A', "a")
        );
        assert_eq!(
            render_entry("* a by @b in x/pull/1 y"),
            "* a by @b in x/pull/1 y"
        );
    }
}

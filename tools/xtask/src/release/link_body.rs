//! The release body side of `scripts/release-notes-link.py`: `read_body`,
//! `augment`, `entry_line`, and the `json.dump(..., indent=2)` of the links.

use crate::ci_operations::ci_metrics_value::Value;
use crate::repository::python_text::{is_space, splitlines};

/// The canonical decimal text of an `int`, as `str(int(digits))`.
pub(crate) fn canonical(digits: &str) -> String {
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() { "0" } else { trimmed }.to_owned()
}

/// `BODY_ENTRY_RE = ^\* .*/pull/(\d+)\s*$`: the pull request of an entry.
/// The greedy `.*` takes the last `/pull/` that satisfies the rest.
pub(crate) fn entry_pr(line: &str) -> Option<String> {
    let rest = line.strip_prefix("* ")?;
    let markers: Vec<(usize, &str)> = rest.match_indices("/pull/").collect();
    markers.into_iter().rev().find_map(|(index, marker)| {
        let after = &rest[index + marker.len()..];
        let end = after
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(after.len());
        let tail_blank = after[end..].chars().all(is_space);
        (end > 0 && tail_blank).then(|| canonical(&after[..end]))
    })
}

/// `TAIL_RE`: where `## New Contributors` or the changelog link starts.
fn is_tail(line: &str) -> bool {
    line.starts_with("## New Contributors") || line.starts_with("**Full Changelog**")
}

/// The parsed body: lines, credited pull requests, and the insertion index.
pub(crate) struct Body {
    pub(crate) lines: Vec<String>,
    pub(crate) credited: Vec<String>,
    pub(crate) insert_at: usize,
}

/// `read_body` over the file's decoded text.
pub(crate) fn read_body(text: &str) -> Body {
    let translated = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<String> = splitlines(&translated)
        .into_iter()
        .map(str::to_owned)
        .collect();
    let tail_at = lines
        .iter()
        .position(|line| is_tail(line))
        .unwrap_or(lines.len());
    let mut credited = Vec::new();
    let mut last_entry = None;
    for (index, line) in lines[..tail_at].iter().enumerate() {
        if let Some(pr) = entry_pr(line) {
            credited.push(pr);
            last_entry = Some(index);
        }
    }
    let insert_at = last_entry.map_or(tail_at, |index| index + 1);
    Body {
        lines,
        credited,
        insert_at,
    }
}

/// `entry_line(repo, pr, title, author)`.
pub(crate) fn entry_line(repo: &str, pr: &str, title: &str, author: &str) -> String {
    format!("* {title} by @{author} in https://github.com/{repo}/pull/{pr}")
}

/// `augment(lines, insert_at, insertions, trailing)`.
pub(crate) fn augment(
    body: &Body,
    insertions: &[(String, Vec<String>)],
    trailing: &[String],
) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut leftover: &[String] = trailing;
    for (index, line) in body.lines.iter().enumerate() {
        if index == body.insert_at && !leftover.is_empty() {
            out.extend(leftover.iter().map(String::as_str));
            leftover = &[];
        }
        out.push(line);
        if index < body.insert_at
            && let Some(pr) = entry_pr(line)
            && let Some((_, entries)) = insertions.iter().find(|(carrier, _)| *carrier == pr)
        {
            out.extend(entries.iter().map(String::as_str));
        }
    }
    out.extend(leftover.iter().map(String::as_str));
    let joined = out.join("\n");
    format!("{}\n", joined.trim_end_matches(is_space))
}

/// `json.dump(links, handle, indent=2)` plus the trailing newline.
pub(crate) fn dump_links(links: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, links, 0);
    out.push('\n');
    out
}

fn write_value(out: &mut String, value: &Value, depth: usize) {
    match value {
        Value::Str(text) => write_string(out, text),
        Value::Object(entries) if entries.is_empty() => out.push_str("{}"),
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push('\n');
                out.push_str(&"  ".repeat(depth + 1));
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, depth + 1);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(depth));
            out.push('}');
        }
        other => out.push_str(&crate::ci_operations::ci_metrics_value::dumps(other, false)),
    }
}

/// `json`'s `ensure_ascii` string encoder: everything outside `' '..='~'`
/// is escaped, `DEL` included.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0_u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(lines: &[&str]) -> Body {
        read_body(&(lines.join("\n") + "\n"))
    }

    #[test]
    fn migration_release_link_body_finds_entries_before_the_tail() {
        let parsed = body(&[
            "## What's Changed",
            "* feat: a by @x in https://github.com/o/r/pull/0012 ",
            "* not /pull/x",
            "",
            "**Full Changelog**: compare",
            "* fix: after the tail in https://github.com/o/r/pull/9",
        ]);
        assert_eq!(parsed.credited, ["12"]);
        assert_eq!(parsed.insert_at, 2);
        assert_eq!(entry_pr("* a /pull/1/pull/2"), Some("2".to_owned()));
        assert_eq!(entry_pr("* a /pull/1 /pull/x"), None);
        assert_eq!(entry_pr("*  /pull/3\t"), Some("3".to_owned()));
    }

    #[test]
    fn migration_release_link_augment_splices_and_appends() {
        let parsed = body(&["* x /pull/1", "* y /pull/2", "", "**Full Changelog**: c"]);
        let insertions = vec![("2".to_owned(), vec!["* recovered".to_owned()])];
        assert_eq!(
            augment(&parsed, &insertions, &["* trailing".to_owned()]),
            "* x /pull/1\n* y /pull/2\n* recovered\n* trailing\n\n**Full Changelog**: c\n"
        );
        let empty = body(&["**Full Changelog**: c"]);
        assert_eq!(
            augment(&empty, &[], &["* t".to_owned()]),
            "* t\n**Full Changelog**: c\n"
        );
    }

    #[test]
    fn migration_release_link_links_json_matches_python() {
        let links = Value::Object(vec![(
            "7".to_owned(),
            Value::Object(vec![
                ("subject".to_owned(), Value::Str("é\u{7f}\"".to_owned())),
                ("trailers".to_owned(), Value::Object(Vec::new())),
            ]),
        )]);
        assert_eq!(
            dump_links(&links),
            "{\n  \"7\": {\n    \"subject\": \"\\u00e9\\u007f\\\"\",\n    \"trailers\": {}\n  }\n}\n"
        );
        assert_eq!(dump_links(&Value::Object(Vec::new())), "{}\n");
    }
}

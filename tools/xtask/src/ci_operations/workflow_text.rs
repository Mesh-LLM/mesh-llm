//! The legacy tool's `re.MULTILINE` workflow patterns, emulated without a
//! regex engine. `^` matches at offset 0 and after `\n`; `$` before `\n` or
//! at the end; a `\s*` before `$` or a capture may cross line breaks, which
//! is why a declaration's value can be read from the following line.

use crate::ci_operations::python_access::{Outcome, require};
use crate::repository::python_text::{self, is_space};

/// Start offsets of every line (`^` positions).
pub(crate) fn line_starts(text: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(text.match_indices('\n').map(|(index, _)| index + 1))
        .filter(|start| *start <= text.len())
        .collect()
}

fn line_end(text: &str, start: usize) -> usize {
    text[start..]
        .find('\n')
        .map_or(text.len(), |offset| start + offset)
}

/// End offset after greedily skipping whitespace (newlines included).
fn skip_space(text: &str, start: usize) -> usize {
    text[start..]
        .char_indices()
        .find(|(_, ch)| !is_space(*ch))
        .map_or(text.len(), |(offset, _)| start + offset)
}

/// `\s*$` from `start`: where the match ends, or `None` if non-space text
/// precedes the line end.
pub(crate) fn blank_to_eol(text: &str, start: usize) -> Option<usize> {
    let stop = skip_space(text, start);
    if stop == text.len() {
        return Some(stop);
    }
    text[start..stop].rfind('\n').map(|offset| start + offset)
}

/// `:\s*(\S[^\n]*)$` from `start`: the capture and the match end.
fn value_after(text: &str, start: usize) -> Option<(&str, usize)> {
    let begin = skip_space(text, start);
    (begin < text.len()).then(|| {
        let end = line_end(text, begin);
        (&text[begin..end], end)
    })
}

/// Every `^<prefix><name>:\s*(\S[^\n]*)$` capture in scan order, where
/// `prefix` is exactly `indent` spaces or (for `None`) one or more.
pub(crate) fn declarations<'a>(text: &'a str, name: &str, indent: Option<usize>) -> Vec<&'a str> {
    let head = format!("{name}:");
    let mut found = Vec::new();
    let mut resume = 0;
    for start in line_starts(text) {
        if start < resume {
            continue;
        }
        let rest = &text[start..];
        let spaces = rest.len() - rest.trim_start_matches(' ').len();
        let indented = match indent {
            Some(width) => spaces == width,
            None => spaces > 0,
        };
        if !indented || !rest[spaces..].starts_with(&head) {
            continue;
        }
        if let Some((value, end)) = value_after(text, start + spaces + head.len()) {
            found.push(value);
            resume = end;
        }
    }
    found
}

/// `scalar(value)`: strip, then drop one pair of matching quotes.
pub(crate) fn scalar(value: &str) -> String {
    let value = python_text::strip(value);
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 2
        && chars[0] == chars[chars.len() - 1]
        && (chars[0] == '"' || chars[0] == '\'')
    {
        return chars[1..chars.len() - 1].iter().collect();
    }
    value.to_owned()
}

/// `one_field(text, name, where, indent)`.
pub(crate) fn one_field(
    text: &str,
    name: &str,
    place: &str,
    indent: Option<usize>,
) -> Outcome<String> {
    let values = declarations(text, name, indent);
    require(values.len() == 1, || {
        format!("{place}: expected exactly one {name} declaration")
    })?;
    Ok(scalar(values[0]))
}

/// Captures of `^ +(?:image|runner_image):\s*([^\n]+)$`, in scan order.
pub(crate) fn image_values(job: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let mut resume = 0;
    for start in line_starts(job) {
        if start < resume {
            continue;
        }
        let rest = &job[start..];
        let spaces = rest.len() - rest.trim_start_matches(' ').len();
        let body = &rest[spaces..];
        let Some(head) = ["image:", "runner_image:"]
            .into_iter()
            .find(|head| body.starts_with(head))
        else {
            continue;
        };
        if spaces == 0 {
            continue;
        }
        let after = start + spaces + head.len();
        match value_after(job, after) {
            Some((value, end)) => {
                found.push(value);
                resume = end;
            }
            // Trailing whitespace can still match `[^\n]+`; it holds no image.
            None if job[after..].contains(|ch: char| ch != '\n') => {
                found.push("");
                resume = job.len();
            }
            None => {}
        }
    }
    found
}

/// `job_steps(job)`: slices starting at each `^      - `.
pub(crate) fn job_steps(job: &str) -> Vec<&str> {
    let starts: Vec<usize> = line_starts(job)
        .into_iter()
        .filter(|start| job[*start..].starts_with("      - "))
        .collect();
    starts
        .iter()
        .enumerate()
        .map(|(index, start)| &job[*start..starts.get(index + 1).copied().unwrap_or(job.len())])
        .collect()
}

/// Whether some line satisfies `test(line_start_offset)`.
pub(crate) fn any_line(text: &str, test: impl Fn(&str) -> bool) -> bool {
    line_starts(text)
        .into_iter()
        .any(|start| test(&text[start..]))
}

/// `^<literal>\s*$` against the text that starts a line.
pub(crate) fn line_is(rest: &str, literal: &str) -> bool {
    rest.strip_prefix(literal)
        .is_some_and(|tail| blank_to_eol(tail, 0).is_some())
}

/// Matrix rows: `^          - [^\n]+\n(?:^            [^\n]*\n)*`.
pub(crate) fn matrix_rows(job: &str) -> Vec<&str> {
    let mut rows = Vec::new();
    let starts = line_starts(job);
    let mut index = 0;
    while index < starts.len() {
        let start = starts[index];
        let end = line_end(job, start);
        let head = &job[start..end];
        let opens = head
            .strip_prefix("          - ")
            .is_some_and(|rest| !rest.is_empty());
        if !opens || end == job.len() {
            index += 1;
            continue;
        }
        let mut stop = end + 1;
        index += 1;
        while index < starts.len() {
            let next_end = line_end(job, starts[index]);
            if !job[starts[index]..].starts_with("            ") || next_end == job.len() {
                break;
            }
            stop = next_end + 1;
            index += 1;
        }
        rows.push(&job[start..stop]);
    }
    rows
}

/// `^            cuda_major: ['"]?<major>['"]?\s*$` within a matrix row.
pub(crate) fn row_has_cuda_major(row: &str, major: &str) -> bool {
    fn strip_quote(text: &str) -> Option<&str> {
        text.strip_prefix(['\'', '"'])
    }
    any_line(row, |rest| {
        let Some(value) = rest.strip_prefix("            cuda_major: ") else {
            return false;
        };
        let value = strip_quote(value).map_or(value, |inner| inner);
        let Some(tail) = value.strip_prefix(major) else {
            return false;
        };
        blank_to_eol(tail, 0).is_some()
            || strip_quote(tail).is_some_and(|tail| blank_to_eol(tail, 0).is_some())
    })
}

/// Matches of `mesh-llm-sccache-seed-[^\n]*?\$\{\{[^\n]*?\}\}`.
pub(crate) fn seed_keys(job: &str) -> Vec<&str> {
    const HEAD: &str = "mesh-llm-sccache-seed-";
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(offset) = job[from..].find(HEAD) {
        let start = from + offset;
        let line = &job[start..line_end(job, start)];
        let end = line[HEAD.len()..].find("${{").and_then(|open| {
            let close_from = HEAD.len() + open + 3;
            line[close_from..]
                .find("}}")
                .map(|close| close_from + close + 2)
        });
        match end {
            Some(end) => {
                found.push(&line[..end]);
                from = start + end;
            }
            None => from = start + 1,
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_operations_declarations_cross_lines_like_python() {
        let text = "      image:\n        foo\n      image: 'bar'\n";
        assert_eq!(declarations(text, "image", Some(6)), ["foo", "'bar'"]);
        assert_eq!(
            one_field("  a: \"x\"\n", "a", "w", None),
            Ok("x".to_owned())
        );
        assert_eq!(
            one_field("  a: 1\n  a: 2\n", "a", "w", None),
            Err("w: expected exactly one a declaration".to_owned())
        );
        assert_eq!(
            image_values("    image: a\n    runner_image:\n      b\n"),
            ["a", "b"]
        );
    }

    #[test]
    fn migration_ci_operations_matrix_rows_and_seed_keys() {
        let job = "          - id: a\n            cuda_major: '12'\n          - id: b\n            cuda_major: 13\n";
        let rows = matrix_rows(job);
        assert_eq!(rows.len(), 2);
        assert!(row_has_cuda_major(rows[0], "12") && row_has_cuda_major(rows[1], "13"));
        assert!(!row_has_cuda_major(rows[1], "12"));
        let keys = seed_keys(
            "key: mesh-llm-sccache-seed-x-${{ hashFiles('a') }} tail\nmesh-llm-sccache-seed-no\n",
        );
        assert_eq!(keys, ["mesh-llm-sccache-seed-x-${{ hashFiles('a') }}"]);
        assert_eq!(
            job_steps("      - a\n        b\n      - c\n"),
            ["      - a\n        b\n", "      - c\n"]
        );
    }
}

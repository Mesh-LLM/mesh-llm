//! Python `str` semantics the ported checks depend on for byte-identical
//! output: `isspace`, `strip`, `splitlines` and `repr`.

/// Python `str.isspace` for one character: Unicode White_Space plus the
/// C0 separators U+001C..U+001F that Python also treats as whitespace.
pub(crate) fn is_space(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

/// Python `str.strip()`.
pub(crate) fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// Python `str.split()` with no separator.
pub(crate) fn split_whitespace(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_space).filter(|part| !part.is_empty())
}

fn is_line_break(ch: char) -> bool {
    matches!(
        ch,
        '\n' | '\r'
            | '\u{0b}'
            | '\u{0c}'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// Python `str.splitlines()`: every Unicode line boundary, `\r\n` counted
/// once, and no trailing empty line.
pub(crate) fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if !is_line_break(ch) {
            continue;
        }
        lines.push(&text[start..index]);
        start = index + ch.len_utf8();
        if ch == '\r' && chars.peek().is_some_and(|(_, next)| *next == '\n') {
            chars.next();
            start += 1;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Python titlecase letters (category Lt), which `str.isupper` rejects.
fn is_title(ch: char) -> bool {
    matches!(
        ch,
        '\u{1c5}' | '\u{1c8}' | '\u{1cb}' | '\u{1f2}'
            | '\u{1f88}'..='\u{1f8f}'
            | '\u{1f98}'..='\u{1f9f}'
            | '\u{1fa8}'..='\u{1faf}'
            | '\u{1fbc}' | '\u{1fcc}' | '\u{1ffc}'
    )
}

/// Python `str.isupper()`: at least one uppercase character and no
/// lowercase or titlecase character.
pub(crate) fn is_upper(text: &str) -> bool {
    !text.chars().any(|ch| ch.is_lowercase() || is_title(ch))
        && text.chars().any(char::is_uppercase)
}

/// Python regex `\d` on `str`. Rust exposes no Nd-only predicate, so this is
/// Unicode Numeric; the two differ only for Nl/No characters such as `²`.
pub(crate) fn is_decimal(ch: char) -> bool {
    ch.is_numeric()
}

/// Python `repr(str)` for the quoting and escapes seen in diagnostics.
pub(crate) fn repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut rendered = String::from(quote);
    for ch in text.chars() {
        match ch {
            '\\' => rendered.push_str("\\\\"),
            '\n' => rendered.push_str("\\n"),
            '\r' => rendered.push_str("\\r"),
            '\t' => rendered.push_str("\\t"),
            _ if ch == quote => {
                rendered.push('\\');
                rendered.push(ch);
            }
            _ if ch < ' ' || ('\u{7f}'..='\u{a0}').contains(&ch) => {
                rendered.push_str(&format!("\\x{:02x}", u32::from(ch)));
            }
            _ => rendered.push(ch),
        }
    }
    rendered.push(quote);
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_repository_splitlines_matches_python() {
        assert_eq!(splitlines("a\r\nb\rc\u{2028}d\n"), ["a", "b", "c", "d"]);
        assert_eq!(splitlines("a\n\nb"), ["a", "", "b"]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn migration_repository_repr_matches_python() {
        assert_eq!(repr("HEAD"), "'HEAD'");
        assert_eq!(repr("it's"), "\"it's\"");
        assert_eq!(repr("a'\"\n\u{1}"), "'a\\'\"\\n\\x01'");
    }

    #[test]
    fn migration_repository_isupper_matches_python() {
        assert!(is_upper("API"));
        assert!(is_upper("A1-B"));
        assert!(!is_upper("Api"));
        assert!(!is_upper("123"));
        assert!(!is_upper("\u{1c5}"));
    }

    #[test]
    fn migration_repository_strip_includes_python_separators() {
        assert_eq!(strip("\u{1f} x \u{a0}"), "x");
        assert_eq!(split_whitespace(" a  b ").collect::<Vec<_>>(), ["a", "b"]);
    }
}

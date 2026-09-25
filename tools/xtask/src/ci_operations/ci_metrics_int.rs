//! Python `int(text)` in base 10 for `collect-ci-metrics.py`: argparse's
//! `type=int` options and `int(raw_attempt)` of saved run JSON.

use crate::repository::python_text::strip;

/// Python `int(text)` in base 10 as canonical decimal text (`-` sign, no
/// leading zeros), or `None` when `int()` raises `ValueError`.
pub(crate) fn python_int_text(text: &str) -> Option<String> {
    let text = strip(text);
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let bytes = digits.as_bytes();
    let well_formed = bytes.first().is_some_and(u8::is_ascii_digit)
        && bytes.last().is_some_and(u8::is_ascii_digit)
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_digit() || *byte == b'_')
        && !digits.contains("__");
    if !well_formed {
        return None;
    }
    let cleaned: String = digits.chars().filter(char::is_ascii_digit).collect();
    let trimmed = cleaned.trim_start_matches('0');
    Some(match (trimmed.is_empty(), negative) {
        (true, _) => "0".to_owned(),
        (false, true) => format!("-{trimmed}"),
        (false, false) => trimmed.to_owned(),
    })
}

/// Python `int(text)` in base 10, saturating beyond `i64`, which
/// preserves every comparison argparse's `type=int` options feed.
pub(crate) fn python_int(text: &str) -> Option<i64> {
    let digits = python_int_text(text)?;
    let saturated = if digits.starts_with('-') {
        i64::MIN
    } else {
        i64::MAX
    };
    Some(digits.parse().unwrap_or(saturated))
}

#[cfg(test)]
mod tests {
    use super::python_int;

    #[test]
    fn migration_ci_operations_python_int_follows_cpython() {
        assert_eq!(python_int(" 7\n"), Some(7));
        assert_eq!(python_int("-1_0"), Some(-10));
        assert_eq!(python_int("+010"), Some(10));
        for rejected in ["", "_1", "1_", "1__0", "1.0", "x", "0x10", "-", "1e3"] {
            assert_eq!(python_int(rejected), None, "{rejected}");
        }
    }
}

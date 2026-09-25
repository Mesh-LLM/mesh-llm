//! `json.loads(raw, object_pairs_hook=..., parse_constant=...)` over bytes
//! with CPython's scanner diagnostics (`Expecting value: line 1 column 1
//! (char 0)` and friends), so malformed evidence fails with legacy wording.
//! Objects keep source order; the hook decides what a duplicate key means.

use crate::ci_plan::document::Json;

/// Builds an object from its pairs, or rejects it with an error message.
pub(crate) type PairsHook = fn(Vec<(String, Json)>) -> Result<Json, String>;

/// Called with each completed object's pairs, then with each constant
/// (`NaN`, `Infinity`, `-Infinity`); either may reject the document.
pub(crate) struct Hooks {
    pub(crate) pairs: PairsHook,
    pub(crate) constant: fn(&str) -> Result<Json, String>,
}

/// CPython 3.13's C scanner raises `RecursionError` beyond 9998 nested
/// containers (its C recursion limit, independent of `sys.getrecursionlimit`).
const RECURSION_LIMIT: usize = 9998;

pub(crate) enum DecodeError {
    /// `ValueError`/`JSONDecodeError`/`UnicodeDecodeError` text.
    Value(String),
    Recursion,
}

pub(crate) fn loads(raw: &[u8], hooks: &Hooks) -> Result<Json, DecodeError> {
    decode(raw, hooks, false)
}

/// Marks a number `serde_json` cannot hold exactly (an integer beyond 64
/// bits or a float that overflows): `{EXACT_NUMBER: "<source text>"}`.
pub(crate) const EXACT_NUMBER: &str = "\u{0}exact-number";

/// `loads` that reports such numbers as [`EXACT_NUMBER`] objects instead of
/// the placeholder the bounded-evidence callers reject.
pub(crate) fn loads_exact(raw: &[u8], hooks: &Hooks) -> Result<Json, DecodeError> {
    decode(raw, hooks, true)
}

fn decode(raw: &[u8], hooks: &Hooks, exact_numbers: bool) -> Result<Json, DecodeError> {
    let raw = raw.strip_prefix(b"\xef\xbb\xbf").unwrap_or(raw);
    let text =
        std::str::from_utf8(raw).map_err(|error| DecodeError::Value(utf8_error(raw, &error)))?;
    let chars: Vec<char> = text.chars().collect();
    let mut scanner = Scanner {
        chars: &chars,
        hooks,
        depth: 0,
        exact_numbers,
    };
    let start = scanner.skip_ws(0);
    let (value, end) = match scanner.scan(start)? {
        Some(found) => found,
        None => return Err(scanner.error("Expecting value", start)),
    };
    let end = scanner.skip_ws(end);
    if end != chars.len() {
        return Err(scanner.error("Extra data", end));
    }
    Ok(value)
}

fn utf8_error(raw: &[u8], error: &std::str::Utf8Error) -> String {
    let start = error.valid_up_to();
    let (end, reason) = match error.error_len() {
        None => (raw.len(), "unexpected end of data"),
        Some(_) if (0xc2..=0xf4).contains(&raw[start]) => (start + 1, "invalid continuation byte"),
        Some(_) => (start + 1, "invalid start byte"),
    };
    if end - start == 1 {
        format!(
            "'utf-8' codec can't decode byte 0x{:02x} in position {start}: {reason}",
            raw[start]
        )
    } else {
        format!(
            "'utf-8' codec can't decode bytes in position {start}-{}: {reason}",
            end - 1
        )
    }
}

pub(crate) struct Scanner<'a> {
    pub(crate) chars: &'a [char],
    hooks: &'a Hooks,
    depth: usize,
    pub(crate) exact_numbers: bool,
}

type Scan = Result<Option<(Json, usize)>, DecodeError>;

impl Scanner<'_> {
    pub(crate) fn at(&self, index: usize) -> Option<char> {
        self.chars.get(index).copied()
    }

    fn skip_ws(&self, mut index: usize) -> usize {
        while matches!(self.at(index), Some(' ' | '\t' | '\n' | '\r')) {
            index += 1;
        }
        index
    }

    pub(crate) fn error(&self, message: &str, pos: usize) -> DecodeError {
        let before = &self.chars[..pos.min(self.chars.len())];
        let line = before.iter().filter(|ch| **ch == '\n').count() + 1;
        let column = match before.iter().rposition(|ch| *ch == '\n') {
            Some(newline) => pos - newline,
            None => pos + 1,
        };
        DecodeError::Value(format!(
            "{message}: line {line} column {column} (char {pos})"
        ))
    }

    fn literal(&self, index: usize, word: &str) -> bool {
        word.chars()
            .enumerate()
            .all(|(offset, ch)| self.at(index + offset) == Some(ch))
    }

    /// `scan_once`: `None` is CPython's `StopIteration` (no value here).
    fn scan(&mut self, index: usize) -> Scan {
        let Some(first) = self.at(index) else {
            return Ok(None);
        };
        match first {
            '"' => self
                .string(index + 1)
                .map(|(text, end)| Some((Json::String(text), end))),
            '{' | '[' => self.nested(first, index + 1),
            'n' if self.literal(index, "null") => Ok(Some((Json::Null, index + 4))),
            't' if self.literal(index, "true") => Ok(Some((Json::Bool(true), index + 4))),
            'f' if self.literal(index, "false") => Ok(Some((Json::Bool(false), index + 5))),
            'N' if self.literal(index, "NaN") => self.constant("NaN", index),
            'I' if self.literal(index, "Infinity") => self.constant("Infinity", index),
            '-' if self.literal(index, "-Infinity") => self.constant("-Infinity", index),
            _ => self.limited_number(index),
        }
    }

    fn constant(&self, name: &str, index: usize) -> Scan {
        let value = (self.hooks.constant)(name).map_err(DecodeError::Value)?;
        Ok(Some((value, index + name.chars().count())))
    }

    fn nested(&mut self, open: char, index: usize) -> Scan {
        self.depth += 1;
        if self.depth > RECURSION_LIMIT {
            return Err(DecodeError::Recursion);
        }
        let result = if open == '{' {
            self.object(index)
        } else {
            self.array(index)
        };
        self.depth -= 1;
        result.map(Some)
    }

    fn value(&mut self, index: usize) -> Result<(Json, usize), DecodeError> {
        self.scan(index)?
            .ok_or_else(|| self.error("Expecting value", index))
    }

    fn object(&mut self, index: usize) -> Result<(Json, usize), DecodeError> {
        let mut pairs = Vec::new();
        let mut index = self.skip_ws(index);
        if self.at(index) != Some('}') {
            loop {
                if self.at(index) != Some('"') {
                    return Err(
                        self.error("Expecting property name enclosed in double quotes", index)
                    );
                }
                let (key, after) = self.string(index + 1)?;
                index = self.skip_ws(after);
                if self.at(index) != Some(':') {
                    return Err(self.error("Expecting ':' delimiter", index));
                }
                let (value, after) = self.value(self.skip_ws(index + 1))?;
                pairs.push((key, value));
                index = self.skip_ws(after);
                if self.at(index) == Some('}') {
                    break;
                }
                if self.at(index) != Some(',') {
                    return Err(self.error("Expecting ',' delimiter", index));
                }
                let comma = index;
                index = self.skip_ws(index + 1);
                if self.at(index) == Some('}') {
                    return Err(self.error("Illegal trailing comma before end of object", comma));
                }
            }
        }
        let value = (self.hooks.pairs)(pairs).map_err(DecodeError::Value)?;
        Ok((value, index + 1))
    }

    fn array(&mut self, index: usize) -> Result<(Json, usize), DecodeError> {
        let mut items = Vec::new();
        let mut index = self.skip_ws(index);
        if self.at(index) != Some(']') {
            loop {
                let (value, after) = self.value(index)?;
                items.push(value);
                index = self.skip_ws(after);
                if self.at(index) == Some(']') {
                    break;
                }
                if self.at(index) != Some(',') {
                    return Err(self.error("Expecting ',' delimiter", index));
                }
                let comma = index;
                index = self.skip_ws(index + 1);
                if self.at(index) == Some(']') {
                    return Err(self.error("Illegal trailing comma before end of array", comma));
                }
            }
        }
        Ok((Json::Array(items), index + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep(pairs: Vec<(String, Json)>) -> Result<Json, String> {
        Ok(Json::Object(pairs))
    }

    fn reject(_: &str) -> Result<Json, String> {
        Err("nonfinite".to_owned())
    }

    fn message(raw: &[u8]) -> String {
        let hooks = Hooks {
            pairs: keep,
            constant: reject,
        };
        match loads(raw, &hooks) {
            Ok(_) => "ok".to_owned(),
            Err(DecodeError::Value(text)) => text,
            Err(DecodeError::Recursion) => "recursion".to_owned(),
        }
    }

    #[test]
    fn migration_ci_operations_decoder_matches_cpython_messages() {
        for (raw, expected) in [
            (&b""[..], "Expecting value: line 1 column 1 (char 0)"),
            (
                b"{",
                "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            (
                b"{\"a\"",
                "Expecting ':' delimiter: line 1 column 5 (char 4)",
            ),
            (
                b"{\"a\":1",
                "Expecting ',' delimiter: line 1 column 7 (char 6)",
            ),
            (
                b"{\"a\":1,}",
                "Illegal trailing comma before end of object: line 1 column 7 (char 6)",
            ),
            (
                b"[1,\n  ]",
                "Illegal trailing comma before end of array: line 1 column 3 (char 2)",
            ),
            (
                b"\"abc",
                "Unterminated string starting at: line 1 column 1 (char 0)",
            ),
            (
                b"\"a\x01\"",
                "Invalid control character at: line 1 column 3 (char 2)",
            ),
            (b"\"\\q\"", "Invalid \\escape: line 1 column 2 (char 1)"),
            (
                b"\"\\u12\"",
                "Invalid \\uXXXX escape: line 1 column 3 (char 2)",
            ),
            (b"1 2", "Extra data: line 1 column 3 (char 2)"),
            (b"-", "Expecting value: line 1 column 1 (char 0)"),
            (b"\xef\xbb\xbf{}", "ok"),
            (b"NaN", "nonfinite"),
            (
                b"\xff",
                "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
            ),
            (
                b"{\"a\":\"\xc3\"}",
                "'utf-8' codec can't decode byte 0xc3 in position 6: invalid continuation byte",
            ),
        ] {
            assert_eq!(message(raw), expected, "{raw:?}");
        }
    }
}

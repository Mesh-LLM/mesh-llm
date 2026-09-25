//! CPython `scanstring` (strict mode) and `_match_number` for the scanner in
//! `python_json_decode`, with the same error positions.

use crate::ci_operations::python_json_decode::{DecodeError, EXACT_NUMBER, Scanner};
use crate::ci_plan::document::Json;
use serde_json::Number;

impl Scanner<'_> {
    /// `number`, plus CPython's 4300-digit `int` conversion limit when
    /// exact numbers are requested (the digits past the limit never parse).
    pub(crate) fn limited_number(
        &self,
        start: usize,
    ) -> Result<Option<(Json, usize)>, DecodeError> {
        let found = self.number(start);
        if let Some((Json::Object(entries), _)) = &found
            && let [(_, Json::String(text))] = entries.as_slice()
        {
            let digits = text.trim_start_matches('-').len();
            if !text.contains(['.', 'e', 'E']) && digits > INT_MAX_STR_DIGITS {
                return Err(DecodeError::Value(format!(
                    "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string \
                     conversion: value has {digits} digits; use \
                     sys.set_int_max_str_digits() to increase the limit"
                )));
            }
        }
        Ok(found)
    }

    /// `scanstring(s, begin)` in strict mode; `begin` follows the quote.
    pub(crate) fn string(&self, begin: usize) -> Result<(String, usize), DecodeError> {
        let mut out = String::new();
        let mut next = begin;
        loop {
            let Some(ch) = self.at(next) else {
                return Err(self.error("Unterminated string starting at", begin - 1));
            };
            match ch {
                '"' => return Ok((out, next + 1)),
                '\\' => next = self.escape(begin, next + 1, &mut out)?,
                ch if (ch as u32) < 0x20 => {
                    return Err(self.error("Invalid control character at", next));
                }
                ch => {
                    out.push(ch);
                    next += 1;
                }
            }
        }
    }

    fn escape(&self, begin: usize, next: usize, out: &mut String) -> Result<usize, DecodeError> {
        let Some(ch) = self.at(next) else {
            return Err(self.error("Unterminated string starting at", begin - 1));
        };
        let simple = match ch {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => '\u{8}',
            'f' => '\u{c}',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'u' => return self.unicode_escape(next + 1, out),
            _ => return Err(self.error("Invalid \\escape", next - 1)),
        };
        out.push(simple);
        Ok(next + 1)
    }

    fn hex4(&self, start: usize) -> Option<u32> {
        (start..start + 4).try_fold(0_u32, |acc, index| {
            Some(acc * 16 + self.at(index)?.to_digit(16)?)
        })
    }

    fn unicode_escape(&self, next: usize, out: &mut String) -> Result<usize, DecodeError> {
        let mut end = next + 4;
        if end >= self.chars.len() {
            return Err(self.error("Invalid \\uXXXX escape", next - 1));
        }
        let mut code = self
            .hex4(next)
            .ok_or_else(|| self.error("Invalid \\uXXXX escape", end - 5))?;
        if (0xd800..0xdc00).contains(&code)
            && end + 6 < self.chars.len()
            && self.at(end) == Some('\\')
            && self.at(end + 1) == Some('u')
        {
            end += 6;
            let low = self
                .hex4(end - 4)
                .ok_or_else(|| self.error("Invalid \\uXXXX escape", end - 5))?;
            if (0xdc00..0xe000).contains(&low) {
                code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
            } else {
                end -= 6;
            }
        }
        // A lone surrogate cannot live in a Rust string; it becomes U+FFFD.
        out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
        Ok(end)
    }

    /// `_match_number`: `None` when no number starts here.
    pub(crate) fn number(&self, start: usize) -> Option<(Json, usize)> {
        let digit = |index: usize| self.at(index).is_some_and(|ch| ch.is_ascii_digit());
        let mut index = start + usize::from(self.at(start) == Some('-'));
        match self.at(index) {
            Some('0') => index += 1,
            Some('1'..='9') => {
                while digit(index) {
                    index += 1;
                }
            }
            _ => return None,
        }
        let mut float = false;
        if self.at(index) == Some('.') && digit(index + 1) {
            float = true;
            index += 1;
            while digit(index) {
                index += 1;
            }
        }
        if matches!(self.at(index), Some('e' | 'E')) {
            let mut exponent = index + 1;
            if matches!(self.at(exponent), Some('+' | '-')) {
                exponent += 1;
            }
            if digit(exponent) {
                float = true;
                while digit(exponent) {
                    exponent += 1;
                }
                index = exponent;
            }
        }
        let text: String = self.chars[start..index].iter().collect();
        Some((number(&text, float, self.exact_numbers), index))
    }
}

const INT_MAX_STR_DIGITS: usize = 4300;

/// Integers beyond 64 bits and non-finite floats never survive the bounded
/// integer check, so they become an arbitrary float that fails it.
fn number(text: &str, float: bool, exact: bool) -> Json {
    let parsed = if float {
        None
    } else {
        text.parse::<i64>().ok().map(Number::from)
    };
    let parsed = parsed
        .or_else(|| {
            (!float)
                .then(|| text.parse::<u64>().ok().map(Number::from))
                .flatten()
        })
        .or_else(|| text.parse::<f64>().ok().and_then(Number::from_f64));
    let exact_int_lost = !float && parsed.as_ref().is_some_and(|number| number.is_f64());
    if exact && (parsed.is_none() || exact_int_lost) {
        return Json::Object(vec![(
            EXACT_NUMBER.to_owned(),
            Json::String(text.to_owned()),
        )]);
    }
    let parsed = parsed.or_else(|| Number::from_f64(0.5));
    parsed.map_or(Json::Null, Json::Number)
}

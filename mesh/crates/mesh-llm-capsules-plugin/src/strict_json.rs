//! Parse a peer's push body strictly: one reading of the bytes, whoever reads
//! them.
//!
//! `serde_json` on its own accepts two things a pushed record must never
//! carry, because another reader of the same bytes may disagree with it:
//!
//! - **A duplicate key** at any depth. `serde_json::Value` keeps the last one
//!   and other readers keep the first, so a body with two `capsule_id`s (or
//!   two `signature`s) could verify under one reading and be stored and cited
//!   under another.
//! - **An integer outside 64 bits.** `serde_json` quietly reads it as a float.
//!
//! This module refuses both, and nesting deeper than [`MAX_JSON_DEPTH`]. What
//! `serde_json` refuses on its own (NaN, Infinity, a lone surrogate, bytes
//! that are not UTF-8, trailing bytes) stays refused, and so does a number
//! that overflows a float.
//!
//! **Numbers under `arbitrary_precision`.** The capsule-emit crate's
//! canonicalizer turns on `serde_json`'s `arbitrary_precision` feature for
//! the whole binary. `serde_json` then hands every number to a visitor as a
//! one-member map keyed by its private [`NUMBER_TOKEN`], holding the number's
//! text, and no longer refuses a float that overflows. So the visitor reads
//! that map back as the number it is and refuses one that is not finite, and
//! a body that spells the token itself as a key is refused: that object
//! could not be told apart from a number.

use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};

/// The key `serde_json` names a number by when it hands one to a visitor
/// under `arbitrary_precision` (its `number::TOKEN`, which it does not
/// export).
const NUMBER_TOKEN: &str = "$serde_json::private::Number";

/// The deepest nesting of arrays and objects a body may have: the most
/// `serde_json` itself will parse (its recursion limit of 128 lets 127
/// through).
pub const MAX_JSON_DEPTH: usize = 127;

/// Why a body was refused. The reason is for logs only; every refusal is the
/// same `request_malformed` on the wire.
#[derive(Debug, PartialEq, Eq)]
pub enum StrictError {
    TooDeep,
    IntegerOutOf64Bits,
    NumberToken,
    Invalid(String),
}

impl fmt::Display for StrictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrictError::TooDeep => write!(f, "nested deeper than {MAX_JSON_DEPTH}"),
            StrictError::IntegerOutOf64Bits => write!(f, "an integer outside 64 bits"),
            StrictError::NumberToken => write!(f, "a key that spells serde_json's number token"),
            StrictError::Invalid(why) => write!(f, "{why}"),
        }
    }
}

/// Parse `body` as one JSON value under the rules in the module doc.
pub fn parse(body: &[u8]) -> Result<Value, StrictError> {
    scan(body)?;
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let value = StrictValue::deserialize(&mut deserializer)
        .map_err(|e| StrictError::Invalid(e.to_string()))?;
    deserializer
        .end()
        .map_err(|e| StrictError::Invalid(e.to_string()))?;
    Ok(value.0)
}

/// One pass over the raw bytes for the three things a parsed value can no
/// longer tell: how deep the nesting went, whether an integer literal
/// fitted 64 bits before `serde_json` turned it into a float, and whether a
/// key spelled [`NUMBER_TOKEN`]. Anything else that is wrong with the
/// bytes is left for the parser to refuse.
fn scan(body: &[u8]) -> Result<(), StrictError> {
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < body.len() {
        match body[i] {
            b'"' => {
                let start = i;
                i += 1;
                while i < body.len() && body[i] != b'"' {
                    // Skip the escaped byte too, so an escaped quote does not
                    // end the string.
                    i += if body[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
                let is_key = body
                    .get(i..)
                    .unwrap_or_default()
                    .iter()
                    .find(|b| !b.is_ascii_whitespace())
                    == Some(&b':');
                if is_key && spells_number_token(&body[start..i.min(body.len())]) {
                    return Err(StrictError::NumberToken);
                }
            }
            b'[' | b'{' => {
                depth += 1;
                if depth > MAX_JSON_DEPTH {
                    return Err(StrictError::TooDeep);
                }
                i += 1;
            }
            b']' | b'}' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                while i < body.len()
                    && matches!(body[i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    i += 1;
                }
                integer_fits(&body[start..i])?;
            }
            _ => i += 1,
        }
    }
    Ok(())
}

/// Whether a string literal (quotes included) decodes to [`NUMBER_TOKEN`].
/// Only a literal with an escape needs decoding to tell.
fn spells_number_token(literal: &[u8]) -> bool {
    let inner = literal
        .strip_prefix(b"\"")
        .and_then(|l| l.strip_suffix(b"\""))
        .unwrap_or_default();
    if !inner.contains(&b'\\') {
        return inner == NUMBER_TOKEN.as_bytes();
    }
    serde_json::from_slice::<String>(literal).is_ok_and(|s| s == NUMBER_TOKEN)
}

/// A number from its text, as `serde_json` hands it over under
/// `arbitrary_precision`: refused unless it is finite as a float.
fn number<E: de::Error>(text: &str) -> Result<Value, E> {
    if !text.parse::<f64>().is_ok_and(f64::is_finite) {
        return Err(E::custom("a number that is not finite"));
    }
    text.parse::<Number>()
        .map(Value::Number)
        .map_err(|_| E::custom("not a JSON number"))
}

/// An integer literal (no fraction, no exponent) must lie in
/// `[-2^63, 2^64)`: what `i64` or `u64` can hold. A literal with a fraction
/// or an exponent is a float, and not this check's business.
fn integer_fits(token: &[u8]) -> Result<(), StrictError> {
    if token.iter().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        return Ok(());
    }
    let (negative, digits) = match token.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, token),
    };
    // A malformed literal ("--1", "1-2", "-") is the parser's to refuse.
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return Ok(());
    }
    let limit: &[u8] = if negative {
        b"9223372036854775808"
    } else {
        b"18446744073709551615"
    };
    let significant = match digits.iter().position(|&b| b != b'0') {
        Some(first) => &digits[first..],
        None => b"0",
    };
    // Equal-length decimal strings compare as their numbers do.
    let fits = significant.len() < limit.len()
        || (significant.len() == limit.len() && significant <= limit);
    if fits {
        Ok(())
    } else {
        Err(StrictError::IntegerOutOf64Bits)
    }
}

/// A `serde_json::Value` built by a visitor that refuses a duplicate key.
struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictVisitor).map(StrictValue)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("a number that is not finite"))
    }

    fn visit_str<E>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(StrictValue(item)) = seq.next_element()? {
            items.push(item);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut object = Map::new();
        let mut first = true;
        while let Some(key) = map.next_key::<String>()? {
            // A number under `arbitrary_precision` (see the module doc): the
            // body itself never spells the token, `scan` saw to that.
            if first && key == NUMBER_TOKEN {
                let text: String = map.next_value()?;
                return number(&text);
            }
            first = false;
            if object.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate key {key:?}")));
            }
            let StrictValue(value) = map.next_value()?;
            object.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(body: &str) -> bool {
        parse(body.as_bytes()).is_err()
    }

    #[test]
    fn an_ordinary_body_parses_and_keeps_its_member_order() {
        let value = parse(br#"{"b": 1, "a": [true, null, "x\"y", -2, 1.5]}"#).unwrap();
        assert_eq!(value["a"][2], "x\"y");
        let keys: Vec<_> = value.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys, ["b", "a"]);
    }

    #[test]
    fn a_duplicate_key_at_any_depth_is_refused() {
        assert!(refused(r#"{"a": 1, "a": 2}"#));
        assert!(refused(r#"{"x": [{"a": 1, "a": 1}]}"#));
        assert!(
            refused(r#"{"a": 1, "\u0061": 2}"#),
            "the same key, spelled with an escape"
        );
        assert!(!refused(r#"{"a": {"a": 1}}"#), "one key per object");
    }

    #[test]
    fn integers_are_held_to_64_bits() {
        assert!(!refused("18446744073709551615"));
        assert!(refused("18446744073709551616"));
        assert!(!refused("-9223372036854775808"));
        assert!(refused("-9223372036854775809"));
        assert!(refused(r#"{"n": 100000000000000000000000000000}"#));
        assert!(
            refused("1e400"),
            "serde_json refuses a float that overflows"
        );
        assert!(
            !refused(r#""18446744073709551616""#),
            "digits inside a string are text"
        );
    }

    /// Under `arbitrary_precision` a number reaches the visitor as a map
    /// keyed by serde_json's private token: it is read back as a number,
    /// and a body spelling the token itself is refused.
    #[test]
    fn numbers_are_numbers_and_the_number_token_is_refused() {
        let value = parse(br#"{"t": 0.7, "n": [1, -2, 1e2, 18446744073709551615]}"#).unwrap();
        assert_eq!(value["t"].as_f64(), Some(0.7));
        assert_eq!(value["n"][0].as_u64(), Some(1));
        assert_eq!(value["n"][1].as_i64(), Some(-2));
        assert_eq!(value["n"][2].as_f64(), Some(100.0));
        assert_eq!(value["n"][3].as_u64(), Some(u64::MAX));
        assert!(refused("-1e400"));
        assert!(refused(r#"{"$serde_json::private::Number": "1"}"#));
        assert!(refused(r#"{"\u0024serde_json::private::Number": "1"}"#));
        assert!(refused(r#"{"a": {"$serde_json::private::Number" : "1"}}"#));
        let text = parse(br#"["$serde_json::private::Number"]"#).unwrap();
        assert_eq!(text[0], "$serde_json::private::Number", "a value is text");
        assert!(!refused(r#"{"$serde_json::private::Numbers": "1"}"#));
    }

    #[test]
    fn nesting_stops_at_the_limit() {
        let at = "[".repeat(MAX_JSON_DEPTH) + &"]".repeat(MAX_JSON_DEPTH);
        let over = "[".repeat(MAX_JSON_DEPTH + 1) + &"]".repeat(MAX_JSON_DEPTH + 1);
        assert!(!refused(&at));
        assert!(refused(&over));
        let brackets_in_a_string = format!(r#"["{}"]"#, "[".repeat(500));
        assert!(!refused(&brackets_in_a_string));
    }

    #[test]
    fn what_serde_json_refuses_stays_refused() {
        for body in [
            "NaN",
            "[Infinity]",
            "{\"x\": \"\\ud800\"}",
            "{} x",
            "",
            "{\"a\":",
            "[1,]",
        ] {
            assert!(refused(body), "{body:?}");
        }
        assert!(parse(b"{\"x\": \"\xff\"}").is_err(), "not UTF-8");
    }

    #[test]
    fn hostile_bytes_never_panic() {
        for body in [
            &b"\"\\"[..],
            b"-",
            b"--1",
            b"1-2",
            b"\"",
            b"{\"a\\",
            b"[-",
            b"\xff\xfe",
        ] {
            let _ = parse(body);
        }
    }
}

//! Exact sampling admission from source tokens after ordinary strict JSON parsing.
use serde_json::value::RawValue;
use std::collections::BTreeMap;

type Fields = BTreeMap<String, Box<RawValue>>;

pub(super) fn admit(raw: &[u8]) -> Result<bool, serde_json::Error> {
    let root: Fields = serde_json::from_slice(raw)?;
    let Some(replay) = root.get("replay") else {
        return Ok(true); // Existing policy owns missing or nonobject replay blocks.
    };
    if !replay.get().trim_start().starts_with('{') {
        return Ok(true);
    }
    let replay: Fields = serde_json::from_str(replay.get())?;
    Ok([("temperature", "0"), ("seed", "42")]
        .into_iter()
        .all(|(field, expected)| {
            replay
                .get(field)
                .is_none_or(|value| exact_decimal(value.get(), expected))
        }))
}

fn exact_decimal(token: &str, expected: &str) -> bool {
    let token = token.trim();
    let negative = token.starts_with('-');
    let unsigned = token.strip_prefix('-').unwrap_or(token);
    if !unsigned.as_bytes().first().is_some_and(u8::is_ascii_digit) {
        return false;
    }
    let (mantissa, exponent) = unsigned.split_once(['e', 'E']).unwrap_or((unsigned, "0"));
    let fraction = mantissa.split_once('.').map_or(0, |(_, tail)| tail.len());
    let digits = mantissa.replace('.', "");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return expected == "0";
    }
    if negative || expected == "0" {
        return false;
    }
    let significant = digits.trim_end_matches('0');
    let trailing = digits.len() - significant.len();
    let Some(power) = exponent
        .parse::<i64>()
        .ok()
        .and_then(|value| value.checked_sub(i64::try_from(fraction).ok()?))
        .and_then(|value| value.checked_add(i64::try_from(trailing).ok()?))
    else {
        return false;
    };
    significant == expected && power == 0
}

#[cfg(test)]
#[path = "sampling_tests.rs"]
mod tests;

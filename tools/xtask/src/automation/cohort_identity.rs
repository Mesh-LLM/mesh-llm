use crate::command::DynResult;
use sha2::{Digest, Sha256};

pub(crate) fn digest(value: &serde_json::Value) -> DynResult<String> {
    let mut encoded = String::new();
    encode(value, &mut encoded)?;
    Ok(hex::encode(Sha256::digest(encoded.as_bytes())))
}

fn encode(value: &serde_json::Value, output: &mut String) -> DynResult<()> {
    match value {
        serde_json::Value::Null => output.push_str("null"),
        serde_json::Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => {
            if value.is_f64() {
                let mut number = String::new();
                crate::automation::codepoint_json::emission::write_float(
                    &mut number,
                    value.as_f64().ok_or("invalid cohort number")?,
                );
                if let Some((mantissa, exponent)) = number.split_once('e') {
                    use std::fmt::Write;
                    let exponent: i32 = exponent.parse()?;
                    write!(output, "{mantissa}e{exponent:+03}")?;
                } else {
                    output.push_str(&number);
                }
            } else {
                output.push_str(&value.to_string());
            }
        }
        serde_json::Value::String(value) => string(value, output)?,
        serde_json::Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                encode(value, output)?;
            }
            output.push(']');
        }
        serde_json::Value::Object(entries) => {
            output.push('{');
            let mut ordered = entries.iter().collect::<Vec<_>>();
            ordered.sort_by_key(|(key, _)| *key);
            for (index, (key, value)) in ordered.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                string(key, output)?;
                output.push(':');
                encode(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn string(value: &str, output: &mut String) -> DynResult<()> {
    let encoded = serde_json::to_string(value)?;
    for character in encoded.chars() {
        let codepoint = u32::from(character);
        if codepoint < 127 {
            output.push(character);
        } else if codepoint <= 65535 {
            use std::fmt::Write;
            write!(output, "\\u{codepoint:04x}")?;
        } else {
            use std::fmt::Write;
            let adjusted = codepoint - 65536;
            let high = 0xd800 + (adjusted >> 10);
            let low = 0xdc00 + (adjusted & 1023);
            write!(output, "\\u{high:04x}\\u{low:04x}")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_cohort_identity_uses_ascii_escaped_compact_sorted_bytes() {
        let value = serde_json::json!({"z":"\u{1f600}","a":"\u{e9}"});
        let mut encoded = String::new();
        encode(&value, &mut encoded).unwrap();
        assert_eq!(encoded, r#"{"a":"\u00e9","z":"\ud83d\ude00"}"#);
    }

    #[test]
    fn control_and_float_extensions_preserve_canonical_encoding() {
        let value: serde_json::Value =
            serde_json::from_str("{\"z\":1e-7,\"a\":\"\\u007f\"}").unwrap();
        let mut encoded = String::new();
        encode(&value, &mut encoded).unwrap();
        assert_eq!(encoded, r#"{"a":"\u007f","z":1e-07}"#);
    }
}

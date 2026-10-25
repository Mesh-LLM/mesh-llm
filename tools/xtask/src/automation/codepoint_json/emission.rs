use super::strings::JsonString;
use crate::prepared_input::value_format::float_repr;

pub(in crate::automation) fn write_float(output: &mut String, number: f64) {
    if number.is_nan() {
        output.push_str("NaN");
    } else if number.is_infinite() {
        output.push_str(if number.is_sign_negative() {
            "-Infinity"
        } else {
            "Infinity"
        });
    } else {
        output.push_str(&float_repr(number));
    }
}

pub(in crate::automation) fn write_string(output: &mut String, text: &JsonString) {
    output.push('"');
    for code in text.codepoints() {
        let Some(character) = char::from_u32(code) else {
            output.push_str(&format!("\\u{code:04x}"));
            continue;
        };
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{8}' => output.push_str("\\b"),
            '\u{c}' => output.push_str("\\f"),
            ' '..='~' => output.push(character),
            _ => {
                let mut units = [0_u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    output.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    output.push('"');
}

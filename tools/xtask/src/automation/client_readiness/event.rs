use crate::process::{LineEnding, ObservedLine};
use serde_json::Value;

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/parser.rs"]
mod tests;

pub(crate) fn matches(line: ObservedLine<'_>) -> bool {
    match line.ending {
        LineEnding::Eof => return false,
        LineEnding::Lf => (),
    }
    let text = String::from_utf8_lossy(line.bytes);
    if !bounded_depth(&text) {
        return false;
    }
    let Ok(Value::Object(event)) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    let structured = [
        ("event", "passive_mode"),
        ("status", "ready"),
        ("role", "client"),
    ]
    .iter()
    .all(|(key, expected)| event.get(*key).and_then(Value::as_str) == Some(*expected));
    structured
        || event
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(phrase)
}

fn phrase(value: &str) -> bool {
    value.to_lowercase().contains("client ready")
}

fn bounded_depth(text: &str) -> bool {
    let mut depth = 0_u8;
    let mut quoted = false;
    let mut escaped = false;
    for byte in text.bytes() {
        if quoted {
            if escaped {
                escaped = false;
            } else {
                match byte {
                    b'\\' => escaped = true,
                    b'"' => quoted = false,
                    _ => (),
                }
            }
        } else {
            match byte {
                b'"' => quoted = true,
                b'{' | b'[' => {
                    if depth == 64 {
                        return false;
                    }
                    depth += 1;
                }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => (),
            }
        }
    }
    true
}

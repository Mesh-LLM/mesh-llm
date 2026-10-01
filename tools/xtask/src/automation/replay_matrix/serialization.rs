use crate::automation::codepoint_json::emission::{write_float, write_string};
use crate::automation::codepoint_json::value::Value;

pub(super) fn render(value: &Value) -> String {
    let mut output = String::new();
    write_value(&mut output, value);
    output.push('\n');
    output
}

fn write_value(output: &mut String, value: &Value) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(flag) => output.push_str(if *flag { "true" } else { "false" }),
        Value::Int(integer) => output.push_str(&integer.to_string()),
        Value::BigInt(text) => output.push_str(text),
        Value::Float(number) => write_float(output, *number),
        Value::Str(text) => write_string(output, text),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                write_value(output, value);
            }
            output.push(']');
        }
        Value::Object(entries) => {
            let mut ordered = entries.iter().collect::<Vec<_>>();
            ordered.sort_by(|(left, _), (right, _)| left.cmp(right));
            output.push('{');
            for (index, (key, value)) in ordered.into_iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                write_string(output, key);
                output.push_str(": ");
                write_value(output, value);
            }
            output.push('}');
        }
    }
}

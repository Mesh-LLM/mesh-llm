use crate::automation::codepoint_json::emission::{write_float, write_string};
use crate::automation::codepoint_json::value::Value;

pub(in crate::automation) fn render(value: &Value) -> String {
    let mut output = String::new();
    write_value(&mut output, value, 0);
    output.push('\n');
    output
}

fn write_value(output: &mut String, value: &Value, depth: usize) {
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
                separator(output, index, depth + 1);
                write_value(output, value, depth + 1);
            }
            close(output, depth, ']', values.is_empty());
        }
        Value::Object(entries) => {
            let mut ordered = entries.iter().collect::<Vec<_>>();
            ordered.sort_by(|(left, _), (right, _)| left.cmp(right));
            output.push('{');
            for (index, (key, value)) in ordered.into_iter().enumerate() {
                separator(output, index, depth + 1);
                write_string(output, key);
                output.push_str(": ");
                write_value(output, value, depth + 1);
            }
            close(output, depth, '}', entries.is_empty());
        }
    }
}

fn separator(output: &mut String, index: usize, depth: usize) {
    if index > 0 {
        output.push(',');
    }
    output.push('\n');
    output.push_str(&"  ".repeat(depth));
}

fn close(output: &mut String, depth: usize, bracket: char, empty: bool) {
    if !empty {
        output.push('\n');
        output.push_str(&"  ".repeat(depth));
    }
    output.push(bracket);
}

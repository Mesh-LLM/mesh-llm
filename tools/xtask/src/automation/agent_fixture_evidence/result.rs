use crate::command::DynResult;
use serde_json::Value;

const LINE_LIMIT: usize = 100_000;
const NODE_LIMIT: usize = 1_000_000;
const DEPTH_LIMIT: usize = 64;

fn failed(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("error")
        || value.get("isError").and_then(Value::as_bool) == Some(true)
        || value.get("is_error").and_then(Value::as_bool) == Some(true)
        || (matches!(
            value.get("type").and_then(Value::as_str),
            Some("tool_result" | "toolResult")
        ) && (value.get("success").and_then(Value::as_bool) == Some(false)
            || matches!(
                value.get("status").and_then(Value::as_str),
                Some("failed" | "error")
            )))
}

fn collect(
    value: &Value,
    names: &mut Vec<String>,
    depth: usize,
    nodes: &mut usize,
) -> DynResult<()> {
    *nodes += 1;
    if depth > DEPTH_LIMIT || *nodes > NODE_LIMIT {
        return Err("agent fixture evidence exceeds nesting or node limit".into());
    }
    match value {
        Value::Object(object) => {
            if failed(value) {
                return Err("agent fixture evidence reports a failed result".into());
            }
            let direct = object
                .get("toolName")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .or_else(|| object.get("tool_name").and_then(Value::as_str))
                .filter(|name| !name.is_empty());
            let nested = if matches!(
                object.get("type").and_then(Value::as_str),
                Some("tool_call" | "tool_request" | "toolRequest")
            ) {
                ["toolCall", "tool_call", "toolRequest", "tool_request"]
                    .into_iter()
                    .filter_map(|key| object.get(key))
                    .filter_map(|item| item.get("name").and_then(Value::as_str))
                    .find(|name| !name.is_empty())
            } else {
                None
            };
            if let Some(name) = direct.or(nested) {
                names.push(name.to_owned());
            }
            for item in object.values() {
                collect(item, names, depth + 1, nodes)?;
            }
        }
        Value::Array(items) => {
            for item in items {
                collect(item, names, depth + 1, nodes)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn check_depth(line: &str) -> DynResult<()> {
    if !line.trim_start().starts_with(['{', '[']) {
        return Ok(());
    }
    let mut depth = 0_usize;
    let mut quoted = false;
    let mut escaped = false;
    for byte in line.bytes() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else {
            match byte {
                b'"' => quoted = true,
                b'{' | b'[' => {
                    depth += 1;
                    if depth > DEPTH_LIMIT + 1 {
                        return Err("agent fixture evidence exceeds nesting limit".into());
                    }
                }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
    }
    Ok(())
}

pub(super) fn validate(bytes: &[u8], label: &str, require_tools: bool) -> DynResult<String> {
    let raw = String::from_utf8_lossy(bytes);
    let mut tools = Vec::new();
    let mut nodes = 0;
    for (index, line) in raw.lines().enumerate() {
        if index >= LINE_LIMIT {
            return Err("agent fixture evidence exceeds line limit".into());
        }
        check_depth(line)?;
        // Plain answer lines and malformed non-JSON client log lines are allowed.
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            collect(&value, &mut tools, 0, &mut nodes)?;
        }
    }
    for expected in [
        "CODEWORD=signal-7429",
        "CHECKSUM=FS-319-DELTA",
        "PRIME_SUM=10",
        "QUESTION=facts/signal.md",
    ] {
        if !raw.lines().any(|line| line.trim_end() == expected) {
            return Err(format!("{label} answer did not include expected fact: {expected}").into());
        }
    }
    let edits = [
        "edit",
        "write",
        "text_editor",
        "developer__text_editor",
        "patch",
    ];
    if require_tools
        && (tools.len() < 3 || !tools.iter().any(|tool| edits.contains(&tool.as_str())))
    {
        return Err(
            format!("{label} did not report expected filesystem/coding tool events").into(),
        );
    }
    let mut output = format!("{label} live coding smoke passed\n");
    if !tools.is_empty() {
        output.push_str(&format!("  tools: {}\n", tools.join(", ")));
    }
    Ok(output)
}

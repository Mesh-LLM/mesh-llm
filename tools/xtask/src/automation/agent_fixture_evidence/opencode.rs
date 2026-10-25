//! OpenCode's actual session and multi-turn tool/answer evidence contracts.
use crate::command::DynResult;
use serde_json::Value;

const LINE_LIMIT: usize = 100_000;
const NODE_LIMIT: usize = 1_000_000;
const DEPTH_LIMIT: usize = 64;
const FACTS: [&str; 4] = [
    "CODEWORD=signal-7429",
    "CHECKSUM=FS-319-DELTA",
    "PRIME_SUM=10",
    "QUESTION=facts/signal.md",
];
const FILESYSTEM: [&str; 7] = [
    "bash",
    "read",
    "grep",
    "glob",
    "edit",
    "write",
    "apply_patch",
];
const EDITS: [&str; 3] = ["edit", "write", "apply_patch"];

fn failed(event: &Value) -> bool {
    if event.get("type").and_then(Value::as_str) == Some("error") {
        return true;
    }
    if event.get("type").and_then(Value::as_str) != Some("tool_use") {
        return false;
    }
    let part = &event["part"];
    [part, &part["state"]].into_iter().any(|state| {
        state.get("success").and_then(Value::as_bool) == Some(false)
            || matches!(
                state.get("status").and_then(Value::as_str),
                Some("error" | "failed")
            )
            || state.get("error").is_some_and(|error| !error.is_null())
    })
}

fn events(bytes: &[u8], mut visit: impl FnMut(&Value) -> DynResult<()>) -> DynResult<()> {
    for (index, raw) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if index >= LINE_LIMIT {
            return Err("OpenCode evidence exceeds line limit".into());
        }
        // Non-event logging, scalar JSON and malformed log lines are not events.
        let Ok(line) = std::str::from_utf8(raw) else {
            continue;
        };
        let Ok(event @ Value::Object(_)) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if failed(&event) {
            return Err("OpenCode evidence reports a failed event".into());
        }
        visit(&event)?;
    }
    Ok(())
}

pub(super) fn session(bytes: &[u8]) -> DynResult<String> {
    let mut session = None;
    events(bytes, |event| {
        if session.is_none()
            && let Some(identity) = event.get("sessionID").and_then(Value::as_str)
            && !identity.trim().is_empty()
            && !identity.chars().any(char::is_control)
        {
            session = Some(identity.to_owned());
        }
        Ok(())
    })?;
    session
        .map(|identity| format!("{identity}\n"))
        .ok_or_else(|| "OpenCode turn 1 did not emit a usable sessionID".into())
}

fn record_facts(text: &str, found: &mut [bool; 4]) {
    for line in text.lines() {
        for (index, expected) in FACTS.into_iter().enumerate() {
            found[index] |= line.trim_end() == expected;
        }
    }
}

fn text_values(
    value: &Value,
    found: &mut [bool; 4],
    nodes: &mut usize,
    depth: usize,
) -> DynResult<()> {
    *nodes += 1;
    if depth > DEPTH_LIMIT || *nodes > NODE_LIMIT {
        return Err("OpenCode text evidence exceeds nesting or node limit".into());
    }
    match value {
        Value::String(text) => record_facts(text, found),
        Value::Array(items) => {
            for item in items {
                text_values(item, found, nodes, depth + 1)?;
            }
        }
        Value::Object(fields) => {
            for (name, item) in fields {
                if matches!(name.as_str(), "text" | "content" | "message" | "delta")
                    || matches!(item, Value::Object(_) | Value::Array(_))
                {
                    text_values(item, found, nodes, depth + 1)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn result(bytes: &[u8]) -> DynResult<String> {
    let mut tools = Vec::new();
    let mut found = [false; 4];
    let mut nodes = 0;
    events(bytes, |event| {
        match event.get("type").and_then(Value::as_str) {
            Some("tool_use") => {
                let part = &event["part"];
                if let Some(name) = part
                    .get("tool")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .or_else(|| {
                        part.get("name")
                            .and_then(Value::as_str)
                            .filter(|name| !name.is_empty())
                    })
                {
                    tools.push(name.to_owned()); // One name per actual event; repeated calls remain separate.
                }
            }
            Some("text" | "message" | "assistant") => {
                text_values(event, &mut found, &mut nodes, 0)?
            }
            _ => {}
        }
        Ok(())
    })?;
    // OpenCode may print its answer as plain lines alongside structured events.
    record_facts(&String::from_utf8_lossy(bytes), &mut found);
    if let Some(index) = found.iter().position(|present| !present) {
        return Err(format!(
            "OpenCode answer did not include expected fact: {}",
            FACTS[index]
        )
        .into());
    }
    let filesystem = tools
        .iter()
        .filter(|tool| FILESYSTEM.contains(&tool.as_str()))
        .count();
    if tools.len() < 4 || filesystem < 4 || !tools.iter().any(|tool| EDITS.contains(&tool.as_str()))
    {
        return Err(
            "OpenCode did not report expected multi-turn filesystem/coding tool events".into(),
        );
    }
    Ok(format!(
        "OpenCode multi-turn coding smoke passed\n  tools: {}\n",
        tools.join(", ")
    ))
}

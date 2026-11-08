use super::*;

// ─── Helpers ─────────────────────────────────────────────────────────

pub(crate) fn api_key_or_skip(test: &str) -> Option<String> {
    match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) if !k.trim().is_empty() => Some(k),
        _ => {
            eprintln!("[{test}] OPENROUTER_API_KEY not set — skipping live eval");
            None
        }
    }
}

pub(crate) fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut boundary = n;
        while !s.is_char_boundary(boundary) {
            boundary -= 1;
        }
        format!("{}...", &s[..boundary])
    }
}

pub(crate) fn tool_schema(name: &str, params: &[(&str, &str)]) -> Value {
    let props: serde_json::Map<String, Value> = params
        .iter()
        .map(|(p, ty)| (p.to_string(), json!({"type": ty})))
        .collect();
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": format!("{name} tool"),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": params.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            }
        }
    })
}

pub(crate) fn agent_tools() -> Value {
    json!([
        tool_schema("list_dir", &[("path", "string")]),
        tool_schema("read_file", &[("path", "string")]),
        tool_schema("search", &[("pattern", "string"), ("path", "string")]),
        tool_schema("run_command", &[("cmd", "string")]),
    ])
}

pub(crate) fn user_turn(prompt: &str, tools: Option<Value>) -> Value {
    let mut body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    });
    if let Some(t) = tools {
        body.as_object_mut().unwrap().insert("tools".into(), t);
    }
    body
}

/// Assistant text, mirroring the in-tree backend's fallback chain.
///
/// Reasoning models routinely spend their whole budget in `reasoning` and
/// return `content: null` (the failure Hermes' troubleshooting doc also
/// documents). Reading `content` alone silently dropped 20/30 committee trials,
/// so fall back to `reasoning` before declaring the response empty.
pub(crate) fn response_text(body: &Value) -> String {
    let msg = body.pointer("/choices/0/message");
    let content = msg
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !content.is_empty() {
        return content.to_string();
    }
    msg.and_then(|m| m.get("reasoning"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}

pub(crate) fn response_tool_calls(body: &Value) -> Vec<(String, String)> {
    body.pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|tcs| {
            tcs.iter()
                .filter_map(|tc| {
                    Some((
                        tc.pointer("/function/name")?.as_str()?.to_string(),
                        tc.pointer("/function/arguments")?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

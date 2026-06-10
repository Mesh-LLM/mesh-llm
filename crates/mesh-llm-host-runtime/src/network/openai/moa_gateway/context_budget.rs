use serde_json::Value;

const MOA_BASE_RESERVE_TOKENS: u32 = 512;
const MOA_TOOL_SCHEMA_RESERVE_TOKENS: u32 = 512;
const MOA_TOOL_RESULT_RESERVE_TOKENS: u32 = 1_024;
const TOOL_RESULT_SCAN_MAX_DEPTH: usize = 32;

pub(in crate::network::openai::moa_gateway) fn add_moa_context_reserve(
    body: &Value,
    required_tokens: Option<u32>,
) -> Option<u32> {
    required_tokens.map(|tokens| tokens.saturating_add(moa_context_reserve_tokens(body)))
}

fn moa_context_reserve_tokens(body: &Value) -> u32 {
    let mut reserve = MOA_BASE_RESERVE_TOKENS;
    if body_has_tools(body) {
        reserve = reserve.saturating_add(MOA_TOOL_SCHEMA_RESERVE_TOKENS);
    }
    if body_has_tool_result(body) {
        reserve = reserve.saturating_add(MOA_TOOL_RESULT_RESERVE_TOKENS);
    }
    reserve
}

fn body_has_tools(body: &Value) -> bool {
    body.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
}

fn body_has_tool_result(body: &Value) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| messages.iter().any(value_is_tool_result))
        || value_contains_responses_tool_result(body)
}

fn value_contains_responses_tool_result(value: &Value) -> bool {
    value_contains_responses_tool_result_at_depth(value, 0)
}

fn value_contains_responses_tool_result_at_depth(value: &Value, depth: usize) -> bool {
    if depth > TOOL_RESULT_SCAN_MAX_DEPTH {
        return false;
    }
    match value {
        Value::Array(values) => values
            .iter()
            .any(|nested| value_contains_responses_tool_result_at_depth(nested, depth + 1)),
        Value::Object(object) => {
            value_is_tool_result(value)
                || object
                    .iter()
                    .filter(|(key, _)| matches!(key.as_str(), "output" | "items" | "content"))
                    .any(|(_, nested)| {
                        value_contains_responses_tool_result_at_depth(nested, depth + 1)
                    })
        }
        _ => false,
    }
}

fn value_is_tool_result(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    if object.get("role").and_then(Value::as_str) == Some("tool") {
        return true;
    }
    matches!(
        object.get("type").and_then(Value::as_str),
        Some("tool_result" | "tool_call_output" | "function_call_output")
    ) || object.contains_key("tool_call_id")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn adds_base_reserve_for_plain_moa_request() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});

        assert_eq!(add_moa_context_reserve(&body, Some(1_000)), Some(1_512));
    }

    #[test]
    fn adds_tool_schema_and_tool_result_reserve_for_chat_shape() {
        let body = json!({
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "messages": [
                {"role": "user", "content": "read"},
                {"role": "tool", "tool_call_id": "call_1", "content": "result"}
            ]
        });

        assert_eq!(add_moa_context_reserve(&body, Some(1_000)), Some(3_048));
    }

    #[test]
    fn detects_responses_api_tool_outputs() {
        let body = json!({
            "input": "continue",
            "output": [{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result"
            }]
        });

        assert_eq!(add_moa_context_reserve(&body, Some(1_000)), Some(2_536));
    }

    #[test]
    fn responses_tool_output_scan_is_depth_limited() {
        let mut nested = json!({"type": "function_call_output", "output": "late"});
        for _ in 0..40 {
            nested = json!({"output": [nested]});
        }
        let body = json!({"input": nested});

        assert_eq!(add_moa_context_reserve(&body, Some(1_000)), Some(1_512));
    }

    #[test]
    fn preserves_unknown_required_tokens() {
        let body = json!({"messages": []});

        assert_eq!(add_moa_context_reserve(&body, None), None);
    }
}

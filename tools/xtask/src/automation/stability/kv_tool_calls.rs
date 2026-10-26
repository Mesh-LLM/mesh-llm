//! typed KV tool identities and sanitized growing conversation history.
use super::{
    kv_requests::{Key, TOOL},
    responses,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    key: Key,
}

pub(super) struct Call {
    pub id: String,
    pub key: Key,
}

pub(super) fn extract(response: &Value, expected: Key) -> Result<Call, String> {
    if responses::first_choice(response)?["finish_reason"] != "tool_calls" {
        return Err("KV tool turn did not finish with tool_calls".into());
    }
    let calls = responses::message(response)?["tool_calls"]
        .as_array()
        .filter(|calls| calls.len() == 1)
        .ok_or("KV tool turn requires exactly one identified call")?;
    let raw = &calls[0];
    let id = raw["id"]
        .as_str()
        .filter(|id| !id.trim().is_empty() && id.len() <= 65536)
        .ok_or("KV tool call has an invalid id")?;
    if raw["type"] != "function" || raw["function"]["name"] != TOOL {
        return Err("KV tool call has an unexpected type or function".into());
    }
    let arguments = &raw["function"]["arguments"];
    let decoded: Arguments = match arguments {
        Value::String(text) => serde_json::from_str(text),
        Value::Object(_) => serde_json::from_value(arguments.clone()),
        _ => return Err("KV tool arguments must be a JSON string or object".into()),
    }
    .map_err(|_| "KV tool arguments require one supported fixture key")?;
    if decoded.key != expected {
        return Err(format!(
            "KV tool returned key={} but this turn required {}",
            decoded.key.name(),
            expected.name()
        ));
    }
    Ok(Call {
        id: id.into(),
        key: decoded.key,
    })
}

pub(super) fn append_tool(
    messages: &mut Vec<Value>,
    response: &Value,
    call: &Call,
) -> Result<(), String> {
    let content = &responses::message(response)?["content"];
    if !content.is_null() && !content.is_string() {
        return Err("KV assistant tool content must be text or null".into());
    }
    messages.push(json!({"role":"assistant","content":content,
        "tool_calls":[{"id":call.id,"type":"function","function":{"name":TOOL,
            "arguments":serde_json::to_string(&json!({"key":call.key}))
                .map_err(|_| "KV tool arguments could not be serialized")?}}]}));
    messages.push(json!({"role":"tool","tool_call_id":call.id,"name":TOOL,
        "content":serde_json::to_string(&json!({"key":call.key,"value":call.key.fact()}))
            .map_err(|_| "KV tool result could not be serialized")?}));
    Ok(())
}

pub(super) fn text_message(response: &Value, expected: &[&str]) -> Result<Value, String> {
    let content = responses::final_content(response)?;
    for expected in expected {
        responses::validate_answer(content, expected)?;
    }
    Ok(json!({"role":"assistant","content":content}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automation::stability::kv_requests::PIN;
    fn tool(key: &str) -> Value {
        json!({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
            "reasoning_content":"private reasoning","tool_calls":[{"id":"call-kv","type":"function",
                "private_field":"not history","function":{"name":TOOL,"arguments":json!({"key":key}).to_string()}}]}}]})
    }
    #[test]
    fn kv_tool_identity_requires_one_call_and_the_requested_fact() {
        assert!(extract(&tool("primary"), Key::Primary).is_ok());
        assert!(extract(&tool("secondary"), Key::Secondary).is_ok());
        assert!(extract(&tool("secondary"), Key::Primary).is_err());
        for mutation in 0..7 {
            let mut response = tool("primary");
            let call = &mut response["choices"][0]["message"]["tool_calls"][0];
            match mutation {
                0 => call["id"] = json!("  "),
                1 => call["type"] = json!("foreign"),
                2 => call["function"]["name"] = json!("foreign"),
                3 => call["function"]["arguments"] = json!("{bad"),
                4 => call["function"]["arguments"] = json!({"key":"primary","extra":true}),
                5 => response["choices"][0]["finish_reason"] = json!("stop"),
                _ => response["choices"][0]["message"]["tool_calls"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({})),
            }
            assert!(
                extract(&response, Key::Primary).is_err(),
                "mutation {mutation}"
            );
        }
    }
    #[test]
    fn kv_tool_history_keeps_identity_and_fact_without_private_assistant_fields() {
        let response = tool("primary");
        let call = extract(&response, Key::Primary).unwrap();
        let mut history = vec![];
        append_tool(&mut history, &response, &call).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(
            history[0]["tool_calls"][0]["id"],
            history[1]["tool_call_id"]
        );
        assert!(history[0].get("reasoning_content").is_none());
        assert!(history[0]["tool_calls"][0].get("private_field").is_none());
        let result: Value = serde_json::from_str(history[1]["content"].as_str().unwrap()).unwrap();
        assert_eq!(result["value"], Key::Primary.fact());
    }
    #[test]
    fn kv_recall_requires_visible_pin_and_both_tool_facts() {
        let valid = format!("{PIN} {} {}", Key::Primary.fact(), Key::Secondary.fact());
        let response =
            json!({"choices":[{"message":{"content":valid,"reasoning_content":"private"}}]});
        let expected = [PIN, Key::Primary.fact(), Key::Secondary.fact()];
        let message = text_message(&response, &expected).unwrap();
        assert!(message.get("reasoning_content").is_none());
        for text in [
            format!("<think>{valid}</think>wrong"),
            format!("{PIN} {}", Key::Primary.fact()),
        ] {
            assert!(
                text_message(
                    &json!({"choices":[{"message":{"content":text}}]}),
                    &expected
                )
                .is_err()
            );
        }
        assert!(
            text_message(
                &json!({"choices":[{"message":{"content":valid,"tool_calls":[{}]}}]}),
                &expected
            )
            .is_err()
        );
    }
}

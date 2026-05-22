use serde_json::{Map, Value};
use std::collections::HashMap;

fn synthetic_id_seed() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn synthetic_id_component(value: &str) -> String {
    let mut component = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            component.push(ch);
        } else if !component.ends_with('_') {
            component.push('_');
        }
    }
    component.trim_matches('_').to_string()
}

fn chat_completion_json_seed(object: &Map<String, Value>) -> String {
    object
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(synthetic_id_component)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| synthetic_id_seed().to_string())
}

pub(super) fn normalize_chat_completion_json_body(body: &[u8]) -> Option<Vec<u8>> {
    let mut value = serde_json::from_slice::<Value>(body).ok()?;
    let object = value.as_object_mut()?;
    let seed = chat_completion_json_seed(object);
    let choices = object.get_mut("choices")?.as_array_mut()?;
    for (choice_position, choice) in choices.iter_mut().enumerate() {
        let Some(tool_calls) = choice
            .get_mut("message")
            .and_then(Value::as_object_mut)
            .and_then(|message| message.get_mut("tool_calls"))
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        normalize_chat_completion_json_tool_call_ids(&seed, choice_position, tool_calls);
    }
    serde_json::to_vec(&value).ok()
}

fn normalize_chat_completion_json_tool_call_ids(
    seed: &str,
    choice_position: usize,
    tool_calls: &mut [Value],
) {
    for (tool_position, tool_call) in tool_calls.iter_mut().enumerate() {
        let Some(tool_call_object) = tool_call.as_object_mut() else {
            continue;
        };
        let has_id = tool_call_object
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .is_some_and(|value| !value.is_empty());
        if has_id {
            continue;
        }
        let id = format!("call_mesh_{seed}_{choice_position}_{tool_position}");
        tool_call_object.insert("id".into(), Value::String(id));
    }
}

#[derive(Debug, Default)]
pub(super) struct ChatStreamNormalizationState {
    completion_id: Option<String>,
    tool_call_ids: HashMap<u64, String>,
    synthetic_seed: Option<u128>,
}

impl ChatStreamNormalizationState {
    fn seed(&mut self) -> u128 {
        *self.synthetic_seed.get_or_insert_with(synthetic_id_seed)
    }

    fn completion_id(&mut self, object: &Map<String, Value>) -> String {
        if let Some(existing) = self.completion_id.clone() {
            return existing;
        }
        let id = object
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("chatcmpl-mesh-{}", self.seed()));
        self.completion_id = Some(id.clone());
        id
    }

    fn tool_call_id(&mut self, index: u64, tool_call: &Map<String, Value>) -> String {
        if let Some(existing) = self.tool_call_ids.get(&index) {
            return existing.clone();
        }
        let id = tool_call
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("call_mesh_{}_{}", self.seed(), index));
        self.tool_call_ids.insert(index, id.clone());
        id
    }

    pub(super) fn normalize_data(&mut self, data: &str) -> String {
        let Ok(mut value) = serde_json::from_str::<Value>(data) else {
            return data.to_string();
        };
        let Some(object) = value.as_object_mut() else {
            return data.to_string();
        };

        let completion_id = self.completion_id(object);
        object.insert("id".into(), Value::String(completion_id));

        let Some(choices) = object.get_mut("choices").and_then(Value::as_array_mut) else {
            return serde_json::to_string(&value).unwrap_or_else(|_| data.to_string());
        };
        for choice in choices {
            let Some(tool_calls) = choice
                .get_mut("delta")
                .and_then(Value::as_object_mut)
                .and_then(|delta| delta.get_mut("tool_calls"))
                .and_then(Value::as_array_mut)
            else {
                continue;
            };
            for (position, tool_call) in tool_calls.iter_mut().enumerate() {
                let Some(tool_call_object) = tool_call.as_object_mut() else {
                    continue;
                };
                let index = tool_call_object
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(position as u64);
                let id = self.tool_call_id(index, tool_call_object);
                tool_call_object.insert("id".into(), Value::String(id));
            }
        }

        serde_json::to_string(&value).unwrap_or_else(|_| data.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_completion_json_normalizer_adds_missing_tool_call_id() {
        let body = br#"{"id":"chatcmpl-a","object":"chat.completion","created":1,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"read_file","arguments":"{\"path\":\"AGENTS.md\"}"}}]},"finish_reason":"tool_calls"}]}"#;
        let normalized = normalize_chat_completion_json_body(body).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&normalized).unwrap();

        assert_eq!(parsed["id"], "chatcmpl-a");
        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_mesh_chatcmpl_a_0_0"
        );
    }

    #[test]
    fn chat_completion_json_normalizer_preserves_existing_tool_call_ids() {
        let body = br#"{"id":"chatcmpl-a","object":"chat.completion","choices":[{"message":{"tool_calls":[{"id":"call_existing","type":"function","function":{"name":"read_file","arguments":"{}"}}]}}]}"#;
        let normalized = normalize_chat_completion_json_body(body).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&normalized).unwrap();

        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_existing"
        );
    }

    #[test]
    fn chat_completion_json_normalizer_keeps_ids_unique_across_choices() {
        let body = br#"{"id":"chatcmpl-a","object":"chat.completion","choices":[{"message":{"tool_calls":[{"type":"function","function":{"name":"first","arguments":"{}"}},{"type":"function","function":{"name":"second","arguments":"{}"}}]}},{"message":{"tool_calls":[{"type":"function","function":{"name":"third","arguments":"{}"}}]}}]}"#;
        let normalized = normalize_chat_completion_json_body(body).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&normalized).unwrap();

        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_mesh_chatcmpl_a_0_0"
        );
        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][1]["id"],
            "call_mesh_chatcmpl_a_0_1"
        );
        assert_eq!(
            parsed["choices"][1]["message"]["tool_calls"][0]["id"],
            "call_mesh_chatcmpl_a_1_0"
        );
    }

    #[test]
    fn chat_stream_normalizer_adds_missing_tool_call_id() {
        let mut state = ChatStreamNormalizationState {
            synthetic_seed: Some(42),
            ..Default::default()
        };
        let normalized = state.normalize_data(
            r#"{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"read_file","arguments":"{\"path\":\"AGENTS.md\"}"}}]},"finish_reason":null}]}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&normalized).unwrap();

        assert_eq!(parsed["id"], "chatcmpl-a");
        assert_eq!(
            parsed["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_mesh_42_0"
        );
    }

    #[test]
    fn chat_stream_normalizer_keeps_completion_and_tool_ids_stable() {
        let mut state = ChatStreamNormalizationState {
            synthetic_seed: Some(7),
            ..Default::default()
        };

        let first = state.normalize_data(
            r#"{"id":"chatcmpl-first","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        );
        let second = state.normalize_data(
            r#"{"id":"chatcmpl-second","object":"chat.completion.chunk","created":2,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":"{}"}}]},"finish_reason":null}]}"#,
        );
        let third = state.normalize_data(
            r#"{"id":"chatcmpl-third","object":"chat.completion.chunk","created":3,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":" more"}}]},"finish_reason":null}]}"#,
        );
        let first: serde_json::Value = serde_json::from_str(&first).unwrap();
        let second: serde_json::Value = serde_json::from_str(&second).unwrap();
        let third: serde_json::Value = serde_json::from_str(&third).unwrap();

        assert_eq!(first["id"], "chatcmpl-first");
        assert_eq!(second["id"], "chatcmpl-first");
        assert_eq!(third["id"], "chatcmpl-first");
        assert_eq!(
            second["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_mesh_7_0"
        );
        assert_eq!(
            third["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_mesh_7_0"
        );
    }
}

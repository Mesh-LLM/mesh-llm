//! Ordered conversation content shared by generation and protocol adapters.
//! Unknown roles and opaque parts remain representable: adaptation must not
//! introduce validation beyond the existing frontend contract.

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct GenerationMessage {
    pub role: String,
    #[serde(
        default,
        deserialize_with = "deserialize_present_message_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub content: Option<MessageContent>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

fn deserialize_present_message_content<'de, D>(
    deserializer: D,
) -> Result<Option<MessageContent>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    serde_json::from_value(value)
        .map(Some)
        .map_err(D::Error::custom)
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
    Other(Value),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MessageContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl MessageContentPart {
    /// Return the URL supplied by any accepted OpenAI multimodal container
    /// shape. The stage runtime may still decide whether that URL is local,
    /// data-backed, or requires an upstream fetch.
    pub fn media_url(&self) -> Option<String> {
        for key in [
            "image_url",
            "input_image",
            "image",
            "input_audio",
            "audio",
            "audio_url",
            "url",
        ] {
            if let Some(value) = self.extra.get(key) {
                if let Some(url) = value.as_str() {
                    return Some(url.to_string());
                }
                if let Some(url) = value.get("url").and_then(Value::as_str) {
                    return Some(url.to_string());
                }
            }
        }
        None
    }

    /// Return an inline base64 payload supplied by any accepted OpenAI
    /// multimodal container shape.
    pub fn media_data(&self) -> Option<String> {
        for key in ["input_audio", "audio", "image", "input_image", "image_url"] {
            if let Some(data) = self
                .extra
                .get(key)
                .and_then(|value| value.get("data"))
                .and_then(Value::as_str)
            {
                return Some(data.to_string());
            }
        }
        None
    }
}

/// Ensure every OpenAI function tool call has one non-empty, unique ID while
/// preserving IDs supplied by the model or an upstream runtime.
pub fn ensure_tool_call_ids(tool_calls: &mut [Value]) {
    let mut emitted_ids = HashSet::new();
    for tool_call in tool_calls {
        let Some(object) = tool_call.as_object_mut() else {
            continue;
        };
        let valid_unseen_id = object
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .filter(|id| emitted_ids.insert((*id).to_string()));
        if valid_unseen_id.is_none() {
            let id = loop {
                let candidate = format!("call_{}", uuid::Uuid::new_v4().simple());
                if emitted_ids.insert(candidate.clone()) {
                    break candidate;
                }
            };
            object.insert("id".to_string(), Value::String(id));
        }
    }
}

pub fn messages_to_plain_prompt(messages: &[GenerationMessage]) -> String {
    messages
        .iter()
        .filter_map(|message| message.content.as_ref())
        .filter_map(message_content_to_text)
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn message_content_to_text(content: &MessageContent) -> Option<String> {
    match content {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter(|part| part.content_type == "text")
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            Some(text)
        }
        MessageContent::Other(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn media_shape_golden_table_accepts_string_and_nested_url_containers() {
        let cases = [
            (
                json!({"type": "image_url", "image_url": "data:image/png;base64,abc"}),
                Some("data:image/png;base64,abc"),
                None,
            ),
            (
                json!({"type": "input_image", "input_image": {"url": "mesh://blob/a/b"}}),
                Some("mesh://blob/a/b"),
                None,
            ),
            (
                json!({"type": "input_audio", "input_audio": {"data": "YWJj"}}),
                None,
                Some("YWJj"),
            ),
        ];

        for (value, expected_url, expected_data) in cases {
            let part: MessageContentPart = serde_json::from_value(value).unwrap();
            assert_eq!(part.media_url().as_deref(), expected_url);
            assert_eq!(part.media_data().as_deref(), expected_data);
        }
    }

    #[test]
    fn ensure_tool_call_ids_golden_table_preserves_and_deduplicates_ids() {
        let mut calls = vec![
            json!({"id": "call_existing", "type": "function"}),
            json!({"id": "call_existing", "type": "function"}),
            json!({"type": "function"}),
        ];

        ensure_tool_call_ids(&mut calls);

        assert_eq!(calls[0]["id"], "call_existing");
        assert_ne!(calls[1]["id"], "call_existing");
        assert_ne!(calls[2]["id"], "call_existing");
        assert_ne!(calls[1]["id"], calls[2]["id"]);
    }
}

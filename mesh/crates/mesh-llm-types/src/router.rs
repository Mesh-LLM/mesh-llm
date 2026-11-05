//! Pure request classification and media requirements shared by host and client routers.

use serde_json::Value;

/// Structural requirements used by both routing surfaces.
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    pub needs_tools: bool,
    pub has_media_inputs: bool,
}

/// Media modalities requested by a chat completion body.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaRequirements {
    pub has_media: bool,
    pub needs_vision: bool,
    pub needs_audio: bool,
}

impl MediaRequirements {
    /// Whether routing needs a runtime with vision or audio support.
    #[must_use]
    pub const fn requires_runtime_modality(self) -> bool {
        self.needs_vision || self.needs_audio
    }
}

/// Strip a split GGUF suffix such as `-00001-of-00004` from a model name.
#[must_use]
pub fn strip_split_suffix(name: &str) -> &str {
    if let Some(idx) = name.rfind("-of-") {
        let after = &name[idx + 4..];
        if !after.is_empty()
            && after.chars().all(|c| c.is_ascii_digit())
            && let Some(dash) = name[..idx].rfind('-')
        {
            let between = &name[dash + 1..idx];
            if !between.is_empty() && between.chars().all(|c| c.is_ascii_digit()) {
                return &name[..dash];
            }
        }
    }
    name
}

/// Owned counterpart to [`strip_split_suffix`].
#[must_use]
pub fn strip_split_suffix_owned(name: &str) -> String {
    strip_split_suffix(name).to_owned()
}

/// Inspect declared tools and media without inferring intent from text.
#[must_use]
pub fn classify(body: &Value) -> Classification {
    Classification {
        needs_tools: detect_tool_requirement(body),
        has_media_inputs: media_requirements(body).has_media,
    }
}

fn detect_tool_requirement(body: &Value) -> bool {
    has_tools_schema(body) || has_tool_blocks(body)
}

fn has_tools_schema(body: &Value) -> bool {
    body.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
}

fn has_tool_blocks(body: &Value) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|blocks| {
                        blocks.iter().any(|block| {
                            matches!(
                                block.get("type").and_then(Value::as_str),
                                Some("tool_use") | Some("tool_result")
                            )
                        })
                    })
            })
        })
}

/// Extract requested image/audio/file modalities from OpenAI/Anthropic content blocks.
#[must_use]
pub fn media_requirements(body: &Value) -> MediaRequirements {
    let mut requirements = MediaRequirements::default();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return requirements;
    };

    for message in messages {
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in blocks {
            match block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
            {
                "image_url" | "input_image" | "image" => {
                    requirements.has_media = true;
                    requirements.needs_vision = true;
                }
                "audio_url" | "input_audio" | "audio" => {
                    requirements.has_media = true;
                    requirements.needs_audio = true;
                }
                "file" | "input_file" => {
                    requirements.has_media = true;
                }
                _ => {
                    if block.get("image_url").is_some() || block.get("image").is_some() {
                        requirements.has_media = true;
                        requirements.needs_vision = true;
                    }
                    if block.get("audio_url").is_some() || block.get("audio").is_some() {
                        requirements.has_media = true;
                        requirements.needs_audio = true;
                    }
                }
            }
        }
    }

    requirements
}

#[cfg(test)]
mod tests {
    use super::{classify, media_requirements, strip_split_suffix};
    use serde_json::json;

    #[test]
    fn detects_declared_tools() {
        let body = json!({
            "tools": [{"type": "function", "function": {"name": "run"}}],
            "messages": [{"role": "user", "content": "write code to sort this"}]
        });
        let result = classify(&body);
        assert!(result.needs_tools);
    }

    #[test]
    fn plain_text_has_no_structural_requirements() {
        let body = json!({"messages": [{"role": "user", "content": "Hello"}]});
        let result = classify(&body);
        assert!(!result.needs_tools);
        assert!(!result.has_media_inputs);
    }

    #[test]
    fn prompt_keywords_and_history_do_not_change_structural_requirements() {
        let plain = json!({"messages": [{"role": "user", "content": "hi"}]});
        let keywords = json!({"messages": [
            {"role": "system", "content": "You are a developer and coding assistant"},
            {"role": "user", "content": "Prove step by step, design a system architecture, write code, explain in depth"}
        ]});
        assert_eq!(classify(&plain), classify(&keywords));
    }

    #[test]
    fn detects_image_and_audio_blocks() {
        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,..."},
                    {"type": "input_audio", "audio": {"data": "..."}}
                ]
            }]
        });
        let result = media_requirements(&body);
        assert!(result.has_media);
        assert!(result.needs_vision);
        assert!(result.needs_audio);
        assert!(result.requires_runtime_modality());
    }

    #[test]
    fn strips_only_numeric_split_suffixes() {
        assert_eq!(strip_split_suffix("Model-Q4-00001-of-00004"), "Model-Q4");
        assert_eq!(strip_split_suffix("Model-Q4-of-four"), "Model-Q4-of-four");
        assert_eq!(strip_split_suffix("Model"), "Model");
    }
}

//! Pure request classification and media requirements shared by host and client routers.

use serde_json::Value;

/// Request category inferred from message text and structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Code,
    Reasoning,
    Chat,
    ToolCall,
    Creative,
    /// Factual lookup, summarization, or knowledge retrieval.
    Info,
    /// Image generation or analysis.
    Image,
}

/// Approximate request complexity inferred from message text and history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    Quick,
    Moderate,
    Deep,
}

/// Pure classification result used by both routing surfaces.
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    pub category: Category,
    pub complexity: Complexity,
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

/// Classify a chat completion request using deterministic text/structure heuristics.
///
/// Tool presence is an attribute, not a category override: a code request with
/// tools remains [`Category::Code`] while setting `needs_tools`.
#[must_use]
pub fn classify(body: &Value) -> Classification {
    let text = collect_message_text(body);
    let lower = text.to_lowercase();
    let media = media_requirements(body);
    let needs_tools = detect_tool_requirement(body);
    let last_user_len = last_user_message_len(body);
    let scores = router_signal_scores(&lower);
    let category = classify_category(scores, detect_system_code_hint(body), media, needs_tools);
    let complexity = classify_complexity(scores, last_user_len, message_count(body));

    Classification {
        category,
        complexity,
        needs_tools,
        has_media_inputs: media.has_media,
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

fn count_signals(lower: &str, signals: &[&str]) -> usize {
    signals
        .iter()
        .filter(|signal| lower.contains(*signal))
        .count()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RouterSignalScores {
    code: usize,
    reasoning: usize,
    creative: usize,
    info: usize,
    image: usize,
    deep: usize,
}

fn router_signal_scores(lower: &str) -> RouterSignalScores {
    RouterSignalScores {
        code: count_signals(
            lower,
            &[
                "```",
                "def ",
                "fn ",
                "func ",
                "class ",
                "import ",
                "function",
                "const ",
                "let ",
                "var ",
                "return ",
                "write a program",
                "write code",
                "implement",
                "refactor",
                "debug",
                "fix the bug",
                "write a script",
                "code review",
                "pull request",
                "git ",
                "compile",
                "syntax",
                "python",
                "javascript",
                "typescript",
                " rust ",
                "golang",
                "java ",
                "c++",
                " ruby ",
                " swift ",
                "kotlin",
                "algorithm",
                "binary search",
                " sort ",
                "regex",
                " api ",
                " http ",
                " sql ",
                "database",
                " query ",
            ],
        ),
        reasoning: count_signals(
            lower,
            &[
                "prove",
                "explain why",
                "step by step",
                "calculate",
                "solve",
                "derive",
                "what is the probability",
                "how many",
                "analyze",
                "compare and contrast",
                "evaluate",
                "mathematical",
                "theorem",
                "equation",
                "logic",
                "think carefully",
                "reason about",
            ],
        ),
        creative: count_signals(
            lower,
            &[
                "write a story",
                "write a poem",
                "creative",
                "imagine",
                "fiction",
                "narrative",
                "compose",
                "brainstorm",
                "write a song",
                "screenplay",
                "dialogue",
            ],
        ),
        info: count_signals(
            lower,
            &[
                "what is",
                "who is",
                "when did",
                "where is",
                "how does",
                "define ",
                "explain ",
                "summarize",
                "summary",
                "overview",
                "tell me about",
                "describe ",
                "what are the",
                "list the",
                "difference between",
                "compare ",
                "history of",
            ],
        ),
        image: count_signals(
            lower,
            &[
                "image",
                "picture",
                "photo",
                "draw",
                "generate an image",
                "visualize",
                "diagram",
                "screenshot",
                "describe this image",
            ],
        ),
        deep: count_signals(
            lower,
            &[
                "architect",
                "design a system",
                "trade-off",
                "tradeoff",
                "in depth",
                "comprehensive",
                "thorough",
                "detailed analysis",
                "long-term",
                "strategy",
                "plan for",
                "review this codebase",
                "rewrite",
                "from scratch",
            ],
        ),
    }
}

fn detect_system_code_hint(body: &Value) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message.get("role").and_then(Value::as_str) == Some("system")
                    && message
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|content| {
                            let system = content.to_lowercase();
                            system.contains("developer")
                                || system.contains("coding")
                                || system.contains("programmer")
                        })
            })
        })
}

fn classify_category(
    scores: RouterSignalScores,
    system_code: bool,
    media: MediaRequirements,
    needs_tools: bool,
) -> Category {
    if system_code
        || scores.code >= 2
        || (scores.code >= 1 && scores.reasoning == 0 && scores.creative == 0)
    {
        Category::Code
    } else if scores.reasoning >= 2 {
        Category::Reasoning
    } else if scores.creative >= 1 {
        Category::Creative
    } else if media.needs_vision || scores.image >= 1 {
        Category::Image
    } else if needs_tools && scores.code == 0 && scores.reasoning == 0 && scores.creative == 0 {
        Category::ToolCall
    } else if scores.info >= 2 && scores.code == 0 {
        Category::Info
    } else {
        Category::Chat
    }
}

fn message_count(body: &Value) -> usize {
    body.get("messages")
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

fn classify_complexity(
    scores: RouterSignalScores,
    last_user_len: usize,
    total_messages: usize,
) -> Complexity {
    if scores.deep >= 1 || last_user_len > 500 || total_messages > 10 {
        Complexity::Deep
    } else if last_user_len < 60 && total_messages <= 2 && scores.reasoning == 0 && scores.deep == 0
    {
        Complexity::Quick
    } else {
        Complexity::Moderate
    }
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

fn last_user_message_len(body: &Value) -> usize {
    body.get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .rev()
                .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        })
        .map(message_text)
        .map_or(0, |text| text.len())
}

fn collect_message_text(body: &Value) -> String {
    let mut text = String::new();
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages {
            let content = message_text(message);
            if !content.is_empty() {
                text.push_str(&content);
                text.push('\n');
            }
        }
    }
    text
}

fn message_text(message: &Value) -> String {
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        return text.to_owned();
    }

    if let Some(blocks) = message.get("content").and_then(Value::as_array) {
        let mut text = String::new();
        for block in blocks {
            if let Some(value) = block.get("text").and_then(Value::as_str) {
                text.push_str(value);
                text.push('\n');
            }
        }
        return text;
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::{Category, Complexity, classify, media_requirements, strip_split_suffix};
    use serde_json::json;

    #[test]
    fn classifies_code_and_tool_requests_without_overriding_category() {
        let body = json!({
            "tools": [{"type": "function", "function": {"name": "run"}}],
            "messages": [{"role": "user", "content": "write code to sort this"}]
        });
        let result = classify(&body);
        assert_eq!(result.category, Category::Code);
        assert!(result.needs_tools);
    }

    #[test]
    fn classifies_short_plain_requests_as_quick_chat() {
        let body = json!({"messages": [{"role": "user", "content": "Hello"}]});
        let result = classify(&body);
        assert_eq!(result.category, Category::Chat);
        assert_eq!(result.complexity, Complexity::Quick);
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

//! Fallback for models without working native tool calls: when the assistant
//! replies with plain text containing a fenced `sh`/`bash`/`shell` block (or a
//! raw `<tool_call>{...}</tool_call>` the server failed to parse), run the
//! first such command and feed its output back as the next user message.

use std::path::Path;

use goose_provider_types::conversation::message::{Message, MessageContent};
use rmcp::model::Role;

const SHELL_TAGS: &[&str] = &["sh", "bash", "shell", "console", "zsh"];

/// The command to run for this assistant message, if the fallback applies.
pub fn extract(message: &Message) -> Option<String> {
    if message.role != Role::Assistant || message.error_kind().is_some() {
        return None;
    }
    if message
        .content
        .iter()
        .any(|c| matches!(c, MessageContent::ToolRequest(_)))
    {
        return None;
    }
    let text = message.as_concat_text();
    raw_tool_call(&text).or_else(|| fenced_block(&text))
}

fn fenced_block(text: &str) -> Option<String> {
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let Some(tag) = line.trim_start().strip_prefix("```") else {
            continue;
        };
        let wanted = SHELL_TAGS.contains(&tag.trim().to_ascii_lowercase().as_str());
        let mut body = Vec::new();
        for inner in lines.by_ref() {
            if inner.trim_start().starts_with("```") {
                break;
            }
            body.push(inner.strip_prefix("$ ").unwrap_or(inner));
        }
        let command = body.join("\n").trim().to_string();
        if wanted && !command.is_empty() {
            return Some(command);
        }
    }
    None
}

/// `<tool_call>{"name":"shell","arguments":{"command":"..."}}</tool_call>`
/// (Qwen/Hermes style) or the Gemma `<|tool_call>` variant left in content.
fn raw_tool_call(text: &str) -> Option<String> {
    let start = text.find("tool_call>")? + "tool_call>".len();
    let rest = &text[start..];
    let json_start = rest.find('{')?;
    let mut de =
        serde_json::Deserializer::from_str(&rest[json_start..]).into_iter::<serde_json::Value>();
    let value = de.next()?.ok()?;
    let args = value.get("arguments").unwrap_or(&value);
    let args = match args {
        serde_json::Value::String(s) => serde_json::from_str(s).ok()?,
        other => other.clone(),
    };
    args.get("command")?.as_str().map(str::to_string)
}

/// Run the command and build the user message carrying its output.
pub async fn run(cwd: &Path, command: &str) -> Message {
    let result = crate::shell::run(cwd, command, None).await;
    let output = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect::<Vec<_>>()
        .join("\n");
    let status = if result.is_error == Some(true) {
        "failed"
    } else {
        "ran"
    };
    Message::user().with_text(format!(
        "Your command {status}. Output:\n```\n{output}\n```\n\
         Continue. When the task is done, reply in plain text without a code block."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(text: &str) -> Message {
        Message::assistant().with_text(text)
    }

    #[test]
    fn picks_first_shell_fence() {
        let m = assistant("Let me look.\n```bash\n$ ls -la\n```\nthen\n```sh\npwd\n```");
        assert_eq!(extract(&m).as_deref(), Some("ls -la"));
    }

    #[test]
    fn ignores_non_shell_fences_and_plain_text() {
        assert_eq!(extract(&assistant("```python\nprint(1)\n```")), None);
        assert_eq!(extract(&assistant("Done: 42 lines.")), None);
    }

    #[test]
    fn parses_raw_tool_call() {
        let m = assistant(
            "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"wc -l a.txt\"}}</tool_call>",
        );
        assert_eq!(extract(&m).as_deref(), Some("wc -l a.txt"));
        let m = assistant(
            "<|tool_call>{\"name\":\"shell\",\"arguments\":\"{\\\"command\\\":\\\"echo hi\\\"}\"}",
        );
        assert_eq!(extract(&m).as_deref(), Some("echo hi"));
    }
}

//! The `shell` tool: one command through `sh -c`, bounded in time and output.

use std::path::Path;
use std::time::Duration;

use rmcp::model::{CallToolResult, ContentBlock as Content, Tool};
use tokio::process::Command;

/// Output beyond this many bytes is cut to a head and tail; the full output is
/// kept in a temp file the model can read back with `sed`/`grep`.
const MAX_OUTPUT_BYTES: usize = 12_000;
const DEFAULT_TIMEOUT_SECS: u64 = 300;

pub const NAME: &str = "shell";

pub fn tool() -> Tool {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": { "type": "string", "description": "Shell command to run with sh -c." },
            "timeout_secs": { "type": "integer", "description": "Optional timeout in seconds (default 300)." }
        }
    });
    Tool::new(
        NAME,
        "Run a shell command in the working directory and return its exit code and output.",
        schema.as_object().cloned().unwrap_or_default(),
    )
}

pub async fn run(cwd: &Path, command: &str, timeout_secs: Option<u64>) -> CallToolResult {
    let timeout = Duration::from_secs(timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS));
    let child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    let output = match tokio::time::timeout(timeout, child).await {
        Err(_) => {
            return CallToolResult::error(vec![Content::text(format!(
                "command timed out after {}s",
                timeout.as_secs()
            ))]);
        }
        Ok(Err(error)) => {
            return CallToolResult::error(vec![Content::text(format!(
                "failed to start sh: {error}"
            ))]);
        }
        Ok(Ok(output)) => output,
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&stderr);
    }
    let code = output
        .status
        .code()
        .map_or_else(|| "killed by signal".to_string(), |c| c.to_string());
    let body = format!("exit code: {code}\n{}", truncate(&text));
    if output.status.success() {
        CallToolResult::success(vec![Content::text(body)])
    } else {
        CallToolResult::error(vec![Content::text(body)])
    }
}

fn truncate(text: &str) -> String {
    if text.len() <= MAX_OUTPUT_BYTES {
        return text.to_string();
    }
    let saved = save_full_output(text);
    let half = MAX_OUTPUT_BYTES / 2;
    let head = &text[..floor_char_boundary(text, half)];
    let tail = &text[floor_char_boundary(text, text.len() - half)..];
    format!(
        "{head}\n... [{} bytes omitted; full output in {saved}] ...\n{tail}",
        text.len() - head.len() - tail.len()
    )
}

fn save_full_output(text: &str) -> String {
    let path = std::env::temp_dir().join(format!(
        "mesh-agent-shell-{}-{}.log",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    match std::fs::write(&path, text) {
        Ok(()) => path.display().to_string(),
        Err(error) => format!("(not saved: {error})"),
    }
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(result: &CallToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|c| c.as_text().map(|t| t.text.clone()))
            .collect()
    }

    #[tokio::test]
    async fn reports_exit_code_and_both_streams() {
        let dir = tempfile::tempdir().unwrap();
        let result = run(dir.path(), "echo out; echo err >&2; exit 3", None).await;
        assert_eq!(result.is_error, Some(true));
        let text = text_of(&result);
        assert!(text.starts_with("exit code: 3"), "{text}");
        assert!(text.contains("out") && text.contains("err"), "{text}");
    }

    #[tokio::test]
    async fn times_out() {
        let dir = tempfile::tempdir().unwrap();
        let result = run(dir.path(), "sleep 5", Some(1)).await;
        assert!(text_of(&result).contains("timed out"));
    }

    #[test]
    fn truncates_long_output_and_keeps_the_full_copy() {
        let long = "x".repeat(MAX_OUTPUT_BYTES * 2);
        let cut = truncate(&long);
        assert!(cut.len() < long.len());
        let path = cut
            .split("full output in ")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), long);
        let _ = std::fs::remove_file(path);
    }
}

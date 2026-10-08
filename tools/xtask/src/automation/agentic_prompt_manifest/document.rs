use serde::Serialize;
use serde_json::Value;

use super::selection::{Selection, Trajectory};
use crate::command::DynResult;

#[derive(Debug, Serialize)]
pub(super) struct Prompt {
    pub family: String,
    pub prompt: String,
}

#[derive(Debug, Serialize)]
pub(super) struct Manifest<'a> {
    pub metadata: Metadata<'a>,
    pub prompts: Vec<Prompt>,
}

#[derive(Debug, Serialize)]
pub(super) struct Metadata<'a> {
    pub dataset: &'a str,
    pub dataset_revision: &'a str,
    pub selection: SelectionDocument<'a>,
    pub requests_per_family: usize,
    pub rows: &'a [Trajectory],
}

#[derive(Debug, Serialize)]
pub(super) struct SelectionDocument<'a> {
    #[serde(flatten)]
    pub selection: &'a Selection,
    pub order: &'static str,
}

pub(super) fn flatten_messages(input: &str) -> DynResult<String> {
    let messages: Value = serde_json::from_str(input)?;
    let messages = messages
        .as_array()
        .filter(|rows| !rows.is_empty())
        .ok_or("trajectory messages_json must contain a nonempty array")?;
    let mut sections = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let message = message
            .as_object()
            .ok_or_else(|| format!("trajectory message {index} must be an object"))?;
        let role = match message.get("role") {
            None => "unknown",
            Some(Value::String(role)) => role,
            _ => return Err(format!("trajectory message {index} role must be a string").into()),
        };
        let mut content = match message.get("content") {
            None => String::new(),
            Some(Value::String(text)) => text.clone(),
            Some(value) => super::structured_content::render(value)?,
        };
        match message.get("tool_calls_json") {
            None | Some(Value::Null) => {}
            Some(Value::String(calls)) if calls.is_empty() => {}
            Some(Value::String(calls)) => {
                content.push_str(&format!("\n<tool_calls>{calls}</tool_calls>"))
            }
            _ => {
                return Err(format!(
                    "trajectory message {index} tool_calls_json must be a string or null"
                )
                .into());
            }
        }
        sections.push(format!("<{role}>\n{content}"));
    }
    Ok(sections.join("\n\n"))
}

pub(super) fn build_manifest<'a>(
    trajectories: &'a [Trajectory],
    requests_per_family: usize,
    dataset_revision: &'a str,
    selection: &'a Selection,
) -> DynResult<Manifest<'a>> {
    if trajectories.is_empty() || requests_per_family == 0 {
        return Err("trajectories and requests per family must be nonempty".into());
    }
    let capacity = trajectories
        .len()
        .checked_mul(requests_per_family)
        .ok_or("prompt count exceeds supported size")?;
    let prefixes = trajectories
        .iter()
        .map(|row| flatten_messages(&row.messages_json))
        .collect::<DynResult<Vec<_>>>()?;
    let mut prompts = Vec::new();
    prompts.try_reserve_exact(capacity)?;
    for request in 0..requests_per_family {
        for (family, prefix) in prefixes.iter().enumerate() {
            prompts.push(Prompt {
                family: format!("trajectory-{family}"),
                prompt: format!("{prefix}\n\n<user>\nBenchmark branch {request}: summarize the latest repository state in one sentence."),
            });
        }
    }
    let metadata = Metadata {
        dataset: "thoughtworks/agentic-coding-trajectories",
        dataset_revision,
        selection: SelectionDocument {
            selection,
            order: "md5(session_id)",
        },
        requests_per_family,
        rows: trajectories,
    };
    Ok(Manifest { metadata, prompts })
}

//! Shared agent fixture evidence. Hidden implementation execution stays separate.
mod opencode;
mod result;
use crate::{command::DynResult, repository::check_report::CheckReport};
use serde::Deserialize;
use std::{fs::File, io::Read, path::Path};

const INPUT_LIMIT: usize = 8 * 1024 * 1024;

fn read(path: &Path) -> DynResult<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take((INPUT_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > INPUT_LIMIT {
        return Err("agent fixture evidence exceeds 8 MiB input limit".into());
    }
    Ok(bytes)
}

#[derive(Deserialize)]
struct Response {
    choices: Vec<Choice>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}
#[derive(Deserialize)]
struct Choice {
    message: Message,
}
#[derive(Deserialize)]
struct Message {
    content: String,
}

fn soak(bytes: &[u8], label: &str) -> DynResult<String> {
    let response: Response = serde_json::from_slice(bytes)?;
    if response.error.is_some()
        || !response.choices.first().is_some_and(|choice| {
            choice
                .message
                .content
                .contains("LONG_SOAK=ALPHA-719|MID-482|OMEGA-503")
        })
    {
        return Err(format!("{label} long prompt sentinel validation failed").into());
    }
    Ok(format!("{label} long prompt soak passed\n"))
}

#[derive(Deserialize)]
struct ProbeResponse {
    object: String,
    choices: Vec<serde_json::Value>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

fn probe(bytes: &[u8], label: &str) -> DynResult<String> {
    let response: ProbeResponse = serde_json::from_slice(bytes)?;
    if response.object != "chat.completion"
        || response.choices.is_empty()
        || response.error.is_some()
    {
        return Err(format!("{label} compatibility probe response is invalid").into());
    }
    Ok(String::new())
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let output = match args {
        [verb, path] if verb == "opencode-session" => opencode::session(&read(Path::new(path))?)?,
        [verb, path] if verb == "opencode-result" => opencode::result(&read(Path::new(path))?)?,
        [verb, path, label] if verb == "probe" => probe(&read(Path::new(path))?, label)?,
        [verb, path, label] if verb == "soak" => soak(&read(Path::new(path))?, label)?,
        [verb, path, label, required] if verb == "result" => {
            let required = match required.to_ascii_lowercase().as_str() {
                "true" => true, "false" => false,
                _ => return Err("tool event requirement must be true or false".into()),
            };
            result::validate(&read(Path::new(path))?, label, required)?
        }
        _ => return Err("usage: automation agent-fixture-evidence {soak RESPONSE LABEL | probe RESPONSE LABEL | result JSONL LABEL REQUIRE_TOOLS | opencode-session JSONL | opencode-result JSONL}".into()),
    };
    CheckReport::success(output).emit()
}

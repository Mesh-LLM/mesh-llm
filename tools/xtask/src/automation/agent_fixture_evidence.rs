//! Shared agent fixture evidence. Hidden implementation execution stays separate.
mod opencode;
mod result;
mod surface;
use crate::{command::DynResult, repository::check_report::CheckReport};
use serde::Deserialize;
use std::{fs::File, io::Read, path::Path};

const INPUT_LIMIT: usize = 8 * 1024 * 1024;
const SURFACE_INPUT_LIMIT: usize = 64 * 1024 * 1024;

fn read(path: &Path) -> DynResult<Vec<u8>> {
    read_limit(path, INPUT_LIMIT)
}

fn read_limit(path: &Path, limit: usize) -> DynResult<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(format!(
            "agent fixture evidence exceeds {} MiB input limit",
            limit / (1024 * 1024)
        )
        .into());
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
        [verb, path, minimum] if verb == "surface" => surface::validate(&read_limit(Path::new(path), SURFACE_INPUT_LIMIT)?, minimum.parse()?)?,
        [verb, path, label, required] if verb == "result" => {
            let required = match required.to_ascii_lowercase().as_str() {
                "true" => true, "false" => false,
                _ => return Err("tool event requirement must be true or false".into()),
            };
            result::validate(&read(Path::new(path))?, label, required)?
        }
        _ => return Err("usage: automation agent-fixture-evidence {soak RESPONSE LABEL | probe RESPONSE LABEL | result JSONL LABEL REQUIRE_TOOLS | opencode-session JSONL | opencode-result JSONL | surface JSONL MIN_BODY_BYTES}".into()),
    };
    CheckReport::success(output).emit()
}

#[cfg(test)]
mod input_bound_tests {
    use super::*;
    #[test]
    fn bounded_reader_accepts_the_boundary_and_rejects_the_first_surplus_byte() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("evidence");
        std::fs::write(&path, b"0123456789abcdef").unwrap();
        assert_eq!(read_limit(&path, 16).unwrap(), b"0123456789abcdef");
        std::fs::write(&path, b"0123456789abcdefg").unwrap();
        assert!(read_limit(&path, 16).is_err());
    }
    #[test]
    fn surface_reader_accepts_large_capture_while_generic_reader_keeps_existing_bound() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("capture");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len((INPUT_LIMIT + 1) as u64).unwrap();
        assert!(read(&path).is_err());
        assert_eq!(
            read_limit(&path, SURFACE_INPUT_LIMIT).unwrap().len(),
            INPUT_LIMIT + 1
        );
    }
}

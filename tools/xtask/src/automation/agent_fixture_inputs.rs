//! Input evidence and long-context requests for the actual agent fixture.
use crate::{command::DynResult, product::digest::file_sha256};
use serde::Serialize;
use std::{fs, path::Path};

const MAX_DOCUMENT_CHARS: usize = 8 * 1024 * 1024;
const HEADER: &str = "This is a long-context CI soak document. Extract the three sentinel values. Return exactly LONG_SOAK=ALPHA-719|MID-482|OMEGA-503 and no extra text.\n\n";
const FILLER: &str = "FILLER: mesh long prompt soak line with predictable neutral text. Do not use this filler as the answer.\n";

#[derive(Serialize)]
struct Message<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: [Message<'a>; 2],
    stream: bool,
    max_tokens: u32,
    temperature: u32,
}

fn document(target: usize) -> String {
    let prefix = "SENTINEL_START=ALPHA-719\n";
    let middle = "\nSENTINEL_MIDDLE=MID-482\n";
    let suffix = "\nSENTINEL_END=OMEGA-503\n";
    let remaining =
        target.saturating_sub(HEADER.len() + prefix.len() + middle.len() + suffix.len());
    let left = ((remaining / 2) / FILLER.len()).max(1);
    let right = (remaining.saturating_sub(left * FILLER.len()) / FILLER.len()).max(1);
    format!(
        "{HEADER}{prefix}{}{middle}{}{suffix}",
        FILLER.repeat(left),
        FILLER.repeat(right)
    )
}

fn write_soak(model: &str, target: &str, path: &Path) -> DynResult<()> {
    if model.is_empty() || model.len() > 65536 {
        return Err("agent soak model must be nonempty and at most 64 KiB".into());
    }
    if target.is_empty() || !target.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("agent soak target must be a positive decimal character count".into());
    }
    let target: usize = target.parse()?;
    if !(1..=MAX_DOCUMENT_CHARS).contains(&target) {
        return Err("agent soak target must be between 1 and 8388608 characters".into());
    }
    let content = document(target);
    let request = Request {
        model,
        messages: [
            Message {
                role: "system",
                content: "You are a precise long-context extraction probe.",
            },
            Message {
                role: "user",
                content: &content,
            },
        ],
        stream: false,
        max_tokens: 64,
        temperature: 0,
    };
    let bytes = serde_json::to_vec(&request)?;
    fs::write(path, bytes)?;
    Ok(())
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [verb, path] if verb == "sha256" => {
            let path = Path::new(path);
            if !fs::metadata(path)?.is_file() {
                return Err("agent fixture hash input must be a regular file".into());
            }
            let digest = file_sha256(path).map_err(|failure| failure.error)?;
            println!("{digest}");
            Ok(())
        }
        [verb, model, target, path] if verb == "soak" => write_soak(model, target, Path::new(path)),
        _ => Err(
            "usage: automation agent-fixture-inputs {sha256 FILE | soak MODEL TARGET_CHARS OUTPUT}"
                .into(),
        ),
    }
}

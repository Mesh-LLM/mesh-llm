//! Pinned benchmark dataset acquisition and deterministic prompt corpora.
mod acquisition;
pub mod cli;
mod config;
mod document;
mod edit_loop;
mod projections;
mod sampling;
#[cfg(test)]
mod tests;

use crate::DynResult;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(value) => value
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .trim()
            .to_owned(),
        value => value.to_string(),
    }
}

fn field(row: &Value, key: &str) -> String {
    text(&row[key])
}

fn prompt_budget(prompt: &str, max: usize, target: Option<usize>) -> DynResult<String> {
    if !(128..=1_000_000).contains(&max) || target.is_some_and(|n| n == 0 || n > max) {
        return Err("prompt budget must be 128..1000000 and target within budget".into());
    }
    let mut prompt = text(&Value::String(prompt.to_owned()));
    if let Some(target) = target.filter(|target| prompt.chars().count() < *target) {
        let excerpt = prompt.clone();
        prompt = "Long-context stress packet built from HF-sourced text. Use this tier for context-capacity and transport stress, not quality scoring.".to_owned();
        let mut index = 1;
        let mut count = prompt.chars().count();
        while count < target {
            let chunk = format!("\n\nSource excerpt repeat {index}:\n{excerpt}");
            count += chunk.chars().count();
            prompt.push_str(&chunk);
            index += 1;
        }
    }
    if prompt.chars().count() <= max {
        return Ok(prompt);
    }
    let marker = "\n\n...[truncated for benchmark prompt budget]...\n\n";
    let available = max - marker.chars().count();
    let head = available * 2 / 3;
    let tail = available - head;
    let prefix: String = prompt.chars().take(head).collect();
    let suffix: String = prompt
        .chars()
        .rev()
        .take(tail)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    Ok(format!(
        "{}{marker}{}",
        prefix.trim_end(),
        suffix.trim_start()
    ))
}

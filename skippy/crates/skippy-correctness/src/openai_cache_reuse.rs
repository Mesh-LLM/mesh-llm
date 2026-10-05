use std::{fs, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::cli::OpenAiCacheReuseArgs;

struct Completion {
    content: String,
    prompt_tokens: u64,
    cached_tokens: u64,
}

pub fn run(args: OpenAiCacheReuseArgs) -> Result<()> {
    ensure!(args.max_tokens > 0, "--max-tokens must be positive");
    let prompt = fs::read_to_string(&args.prompt_file)
        .with_context(|| format!("read prompt file {}", args.prompt_file.display()))?;
    ensure!(!prompt.trim().is_empty(), "prompt file is empty");
    let client = Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs.max(1)))
        .build()
        .context("build OpenAI cache check client")?;
    let url = format!("{}/chat/completions", args.base_url.trim_end_matches('/'));
    let first_turn = vec![json!({"role": "user", "content": prompt})];
    let seed = complete(&client, &url, &args.model, &first_turn, args.max_tokens)?;
    let repeat = complete(&client, &url, &args.model, &first_turn, args.max_tokens)?;
    require_hit("repeat", &repeat)?;

    let mut growing_turn = first_turn;
    growing_turn.push(json!({"role": "assistant", "content": seed.content}));
    growing_turn.push(json!({"role": "user", "content": "Summarize your answer in one word."}));
    let grown_seed = complete(&client, &url, &args.model, &growing_turn, args.max_tokens)?;
    require_hit("growing chat", &grown_seed)?;
    let grown_repeat = complete(&client, &url, &args.model, &growing_turn, args.max_tokens)?;
    require_hit("growing chat repeat", &grown_repeat)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "pass",
            "repeat": {
                "cached_tokens": repeat.cached_tokens,
                "prompt_tokens": repeat.prompt_tokens
            },
            "growing_chat_repeat": {
                "cached_tokens": grown_repeat.cached_tokens,
                "prompt_tokens": grown_repeat.prompt_tokens
            },
            "growing_chat": {
                "cached_tokens": grown_seed.cached_tokens,
                "prompt_tokens": grown_seed.prompt_tokens
            }
        }))?
    );
    Ok(())
}

fn complete(
    client: &Client,
    url: &str,
    model: &str,
    messages: &[Value],
    max_tokens: u32,
) -> Result<Completion> {
    let response = client
        .post(url)
        .json(&json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "reasoning_effort": "none"
        }))
        .send()
        .context("send chat completion")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        bail!("chat completion returned {status}: {body}");
    }
    let body: Value = response.json().context("parse chat completion")?;
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .context("chat completion has no message content")?
        .to_string();
    let prompt_tokens = body["usage"]["prompt_tokens"]
        .as_u64()
        .context("chat completion has no prompt token count")?;
    let cached_tokens = body["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    Ok(Completion {
        content,
        prompt_tokens,
        cached_tokens,
    })
}

fn require_hit(label: &str, completion: &Completion) -> Result<()> {
    ensure!(
        completion.cached_tokens > 0 && completion.cached_tokens <= completion.prompt_tokens,
        "{label} did not reuse a prompt prefix: cached={} prompt={}",
        completion.cached_tokens,
        completion.prompt_tokens
    );
    Ok(())
}

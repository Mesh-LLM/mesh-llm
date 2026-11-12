use std::{fs, thread, time::Duration};

use anyhow::{Context, Result, anyhow, bail, ensure};
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
    ensure!(
        seed.cached_tokens == 0,
        "cold seed unexpectedly reused a prompt prefix"
    );
    ensure!(!seed.content.is_empty(), "cold seed returned empty content");
    let repeat = complete(&client, &url, &args.model, &first_turn, args.max_tokens)?;
    require_hit("repeat", &repeat)?;
    require_content("repeat", &seed, &repeat)?;

    let mut growing_turn = first_turn;
    growing_turn.push(json!({"role": "assistant", "content": seed.content.clone()}));
    growing_turn.push(json!({"role": "user", "content": "Summarize your answer in one word."}));
    let grown_seed = complete(&client, &url, &args.model, &growing_turn, args.max_tokens)?;
    require_hit("growing chat", &grown_seed)?;
    ensure!(
        !grown_seed.content.is_empty(),
        "growing chat returned empty content"
    );
    let grown_repeat = complete(&client, &url, &args.model, &growing_turn, args.max_tokens)?;
    require_hit("growing chat repeat", &grown_repeat)?;
    require_content("growing chat repeat", &grown_seed, &grown_repeat)?;
    let wave_branches = if args.wave > 1 {
        Some(check_wave(
            &client,
            &url,
            &args,
            &growing_turn,
            &grown_seed,
        )?)
    } else {
        None
    };
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": "pass",
            "content_parity": "exact",
            "wave": wave_branches.map(|branches| json!({
                "requests": args.wave,
                "copies": "exact",
                "branches": branches
            })),
            "cold_content": seed.content,
            "repeat": {
                "cached_tokens": repeat.cached_tokens,
                "prompt_tokens": repeat.prompt_tokens,
                "content": repeat.content
            },
            "growing_chat_repeat": {
                "cached_tokens": grown_repeat.cached_tokens,
                "prompt_tokens": grown_repeat.prompt_tokens,
                "content": grown_repeat.content
            },
            "growing_chat": {
                "cached_tokens": grown_seed.cached_tokens,
                "prompt_tokens": grown_seed.prompt_tokens,
                "content": grown_seed.content
            }
        }))?
    );
    Ok(())
}

/// The concurrent half of the check: the requests of a wave share one batch,
/// one KV pool and one sampling pass, and each must come out as it does alone.
///
/// Returns each branch's content.
fn check_wave(
    client: &Client,
    url: &str,
    args: &OpenAiCacheReuseArgs,
    growing_turn: &[Value],
    grown_seed: &Completion,
) -> Result<Vec<String>> {
    let wave = usize::from(args.wave);
    let copies = vec![growing_turn.to_vec(); wave];
    let together = complete_wave(client, url, &args.model, &copies, args.max_tokens)?;
    for (index, copy) in together.iter().enumerate() {
        let label = format!("wave copy {index}");
        require_hit(&label, copy)?;
        require_content(&label, grown_seed, copy)?;
    }

    let branches: Vec<Vec<Value>> = (0..wave)
        .map(|index| {
            let mut turn = growing_turn.to_vec();
            turn.push(json!({"role": "assistant", "content": grown_seed.content.clone()}));
            turn.push(json!({
                "role": "user",
                "content": format!("Reply with the number {index} and nothing else.")
            }));
            turn
        })
        .collect();
    let batched = complete_wave(client, url, &args.model, &branches, args.max_tokens)?;
    for (index, (turn, batched)) in branches.iter().zip(&batched).enumerate() {
        let label = format!("wave branch {index}");
        require_hit(&label, batched)?;
        ensure!(
            !batched.content.is_empty(),
            "{label} returned empty content"
        );
        let alone = complete(client, url, &args.model, turn, args.max_tokens)?;
        let label = format!("wave branch {index} alone");
        require_hit(&label, &alone)?;
        require_content(&label, batched, &alone)?;
    }
    Ok(batched.into_iter().map(|branch| branch.content).collect())
}

/// Sends every turn at once and returns the completions in turn order.
fn complete_wave(
    client: &Client,
    url: &str,
    model: &str,
    turns: &[Vec<Value>],
    max_tokens: u32,
) -> Result<Vec<Completion>> {
    thread::scope(|scope| {
        let requests: Vec<_> = turns
            .iter()
            .map(|messages| scope.spawn(move || complete(client, url, model, messages, max_tokens)))
            .collect();
        requests
            .into_iter()
            .map(|request| {
                request
                    .join()
                    .map_err(|_| anyhow!("a wave request panicked"))?
            })
            .collect()
    })
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
            "temperature": 0,
            "seed": 42,
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
        .context("chat completion has no unsigned cached token count")?;
    Ok(Completion {
        content,
        prompt_tokens,
        cached_tokens,
    })
}

fn require_content(label: &str, expected: &Completion, actual: &Completion) -> Result<()> {
    ensure!(
        actual.content == expected.content,
        "{label} generated-content mismatch: expected={:?} actual={:?}",
        expected.content,
        actual.content
    );
    Ok(())
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

#[cfg(test)]
#[path = "openai_cache_reuse/tests.rs"]
mod tests;

//! Prefix-reuse retention probe for AgentX serving.
//!
//! Drives interleaved multi-turn conversations against one OpenAI-compatible
//! server and requires conversation-specific cached-prefix retention after a
//! warm-up turn. Each conversation sends turn 1 as a warm-up whose record the
//! cache cannot yet have, then grows the real chat history with the server's
//! own assistant replies. A scored turn demonstrates retention only when its
//! `cached_tokens` cover a defined fraction of that conversation's previous
//! prompt. This is a usage-based heuristic, not a token-level proof of which
//! conversation was retained. Missing usage, HTTP errors, and retention misses fail the probe.
//!
//! This is the typed owner of the retired `prefix-reuse-probe.py` script.
use crate::command::DynResult;
use serde_json::{Value, json};
#[cfg(test)]
use std::io::{BufReader, Read, Write};
#[cfg(test)]
use std::net::TcpStream;
use std::time::Duration;

pub(super) const PREFIX_REUSE_PROBE_USAGE: &str = "automation agentx prefix-reuse-probe --port PORT [--conversations N] [--turns T] \
     [--block-chars B] [--max-tokens M] [--stagger-secs S]";

/// A scored turn demonstrates conversation-specific retention when its cached
/// tokens cover at least this fraction of the conversation's previous prompt.
/// A minimum prompt block keeps the short shared system message from dominating
/// normal model prompts; unusual chat templates still need evidence inspection.
const CONVERSATION_RETENTION_FRACTION: f64 = 0.5;

/// A probe needs one warm-up turn plus at least one scored turn to say
/// anything about retention.
const MIN_TURNS: u32 = 2;
const MIN_BLOCK_CHARS: usize = 240;

const HTTP_TIMEOUT_SECS: u64 = 600;

pub(super) struct ProbeConfig {
    port: u16,
    conversations: usize,
    turns: u32,
    block_chars: usize,
    max_tokens: u32,
    stagger_secs: u64,
}

pub(super) struct TurnRecord {
    conversation: usize,
    turn: u32,
    prompt_tokens: u64,
    cached_tokens: u64,
    failure: Option<String>,
}

pub(super) struct ProbeReport {
    pub records: Vec<TurnRecord>,
    /// Transport, usage, and retention failures; empty means the probe
    /// accepted the server's retention.
    pub failures: Vec<String>,
}

impl ProbeReport {
    pub(super) fn accepted(&self) -> bool {
        self.failures.is_empty()
    }
}

pub(super) fn command(args: &[String]) -> DynResult<()> {
    let config = parse_config(args)?;
    let report = run_probe(&config);
    print_report(&report);
    if report.accepted() {
        println!(
            "PREFIX_REUSE_PROBE accepted: {} conversations x {} turns retained conversation-specific prefixes",
            config.conversations, config.turns
        );
        Ok(())
    } else {
        Err(format!(
            "prefix-reuse probe failed with {} failure(s); retention is broken",
            report.failures.len()
        )
        .into())
    }
}

fn parse_config(args: &[String]) -> DynResult<ProbeConfig> {
    let mut port = None;
    let mut conversations = 4usize;
    let mut turns = 6u32;
    let mut block_chars = 6300usize;
    let mut max_tokens = 96u32;
    let mut stagger_secs = 1u64;
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let value = iter.next().ok_or_else(|| {
            format!("missing value for {flag}; usage: {PREFIX_REUSE_PROBE_USAGE}")
        })?;
        match flag.as_str() {
            "--port" => port = Some(value.parse::<u16>().map_err(|_| "--port needs a number")?),
            "--conversations" => {
                conversations = value
                    .parse::<usize>()
                    .map_err(|_| "--conversations needs a number")?
            }
            "--turns" => turns = value.parse::<u32>().map_err(|_| "--turns needs a number")?,
            "--block-chars" => {
                block_chars = value
                    .parse::<usize>()
                    .map_err(|_| "--block-chars needs a number")?
            }
            "--max-tokens" => {
                max_tokens = value
                    .parse::<u32>()
                    .map_err(|_| "--max-tokens needs a number")?
            }
            "--stagger-secs" => {
                stagger_secs = value
                    .parse::<u64>()
                    .map_err(|_| "--stagger-secs needs a number")?
            }
            other => {
                return Err(
                    format!("unknown flag {other}; usage: {PREFIX_REUSE_PROBE_USAGE}").into(),
                );
            }
        }
    }
    let port =
        port.ok_or_else(|| format!("--port is required; usage: {PREFIX_REUSE_PROBE_USAGE}"))?;
    if conversations == 0 {
        return Err("--conversations must be at least 1".into());
    }
    if turns < MIN_TURNS {
        return Err(format!(
            "--turns must cover one warm-up turn plus scored turns (at least {MIN_TURNS})"
        )
        .into());
    }
    if block_chars < MIN_BLOCK_CHARS {
        return Err(format!("--block-chars must be at least {MIN_BLOCK_CHARS} so the shared template does not dominate").into());
    }
    if max_tokens == 0 || port == 0 {
        return Err("--max-tokens and --port must be positive".into());
    }
    Ok(ProbeConfig {
        port,
        conversations,
        turns,
        block_chars,
        max_tokens,
        stagger_secs,
    })
}

fn system_message() -> Value {
    json!({"role": "system", "content": "You are a helpful assistant."})
}

fn user_content(conversation: usize, turn: u32, block_chars: usize) -> String {
    // The conversation marker lets evidence readers (and test fixtures)
    // attribute a request to its conversation; the stable block gives every
    // conversation its own large cacheable prefix.
    let marker = format!("conversation-{conversation} ");
    let stable = marker.repeat(block_chars.div_ceil(marker.len()));
    format!("{stable}Q{conversation}-{turn}: name a color.")
}

fn chat_body(
    history: &[(String, String)],
    conversation: usize,
    turn: u32,
    block_chars: usize,
    max_tokens: u32,
) -> Value {
    let mut messages = vec![system_message()];
    for (user, assistant) in history {
        messages.push(json!({"role": "user", "content": user}));
        messages.push(json!({"role": "assistant", "content": assistant}));
    }
    messages.push(json!({
        "role": "user",
        "content": user_content(conversation, turn, block_chars)
    }));
    json!({
        "model": "m",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": false
    })
}

fn post_chat(port: u16, body: &Value) -> Result<Value, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_secs(HTTP_TIMEOUT_SECS),
            crate::automation::openai_exchange::json_completion::request_value(
                &format!("http://127.0.0.1:{port}/v1"),
                body,
            ),
        )
        .await
        .map_err(|_| "completion deadline exceeded".to_string())?
        .map_err(|error| error.to_string())
    })
}

fn extract_usage(response: &Value) -> Result<(u64, u64, u64), String> {
    let usage = response
        .get("usage")
        .ok_or_else(|| "usage missing from completion response".to_string())?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .filter(|tokens| *tokens > 0)
        .ok_or_else(|| "usage.prompt_tokens missing or zero".to_string())?;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| "usage.completion_tokens missing".to_string())?;
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .ok_or_else(|| "usage.cached_tokens missing or invalid".to_string())?;
    if cached_tokens > prompt_tokens {
        return Err("cached usage exceeds prompt usage".into());
    }
    Ok((prompt_tokens, cached_tokens, completion_tokens))
}

fn assistant_reply(response: &Value) -> Result<String, String> {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .filter(|reply| !reply.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| "assistant reply missing from completion response".to_string())
}

/// Scored turns must keep conversation-specific reuse: enough cached tokens
/// to cover the retention fraction of that conversation's previous prompt.
fn score_turn(record: &TurnRecord, previous_prompt_tokens: Option<u64>) -> Option<String> {
    let previous = previous_prompt_tokens?;
    let required = previous.div_ceil(2);
    (record.cached_tokens < required).then(|| {
        format!(
            "conversation {} turn {}: cached {} tokens covers less than {:.0}% of the previous prompt ({previous} tokens)",
            record.conversation,
            record.turn,
            record.cached_tokens,
            CONVERSATION_RETENTION_FRACTION * 100.0
        )
    })
}

/// Every post-warm-up request must retain its conversation-specific prefix.
fn retention_failures(records: &[TurnRecord]) -> Vec<String> {
    let mut conversations = std::collections::BTreeMap::<usize, Vec<&TurnRecord>>::new();
    for record in records {
        conversations
            .entry(record.conversation)
            .or_default()
            .push(record);
    }
    let mut failures = Vec::new();
    for (conversation, turns) in conversations {
        let mut previous_prompt = None;
        for record in turns {
            if record.failure.is_none() && record.turn >= 2 {
                if let Some(failure) = score_turn(record, previous_prompt) {
                    failures.push(failure);
                } else if previous_prompt.is_none() {
                    failures.push(format!(
                        "conversation {conversation}: missing warm-up usage"
                    ));
                }
            }
            if record.failure.is_none() {
                previous_prompt = Some(record.prompt_tokens);
            }
        }
    }
    failures
}

fn run_probe(config: &ProbeConfig) -> ProbeReport {
    let mut histories = vec![Vec::new(); config.conversations];
    let mut records = Vec::new();
    // Scope each round: all conversations complete a turn before any resumes.
    // A panicked worker fails its record instead of stranding peers at a barrier.
    for turn in 1..=config.turns {
        std::thread::scope(|scope| {
            let handles: Vec<_> = histories
                .iter_mut()
                .enumerate()
                .map(|(conversation, history)| {
                    (
                        conversation,
                        scope.spawn(move || request_turn(config, history, conversation, turn)),
                    )
                })
                .collect();
            for (conversation, handle) in handles {
                records.push(handle.join().unwrap_or_else(|_| TurnRecord {
                    conversation,
                    turn,
                    prompt_tokens: 0,
                    cached_tokens: 0,
                    failure: Some(format!(
                        "conversation {conversation} turn {turn}: worker panicked"
                    )),
                }));
            }
        });
    }
    let mut failures = retention_failures(&records);
    failures.extend(records.iter().filter_map(|record| record.failure.clone()));
    ProbeReport { records, failures }
}

fn request_turn(
    config: &ProbeConfig,
    history: &mut Vec<(String, String)>,
    conversation: usize,
    turn: u32,
) -> TurnRecord {
    if turn == 1 && config.stagger_secs > 0 {
        std::thread::sleep(Duration::from_secs(
            config.stagger_secs * conversation as u64,
        ));
    }
    let body = chat_body(
        history,
        conversation,
        turn,
        config.block_chars,
        config.max_tokens,
    );
    let result = post_chat(config.port, &body).and_then(|response| {
        let (prompt, cached, _) = extract_usage(&response)?;
        let reply = assistant_reply(&response)?;
        history.push((user_content(conversation, turn, config.block_chars), reply));
        Ok((prompt, cached))
    });
    match result {
        Ok((prompt_tokens, cached_tokens)) => TurnRecord {
            conversation,
            turn,
            prompt_tokens,
            cached_tokens,
            failure: None,
        },
        Err(error) => TurnRecord {
            conversation,
            turn,
            prompt_tokens: 0,
            cached_tokens: 0,
            failure: Some(format!("conversation {conversation} turn {turn}: {error}")),
        },
    }
}

fn print_report(report: &ProbeReport) {
    for record in &report.records {
        match record.failure.as_deref() {
            Some(failure) => println!("  ERROR {failure}"),
            None => {
                let percent = (u128::from(record.cached_tokens) * 100)
                    .checked_div(u128::from(record.prompt_tokens))
                    .unwrap_or(0);
                println!(
                    "  RESULT conversation {} turn {}: prompt {} cached {percent}% completion-source turn",
                    record.conversation, record.turn, record.prompt_tokens
                );
            }
        }
    }
    for failure in &report.failures {
        println!("  RETENTION_FAILURE {failure}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::BufRead;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// How the mock server simulates prefix-cache behavior.
    #[derive(Clone, Copy, PartialEq)]
    enum MockCache {
        /// cached_tokens equal the conversation's previous full prompt.
        Healthy,
        /// No cache retention at all.
        Collapsed,
        /// Only the shared system prefix stays cached.
        SystemOnly,
        /// Respond without a usage object.
        MissingUsage,
    }

    struct MockServer {
        port: u16,
    }

    impl MockServer {
        fn spawn(cache: MockCache) -> Self {
            Self::spawn_with(cache, None)
        }
        fn spawn_with(cache: MockCache, fail_conversation: Option<usize>) -> Self {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let port = listener.local_addr().unwrap().port();
            let prompts = Arc::new(Mutex::new(BTreeMap::new()));
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { continue };
                    let prompts = prompts.clone();
                    std::thread::spawn(move || {
                        let _ = serve_one(stream, cache, fail_conversation, prompts);
                    });
                }
            });
            MockServer { port }
        }
    }

    fn serve_one(
        stream: TcpStream,
        cache: MockCache,
        fail_conversation: Option<usize>,
        prompts: Arc<Mutex<BTreeMap<usize, u64>>>,
    ) -> std::io::Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line)?;
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line == "\r\n" || line.is_empty() {
                break;
            }
            if let Some(value) = line
                .to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|rest| rest.trim().parse::<usize>().ok())
            {
                content_length = value;
            }
        }
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body)?;
        let body: Value = serde_json::from_slice(&body).unwrap();
        let conversation = body["messages"]
            .as_array()
            .and_then(|messages| messages.iter().find(|m| m["role"] == "user"))
            .and_then(|m| m["content"].as_str())
            .and_then(|content| {
                content
                    .split_whitespace()
                    .next()
                    .and_then(|marker| marker.strip_prefix("conversation-")?.parse::<usize>().ok())
            })
            .unwrap_or(0);
        let turn = body["messages"]
            .as_array()
            .and_then(|messages| messages.last())
            .and_then(|m| m["content"].as_str())
            .and_then(|content| {
                content
                    .rsplit_once("Q")
                    .and_then(|(_, tail)| tail.split('-').nth(1))
                    .and_then(|tail| tail.split(':').next())
                    .and_then(|turn| turn.parse::<u32>().ok())
            })
            .unwrap_or(1);
        let chars: usize = body["messages"]
            .as_array()
            .map(|messages| {
                messages
                    .iter()
                    .filter_map(|m| m["content"].as_str())
                    .map(str::len)
                    .sum()
            })
            .unwrap_or(0);
        let prompt_tokens = (chars / 4).max(1) as u64;
        let mut stream = stream;
        if fail_conversation == Some(conversation) && turn >= 2 {
            let payload = format!("{{\"error\":\"mock failure conv {conversation} turn {turn}\"}}");
            let response = format!(
                "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            stream.write_all(response.as_bytes())?;
            return Ok(());
        }
        if cache == MockCache::MissingUsage {
            let payload =
                json!({"choices": [{"message": {"role": "assistant", "content": "blue"}}]});
            write_json(&mut stream, &payload)?;
            return Ok(());
        }
        let mut prompts = prompts.lock().unwrap();
        let cached = match cache {
            MockCache::Healthy => prompts.get(&conversation).copied().unwrap_or(0),
            MockCache::Collapsed => 0,
            MockCache::SystemOnly => 9,
            MockCache::MissingUsage => unreachable!(),
        };
        prompts.insert(conversation, prompt_tokens);
        let payload = json!({
            "choices": [{"message": {"role": "assistant", "content": format!("color for turn {turn}")}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": cached}
            }
        });
        write_json(&mut stream, &payload)
    }

    fn write_json(stream: &mut TcpStream, payload: &Value) -> std::io::Result<()> {
        let body = serde_json::to_vec(payload).unwrap();
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
        write!(stream, "{:x}\r\n", body.len())?;
        stream.write_all(&body)?;
        stream.write_all(b"\r\n0\r\n\r\n")
    }

    fn probe_config(port: u16) -> ProbeConfig {
        ProbeConfig {
            port,
            conversations: 3,
            turns: 3,
            block_chars: 240,
            max_tokens: 8,
            stagger_secs: 0,
        }
    }

    #[test]
    fn healthy_retention_passes_with_conversation_specific_reuse() {
        let server = MockServer::spawn(MockCache::Healthy);
        let report = run_probe(&probe_config(server.port));
        assert!(
            report.accepted(),
            "healthy retention must pass; failures: {:?}",
            report.failures
        );
        // The warm-up turn must actually be sent: every conversation has a
        // turn 1 record that the mock answered without any cache.
        for conversation in 0..3 {
            let warmup = report
                .records
                .iter()
                .find(|record| record.conversation == conversation && record.turn == 1);
            assert!(warmup.is_some(), "turn 1 warm-up must be sent");
        }
    }

    #[test]
    fn collapsed_cache_fails_every_conversation() {
        let server = MockServer::spawn(MockCache::Collapsed);
        let report = run_probe(&probe_config(server.port));
        assert!(!report.accepted());
        assert_eq!(
            report
                .failures
                .iter()
                .filter(|failure| failure.contains("cached 0 tokens"))
                .count(),
            6,
            "every post-warm-up miss must fail"
        );
    }

    #[test]
    fn shared_system_reuse_alone_fails_the_conversation_threshold() {
        let server = MockServer::spawn(MockCache::SystemOnly);
        let report = run_probe(&probe_config(server.port));
        assert!(
            !report.accepted(),
            "a positive cached count from the shared system prefix must not pass"
        );
    }

    #[test]
    fn missing_usage_fails_the_probe() {
        let server = MockServer::spawn(MockCache::MissingUsage);
        let report = run_probe(&probe_config(server.port));
        assert!(!report.accepted());
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("usage missing")),
            "missing usage must be a failure: {:?}",
            report.failures
        );
    }

    #[test]
    fn http_errors_fail_the_probe() {
        let server = MockServer::spawn_with(MockCache::Healthy, Some(1));
        let report = run_probe(&probe_config(server.port));
        assert!(!report.accepted());
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("HTTP 500")),
            "HTTP errors must be failures: {:?}",
            report.failures
        );
    }

    #[test]
    fn probe_requires_a_scored_turn_after_warmup() {
        assert!(
            parse_config(&[
                "--port".to_string(),
                "1".to_string(),
                "--turns".to_string(),
                "1".to_string()
            ])
            .is_err()
        );
        assert!(parse_config(&[]).is_err(), "--port is required");
    }

    #[test]
    fn scored_threshold_measures_against_the_previous_prompt() {
        let record = |cached: u64| TurnRecord {
            conversation: 0,
            turn: 2,
            prompt_tokens: 400,
            cached_tokens: cached,
            failure: None,
        };
        let mut warmup = record(0);
        warmup.turn = 1;
        let mut records = vec![warmup, record(200), record(199)];
        assert_eq!(retention_failures(&records).len(), 1);
        records[2].turn = 1;
        assert!(
            retention_failures(&records).is_empty(),
            "warm-up turn is never scored"
        );
        records[2].turn = 2;
        records[2].failure = Some("transport".into());
        assert!(
            retention_failures(&records)
                .iter()
                .all(|failure| !failure.contains("cached")),
            "failed transports are reported as transport failures, not retention misses"
        );
    }
    #[test]
    fn invalid_usage_and_empty_assistant_are_rejected() {
        let response = |prompt: Value, cached: Value| {
            json!({
                "usage": {"prompt_tokens": prompt, "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": cached}}
            })
        };
        for (prompt, cached) in [
            (json!(0), json!(0)),
            (json!(10), json!(11)),
            (json!(10), json!(-1)),
            (json!(10), Value::Null),
            (json!(10), json!(1.5)),
        ] {
            assert!(extract_usage(&response(prompt, cached)).is_err());
        }
        for content in [json!(""), json!(" \n"), Value::Null] {
            assert!(
                assistant_reply(&json!({"choices": [{"message": {"content": content}}]})).is_err()
            );
        }
    }

    #[test]
    fn tiny_blocks_and_zero_generation_are_not_valid_probes() {
        for (flag, value) in [
            ("--block-chars", "1"),
            ("--block-chars", "239"),
            ("--max-tokens", "0"),
        ] {
            assert!(
                parse_config(&["--port".into(), "1".into(), flag.into(), value.into()]).is_err()
            );
        }
    }
}

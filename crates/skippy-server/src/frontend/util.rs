use crate::runtime_state::RuntimeState;
use openai_frontend::FinishReason;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use sha2::Digest;
use sha2::Sha256;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

#[cfg(test)]
use anyhow::{Result, anyhow};
#[cfg(test)]
use skippy_protocol::binary::recv_ready;
#[cfg(test)]
use std::{net::TcpStream, time::Duration};

pub(super) fn trim_at_stop<'a>(text: &'a str, stop_values: &[&str]) -> &'a str {
    let first_stop = stop_values
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min();
    match first_stop {
        Some(index) => &text[..index],
        None => text,
    }
}

pub(super) fn generation_stop_values(
    stop: Option<&openai_frontend::StopSequence>,
    chat_metadata: Option<&str>,
) -> Vec<String> {
    let mut values: Vec<String> = stop
        .map(|stop| stop.values().into_iter().map(str::to_string).collect())
        .unwrap_or_default();
    let additional_stops = chat_metadata
        .and_then(|metadata| serde_json::from_str::<serde_json::Value>(metadata).ok())
        .and_then(|value| {
            value
                .get("additional_stops")
                .and_then(serde_json::Value::as_array)
                .cloned()
        });
    if let Some(stops) = additional_stops {
        values.extend(
            stops
                .iter()
                .filter_map(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
        );
    }
    values
}

pub(super) fn valid_utf8_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(error) => error.valid_up_to(),
    }
}

pub(super) fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

pub(super) fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(super) fn stable_wire_id(parts: &[&[u8]]) -> u64 {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let id = u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("sha256 digest has an 8-byte prefix"),
    );
    if id == 0 { 1 } else { id }
}

pub(super) fn detokenize_bytes_with_runtime(
    runtime: &Arc<Mutex<RuntimeState>>,
    token_ids: &[i32],
) -> OpenAiResult<Vec<u8>> {
    let runtime = runtime
        .lock()
        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
    runtime
        .model
        .detokenize_bytes(token_ids)
        .map_err(openai_backend_error)
}

pub(super) fn token_is_eog_with_runtime(
    runtime: &Arc<Mutex<RuntimeState>>,
    token_id: i32,
) -> OpenAiResult<bool> {
    let runtime = runtime
        .lock()
        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
    runtime
        .model
        .token_is_eog(token_id)
        .map_err(openai_backend_error)
}

pub(super) fn ms_to_us(ms: f64) -> i64 {
    (ms * 1000.0).round() as i64
}

pub(super) fn us_to_ms(us: i64) -> f64 {
    us as f64 / 1000.0
}

pub(super) fn openai_backend_error(error: anyhow::Error) -> OpenAiError {
    OpenAiError::backend(error.to_string())
}

pub(super) fn openai_io_error(error: std::io::Error) -> OpenAiError {
    OpenAiError::backend(error.to_string())
}

#[cfg(test)]
pub(super) fn connect_endpoint_ready(endpoint: &str, timeout_secs: u64) -> Result<TcpStream> {
    let endpoint = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match TcpStream::connect(endpoint) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                match recv_ready(&mut stream) {
                    Ok(()) => return Ok(stream),
                    Err(error) => {
                        last_error = Some(anyhow!(error).context("ready handshake failed"))
                    }
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

pub(super) fn finish_reason_for_generation(exhausted_max_tokens: bool) -> FinishReason {
    if exhausted_max_tokens {
        FinishReason::Length
    } else {
        FinishReason::Stop
    }
}

pub(super) fn ensure_context_capacity(
    prompt_token_count: usize,
    max_tokens: u32,
    ctx_size: usize,
) -> OpenAiResult<()> {
    let requested_tokens = prompt_token_count.saturating_add(max_tokens as usize);
    if requested_tokens > ctx_size {
        return Err(OpenAiError::context_length_exceeded(format!(
            "requested prompt plus completion tokens ({requested_tokens}) exceed context window ({ctx_size})"
        )));
    }
    Ok(())
}

pub(super) fn context_budget_completion_tokens(
    prompt_token_count: usize,
    ctx_size: usize,
) -> OpenAiResult<u32> {
    if prompt_token_count > ctx_size {
        return Err(OpenAiError::context_length_exceeded(format!(
            "requested prompt tokens ({prompt_token_count}) exceed context window ({ctx_size})"
        )));
    }
    Ok(ctx_size
        .saturating_sub(prompt_token_count)
        .min(u32::MAX as usize) as u32)
}

//! OpenAI generation driver for a mesh-managed chain of MLX layer stages.
//!
//! The coordinator owns tokenizer/chat-template sidecars only. Model weights
//! remain in the stage engines selected by the mesh topology. Generation uses
//! the existing Skippy binary stage wire: one final prefill followed by greedy
//! decode frames, with a stop frame resetting every stage in the chain.

use std::fs;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use safemlx_lm_utils::tokenizer::{Tokenizer as ChatTokenizer, load_model_chat_template_from_file};
use serde_json::{Map, Value, json};
use skippy_protocol::binary::{
    StageReply, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
    WireReplyKind, recv_ready, recv_reply, write_stage_message,
};
use tokio::sync::mpsc;

use crate::engine::{FinishReason, GenerateRequest, TokenMsg};

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct MlxDistributedEngineConfig {
    pub model_dir: PathBuf,
    pub model_id: String,
    pub stage_addr: SocketAddr,
    pub wire_dtype: WireActivationDType,
    pub default_max_tokens: usize,
    pub max_tokens_cap: usize,
    pub context_tokens: usize,
}

struct Job {
    request: GenerateRequest,
    reply: mpsc::UnboundedSender<TokenMsg>,
}

pub struct MlxDistributedEngine {
    jobs: mpsc::UnboundedSender<Job>,
    config: MlxDistributedEngineConfig,
}

impl MlxDistributedEngine {
    pub fn spawn(config: MlxDistributedEngineConfig) -> Result<Self> {
        let (jobs, mut job_rx) = mpsc::unbounded_channel::<Job>();
        let worker_config = config.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        thread::Builder::new()
            .name("mlx-distributed-engine".to_string())
            .spawn(move || {
                let tokenizer = DistributedTokenizer::load(&worker_config.model_dir);
                match tokenizer {
                    Ok(mut tokenizer) => {
                        let _ = ready_tx.send(Ok(()));
                        while let Some(job) = job_rx.blocking_recv() {
                            if let Err(error) = generate_one(&worker_config, &mut tokenizer, &job) {
                                let _ = job.reply.send(TokenMsg::Error(format!("{error:#}")));
                            }
                        }
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(format!("{error:#}")));
                    }
                }
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self { jobs, config }),
            Ok(Err(error)) => Err(anyhow!("MLX distributed tokenizer load failed: {error}")),
            Err(_) => Err(anyhow!("MLX distributed worker exited before readiness")),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    pub fn clamp_max_tokens(&self, requested: Option<usize>) -> usize {
        requested
            .unwrap_or(self.config.default_max_tokens)
            .clamp(1, self.config.max_tokens_cap)
    }

    pub fn submit(&self, request: GenerateRequest) -> mpsc::UnboundedReceiver<TokenMsg> {
        let (reply, rx) = mpsc::unbounded_channel();
        if self
            .jobs
            .send(Job {
                request,
                reply: reply.clone(),
            })
            .is_err()
        {
            let _ = reply.send(TokenMsg::Error(
                "MLX distributed worker is not running".to_string(),
            ));
        }
        rx
    }
}

struct DistributedTokenizer {
    tokenizer: ChatTokenizer,
    chat_template: Option<String>,
    eos: Vec<u32>,
}

impl DistributedTokenizer {
    fn load(model_dir: &Path) -> Result<Self> {
        let mut tokenizer = ChatTokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|error| anyhow!("tokenizer.json: {error}"))?;
        tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(model_dir)?);
        let chat_template = load_chat_template(model_dir)?;
        let config: Value = serde_json::from_slice(
            &fs::read(model_dir.join("config.json")).context("read MLX config.json")?,
        )
        .context("parse MLX config.json")?;
        Ok(Self {
            tokenizer,
            chat_template,
            eos: eos_token_ids(&config),
        })
    }

    fn prompt_tokens(&mut self, request: &GenerateRequest) -> Result<Vec<i32>> {
        let (prompt, add_special_tokens) = self.render_prompt(request)?;
        self.tokenizer
            .encode(prompt, add_special_tokens)
            .map_err(|error| anyhow!("encode distributed MLX prompt: {error}"))?
            .get_ids()
            .iter()
            .copied()
            .map(|token| i32::try_from(token).context("token id exceeds i32"))
            .collect()
    }

    fn render_prompt(&mut self, request: &GenerateRequest) -> Result<(String, bool)> {
        if let Some(prompt) = request.raw_prompt.as_ref() {
            return Ok((prompt.clone(), true));
        }
        let messages = request
            .messages
            .iter()
            .map(|turn| json!({"role": turn.role, "content": turn.content}))
            .collect::<Vec<_>>();
        let Some(template) = self.chat_template.clone() else {
            return Ok((fallback_prompt(request), true));
        };
        let rendered = self
            .tokenizer
            .apply_chat_template_json(template, vec![messages], None, "mesh-llm-mlx", true, None)
            .map_err(|error| anyhow!("render MLX chat template: {error}"))?
            .into_iter()
            .next()
            .context("MLX chat template returned no prompt")?;
        Ok((rendered, false))
    }
}

fn load_chat_template(model_dir: &Path) -> Result<Option<String>> {
    let tokenizer_config = model_dir.join("tokenizer_config.json");
    if tokenizer_config.is_file()
        && let Some(template) = load_model_chat_template_from_file(&tokenizer_config)?
    {
        return Ok(Some(template));
    }
    let standalone = model_dir.join("chat_template.jinja");
    if standalone.is_file() {
        return fs::read_to_string(standalone)
            .map(Some)
            .context("read MLX chat_template.jinja");
    }
    Ok(None)
}

fn load_tokenizer_template_kwargs(model_dir: &Path) -> Result<Map<String, Value>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.is_file() {
        return Ok(Map::new());
    }
    let value: Value =
        serde_json::from_slice(&fs::read(config_path).context("read MLX tokenizer_config.json")?)
            .context("parse MLX tokenizer_config.json")?;
    Ok(tokenizer_template_kwargs(&value))
}

fn tokenizer_template_kwargs(value: &Value) -> Map<String, Value> {
    value
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(key, value)| key.ends_with("_token") && (value.is_string() || value.is_null()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn eos_token_ids(config: &Value) -> Vec<u32> {
    let value = config
        .get("eos_token_id")
        .or_else(|| config.get("text_config")?.get("eos_token_id"));
    match value {
        Some(Value::Number(value)) => value
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .into_iter()
            .collect(),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_u64)
            .filter_map(|id| u32::try_from(id).ok())
            .collect(),
        _ => Vec::new(),
    }
}

fn fallback_prompt(request: &GenerateRequest) -> String {
    request
        .messages
        .iter()
        .map(|turn| format!("{}: {}", turn.role, turn.content))
        .chain(std::iter::once("assistant:".to_string()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_one(
    config: &MlxDistributedEngineConfig,
    tokenizer: &mut DistributedTokenizer,
    job: &Job,
) -> Result<()> {
    let prompt = tokenizer.prompt_tokens(&job.request)?;
    ensure!(!prompt.is_empty(), "distributed MLX prompt has no tokens");
    let requested_max_tokens = job.request.max_tokens.clamp(1, config.max_tokens_cap);
    let max_tokens =
        context_bounded_max_tokens(prompt.len(), requested_max_tokens, config.context_tokens)?;
    let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed).max(1);
    let session_id = request_id;
    let mut stream = connect_stage(config.stage_addr)?;
    let result = generate_tokens(
        &mut stream,
        config.wire_dtype,
        request_id,
        session_id,
        &prompt,
        max_tokens,
        &tokenizer.eos,
        &tokenizer.tokenizer,
        &job.reply,
    );
    let stop_result = stop_session(&mut stream, config.wire_dtype, request_id, session_id);
    result.and(stop_result)
}

fn connect_stage(addr: SocketAddr) -> Result<TcpStream> {
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
    const GENERATION_IO_TIMEOUT: Duration = Duration::from_secs(5 * 60);
    let mut stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .with_context(|| format!("connect MLX stage 0 at {addr}"))?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(CONNECT_TIMEOUT))?;
    stream.set_write_timeout(Some(CONNECT_TIMEOUT))?;
    recv_ready(&mut stream).context("MLX stage 0 did not become ready")?;
    stream.set_read_timeout(Some(GENERATION_IO_TIMEOUT))?;
    stream.set_write_timeout(Some(GENERATION_IO_TIMEOUT))?;
    Ok(stream)
}

#[allow(clippy::too_many_arguments)]
fn generate_tokens(
    stream: &mut TcpStream,
    dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
    prompt: &[i32],
    max_tokens: usize,
    eos: &[u32],
    tokenizer: &ChatTokenizer,
    reply: &mpsc::UnboundedSender<TokenMsg>,
) -> Result<()> {
    send_message(
        stream,
        &prefill_message(dtype, request_id, session_id, prompt),
        dtype,
    )?;
    let mut predicted = predicted_reply(stream)?;
    let mut generated = Vec::with_capacity(max_tokens);
    let mut decoder = tokenizer.decode_stream(true);
    let mut emitted = String::new();
    let mut finish = FinishReason::Length;
    while generated.len() < max_tokens {
        let token = u32::try_from(predicted).context("negative predicted token")?;
        if eos.contains(&token) {
            finish = FinishReason::Stop;
            break;
        }
        generated.push(token);
        if let Some(delta) = decoder
            .step(token)
            .map_err(|error| anyhow!("decode distributed MLX token: {error}"))?
        {
            emitted.push_str(&delta);
            let _ = reply.send(TokenMsg::Delta(delta));
        }
        if reply.is_closed() {
            return Ok(());
        }
        if generated.len() == max_tokens {
            break;
        }
        send_message(
            stream,
            &decode_message(
                dtype,
                request_id,
                session_id,
                prompt.len(),
                generated.len() - 1,
                predicted,
            ),
            dtype,
        )?;
        predicted = predicted_reply(stream)?;
    }
    emit_decode_suffix(tokenizer, &generated, &emitted, reply)?;
    let _ = reply.send(TokenMsg::Done {
        finish_reason: finish,
        prompt_tokens: u32::try_from(prompt.len()).unwrap_or(u32::MAX),
        completion_tokens: u32::try_from(generated.len()).unwrap_or(u32::MAX),
    });
    Ok(())
}

fn emit_decode_suffix(
    tokenizer: &ChatTokenizer,
    token_ids: &[u32],
    emitted: &str,
    reply: &mpsc::UnboundedSender<TokenMsg>,
) -> Result<()> {
    let decoded = tokenizer
        .decode(token_ids, true)
        .map_err(|error| anyhow!("finalize distributed MLX decode: {error}"))?;
    if let Some(suffix) = decoded.strip_prefix(emitted)
        && !suffix.is_empty()
    {
        let _ = reply.send(TokenMsg::Delta(suffix.to_string()));
    }
    Ok(())
}

fn context_bounded_max_tokens(
    prompt_tokens: usize,
    requested_max_tokens: usize,
    context_tokens: usize,
) -> Result<usize> {
    let available = context_tokens.saturating_sub(prompt_tokens);
    ensure!(
        available > 0,
        "distributed MLX prompt uses {prompt_tokens} tokens, exceeding the {context_tokens}-token context"
    );
    Ok(requested_max_tokens.min(available))
}

fn prefill_message(
    dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
    tokens: &[i32],
) -> StageWireMessage {
    let token_count = i32::try_from(tokens.len()).unwrap_or(i32::MAX);
    let mut state = StageStateHeader::new(WireMessageKind::PrefillFinalEmbd, dtype);
    state.prompt_token_count = token_count;
    StageWireMessage {
        kind: WireMessageKind::PrefillFinalEmbd,
        pos_start: 0,
        token_count,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: tokens.to_vec(),
        positions: (0..token_count).collect(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

fn decode_message(
    dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
    prompt_tokens: usize,
    decode_step: usize,
    current_token: i32,
) -> StageWireMessage {
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, dtype);
    state.prompt_token_count = i32::try_from(prompt_tokens).unwrap_or(i32::MAX);
    state.decode_step = i32::try_from(decode_step).unwrap_or(i32::MAX);
    state.current_token = current_token;
    let position = prompt_tokens.saturating_add(decode_step);
    StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: i32::try_from(position).unwrap_or(i32::MAX),
        token_count: 1,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: vec![i32::try_from(position).unwrap_or(i32::MAX)],
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

fn send_message(
    stream: &mut TcpStream,
    message: &StageWireMessage,
    dtype: WireActivationDType,
) -> Result<()> {
    write_stage_message(&mut *stream, message, dtype)?;
    stream.flush()?;
    Ok(())
}

fn predicted_reply(stream: &mut TcpStream) -> Result<i32> {
    let reply = recv_reply(stream)?;
    ensure!(
        matches!(
            reply.kind,
            WireReplyKind::PredictedToken | WireReplyKind::PredictedTokens
        ),
        "MLX stage chain returned {:?}, expected a prediction",
        reply.kind
    );
    prediction(&reply).context("MLX stage chain returned no predicted token")
}

fn prediction(reply: &StageReply) -> Option<i32> {
    reply
        .predicted_tokens
        .first()
        .copied()
        .or_else(|| (reply.kind == WireReplyKind::PredictedToken).then_some(reply.predicted))
}

fn stop_session(
    stream: &mut TcpStream,
    dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
) -> Result<()> {
    send_message(
        stream,
        &StageWireMessage::stop_with_identity(dtype, request_id, session_id),
        dtype,
    )?;
    let reply = recv_reply(stream)?;
    ensure!(
        reply.kind == WireReplyKind::Ack,
        "MLX stage stop was not acknowledged"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_frame_requests_prediction_for_complete_prompt() {
        let message = prefill_message(WireActivationDType::F16, 7, 9, &[1, 2, 3]);

        assert_eq!(message.kind, WireMessageKind::PrefillFinalEmbd);
        assert_eq!(message.token_count, 3);
        assert_eq!(message.tokens, vec![1, 2, 3]);
        assert_eq!(message.positions, vec![0, 1, 2]);
        assert_eq!(message.state.prompt_token_count, 3);
        assert!(message.kind.requires_predicted_reply());
    }

    #[test]
    fn decode_frame_carries_current_token_and_absolute_position() {
        let message = decode_message(WireActivationDType::F16, 7, 9, 12, 2, 42);

        assert_eq!(message.kind, WireMessageKind::DecodeEmbd);
        assert_eq!(message.pos_start, 14);
        assert_eq!(message.positions, vec![14]);
        assert_eq!(message.state.current_token, 42);
        assert_eq!(message.state.decode_step, 2);
        assert_eq!(message.state.prompt_token_count, 12);
    }

    #[test]
    fn prediction_prefers_multi_token_sideband() {
        let reply = StageReply {
            kind: WireReplyKind::PredictedTokens,
            predicted: 3,
            predicted_tokens: vec![4, 5],
            stats: Default::default(),
        };

        assert_eq!(prediction(&reply), Some(4));
    }

    #[test]
    fn scalar_prediction_falls_back_to_scalar_field() {
        let reply = StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted: 3,
            predicted_tokens: Vec::new(),
            stats: Default::default(),
        };

        assert_eq!(prediction(&reply), Some(3));
    }

    #[test]
    fn empty_prediction_batch_is_rejected() {
        let reply = StageReply {
            kind: WireReplyKind::PredictedTokens,
            predicted: 0,
            predicted_tokens: Vec::new(),
            stats: Default::default(),
        };

        assert_eq!(prediction(&reply), None);
    }

    #[test]
    fn generation_is_bounded_by_remaining_context() {
        assert_eq!(context_bounded_max_tokens(12, 8, 16).unwrap(), 4);
        assert!(context_bounded_max_tokens(16, 8, 16).is_err());
    }

    #[test]
    fn tokenizer_template_kwargs_keep_only_special_token_values() {
        let value = json!({
            "bos_token": "<s>",
            "eos_token": null,
            "chat_template": "ignored",
            "added_token": {"content": "ignored"},
            "padding_side": "left"
        });

        let kwargs = tokenizer_template_kwargs(&value);

        assert_eq!(kwargs.len(), 2);
        assert_eq!(kwargs["bos_token"], "<s>");
        assert!(kwargs["eos_token"].is_null());
    }
}

//! MLX generation engine backed by a dedicated OS worker thread.
//!
//! MLX arrays, streams, and the loaded model wrap raw C pointers and are neither
//! `Send` nor `Sync`. Rather than fight that, we confine every MLX object to a
//! single worker thread that owns them for its whole life, and talk to it only
//! with `Send` messages:
//!
//! - a `Send + Sync` job channel (tokio unbounded) carries generation requests;
//! - each job carries a per-request token channel the worker streams results on.
//!
//! This also naturally serializes GPU access (one generation at a time), which
//! matches how goose drives safemlx today.

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use safemlx::transforms::async_eval;
use safemlx::{Device, DeviceType, Stream};
use safemlx_lm::models::input::{InputPart, ModelInput};
use safemlx_lm::models::{LoadedModel, ModelLoadOptions};
use safemlx_lm::sampler::DefaultSampler;

use crate::stage::MlxWeightQuantization;

/// How the worker should load and run a model.
#[derive(Clone, Debug)]
pub struct MlxEngineConfig {
    pub model_dir: PathBuf,
    pub model_id: String,
    pub default_max_tokens: usize,
    pub max_tokens_cap: usize,
    /// Quantize eligible dense checkpoint tensors as they are loaded. Already
    /// quantized checkpoints with matching metadata load directly.
    pub weight_quantization: Option<MlxWeightQuantization>,
    /// Auto policy may retry native loading for a quantization incompatibility
    /// or a narrowly recognized benign strict-loader rejection.
    pub allow_native_quantization_fallback: bool,
}

/// Selects load-time affine-4 only for dense, unquantized model families that
/// the pinned safemlx runtime can quantize safely. Frontier grouped-expert
/// families and checkpoints already declaring a representation load natively.
pub fn automatic_weight_quantization(model_dir: &Path) -> Result<Option<MlxWeightQuantization>> {
    if model_dir.is_file() {
        return Ok(None);
    }
    let config: Value = serde_json::from_slice(
        &fs::read(model_dir.join("config.json"))
            .map_err(|error| anyhow!("read MLX config.json: {error}"))?,
    )
    .map_err(|error| anyhow!("parse MLX config.json: {error}"))?;
    Ok(automatic_weight_quantization_for_config(&config))
}

fn automatic_weight_quantization_for_config(config: &Value) -> Option<MlxWeightQuantization> {
    let model_type = config.get("model_type").and_then(Value::as_str);
    let text_config = config.get("text_config");
    let text_model_type = text_config
        .and_then(|value| value.get("model_type"))
        .and_then(Value::as_str);
    let is_grouped_expert = |model_type: Option<&str>| {
        matches!(
            model_type,
            Some("inkling_mm_model" | "nemotron_h" | "nemotron_h_moe")
        )
    };
    let unsupported_grouped_experts =
        is_grouped_expert(model_type) || is_grouped_expert(text_model_type);
    let declares_representation = declares_weight_representation(config)
        || text_config.is_some_and(declares_weight_representation);
    (!unsupported_grouped_experts && !declares_representation).then_some(
        MlxWeightQuantization::Affine {
            group_size: 64,
            bits: 4,
        },
    )
}

fn declares_weight_representation(config: &Value) -> bool {
    ["quantization", "quantization_config", "compression_config"]
        .iter()
        .any(|key| config.get(*key).is_some_and(|value| !value.is_null()))
}

/// One chat turn, in `Send` form (no MLX types).
#[derive(Clone, Debug)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// A generation request handed to the worker.
#[derive(Debug)]
pub struct GenerateRequest {
    pub messages: Vec<ChatTurn>,
    /// If set, skip the chat template and feed this text verbatim.
    pub raw_prompt: Option<String>,
    pub max_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

/// Streamed output from the worker for one request.
#[derive(Debug)]
pub enum TokenMsg {
    Delta(String),
    Done {
        finish_reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Error(String),
}

struct Job {
    req: GenerateRequest,
    reply: mpsc::UnboundedSender<TokenMsg>,
}

/// Handle to the MLX worker thread. `Send + Sync`, safe to share in an `Arc`.
pub struct MlxEngine {
    job_tx: mpsc::UnboundedSender<Job>,
    config: MlxEngineConfig,
}

impl MlxEngine {
    /// Spawns the worker and blocks until the model has finished loading.
    /// Call from a blocking context (e.g. `tokio::task::spawn_blocking`).
    pub fn spawn(config: MlxEngineConfig) -> Result<Self> {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

        let worker_config = config.clone();
        thread::Builder::new()
            .name("mlx-engine".into())
            .spawn(move || run_worker(worker_config, job_rx, ready_tx))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self { job_tx, config }),
            Ok(Err(e)) => Err(anyhow!("MLX model load failed: {e}")),
            Err(_) => Err(anyhow!("MLX worker exited before signalling readiness")),
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

    /// Submits a request and returns the channel its tokens will stream on.
    pub fn submit(&self, req: GenerateRequest) -> mpsc::UnboundedReceiver<TokenMsg> {
        let (tx, rx) = mpsc::unbounded_channel();
        if self
            .job_tx
            .send(Job {
                req,
                reply: tx.clone(),
            })
            .is_err()
        {
            let _ = tx.send(TokenMsg::Error("MLX worker is not running".into()));
        }
        rx
    }
}

struct LoadedEngine {
    model: LoadedModel,
    stream: Stream,
    tokenizer: tokenizers::Tokenizer,
    eos: Vec<u32>,
}

fn load_engine(config: &MlxEngineConfig) -> Result<LoadedEngine> {
    // Metal GPU stream for compute, CPU stream for weight staging (goose's split).
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));

    let started = Instant::now();
    let options = config
        .weight_quantization
        .map(MlxWeightQuantization::safemlx)
        .transpose()?
        .map_or_else(
            ModelLoadOptions::default,
            ModelLoadOptions::with_quantization,
        );
    let mut used_native_fallback = false;
    let model = match LoadedModel::load_with_options(
        &config.model_dir,
        options,
        &stream,
        &weights_stream,
    ) {
        Ok(model) => model,
        Err(error)
            if config.allow_native_quantization_fallback
                && config.weight_quantization.is_some()
                && optional_quantization_incompatible(&config.model_dir, &error) =>
        {
            used_native_fallback = true;
            tracing::warn!(
                model = %config.model_id,
                %error,
                "MLX automatic quantization is incompatible; retrying native checkpoint representation"
            );
            let _ = stream.synchronize();
            LoadedModel::load(&config.model_dir, &stream, &weights_stream).map_err(|native_error| {
                anyhow!(
                    "load {} with automatic quantization failed ({error}); native fallback also failed: {native_error}",
                    config.model_dir.display()
                )
            })?
        }
        Err(error) => return Err(anyhow!("load {}: {error}", config.model_dir.display())),
    };
    stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;

    let tokenizer = tokenizers::Tokenizer::from_file(config.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
    let eos = model.eos_token_ids().to_vec();
    let quantization_label = if used_native_fallback {
        "checkpoint-fallback".to_string()
    } else {
        config
            .weight_quantization
            .map_or_else(|| "checkpoint".to_string(), MlxWeightQuantization::label)
    };

    tracing::info!(
        model = %config.model_id,
        kind = model.model_type(),
        weight_quantization = %quantization_label,
        load_secs = started.elapsed().as_secs_f64(),
        "MLX model loaded"
    );
    Ok(LoadedEngine {
        model,
        stream,
        tokenizer,
        eos,
    })
}

fn optional_quantization_incompatible(model_dir: &Path, error: &safemlx_lm::error::Error) -> bool {
    match error {
        safemlx_lm::error::Error::Quantization(_) => true,
        safemlx_lm::error::Error::StrictLoadValidation { missing, unused } => {
            known_tied_qwen_lm_head_failure(model_dir, missing, unused)
        }
        _ => false,
    }
}

fn known_tied_qwen_lm_head_failure(
    model_dir: &Path,
    missing: &[String],
    unused: &[String],
) -> bool {
    if !missing.is_empty() || unused != ["lm_head.weight"] {
        return false;
    }
    let Ok(bytes) = fs::read(model_dir.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    config.get("model_type").and_then(Value::as_str) == Some("qwen3")
        && config.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true)
}

fn run_worker(
    config: MlxEngineConfig,
    mut job_rx: mpsc::UnboundedReceiver<Job>,
    ready_tx: std::sync::mpsc::Sender<Result<(), String>>,
) {
    let mut engine = match load_engine(&config) {
        Ok(engine) => {
            let _ = ready_tx.send(Ok(()));
            engine
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e.to_string()));
            return;
        }
    };

    while let Some(job) = job_rx.blocking_recv() {
        let reply = job.reply.clone();
        if let Err(e) = generate_one(&config, &mut engine, job) {
            let _ = reply.send(TokenMsg::Error(e.to_string()));
        }
    }
}

fn build_prompt(model: &mut LoadedModel, req: &GenerateRequest) -> Result<(String, bool)> {
    if let Some(raw) = &req.raw_prompt {
        return Ok((raw.clone(), true));
    }
    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|turn| json!({"role": turn.role, "content": turn.content}))
        .collect();
    let rendered = model
        .apply_chat_template_json(vec![messages], None, true)
        .map_err(|e| anyhow!("chat template: {e}"))?;
    match rendered {
        Some(prompt) => Ok((prompt, false)),
        None => {
            let fallback = req
                .messages
                .last()
                .map(|turn| turn.content.clone())
                .unwrap_or_default();
            Ok((fallback, true))
        }
    }
}

fn generate_one(config: &MlxEngineConfig, engine: &mut LoadedEngine, job: Job) -> Result<()> {
    let LoadedEngine {
        model,
        stream,
        tokenizer,
        eos,
    } = engine;
    let reply = job.reply;

    let (prompt, add_special) = build_prompt(model, &job.req)?;
    let tokens = model
        .encode_to_array(&prompt, add_special, stream)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_tokens = tokens.shape()[1] as u32;
    let available_tokens = config.max_tokens_cap.saturating_sub(prompt_tokens as usize);
    anyhow::ensure!(
        available_tokens > 0,
        "MLX prompt uses {prompt_tokens} tokens, exceeding the {}-token context",
        config.max_tokens_cap
    );
    let max_tokens = job.req.max_tokens.min(available_tokens);

    let mut cache = model.new_cache();
    let parts = [InputPart::text_token_ids(&tokens)];
    let input = ModelInput::new(&parts);
    let mut generator = model.generate_input_with_cache_sampler(
        &mut cache,
        0.0,
        input,
        None,
        stream,
        DefaultSampler,
    );

    let mut ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut decoder = tokenizer.decode_stream(true);
    let mut emitted = String::new();
    let mut finish = FinishReason::Length;

    let mut current = generator.next().transpose().map_err(|e| anyhow!("{e}"))?;
    for index in 0..max_tokens {
        let Some(token) = current.take() else {
            finish = FinishReason::Stop;
            break;
        };

        // Start the next decode before reading this token back (mlx-lm's
        // one-token async pipeline overlaps compute with host readback).
        let next = if index + 1 < max_tokens {
            let next = generator.next();
            if let Some(Ok(next_token)) = next.as_ref() {
                async_eval([next_token]).map_err(|e| anyhow!("async_eval: {e}"))?;
            }
            next
        } else {
            None
        };

        let token_id = token.item::<u32>(&*stream);
        if eos.contains(&token_id) {
            finish = FinishReason::Stop;
            break;
        }
        ids.push(token_id);

        if let Some(delta) = decoder
            .step(token_id)
            .map_err(|error| anyhow!("decode: {error}"))?
        {
            emitted.push_str(&delta);
            if reply.send(TokenMsg::Delta(delta)).is_err() {
                return Ok(()); // client hung up
            }
        }

        if reply.is_closed() {
            return Ok(());
        }
        current = next.transpose().map_err(|e| anyhow!("{e}"))?;
    }

    let decoded = tokenizer
        .decode(&ids, true)
        .map_err(|error| anyhow!("finalize decode: {error}"))?;
    if let Some(suffix) = decoded.strip_prefix(&emitted)
        && !suffix.is_empty()
    {
        let _ = reply.send(TokenMsg::Delta(suffix.to_string()));
    }

    let _ = reply.send(TokenMsg::Done {
        finish_reason: finish,
        prompt_tokens,
        completion_tokens: ids.len() as u32,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_quantization_preserves_frontier_and_prequantized_models() {
        assert_eq!(
            automatic_weight_quantization_for_config(&json!({"model_type": "llama"})),
            Some(MlxWeightQuantization::Affine {
                group_size: 64,
                bits: 4
            })
        );
        assert_eq!(
            automatic_weight_quantization_for_config(&json!({"model_type": "inkling_mm_model"})),
            None
        );
        assert_eq!(
            automatic_weight_quantization_for_config(&json!({"model_type": "nemotron_h"})),
            None
        );
        assert_eq!(
            automatic_weight_quantization_for_config(&json!({
                "model_type": "multimodal_wrapper",
                "text_config": {"model_type": "nemotron_h"}
            })),
            None
        );
        assert_eq!(
            automatic_weight_quantization_for_config(&json!({
                "model_type": "qwen3",
                "quantization_config": {"bits": 8}
            })),
            None
        );
        assert_eq!(
            automatic_weight_quantization_for_config(&json!({
                "model_type": "qwen3_vl",
                "text_config": {"compression_config": {"format": "mxfp4"}}
            })),
            None
        );
    }

    #[test]
    fn published_mlx_lm_quantization_without_mode_is_supported() {
        let quantization = serde_json::from_value::<safemlx_lm::quantization::WeightQuantization>(
            json!({"group_size": 64, "bits": 4}),
        )
        .unwrap();
        assert_eq!(
            quantization,
            safemlx_lm::quantization::WeightQuantization::Affine(
                safemlx_lm::quantization::AffineQuantization::new(64, 4).unwrap()
            )
        );
    }

    #[test]
    fn automatic_quantization_retries_quantization_failures() {
        assert!(optional_quantization_incompatible(
            Path::new("unused"),
            &safemlx_lm::error::Error::Quantization("unsupported".to_string())
        ));
        assert!(!optional_quantization_incompatible(
            Path::new("unused"),
            &safemlx_lm::error::Error::UnsupportedArchitecture("unsupported".to_string())
        ));
    }

    #[test]
    fn automatic_quantization_retries_only_known_tied_qwen_strict_failure() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join("config.json"),
            br#"{"model_type":"qwen3","tie_word_embeddings":true}"#,
        )
        .unwrap();
        let error = safemlx_lm::error::Error::StrictLoadValidation {
            missing: Vec::new(),
            unused: vec!["lm_head.weight".to_string()],
        };
        assert!(optional_quantization_incompatible(temp.path(), &error));

        let missing_core = safemlx_lm::error::Error::StrictLoadValidation {
            missing: vec!["model.layers.0.self_attn.q_proj.weight".to_string()],
            unused: vec!["lm_head.weight".to_string()],
        };
        assert!(!optional_quantization_incompatible(
            temp.path(),
            &missing_core
        ));
        let unexpected_unused = safemlx_lm::error::Error::StrictLoadValidation {
            missing: Vec::new(),
            unused: vec![
                "lm_head.weight".to_string(),
                "unexpected.weight".to_string(),
            ],
        };
        assert!(!optional_quantization_incompatible(
            temp.path(),
            &unexpected_unused
        ));
        let shape_mismatch = safemlx_lm::error::Error::StrictLoadValidation {
            missing: vec!["model.layers.0.mlp.down_proj.weight".to_string()],
            unused: vec![
                "model.layers.0.mlp.down_proj.weight -> model.layers.0.mlp.down_proj.weight: expected [1024, 3072], got [1024, 2048]".to_string(),
            ],
        };
        assert!(!optional_quantization_incompatible(
            temp.path(),
            &shape_mismatch
        ));

        fs::write(
            temp.path().join("config.json"),
            br#"{"model_type":"llama","tie_word_embeddings":true}"#,
        )
        .unwrap();
        assert!(!optional_quantization_incompatible(temp.path(), &error));
    }
}

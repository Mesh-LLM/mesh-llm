//! Proves that two MLX stages loaded from disjoint partial SafeTensors files
//! reproduce whole-model greedy decode without a complete checkpoint file.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use safemlx::module::{Module, ModuleParameters, ModuleParametersExt};
use safemlx::ops::indexing::{NewAxis, TryIndexOp};
use safemlx::{arange, Array, Device, DeviceType, Dtype, Stream};
use safemlx_lm::cache::{ConcatKeyValueCache, KeyValueCache};
use safemlx_lm::models::common::linear::project_logits_maybe_quantized;
use safemlx_lm::models::llama::{self, AttentionInput, TransformerBlock};
use safemlx_lm::weights::{
    load_safetensors_lenient, load_safetensors_strict, StrictLoadConfig, StrictLoadReport,
};
use skippy_protocol::binary::{
    encode_f32_activation_payload, read_stage_message, write_stage_message, StageStateHeader,
    StageWireMessage, WireActivationDType, WireMessageKind,
};

#[derive(Debug, Parser)]
#[command(about = "Compare whole-model MLX with two partial SafeTensors stages")]
struct Args {
    #[arg(long)]
    stage0: PathBuf,

    #[arg(long)]
    stage1: PathBuf,

    /// First layer owned by stage 1.
    #[arg(long, default_value_t = 15)]
    split: usize,

    /// Comma-separated prompt token ids. The mesh sends token ids to stage 0.
    #[arg(long, default_value = "1,1531,314,260,3575,28")]
    tokens: String,

    /// Number of greedy decode steps to compare after prompt prefill.
    #[arg(long, default_value_t = 8)]
    steps: usize,

    /// Activation encoding used at the artificial mesh boundary.
    #[arg(long, value_enum, default_value_t = WireDtype::F16)]
    wire_dtype: WireDtype,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum WireDtype {
    F16,
    F32,
}

struct Models {
    baseline: llama::Model,
    stage0: llama::Model,
    stage1: llama::Model,
}

struct Caches {
    baseline: Vec<Option<ConcatKeyValueCache>>,
    stage0: Vec<Option<ConcatKeyValueCache>>,
    stage1: Vec<Option<ConcatKeyValueCache>>,
}

impl Caches {
    fn new() -> Self {
        Self {
            baseline: Vec::new(),
            stage0: Vec::new(),
            stage1: Vec::new(),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.steps > 0, "--steps must be positive");
    let prompt = parse_tokens(&args.tokens)?;

    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut models = load_models(&args, &stream, &weights_stream)?;
    stream.synchronize()?;

    let mut caches = Caches::new();
    let mut input = prompt;
    let mut generated = Vec::with_capacity(args.steps);
    let mut worst_logit_delta = 0.0_f32;
    let mut boundary_bytes = 0_usize;

    for step in 0..args.steps {
        let tokens = Array::from_slice(&input, &[1, input.len() as i32]);
        let (baseline_logits, baseline_boundary, baseline_final, baseline_normed) =
            baseline_forward(
                &mut models.baseline,
                &tokens,
                &mut caches.baseline,
                args.split,
                &stream,
            )?;
        let (staged_logits, staged_boundary, staged_final, staged_normed, bytes) = staged_forward(
            &mut models,
            &tokens,
            &mut caches,
            args.split,
            args.wire_dtype,
            &stream,
        )?;
        boundary_bytes += bytes;
        let boundary_delta = array_max_abs_delta(&baseline_boundary, &staged_boundary, &stream)?;
        let final_delta = array_max_abs_delta(&baseline_final, &staged_final, &stream)?;
        let norm_delta = array_max_abs_delta(&baseline_normed, &staged_normed, &stream)?;

        let baseline = last_logits(&baseline_logits, &stream)?;
        let staged = last_logits(&staged_logits, &stream)?;
        ensure!(baseline.len() == staged.len(), "logit width mismatch");
        let delta = baseline
            .iter()
            .zip(&staged)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        worst_logit_delta = worst_logit_delta.max(delta);
        let baseline_token = argmax(&baseline)?;
        let staged_token = argmax(&staged)?;
        ensure!(
            baseline_token == staged_token,
            "greedy token diverged at step {step}: baseline={baseline_token}, staged={staged_token}, boundary_delta={boundary_delta}, final_delta={final_delta}, norm_delta={norm_delta}, max_abs_logit_delta={delta}"
        );
        generated.push(staged_token);
        input = vec![staged_token];
        eprintln!(
            "step={step} token={staged_token} boundary_delta={boundary_delta:.6} max_abs_logit_delta={delta:.6} stage_wire_bytes={bytes}"
        );
    }

    println!("PASS: whole-model and two-stage greedy tokens match");
    println!("wire_dtype={:?}", args.wire_dtype);
    println!("split=0..{} | {}..30", args.split, args.split);
    println!("generated_tokens={generated:?}");
    println!("worst_max_abs_logit_delta={worst_logit_delta:.6}");
    println!("stage_wire_bytes={boundary_bytes}");
    Ok(())
}

fn parse_tokens(value: &str) -> Result<Vec<u32>> {
    let tokens = value
        .split(',')
        .map(|token| token.trim().parse::<u32>().context("parse token id"))
        .collect::<Result<Vec<_>>>()?;
    ensure!(!tokens.is_empty(), "--tokens must not be empty");
    Ok(tokens)
}

fn load_models(args: &Args, stream: &Stream, weights_stream: &Stream) -> Result<Models> {
    let model_args = llama::get_llama_model_args(&args.stage0)?;
    let total_layers = usize::try_from(model_args.num_hidden_layers)?;
    ensure!(
        args.split > 0 && args.split < total_layers,
        "--split must be inside 0..{total_layers}"
    );
    let stage0_file = weight_file(&args.stage0);
    let stage1_file = weight_file(&args.stage1);

    let mut baseline = llama::Model::new(model_args.clone(), stream)?;
    let strict = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    load_safetensors_strict(
        &mut baseline,
        &stage0_file,
        weights_stream,
        &strict,
        &mut report,
    )?;
    load_safetensors_strict(
        &mut baseline,
        &stage1_file,
        weights_stream,
        &strict,
        &mut report,
    )?;
    report.finish(&baseline, &strict)?;
    baseline.copy_to_stream(stream)?;

    let mut stage0 = llama::Model::new(model_args.clone(), stream)?;
    load_safetensors_lenient(&mut stage0, &stage0_file, weights_stream)?;
    stage0.model.layers.truncate(args.split);
    stage0.model.num_hidden_layers = i32::try_from(args.split)?;
    stage0.copy_to_stream(stream)?;

    let mut stage1 = llama::Model::new(model_args, stream)?;
    load_safetensors_lenient(&mut stage1, &stage1_file, weights_stream)?;
    for block in &mut stage1.model.layers[args.split..] {
        block.copy_to_stream(stream)?;
    }
    stage1.model.norm.copy_to_stream(stream)?;
    stage1.model.embed_tokens.copy_to_stream(stream)?;
    if let Some(lm_head) = &mut stage1.lm_head {
        lm_head.copy_to_stream(stream)?;
    }
    verify_layer_weights(&baseline, &stage1, args.split, stream)?;

    eprintln!(
        "loaded baseline from union of partial files; stage layers={}+{}",
        stage0.model.layers.len(),
        stage1.model.layers.len() - args.split
    );
    Ok(Models {
        baseline,
        stage0,
        stage1,
    })
}

fn verify_layer_weights(
    baseline: &llama::Model,
    stage1: &llama::Model,
    layer: usize,
    stream: &Stream,
) -> Result<()> {
    let baseline_parameters = baseline.parameters().flatten();
    let stage_parameters = stage1.parameters().flatten();
    let prefix = format!("model.layers.{layer}.");
    let mut checked = 0_usize;
    for (name, baseline_value) in baseline_parameters
        .iter()
        .filter(|(name, _)| name.starts_with(&prefix))
    {
        let stage_value = stage_parameters
            .get(name)
            .with_context(|| format!("stage 1 parameter {name} is absent"))?;
        let delta = array_max_abs_delta(baseline_value, stage_value, stream)?;
        ensure!(delta == 0.0, "stage 1 parameter {name} differs by {delta}");
        checked += 1;
    }
    ensure!(checked > 0, "no parameters found for {prefix}");
    eprintln!("verified {checked} layer-{layer} stage parameters against baseline");
    Ok(())
}

fn weight_file(directory: &Path) -> PathBuf {
    directory.join("model.safetensors")
}

fn baseline_forward(
    model: &mut llama::Model,
    tokens: &Array,
    cache: &mut Vec<Option<ConcatKeyValueCache>>,
    split: usize,
    stream: &Stream,
) -> Result<(Array, Array, Array, Array)> {
    let hidden = model.model.embed_tokens.forward(tokens, stream)?;
    let mask = attention_mask(&hidden, cache, stream)?;
    if cache.is_empty() {
        *cache = (0..model.model.layers.len())
            .map(|_| Some(ConcatKeyValueCache::default()))
            .collect();
    }
    let (first_layers, last_layers) = model.model.layers.split_at_mut(split);
    let (first_cache, last_cache) = cache.split_at_mut(split);
    let hidden = forward_block_slice(first_layers, hidden, mask.as_ref(), first_cache, stream)?;
    let boundary = hidden.clone();
    let hidden = forward_block_slice(last_layers, hidden, mask.as_ref(), last_cache, stream)?;
    let final_hidden = hidden.clone();
    let hidden = model.model.norm.forward(&hidden, stream)?;
    let normed = hidden.clone();
    let logits = project_logits_maybe_quantized(
        &mut model.lm_head,
        &mut model.model.embed_tokens,
        &hidden,
        stream,
    )?;
    Ok((logits, boundary, final_hidden, normed))
}

fn staged_forward(
    models: &mut Models,
    tokens: &Array,
    caches: &mut Caches,
    split: usize,
    wire_dtype: WireDtype,
    stream: &Stream,
) -> Result<(Array, Array, Array, Array, usize)> {
    let mut hidden = models.stage0.model.embed_tokens.forward(tokens, stream)?;
    let mask = attention_mask(&hidden, &caches.stage0, stream)?;
    hidden = forward_blocks(
        &mut models.stage0.model.layers,
        hidden,
        mask.as_ref(),
        &mut caches.stage0,
        stream,
    )?;
    let stage0_boundary = hidden.clone();
    let (hidden, boundary_bytes) = wire_roundtrip(&hidden, wire_dtype, stream)?;
    let hidden = forward_blocks(
        &mut models.stage1.model.layers[split..],
        hidden,
        mask.as_ref(),
        &mut caches.stage1,
        stream,
    )?;
    let final_hidden = hidden.clone();
    let hidden = models.stage1.model.norm.forward(&hidden, stream)?;
    let normed = hidden.clone();
    let logits = project_logits_maybe_quantized(
        &mut models.stage1.lm_head,
        &mut models.stage1.model.embed_tokens,
        &hidden,
        stream,
    )?;
    Ok((
        logits,
        stage0_boundary,
        final_hidden,
        normed,
        boundary_bytes,
    ))
}

fn attention_mask(
    hidden: &Array,
    cache: &[Option<ConcatKeyValueCache>],
    stream: &Stream,
) -> Result<Option<Array>> {
    let sequence = hidden.shape()[1];
    if sequence == 1 {
        return Ok(None);
    }
    let offset = cache
        .first()
        .and_then(Option::as_ref)
        .map_or(0, KeyValueCache::offset);
    let right = arange!(stop = offset + sequence, stream = stream)?;
    let left = arange!(start = offset, stop = offset + sequence, stream = stream)?;
    let left = left.try_index_device((.., NewAxis), stream)?;
    let right = right.try_index_device(NewAxis, stream)?;
    Ok(Some(left.ge(&right, stream)?))
}

fn forward_blocks(
    blocks: &mut [TransformerBlock],
    hidden: Array,
    mask: Option<&Array>,
    cache: &mut Vec<Option<ConcatKeyValueCache>>,
    stream: &Stream,
) -> Result<Array> {
    if cache.is_empty() {
        *cache = (0..blocks.len())
            .map(|_| Some(ConcatKeyValueCache::default()))
            .collect();
    }
    ensure!(cache.len() == blocks.len(), "stage cache length mismatch");
    forward_block_slice(blocks, hidden, mask, cache, stream)
}

fn forward_block_slice(
    blocks: &mut [TransformerBlock],
    mut hidden: Array,
    mask: Option<&Array>,
    cache: &mut [Option<ConcatKeyValueCache>],
    stream: &Stream,
) -> Result<Array> {
    ensure!(cache.len() == blocks.len(), "stage cache length mismatch");
    for (block, layer_cache) in blocks.iter_mut().zip(cache.iter_mut()) {
        hidden = block.forward(
            AttentionInput {
                x: &hidden,
                mask,
                cache: layer_cache.as_mut(),
                generated_sliding_window: None,
            },
            stream,
        )?;
    }
    Ok(hidden)
}

fn wire_roundtrip(
    hidden: &Array,
    wire_dtype: WireDtype,
    stream: &Stream,
) -> Result<(Array, usize)> {
    let shape = hidden.shape().to_vec();
    let compute_dtype = hidden.dtype();
    let token_count = shape[1];
    let hidden_width = shape[2];
    let f32_values = hidden
        .as_dtype(Dtype::Float32, stream)?
        .evaluated()?
        .as_slice::<f32>()
        .to_vec();
    let f32_bytes = f32_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let protocol_dtype = match wire_dtype {
        WireDtype::F16 => WireActivationDType::F16,
        WireDtype::F32 => WireActivationDType::F32,
    };
    let activation =
        encode_f32_activation_payload(protocol_dtype, token_count, hidden_width, &f32_bytes)?;
    let kind = if token_count > 1 {
        WireMessageKind::PrefillEmbd
    } else {
        WireMessageKind::DecodeEmbd
    };
    let mut state = StageStateHeader::new(kind, protocol_dtype);
    state.source_stage_index = 0;
    let message = StageWireMessage {
        kind,
        pos_start: 0,
        token_count,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation,
        raw_bytes: Vec::new(),
    };
    let mut frame = Vec::new();
    write_stage_message(&mut frame, &message, protocol_dtype)?;
    let decoded = read_stage_message(Cursor::new(&frame), hidden_width)?;
    let decoded_f32 = decoded.activation_f32_payload(hidden_width)?;
    let values = decoded_f32
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    let restored = Array::from_slice(&values, &shape).as_dtype(compute_dtype, stream)?;
    Ok((restored, frame.len()))
}

fn last_logits(logits: &Array, stream: &Stream) -> Result<Vec<f32>> {
    let row = logits
        .try_index_device((0, -1, ..), stream)?
        .as_dtype(Dtype::Float32, stream)?;
    Ok(row.evaluated()?.as_slice::<f32>().to_vec())
}

fn array_max_abs_delta(left: &Array, right: &Array, stream: &Stream) -> Result<f32> {
    ensure!(left.shape() == right.shape(), "activation shape mismatch");
    let left = left.as_dtype(Dtype::Float32, stream)?;
    let right = right.as_dtype(Dtype::Float32, stream)?;
    Ok(left
        .evaluated()?
        .as_slice::<f32>()
        .iter()
        .zip(right.evaluated()?.as_slice::<f32>())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max))
}

fn argmax(values: &[f32]) -> Result<u32> {
    let Some((index, _)) = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
    else {
        bail!("cannot argmax empty logits")
    };
    Ok(u32::try_from(index)?)
}

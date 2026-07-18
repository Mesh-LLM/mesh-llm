//! Spike: goose-style solo MLX serving via safemlx-lm.
//!
//! Proves the Phase-2 workflow claim from docs/design/MLX_STAGE_ENGINE_PLAN.md:
//! load an HF safetensors model in Rust, optionally JIT-quantize on load, and
//! generate tokens — with no GGUF and no ahead-of-time quant step.
//!
//! CPU-only (accelerate) because this box has no Metal shader compiler; the
//! generation *path* is what we are validating here, not Metal throughput.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use safemlx::transforms::async_eval;
use safemlx::{Device, DeviceType, Stream};
use safemlx_lm::models::input::{InputPart, ModelInput};
use safemlx_lm::models::{LoadedModel, ModelLoadOptions};
use safemlx_lm::quantization::AffineQuantization;
use safemlx_lm::sampler::DefaultSampler;

#[derive(Parser, Debug)]
#[command(about = "Solo MLX serving spike (load + optional JIT quant + generate)")]
struct Cli {
    /// Model directory (HF safetensors) or GGUF file.
    #[arg(short, long)]
    model: PathBuf,

    /// Prompt text.
    #[arg(default_value = "Explain what a mesh network is in two sentences.")]
    prompt: String,

    /// Max tokens to generate.
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,

    /// JIT-quantize eligible dense weights to this bit width on load (e.g. 4, 8).
    /// Omit to load at source precision.
    #[arg(short, long)]
    quantize: Option<i32>,

    /// Group size for affine quantization.
    #[arg(long, default_value_t = 64)]
    quant_group_size: i32,

    /// Skip the chat template and feed the prompt raw.
    #[arg(long)]
    raw: bool,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // Mirror goose (crates/goose-local-inference/src/mlx.rs): Metal GPU stream for
    // compute, CPU stream for weight staging.
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));

    let load_options = match args.quantize {
        Some(bits) => {
            eprintln!("[load] JIT affine quantization: {bits}-bit, group_size={}", args.quant_group_size);
            ModelLoadOptions::with_quantization(AffineQuantization::new(args.quant_group_size, bits)?)
        }
        None => {
            eprintln!("[load] no quantization (source precision)");
            ModelLoadOptions::default()
        }
    };

    eprintln!("[load] loading {} ...", args.model.display());
    let load_started = Instant::now();
    let mut model = LoadedModel::load_with_options(&args.model, load_options, &stream, &weights_stream)
        .with_context(|| format!("failed to load model from {}", args.model.display()))?;
    stream.synchronize()?;
    let load_elapsed = load_started.elapsed();
    eprintln!(
        "[load] ok: model_type={} in {:.2}s",
        model.model_type(),
        load_elapsed.as_secs_f64()
    );

    let (rendered, add_special) = if args.raw {
        (args.prompt.clone(), true)
    } else {
        match model.apply_chat_template_json(
            vec![vec![serde_json::json!({"role": "user", "content": args.prompt})]],
            None,
            true,
        )? {
            Some(rendered) => (rendered, false),
            None => {
                eprintln!("[prompt] no chat template; feeding raw");
                (args.prompt.clone(), true)
            }
        }
    };

    let tokens = model.encode_to_array(&rendered, add_special, &stream)?;
    let prompt_len = tokens.shape()[1];
    if prompt_len == 0 {
        bail!("prompt produced no tokens");
    }
    eprintln!("[prompt] {prompt_len} tokens");

    let eos = model.eos_token_ids().to_vec();
    let mut cache = model.new_cache();

    let mut output_ids: Vec<u32> = Vec::with_capacity(args.max_tokens);
    let gen_started = Instant::now();
    let mut ttft = None;

    {
        let parts = [InputPart::text_token_ids(&tokens)];
        let input = ModelInput::new(&parts);
        // Greedy sampling (temp=0.0, no prng key) keeps the spike deterministic.
        let mut generator =
            model.generate_input_with_cache_sampler(&mut cache, 0.0, input, None, &stream, DefaultSampler);

        let mut current = generator.next().transpose()?;
        for index in 0..args.max_tokens {
            let Some(token) = current.take() else { break };

            // Kick off the next decode before reading this token back (mlx-lm's
            // one-token async pipeline: overlaps compute with host readback).
            let next = if index + 1 < args.max_tokens {
                let next = generator.next();
                if let Some(Ok(next_token)) = next.as_ref() {
                    async_eval([next_token])?;
                }
                next
            } else {
                None
            };

            let token_id = token.item::<u32>(&stream);
            if ttft.is_none() {
                ttft = Some(gen_started.elapsed());
            }
            output_ids.push(token_id);
            if eos.contains(&token_id) {
                break;
            }
            current = next.transpose()?;
        }
    }

    let gen_elapsed = gen_started.elapsed();
    let text = model.decode(&output_ids, true)?;

    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "\n===== OUTPUT =====\n{text}\n==================")?;

    let decode_tokens = output_ids.len().saturating_sub(1);
    let decode_elapsed = gen_elapsed.saturating_sub(ttft.unwrap_or_default());
    let decode_rate = if decode_elapsed.is_zero() {
        0.0
    } else {
        decode_tokens as f64 / decode_elapsed.as_secs_f64()
    };
    eprintln!(
        "[stats] load={:.2}s prompt_tokens={} generated={} ttft={:.3}s decode={:.1} tok/s",
        load_elapsed.as_secs_f64(),
        prompt_len,
        output_ids.len(),
        ttft.map(|d| d.as_secs_f64()).unwrap_or(0.0),
        decode_rate,
    );

    Ok(())
}

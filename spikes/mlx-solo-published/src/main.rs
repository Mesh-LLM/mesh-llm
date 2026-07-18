//! Fork-free solo MLX serving: depends only on crates.io safemlx-lm 0.4.1.
//!
//! Proves the goose-baseline path (load + generate, no JIT quant) works with no
//! fork. The published crate exposes `LoadedModel::load` but NOT the
//! JIT-quant-on-load API (`ModelLoadOptions` / `with_quantization`), which lives
//! only in the fork's unpublished HEAD — so this binary intentionally has no
//! --quantize flag. It serves source-precision or already-quantized MLX repos.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use safemlx::transforms::async_eval;
use safemlx::{Device, DeviceType, Stream};
use safemlx_lm::models::input::{InputPart, ModelInput};
use safemlx_lm::models::LoadedModel;
use safemlx_lm::sampler::DefaultSampler;

#[derive(Parser, Debug)]
#[command(about = "Fork-free solo MLX serving (crates.io only; load + generate)")]
struct Cli {
    /// Model directory (HF safetensors, source-precision or pre-quantized MLX).
    #[arg(short, long)]
    model: PathBuf,

    /// Prompt text.
    #[arg(default_value = "Explain what a mesh network is in two sentences.")]
    prompt: String,

    /// Max tokens to generate.
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,

    /// Skip the chat template and feed the prompt raw.
    #[arg(long)]
    raw: bool,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // Mirror goose: Metal GPU stream for compute, CPU stream for weight staging.
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));

    eprintln!("[load] loading {} ...", args.model.display());
    let load_started = Instant::now();
    let mut model = LoadedModel::load(&args.model, &stream, &weights_stream)
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
        let mut generator =
            model.generate_input_with_cache_sampler(&mut cache, 0.0, input, None, &stream, DefaultSampler);

        let mut current = generator.next().transpose()?;
        for index in 0..args.max_tokens {
            let Some(token) = current.take() else { break };

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

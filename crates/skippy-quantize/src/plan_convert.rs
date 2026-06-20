use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::gguf_template::metadata_from_hf_config;
use crate::gguf_writer::{
    RawGgufValidation, RawGgufWriteOptions, TensorSelection, validate_raw_safetensors_gguf,
    write_raw_safetensors_gguf,
};
use crate::hf_checkpoint::{
    HfCheckpointPlan, inspect_hf_checkpoint, verify_hf_checkpoint_tensor_streams,
};
use crate::memory_budget::MemorySize;
use crate::tensor_map::TensorNameMap;
use crate::tokenizer_metadata::ensure_native_tokenizer_metadata_supported;
use crate::types::ConvertOutputType;

#[derive(Debug, Parser)]
pub(crate) struct PlanConvertArgs {
    source: PathBuf,
    #[arg(long)]
    max_memory: Option<MemorySize>,
    #[arg(long, default_value_t = 0.60)]
    staging_fraction: f64,
    #[arg(long)]
    verify_streaming: bool,
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    stream_buffer_bytes: usize,
    #[arg(long)]
    write_raw_gguf: Option<PathBuf>,
    #[arg(long)]
    hf_config_metadata: bool,
    #[arg(long)]
    validate_native: bool,
    #[arg(long, value_enum, default_value_t = ConvertOutputType::Bf16)]
    output_type: ConvertOutputType,
    #[arg(long)]
    json: bool,
}

pub(crate) fn run_plan_convert(args: PlanConvertArgs) -> Result<()> {
    let mut plan = inspect_hf_checkpoint(&args.source, args.max_memory, args.staging_fraction)?;
    let mut native_validation = None;
    if args.verify_streaming {
        plan.stream_verification = Some(verify_hf_checkpoint_tensor_streams(
            &args.source,
            args.stream_buffer_bytes,
        )?);
    }
    if args.validate_native {
        ensure_native_tokenizer_metadata_supported(&args.source)?;
        let metadata = metadata_from_hf_config(&args.source, plan.tensor_count)?;
        native_validation = Some(validate_raw_safetensors_gguf(
            &args.source,
            RawGgufWriteOptions {
                buffer_size: args.stream_buffer_bytes,
                metadata: Some(metadata),
                tensor_name_map: TensorNameMap::HfToGguf,
                split: None,
                output_type: Some(args.output_type),
                tensor_selection: TensorSelection::All,
            },
        )?);
    }
    if let Some(output) = args.write_raw_gguf.as_deref() {
        let metadata = if args.hf_config_metadata {
            ensure_native_tokenizer_metadata_supported(&args.source)?;
            Some(metadata_from_hf_config(&args.source, plan.tensor_count)?)
        } else {
            None
        };
        write_raw_safetensors_gguf(
            &args.source,
            output,
            RawGgufWriteOptions {
                buffer_size: args.stream_buffer_bytes,
                metadata,
                tensor_name_map: if args.hf_config_metadata {
                    TensorNameMap::HfToGguf
                } else {
                    TensorNameMap::Raw
                },
                split: None,
                output_type: args.hf_config_metadata.then_some(args.output_type),
                tensor_selection: TensorSelection::All,
            },
        )?;
    }
    if args.json {
        if let Some(native_validation) = native_validation {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "plan": plan,
                    "native_validation": native_validation,
                }))?
            );
        } else {
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
    } else {
        print_plan(&plan);
        if let Some(native_validation) = native_validation {
            print_native_validation(&native_validation);
        }
    }
    Ok(())
}

fn print_plan(plan: &HfCheckpointPlan) {
    println!("checkpoint={}", plan.source.display());
    println!("safetensors={}", plan.safetensor_count);
    println!("tensors={}", plan.tensor_count);
    println!("total_tensor_bytes={}", plan.total_tensor_bytes);
    println!("largest_tensor_bytes={}", plan.largest_tensor_bytes);
    println!("source_windows={}", plan.source_windows.len());
    for window in &plan.source_windows {
        println!(
            "window={} files={} bytes={}",
            window.index,
            window.files.len(),
            window.total_tensor_bytes
        );
    }
    if let Some(verification) = &plan.stream_verification {
        println!(
            "stream_verified=true tensors={} bytes={} buffer_size={}",
            verification.tensor_count, verification.streamed_bytes, verification.buffer_size
        );
    }
}

fn print_native_validation(validation: &RawGgufValidation) {
    println!("native_valid=true");
    println!(
        "native_selected_tensors={}",
        validation.selected_tensor_count
    );
    println!(
        "native_selected_tensor_bytes={}",
        validation.selected_tensor_bytes
    );
    println!("native_metadata_entries={}", validation.metadata_count);
    if let Some(output_type) = validation.output_type.as_deref() {
        println!("native_output_type={output_type}");
    }
}

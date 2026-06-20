use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::ConvertRunnerArgs;
use crate::backend::BackendRunStatus;
use crate::gguf_template::{
    MetadataOptions, metadata_from_hf_config_with_options, mtp_layer_start_from_hf_config,
};
use crate::gguf_writer::{
    GgufSplit, RawGgufWriteOptions, TensorSelection, write_raw_safetensors_gguf,
};
use crate::hf_checkpoint::{inspect_hf_checkpoint, resolve_auto_output_type};
use crate::manifest::Manifest;
use crate::memory_budget::{
    effective_stream_buffer_bytes, enforce_memory_budget, native_convert_stream_working_set_bytes,
};
use crate::splits::SplitWindow;
use crate::tensor_map::TensorNameMap;
use crate::tokenizer_metadata::ensure_native_tokenizer_metadata_supported;

pub(crate) fn build_native_convert_command(
    runner: &ConvertRunnerArgs,
    manifest: &Manifest,
    output_prefix: &Path,
    window: SplitWindow,
) -> Vec<String> {
    let mut command = vec![
        "skippy-quantize".to_string(),
        "run-convert-window".to_string(),
        "--backend".to_string(),
        "native-rust".to_string(),
        "--source".to_string(),
        manifest.source.display().to_string(),
        "--outfile".to_string(),
        output_prefix.display().to_string(),
        "--first-split".to_string(),
        window.first_split.to_string(),
        "--last-split".to_string(),
        window.last_split.to_string(),
        "--expected-splits".to_string(),
        manifest.expected_splits.to_string(),
    ];
    if runner.no_mtp {
        command.push("--no-mtp".to_string());
    }
    if runner.mtp {
        command.push("--mtp".to_string());
    }
    command
}

pub(crate) fn run_native_convert(
    runner: &ConvertRunnerArgs,
    manifest: &Manifest,
    window: SplitWindow,
    output_prefix: &Path,
) -> Result<BackendRunStatus> {
    if runner.dry_run {
        return Ok(BackendRunStatus::from_code(0));
    }
    ensure!(
        window.first_split <= window.last_split,
        "invalid native convert window {}..{}",
        window.first_split,
        window.last_split
    );
    ensure!(
        window.last_split <= manifest.expected_splits,
        "native convert window ends after expected split count"
    );
    let buffer_size = effective_stream_buffer_bytes(runner.stream_buffer_bytes, runner.max_memory)?;
    let output_type = manifest
        .output_type
        .map(|output_type| resolve_auto_output_type(&manifest.source, output_type))
        .transpose()?;
    let estimated_stream_working_set_bytes =
        native_convert_stream_working_set_bytes(buffer_size, output_type)?;
    enforce_memory_budget(
        "native_convert_stream_buffers",
        estimated_stream_working_set_bytes,
        runner.max_memory,
        runner.memory_policy,
    )?;
    let plan = inspect_hf_checkpoint(&manifest.source, runner.max_memory, 0.60)?;
    ensure_native_tokenizer_metadata_supported(&manifest.source)?;
    let mtp_layer_start = mtp_layer_start_from_hf_config(&manifest.source)?;
    let tensor_selection = native_tensor_selection(runner, mtp_layer_start)?;
    let tensor_name_map = native_tensor_name_map(mtp_layer_start);
    for split_index in window.first_split..=window.last_split {
        let output = output_shard_path(output_prefix, split_index, manifest.expected_splits)?;
        println!(
            "native_convert_write split={} total={} output={} buffer_size={} estimated_stream_working_set_bytes={}",
            split_index,
            manifest.expected_splits,
            output.display(),
            buffer_size,
            estimated_stream_working_set_bytes
        );
        let metadata = metadata_from_hf_config_with_options(
            &manifest.source,
            plan.tensor_count,
            MetadataOptions {
                include_mtp: !runner.no_mtp,
            },
        )?;
        write_raw_safetensors_gguf(
            &manifest.source,
            &output,
            RawGgufWriteOptions {
                buffer_size,
                metadata: Some(metadata),
                tensor_name_map,
                split: split_for(split_index, manifest.expected_splits),
                output_type,
                tensor_selection,
            },
        )?;
    }
    Ok(BackendRunStatus::from_code(0))
}

fn native_tensor_selection(
    runner: &ConvertRunnerArgs,
    mtp_layer_start: Option<u32>,
) -> Result<TensorSelection> {
    if runner.no_mtp {
        let layer_start = mtp_layer_start
            .context("--no-mtp requested but config.json does not declare MTP layers")?;
        return Ok(TensorSelection::ExcludeMtp { layer_start });
    }
    if runner.mtp {
        let layer_start = mtp_layer_start
            .context("--mtp requested but config.json does not declare MTP layers")?;
        return Ok(TensorSelection::MtpOnly { layer_start });
    }
    Ok(TensorSelection::All)
}

fn native_tensor_name_map(mtp_layer_start: Option<u32>) -> TensorNameMap {
    mtp_layer_start.map_or(TensorNameMap::HfToGguf, |layer_start| {
        TensorNameMap::HfToGgufWithMtp { layer_start }
    })
}

fn split_for(split_index: u32, split_count: u32) -> Option<GgufSplit> {
    (split_count > 1).then_some(GgufSplit {
        split_index,
        split_count,
    })
}

fn output_shard_path(output_prefix: &Path, split_index: u32, split_count: u32) -> Result<PathBuf> {
    ensure!(split_count > 0, "split_count must be greater than zero");
    ensure!(split_index > 0, "split_index must be greater than zero");
    ensure!(
        split_index <= split_count,
        "split_index {} exceeds split_count {}",
        split_index,
        split_count
    );
    if split_count == 1 {
        return Ok(output_prefix.to_path_buf());
    }
    let file_name = output_prefix
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("invalid output prefix {}", output_prefix.display()))?;
    let base = file_name.strip_suffix(".gguf").with_context(|| {
        format!(
            "output prefix must end in .gguf: {}",
            output_prefix.display()
        )
    })?;
    Ok(output_prefix.with_file_name(format!("{base}-{split_index:05}-of-{split_count:05}.gguf")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_split_output_path_from_prefix() {
        assert_eq!(
            output_shard_path(Path::new("/out/model.gguf"), 2, 7).unwrap(),
            PathBuf::from("/out/model-00002-of-00007.gguf")
        );
        assert_eq!(
            output_shard_path(Path::new("/out/model.gguf"), 1, 1).unwrap(),
            PathBuf::from("/out/model.gguf")
        );
    }
}

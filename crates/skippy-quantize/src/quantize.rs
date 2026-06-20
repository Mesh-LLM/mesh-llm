use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};

use crate::QuantRunnerArgs;
use crate::manifest::Manifest;
use crate::splits::SplitWindow;
use crate::types::TensorType;

pub fn build_quantize_command(
    args: &QuantRunnerArgs,
    manifest: &Manifest,
    staged_first_shard: &Path,
    output_prefix: &Path,
    window: SplitWindow,
) -> Result<Vec<String>> {
    let quant = manifest
        .quant
        .as_deref()
        .context("quantize manifest is missing quant type")?;
    let llama_quantize = args
        .llama_quantize
        .as_deref()
        .context("--llama-quantize is required for external quantization backend")?;
    let mut command = vec![llama_quantize.display().to_string()];
    if args.allow_requantize {
        command.push("--allow-requantize".to_string());
    }
    if args.pure {
        command.push("--pure".to_string());
    }
    if let Some(imatrix) = args.imatrix.as_deref() {
        command.push("--imatrix".to_string());
        command.push(imatrix.display().to_string());
    }
    ensure!(
        args.include_weights.is_empty() || args.exclude_weights.is_empty(),
        "--include-weights and --exclude-weights cannot be used together"
    );
    push_repeated_option(&mut command, "--include-weights", &args.include_weights);
    push_repeated_option(&mut command, "--exclude-weights", &args.exclude_weights);
    push_optional_tensor_type(
        &mut command,
        "--output-tensor-type",
        &args.output_tensor_type,
    )?;
    push_optional_tensor_type(
        &mut command,
        "--token-embedding-type",
        &args.token_embedding_type,
    )?;
    for entry in &args.tensor_type {
        let normalized_entry = normalize_tensor_type_entry(entry)?;
        command.push("--tensor-type".to_string());
        command.push(normalized_entry);
    }
    command.push("--keep-split".to_string());
    if args.dry_run {
        command.push("--dry-run".to_string());
    }
    if args.leave_output_tensor {
        command.push("--leave-output-tensor".to_string());
    }
    if let Some(tensor_type_file) = manifest.tensor_type_file.as_deref() {
        command.push("--tensor-type-file".to_string());
        command.push(tensor_type_file.display().to_string());
    }
    if let Some(prune_layers) = args.prune_layers.as_deref() {
        command.push("--prune-layers".to_string());
        command.push(prune_layers.to_string());
    }
    push_repeated_option(&mut command, "--override-kv", &args.override_kv);
    command.extend([
        "--first-split".to_string(),
        window.first_split.to_string(),
        "--last-split".to_string(),
        window.last_split.to_string(),
        staged_first_shard.display().to_string(),
        output_prefix.display().to_string(),
        quant.to_string(),
    ]);
    if let Some(nthreads) = args.nthreads {
        command.push(nthreads.to_string());
    }
    Ok(command)
}

pub fn ensure_tensor_type_entry(token: &str) -> Result<()> {
    normalize_tensor_type_entry(token).map(|_| ())
}

pub fn normalize_tensor_type_entry(token: &str) -> Result<String> {
    let (name, raw_type) = token
        .split_once('=')
        .ok_or_else(|| anyhow!("malformed tensor type entry {token:?}"))?;
    ensure!(!name.is_empty(), "tensor type entry has empty tensor name");
    ensure_raw_tensor_type(raw_type).with_context(|| {
        format!("unsupported raw ggml tensor type {raw_type:?} in entry {token:?}")
    })?;
    Ok(format!("{}={raw_type}", name.to_ascii_lowercase()))
}

fn push_repeated_option(command: &mut Vec<String>, option: &str, values: &[String]) {
    for value in values {
        command.push(option.to_string());
        command.push(value.clone());
    }
}

fn push_optional_tensor_type(
    command: &mut Vec<String>,
    option: &str,
    raw_type: &Option<String>,
) -> Result<()> {
    if let Some(raw_type) = raw_type.as_deref() {
        ensure_raw_tensor_type(raw_type)?;
        command.push(option.to_string());
        command.push(raw_type.to_string());
    }
    Ok(())
}

fn ensure_raw_tensor_type(raw_type: &str) -> Result<()> {
    ensure!(
        TensorType::parse(raw_type).is_some(),
        "unsupported raw ggml tensor type {raw_type:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::MANIFEST_VERSION;
    use crate::manifest::Manifest;
    use crate::splits::SplitWindow;
    use crate::types::{ConvertOutputType, JobKind};

    use super::*;

    #[test]
    fn builds_quant_window_command() {
        let manifest = Manifest {
            schema_version: MANIFEST_VERSION,
            kind: JobKind::QuantizeGguf,
            source: PathBuf::from("/source"),
            source_prefix: Some("BF16".to_string()),
            target: PathBuf::from("/target"),
            target_prefix: "Q2_K".to_string(),
            output_basename: "out".to_string(),
            expected_splits: 4,
            window_size: 2,
            quant: Some("Q2_K".to_string()),
            output_type: None,
            tensor_type_file: Some(PathBuf::from("/recipe/tensors.txt")),
        };
        let args = crate::QuantRunnerArgs {
            backend: crate::BackendKind::ExternalProcess,
            llama_quantize: Some(PathBuf::from("/bin/llama-quantize")),
            native_runtime_libraries: Vec::new(),
            work_dir: PathBuf::from("/tmp/work"),
            print_only: false,
            dry_run: true,
            allow_requantize: true,
            pure: true,
            imatrix: Some(PathBuf::from("/recipe/imatrix.dat")),
            include_weights: vec!["blk.1".to_string()],
            exclude_weights: Vec::new(),
            output_tensor_type: Some("Q8_0".to_string()),
            token_embedding_type: Some("F16".to_string()),
            tensor_type: vec!["MTP_Head.Weight=NVFP4".to_string()],
            prune_layers: Some("1,2".to_string()),
            override_kv: vec!["general.name=str:test".to_string()],
            nthreads: Some(8),
            leave_output_tensor: true,
            no_stage_source: false,
            keep_staged_source: false,
            spool_dir: None,
            keep_spool: false,
            watchdog_seconds: None,
            max_memory: None,
            memory_policy: crate::memory_budget::MemoryPolicy::Hard,
            record_dir: None,
        };
        let command = build_quantize_command(
            &args,
            &manifest,
            Path::new("/tmp/work/source-window/BF16/model-00001-of-00004.gguf"),
            Path::new("/target/Q2_K/out.gguf"),
            SplitWindow {
                first_split: 3,
                last_split: 4,
            },
        )
        .unwrap();

        assert_eq!(
            command,
            vec![
                "/bin/llama-quantize",
                "--allow-requantize",
                "--pure",
                "--imatrix",
                "/recipe/imatrix.dat",
                "--include-weights",
                "blk.1",
                "--output-tensor-type",
                "Q8_0",
                "--token-embedding-type",
                "F16",
                "--tensor-type",
                "mtp_head.weight=NVFP4",
                "--keep-split",
                "--dry-run",
                "--leave-output-tensor",
                "--tensor-type-file",
                "/recipe/tensors.txt",
                "--prune-layers",
                "1,2",
                "--override-kv",
                "general.name=str:test",
                "--first-split",
                "3",
                "--last-split",
                "4",
                "/tmp/work/source-window/BF16/model-00001-of-00004.gguf",
                "/target/Q2_K/out.gguf",
                "Q2_K",
                "8",
            ]
        );
    }

    #[test]
    fn rejects_conflicting_imatrix_filters() {
        let manifest = Manifest {
            schema_version: MANIFEST_VERSION,
            kind: JobKind::QuantizeGguf,
            source: PathBuf::from("/source"),
            source_prefix: Some("BF16".to_string()),
            target: PathBuf::from("/target"),
            target_prefix: "Q2_K".to_string(),
            output_basename: "out".to_string(),
            expected_splits: 1,
            window_size: 1,
            quant: Some("Q2_K".to_string()),
            output_type: Some(ConvertOutputType::Bf16),
            tensor_type_file: None,
        };
        let args = crate::QuantRunnerArgs {
            backend: crate::BackendKind::ExternalProcess,
            llama_quantize: Some(PathBuf::from("/bin/llama-quantize")),
            native_runtime_libraries: Vec::new(),
            work_dir: PathBuf::from("/tmp/work"),
            print_only: false,
            dry_run: false,
            allow_requantize: false,
            pure: false,
            imatrix: None,
            include_weights: vec!["blk.1".to_string()],
            exclude_weights: vec!["blk.2".to_string()],
            output_tensor_type: None,
            token_embedding_type: None,
            tensor_type: Vec::new(),
            prune_layers: None,
            override_kv: Vec::new(),
            nthreads: None,
            leave_output_tensor: true,
            no_stage_source: false,
            keep_staged_source: false,
            spool_dir: None,
            keep_spool: false,
            watchdog_seconds: None,
            max_memory: None,
            memory_policy: crate::memory_budget::MemoryPolicy::Hard,
            record_dir: None,
        };

        assert!(
            build_quantize_command(
                &args,
                &manifest,
                Path::new("/source/BF16/model-00001-of-00001.gguf"),
                Path::new("/target/Q2_K/out.gguf"),
                SplitWindow {
                    first_split: 1,
                    last_split: 1,
                },
            )
            .is_err()
        );
    }
}

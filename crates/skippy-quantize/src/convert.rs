use std::path::Path;

use anyhow::{Context, Result};

use crate::ConvertRunnerArgs;
use crate::manifest::Manifest;
use crate::splits::SplitWindow;

pub fn build_convert_command(
    args: &ConvertRunnerArgs,
    manifest: &Manifest,
    output_prefix: &Path,
    window: SplitWindow,
) -> Result<Vec<String>> {
    let output_type = manifest
        .output_type
        .context("convert manifest is missing output_type")?;
    let converter = args
        .converter
        .as_deref()
        .context("--converter is required for external conversion backend")?;
    let mut command = vec![
        args.python.clone(),
        converter.display().to_string(),
        "--outtype".to_string(),
        output_type.as_arg().to_string(),
        "--split-max-size".to_string(),
        args.split_max_size.clone(),
        "--outfile".to_string(),
        output_prefix.display().to_string(),
    ];
    push_flag(&mut command, args.vocab_only, "--vocab-only");
    push_flag(&mut command, args.bigendian, "--bigendian");
    push_flag(&mut command, args.use_temp_file, "--use-temp-file");
    push_flag(&mut command, args.no_lazy, "--no-lazy");
    push_optional_value(&mut command, "--model-name", args.model_name.as_deref());
    push_flag(&mut command, args.verbose, "--verbose");
    if let Some(split_max_tensors) = args.split_max_tensors {
        command.push("--split-max-tensors".to_string());
        command.push(split_max_tensors.to_string());
    }
    if window.first_split > 1 {
        command.push("--skip-output-shards-before".to_string());
        command.push(window.first_split.to_string());
    }
    command.push("--stop-output-shards-after".to_string());
    command.push(window.last_split.to_string());
    push_flag(
        &mut command,
        args.no_tensor_first_split,
        "--no-tensor-first-split",
    );
    push_optional_path(&mut command, "--metadata", args.metadata.as_deref());
    push_flag(
        &mut command,
        args.print_supported_models,
        "--print-supported-models",
    );
    push_flag(&mut command, args.dry_run, "--dry-run");
    push_flag(&mut command, args.remote, "--remote");
    push_flag(&mut command, args.mmproj, "--mmproj");
    push_flag(&mut command, args.mtp, "--mtp");
    push_flag(&mut command, args.no_mtp, "--no-mtp");
    push_flag(&mut command, args.mistral_format, "--mistral-format");
    push_flag(
        &mut command,
        args.disable_mistral_community_chat_template,
        "--disable-mistral-community-chat-template",
    );
    push_flag(
        &mut command,
        args.sentence_transformers_dense_modules,
        "--sentence-transformers-dense-modules",
    );
    push_flag(&mut command, args.fuse_gate_up_exps, "--fuse-gate-up-exps");
    push_flag(&mut command, args.fp8_as_q8, "--fp8-as-q8");
    push_optional_value(
        &mut command,
        "--target-model-dir",
        args.target_model_dir.as_deref(),
    );
    command.push(manifest.source.display().to_string());
    Ok(command)
}

fn push_flag(command: &mut Vec<String>, enabled: bool, flag: &str) {
    if enabled {
        command.push(flag.to_string());
    }
}

fn push_optional_path(command: &mut Vec<String>, flag: &str, value: Option<&Path>) {
    if let Some(value) = value {
        command.push(flag.to_string());
        command.push(value.display().to_string());
    }
}

fn push_optional_value(command: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        command.push(flag.to_string());
        command.push(value.to_string());
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::manifest::{MANIFEST_VERSION, Manifest};
    use crate::splits::SplitWindow;
    use crate::types::{ConvertOutputType, JobKind};

    use super::*;

    #[test]
    fn builds_convert_window_command() {
        let manifest = Manifest {
            schema_version: MANIFEST_VERSION,
            kind: JobKind::ConvertHf,
            source: PathBuf::from("/models/GLM-5.2"),
            source_prefix: None,
            target: PathBuf::from("/target"),
            target_prefix: "BF16".to_string(),
            output_basename: "GLM-5.2-BF16".to_string(),
            expected_splits: 306,
            window_size: 3,
            quant: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_type_file: None,
        };
        let args = crate::ConvertRunnerArgs {
            backend: crate::BackendKind::ExternalProcess,
            converter: Some(PathBuf::from("/llama.cpp/convert_hf_to_gguf.py")),
            python: "python3".to_string(),
            split_max_size: "50G".to_string(),
            split_max_tensors: Some(128),
            skip_output_shards_before: None,
            stop_output_shards_after: None,
            remote: true,
            vocab_only: true,
            bigendian: true,
            verbose: true,
            dry_run: true,
            use_temp_file: true,
            no_lazy: true,
            model_name: Some("glm52".to_string()),
            no_tensor_first_split: true,
            metadata: Some(PathBuf::from("/metadata.json")),
            print_supported_models: true,
            mmproj: true,
            mtp: true,
            no_mtp: true,
            mistral_format: true,
            disable_mistral_community_chat_template: true,
            sentence_transformers_dense_modules: true,
            fuse_gate_up_exps: true,
            fp8_as_q8: true,
            target_model_dir: Some("/target-model".to_string()),
            spool_dir: None,
            keep_spool: false,
            watchdog_seconds: None,
            max_memory: None,
            memory_policy: crate::memory_budget::MemoryPolicy::Hard,
            stream_buffer_bytes: 8 * 1024 * 1024,
            print_only: false,
            record_dir: None,
        };
        let command = build_convert_command(
            &args,
            &manifest,
            Path::new("/target/BF16/GLM-5.2-BF16.gguf"),
            SplitWindow {
                first_split: 4,
                last_split: 6,
            },
        )
        .unwrap();

        assert_eq!(
            command,
            vec![
                "python3",
                "/llama.cpp/convert_hf_to_gguf.py",
                "--outtype",
                "bf16",
                "--split-max-size",
                "50G",
                "--outfile",
                "/target/BF16/GLM-5.2-BF16.gguf",
                "--vocab-only",
                "--bigendian",
                "--use-temp-file",
                "--no-lazy",
                "--model-name",
                "glm52",
                "--verbose",
                "--split-max-tensors",
                "128",
                "--skip-output-shards-before",
                "4",
                "--stop-output-shards-after",
                "6",
                "--no-tensor-first-split",
                "--metadata",
                "/metadata.json",
                "--print-supported-models",
                "--dry-run",
                "--remote",
                "--mmproj",
                "--mtp",
                "--no-mtp",
                "--mistral-format",
                "--disable-mistral-community-chat-template",
                "--sentence-transformers-dense-modules",
                "--fuse-gate-up-exps",
                "--fp8-as-q8",
                "--target-model-dir",
                "/target-model",
                "/models/GLM-5.2",
            ]
        );
    }
}

//! Prepare local models through the same API used by Mesh.
use crate::cli::ServeOpenAiArgs;
use anyhow::{Context, Result};
use skippy_api::{SingleStageOptions, hash_cache::SidecarDigestCache};
use skippy_config::load_json;
use skippy_protocol::StageConfig;

use crate::local_resource_planning::{LocalResourcePlanningInput, plan_local_resources};

pub(crate) fn prepare_openai_stage(args: &ServeOpenAiArgs) -> Result<StageConfig> {
    match (&args.config, &args.model_path) {
        (Some(path), None) => {
            load_json(path).with_context(|| format!("load stage config {}", path.display()))
        }
        (None, Some(path)) => {
            // Preserve the final path component so strict source verification
            // can reject user-provided symlinks rather than silently following them.
            let path = if path.is_absolute() {
                path.clone()
            } else {
                std::env::current_dir()
                    .context("resolve local model directory")?
                    .join(path)
            };
            let model_id = args.model_id.clone().unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("local-model")
                    .to_string()
            });
            let mut options = SingleStageOptions::new(&model_id, &path);
            options.ctx_size = args.ctx_size.unwrap_or(options.ctx_size);
            options.n_gpu_layers = args.n_gpu_layers.unwrap_or(options.n_gpu_layers);
            options.generation_concurrency = args
                .generation_concurrency
                .unwrap_or(options.generation_concurrency);
            options.checkpoint_quantization = args.checkpoint_quantization.clone();
            options.native_mtp_enabled = skippy_model_artifact::gguf::supports_native_mtp(&path);
            options.checkpoint_imatrix = args
                .checkpoint_imatrix
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned());
            options.projector_path = args
                .mmproj
                .clone()
                .or_else(|| skippy_model_hf::store::local::find_mmproj_path(&model_id, &path));
            if let Some(projector) = options.projector_path.as_ref() {
                anyhow::ensure!(
                    projector.is_file(),
                    "multimodal projector path is not a file: {}",
                    projector.display()
                );
            }
            options.validate()?;
            let cache = args.hash_cache.clone().map(SidecarDigestCache::open_in);
            let identity = skippy_api::source::synthetic_direct_gguf_package(
                &model_id,
                &path,
                cache.as_ref(),
            )?;
            let plan = plan_local_resources(LocalResourcePlanningInput {
                model_path: &identity.source_model_path,
                model_bytes: identity.source_model_bytes,
                projector_path: options.projector_path.as_deref(),
                n_gpu_layers: options.n_gpu_layers,
                ctx_size_override: args.ctx_size,
                parallel_override: args.generation_concurrency,
                cache_type_k: &options.cache_type_k,
                cache_type_v: &options.cache_type_v,
            });
            options.ctx_size = plan.context_length;
            options.generation_concurrency = plan.slots;
            options.validate()?;
            // Load from the verified source locator (including managed multipart
            // views), not from an independently resolved input path.
            options.model_path = identity.source_model_path.clone();
            skippy_api::single_stage_config(
                &options,
                identity.into(),
                format!("skippy-{}", uuid::Uuid::new_v4()),
            )
        }
        _ => anyhow::bail!("provide exactly one of --config or --model-path"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    fn args(values: &[&str]) -> ServeOpenAiArgs {
        let cli = Cli::try_parse_from(values).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve")
        };
        args.public
    }

    #[test]
    fn local_source_and_stage_config_are_exclusive() {
        for values in [
            vec![
                "skippy",
                "serve",
                "--config",
                "stage.json",
                "--model-path",
                "model.gguf",
            ],
            vec![
                "skippy",
                "serve",
                "--config",
                "stage.json",
                "--ctx-size",
                "512",
            ],
        ] {
            assert!(Cli::try_parse_from(values).is_err());
        }
    }

    #[test]
    fn local_model_rejects_invalid_options_before_reading_weights() {
        let args = args(&[
            "skippy",
            "serve",
            "--model-path",
            "missing.gguf",
            "--ctx-size",
            "0",
        ]);
        assert!(
            prepare_openai_stage(&args)
                .unwrap_err()
                .to_string()
                .contains("ctx_size")
        );
    }

    #[test]
    fn local_model_uses_verified_identity_and_shared_stage_options() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        let mut bytes = Vec::from(*b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        };
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(&mut bytes, "llama");
        for (key, value) in [
            ("llama.block_count", 2u32),
            ("llama.embedding_length", 128),
            ("llama.context_length", 4096),
        ] {
            string(&mut bytes, key);
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let explicit_args = args(&[
            "skippy",
            "serve",
            "--model-path",
            path.to_str().unwrap(),
            "--model-id",
            "tiny",
            "--ctx-size",
            "512",
            "--n-gpu-layers",
            "0",
            "--generation-concurrency",
            "2",
        ]);
        let config = prepare_openai_stage(&explicit_args).unwrap();
        let identity =
            skippy_api::source::synthetic_direct_gguf_package("tiny", &path, None).unwrap();
        assert_eq!(
            config.source_model_sha256.as_deref(),
            Some(identity.source_model_sha256.as_str())
        );
        assert_eq!(
            config.manifest_sha256.as_deref(),
            Some(identity.manifest_sha256.as_str())
        );
        assert_eq!(
            config.model_path,
            Some(identity.source_model_path.to_string_lossy().into_owned())
        );
        assert_eq!(config.layer_end, 2);
        assert_eq!(config.ctx_size, 512);
        assert_eq!(config.lane_count, 2);
        assert_eq!(config.n_gpu_layers, 0);
        assert!(config.resident_tensor_names.is_empty());
        assert!(config.execution_contract.is_empty());
        assert!(config.run_id.starts_with("skippy-"));

        let default_args = args(&["skippy", "serve", "--model-path", path.to_str().unwrap()]);
        let default_config = prepare_openai_stage(&default_args).unwrap();
        assert_eq!(default_config.ctx_size, 4096);
        assert_eq!(default_config.lane_count, 4);
    }

    #[test]
    fn local_model_passes_checkpoint_quantization_and_projector_to_shared_stage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        let mut bytes = Vec::from(*b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        };
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(&mut bytes, "llama");
        for (key, value) in [
            ("llama.block_count", 2u32),
            ("llama.embedding_length", 128),
            ("llama.context_length", 4096),
        ] {
            string(&mut bytes, key);
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let projector = dir.path().join("mmproj-tiny.gguf");
        std::fs::write(&projector, b"projector").unwrap();
        let args = args(&[
            "skippy",
            "serve",
            "--model-path",
            path.to_str().unwrap(),
            "--quant",
            "Q4_K",
            "--mmproj",
            projector.to_str().unwrap(),
        ]);
        let config = prepare_openai_stage(&args).unwrap();
        assert_eq!(config.checkpoint_quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(config.projector_path.as_deref(), projector.to_str());
    }
}

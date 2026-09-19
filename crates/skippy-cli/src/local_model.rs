//! Prepare local models through the same API used by Mesh.
use crate::cli::ServeOpenAiArgs;
use anyhow::{Context, Result};
use skippy_api::{SingleStageOptions, hash_cache::SidecarDigestCache};
use skippy_config::load_json;
use skippy_protocol::StageConfig;

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
            options.ctx_size = args.ctx_size.unwrap_or(4096);
            options.n_gpu_layers = args.n_gpu_layers.unwrap_or(-1);
            options.generation_concurrency = args.generation_concurrency.unwrap_or(1);
            options.validate()?;
            let cache = args.hash_cache.clone().map(SidecarDigestCache::open_in);
            let identity = skippy_api::source::synthetic_direct_gguf_package(
                &model_id,
                &path,
                cache.as_ref(),
            )?;
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
        let Command::ServeOpenAi(args) = cli.command else {
            panic!("expected serve-openai")
        };
        args
    }

    #[test]
    fn local_source_and_stage_config_are_exclusive() {
        for values in [
            vec!["skippy", "serve-openai"],
            vec![
                "skippy",
                "serve-openai",
                "--config",
                "stage.json",
                "--model-path",
                "model.gguf",
            ],
            vec![
                "skippy",
                "serve-openai",
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
            "serve-openai",
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
        let args = args(&[
            "skippy",
            "serve-openai",
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
        let config = prepare_openai_stage(&args).unwrap();
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
        assert!(!config.filter_tensors_on_load);
        assert!(config.run_id.starts_with("skippy-"));
    }
}

//! `mlx-serve` — stand up mesh-llm's real OpenAI-compatible frontend backed by
//! the MLX (Metal) engine, serving an HF safetensors model.
//!
//! This exists to prove the engine serves over the SAME `openai-frontend`
//! surface the shipped binary uses (`router_for(Arc<dyn OpenAiBackend>)`), not a
//! bespoke HTTP handler.

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod real {
    use std::path::PathBuf;
    use std::sync::Arc;

    use anyhow::{Context, Result};
    use clap::{Parser, ValueEnum};
    use skippy_engine_mlx::{MlxBackend, MlxEngine, MlxEngineConfig, MlxWeightQuantization};

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum WeightQuantization {
        Auto,
        None,
        Affine4,
        Affine8,
        MxFp4,
    }

    impl WeightQuantization {
        fn engine(self) -> Option<MlxWeightQuantization> {
            match self {
                Self::Auto | Self::None => None,
                Self::Affine4 => Some(MlxWeightQuantization::Affine {
                    group_size: 64,
                    bits: 4,
                }),
                Self::Affine8 => Some(MlxWeightQuantization::Affine {
                    group_size: 64,
                    bits: 8,
                }),
                Self::MxFp4 => Some(MlxWeightQuantization::MxFp4),
            }
        }
    }

    #[derive(Parser, Debug)]
    #[command(about = "Serve an MLX safetensors model over the mesh-llm OpenAI frontend")]
    struct Cli {
        /// Model directory (HF safetensors: config.json + tokenizer.json + *.safetensors).
        #[arg(short, long)]
        model: PathBuf,

        /// Model id advertised on /v1/models (defaults to the directory name).
        #[arg(long)]
        model_id: Option<String>,

        #[arg(long, default_value_t = 512)]
        default_max_tokens: usize,

        #[arg(long, default_value_t = 4096)]
        max_tokens_cap: usize,

        /// Weight representation for dense checkpoints.
        #[arg(long, value_enum, default_value = "auto")]
        weight_quantization: WeightQuantization,

        #[arg(long, default_value = "127.0.0.1:11434")]
        bind: String,
    }

    pub async fn main() -> Result<()> {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();

        let cli = Cli::parse();
        let model_id = cli.model_id.clone().unwrap_or_else(|| {
            cli.model
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "mlx-model".to_string())
        });

        let weight_quantization = match cli.weight_quantization {
            WeightQuantization::Auto => {
                skippy_engine_mlx::automatic_weight_quantization(&cli.model)?
            }
            explicit => explicit.engine(),
        };
        let allow_native_quantization_fallback =
            matches!(cli.weight_quantization, WeightQuantization::Auto);
        let config = MlxEngineConfig {
            model_dir: cli.model.clone(),
            model_id: model_id.clone(),
            default_max_tokens: cli.default_max_tokens,
            max_tokens_cap: cli.max_tokens_cap,
            weight_quantization,
            allow_native_quantization_fallback,
        };

        tracing::info!("loading MLX model from {} ...", cli.model.display());
        let engine = tokio::task::spawn_blocking(move || MlxEngine::spawn(config))
            .await
            .context("join MLX load task")??;

        let backend = Arc::new(MlxBackend::new(engine));
        let app = openai_frontend::router::router_for(backend);

        let listener = tokio::net::TcpListener::bind(&cli.bind)
            .await
            .with_context(|| format!("bind {}", cli.bind))?;
        tracing::info!(
            "MLX serving '{}' on http://{}/v1  (try: GET /v1/models)",
            model_id,
            cli.bind
        );
        axum::serve(listener, app).await.context("axum serve")?;
        Ok(())
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    real::main().await
}

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
fn main() {
    eprintln!(
        "mlx-serve was built without MLX support.\n\
         Rebuild on Apple Silicon with `--features mlx` to enable the MLX engine."
    );
    std::process::exit(1);
}

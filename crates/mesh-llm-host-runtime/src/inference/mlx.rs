//! MLX (Metal) serving integration — Apple Silicon only.
//!
//! Bridges the `skippy-engine-mlx` crate (which serves HF safetensors models on
//! Metal via `safemlx`) into the host runtime's local-model launch path. Gated
//! behind both the `mlx` cargo feature and `target_os = "macos"`, so it is
//! entirely absent from every other build.
//!
//! Unlike the skippy/GGUF path (which owns an embedded HTTP server in
//! `skippy-server`), MLX serves over the plain `openai-frontend` router; this
//! module stands up that router on a local port and manages its lifecycle with a
//! graceful-shutdown handle mirroring `SkippyHttpHandle`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use skippy_engine_mlx::{
    MlxBackend, MlxDistributedEngine, MlxDistributedEngineConfig, MlxEngine, MlxEngineConfig,
    automatic_weight_quantization,
};
use skippy_protocol::binary::WireActivationDType;

const DEFAULT_MAX_GENERATION_TOKENS: usize = 512;

/// A loaded MLX model plus the OpenAI backend that serves it.
pub(crate) struct MlxModelHandle {
    backend: Arc<MlxBackend>,
}

impl MlxModelHandle {
    /// Loads a safetensors model directory on the MLX (Metal) engine. Blocking:
    /// call from `spawn_blocking`.
    pub(crate) fn load(model_dir: PathBuf, model_id: String, context_length: u32) -> Result<Self> {
        let weight_quantization = automatic_weight_quantization(&model_dir)?;
        let config = MlxEngineConfig {
            model_dir,
            model_id,
            default_max_tokens: (context_length.max(1) as usize).min(DEFAULT_MAX_GENERATION_TOKENS),
            max_tokens_cap: context_length.max(1) as usize,
            weight_quantization,
            allow_native_quantization_fallback: true,
        };
        let engine = MlxEngine::spawn(config)?;
        Ok(Self {
            backend: Arc::new(MlxBackend::new(engine)),
        })
    }

    /// Loads tokenizer/chat-template sidecars and drives a mesh-managed MLX
    /// stage chain rooted at `stage_addr`. No model weights are loaded here.
    pub(crate) fn load_distributed(
        model_dir: PathBuf,
        model_id: String,
        context_length: u32,
        stage_addr: SocketAddr,
        wire_dtype: WireActivationDType,
    ) -> Result<Self> {
        let engine = MlxDistributedEngine::spawn(MlxDistributedEngineConfig {
            model_dir,
            model_id,
            stage_addr,
            wire_dtype,
            default_max_tokens: (context_length.max(1) as usize).min(DEFAULT_MAX_GENERATION_TOKENS),
            max_tokens_cap: context_length.max(1) as usize,
            context_tokens: context_length.max(1) as usize,
        })?;
        Ok(Self {
            backend: Arc::new(MlxBackend::new_distributed(engine)),
        })
    }

    /// Starts an `openai-frontend` HTTP server for this model on `port`.
    pub(crate) async fn start_http(
        &self,
        port: u16,
        death_tx: tokio::sync::oneshot::Sender<()>,
    ) -> Result<MlxHttpHandle> {
        let addr: SocketAddr = ([127, 0, 0, 1], port).into();
        let app = openai_frontend::router::router_for(self.backend.clone());
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("bind MLX OpenAI frontend at {addr}"))?;

        let server = tokio::spawn(async move {
            let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });
            if let Err(error) = serve.await {
                tracing::error!(%error, "MLX openai frontend server error");
            }
            let _ = death_tx.send(());
        });

        Ok(MlxHttpHandle {
            port,
            shutdown_tx: Some(shutdown_tx),
            server: Some(server),
        })
    }
}

/// Lifecycle handle for the MLX model's HTTP server.
pub(crate) struct MlxHttpHandle {
    port: u16,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    server: Option<tokio::task::JoinHandle<()>>,
}

impl MlxHttpHandle {
    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) async fn shutdown(mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(server) = self.server.take() {
            server.await.context("join MLX openai frontend task")?;
        }
        Ok(())
    }
}

impl Drop for MlxHttpHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(server) = self.server.take() {
            server.abort();
        }
    }
}

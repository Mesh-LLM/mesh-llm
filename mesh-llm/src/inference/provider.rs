use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[cfg(target_os = "macos")]
fn builtin_mlx_model_path(path: &Path) -> bool {
    crate::mlx::is_mlx_model_dir(path)
}

#[cfg(not(target_os = "macos"))]
fn builtin_mlx_model_path(_path: &Path) -> bool {
    false
}

#[cfg(target_os = "macos")]
fn builtin_mlx_model_name(path: &Path) -> String {
    if let Some(dir) = crate::mlx::mlx_model_dir(path) {
        if let Some(identity) =
            crate::models::huggingface_identity_for_path(&dir.join("config.json"))
        {
            if let Some(name) = identity.repo_id.rsplit('/').next() {
                return name.to_string();
            }
        }
        if let Some(name) = dir.file_name().and_then(|value| value.to_str()) {
            return name.to_string();
        }
    }

    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

/// Backend-neutral runtime handle for a serving instance.
///
/// This sits above any concrete backend implementation:
/// - llama.cpp subprocesses
/// - MLX in-process serving
/// - future plugin-hosted inference runtimes
#[derive(Clone, Debug)]
pub struct InferenceServerHandle {
    pid: u32,
    expected_exit: Arc<AtomicBool>,
    shutdown_tx: Option<tokio::sync::watch::Sender<bool>>,
}

impl InferenceServerHandle {
    pub(crate) fn process(pid: u32, expected_exit: Arc<AtomicBool>) -> Self {
        Self {
            pid,
            expected_exit,
            shutdown_tx: None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn in_process(shutdown_tx: tokio::sync::watch::Sender<bool>) -> Self {
        Self {
            pid: std::process::id(),
            expected_exit: Arc::new(AtomicBool::new(true)),
            shutdown_tx: Some(shutdown_tx),
        }
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub async fn shutdown(&self) {
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(true);
            return;
        }
        self.expected_exit.store(true, Ordering::Relaxed);
        crate::inference::launch::terminate_process(self.pid).await;
    }
}

/// Backend-neutral runtime result for a started serving instance.
#[derive(Debug)]
pub struct InferenceServerProcess {
    pub handle: InferenceServerHandle,
    pub death_rx: tokio::sync::oneshot::Receiver<()>,
    pub context_length: u32,
}

/// Backend-neutral request to start a serving endpoint for a model.
///
/// Backends can treat `worker_tunnel_ports` / `tensor_split` as the
/// distributed-host shape, or ignore them for single-node serving.
#[derive(Clone, Debug)]
pub struct InferenceEndpointRequest {
    pub model_path: PathBuf,
    pub listen_port: u16,
    pub worker_tunnel_ports: Vec<u16>,
    pub tensor_split: Option<String>,
    pub draft_model_path: Option<PathBuf>,
    pub draft_max: u16,
    pub model_bytes: u64,
    pub local_vram_bytes: u64,
    pub mmproj_path: Option<PathBuf>,
    pub ctx_size_override: Option<u32>,
    pub total_group_vram_bytes: Option<u64>,
}

impl InferenceEndpointRequest {
    pub fn local(
        model_path: impl Into<PathBuf>,
        listen_port: u16,
        model_bytes: u64,
        local_vram_bytes: u64,
    ) -> Self {
        Self {
            model_path: model_path.into(),
            listen_port,
            worker_tunnel_ports: Vec::new(),
            tensor_split: None,
            draft_model_path: None,
            draft_max: 0,
            model_bytes,
            local_vram_bytes,
            mmproj_path: None,
            ctx_size_override: None,
            total_group_vram_bytes: None,
        }
    }

    pub fn with_ctx_size_override(mut self, ctx_size_override: Option<u32>) -> Self {
        self.ctx_size_override = ctx_size_override;
        self
    }

    pub fn with_mmproj_path(mut self, mmproj_path: Option<impl AsRef<Path>>) -> Self {
        self.mmproj_path = mmproj_path.map(|path| path.as_ref().to_path_buf());
        self
    }

    pub fn distributed_host(
        model_path: impl Into<PathBuf>,
        listen_port: u16,
        worker_tunnel_ports: Vec<u16>,
        model_bytes: u64,
        local_vram_bytes: u64,
    ) -> Self {
        Self {
            model_path: model_path.into(),
            listen_port,
            worker_tunnel_ports,
            tensor_split: None,
            draft_model_path: None,
            draft_max: 0,
            model_bytes,
            local_vram_bytes,
            mmproj_path: None,
            ctx_size_override: None,
            total_group_vram_bytes: None,
        }
    }

    pub fn with_tensor_split(mut self, tensor_split: Option<impl Into<String>>) -> Self {
        self.tensor_split = tensor_split.map(Into::into);
        self
    }

    pub fn with_draft_model_path(mut self, draft_model_path: Option<impl AsRef<Path>>) -> Self {
        self.draft_model_path = draft_model_path.map(|path| path.as_ref().to_path_buf());
        self
    }

    pub fn with_draft_max(mut self, draft_max: u16) -> Self {
        self.draft_max = draft_max;
        self
    }

    pub fn with_total_group_vram_bytes(mut self, total_group_vram_bytes: Option<u64>) -> Self {
        self.total_group_vram_bytes = total_group_vram_bytes;
        self
    }
}

/// Backend-neutral request to start a worker-side runtime helper.
#[derive(Clone, Debug, Default)]
pub struct InferenceWorkerRequest {
    pub model_path: Option<PathBuf>,
    pub device_hint: Option<String>,
}

impl InferenceWorkerRequest {
    pub fn with_model_path(mut self, model_path: Option<impl AsRef<Path>>) -> Self {
        self.model_path = model_path.map(|path| path.as_ref().to_path_buf());
        self
    }

    pub fn with_device_hint(mut self, device_hint: Option<impl Into<String>>) -> Self {
        self.device_hint = device_hint.map(Into::into);
        self
    }
}

/// Built-in provider adapter for the current llama.cpp runtime path.
///
/// This is the first step toward a pluggable backend provider interface:
/// core call sites depend on the provider contract, while the built-in
/// provider still delegates to the existing llama-specific launch code.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinLlamaProvider;

impl BuiltinLlamaProvider {
    pub async fn start_endpoint(
        self,
        bin_dir: &Path,
        binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &InferenceEndpointRequest,
    ) -> Result<InferenceServerProcess> {
        crate::inference::launch::start_llama_server(bin_dir, binary_flavor, request).await
    }

    pub async fn start_worker(
        self,
        bin_dir: &Path,
        binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &InferenceWorkerRequest,
    ) -> Result<u16> {
        crate::inference::launch::start_rpc_server(bin_dir, binary_flavor, request).await
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinMlxProvider;

impl BuiltinMlxProvider {
    pub async fn start_endpoint(
        self,
        _bin_dir: &Path,
        _binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &InferenceEndpointRequest,
    ) -> Result<InferenceServerProcess> {
        #[cfg(target_os = "macos")]
        {
            let Some(model_dir) = crate::mlx::mlx_model_dir(request.model_path.as_path()) else {
                anyhow::bail!(
                    "MLX provider expected a normalized MLX model path, got {}",
                    request.model_path.display()
                );
            };
            let model_name = builtin_mlx_model_name(request.model_path.as_path());
            return crate::mlx::start_mlx_server(model_dir, model_name, request.listen_port).await;
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = request;
            anyhow::bail!("MLX provider is only available on macOS");
        }
    }
}

/// Named backend selection seam for built-in inference providers.
///
/// Call sites do not need to know which concrete built-in provider they are
/// using. This branch wires both llama.cpp and MLX through the same seam.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuiltinProviderKind {
    Llama,
    #[cfg(target_os = "macos")]
    Mlx,
}

impl BuiltinProviderKind {
    pub fn backend_label(self) -> &'static str {
        match self {
            BuiltinProviderKind::Llama => "llama",
            #[cfg(target_os = "macos")]
            BuiltinProviderKind::Mlx => "mlx",
        }
    }

    pub async fn start_endpoint(
        self,
        bin_dir: &Path,
        binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &InferenceEndpointRequest,
    ) -> Result<InferenceServerProcess> {
        match self {
            BuiltinProviderKind::Llama => {
                BuiltinLlamaProvider
                    .start_endpoint(bin_dir, binary_flavor, request)
                    .await
            }
            #[cfg(target_os = "macos")]
            BuiltinProviderKind::Mlx => {
                BuiltinMlxProvider
                    .start_endpoint(bin_dir, binary_flavor, request)
                    .await
            }
        }
    }

    pub async fn start_worker(
        self,
        bin_dir: &Path,
        binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &InferenceWorkerRequest,
    ) -> Result<u16> {
        match self {
            BuiltinProviderKind::Llama => {
                BuiltinLlamaProvider
                    .start_worker(bin_dir, binary_flavor, request)
                    .await
            }
            #[cfg(target_os = "macos")]
            BuiltinProviderKind::Mlx => {
                let _ = (bin_dir, binary_flavor, request);
                anyhow::bail!("MLX does not use a worker helper runtime")
            }
        }
    }
}

pub fn select_local_endpoint_provider(request: &InferenceEndpointRequest) -> BuiltinProviderKind {
    #[cfg(target_os = "macos")]
    if builtin_mlx_model_path(request.model_path.as_path()) {
        return BuiltinProviderKind::Mlx;
    }

    BuiltinProviderKind::Llama
}

pub fn select_distributed_endpoint_provider(
    _request: &InferenceEndpointRequest,
) -> BuiltinProviderKind {
    BuiltinProviderKind::Llama
}

pub fn select_worker_provider(_request: &InferenceWorkerRequest) -> BuiltinProviderKind {
    BuiltinProviderKind::Llama
}

pub fn provider_requires_worker_runtime(model_path: &Path) -> bool {
    !builtin_mlx_model_path(model_path)
}

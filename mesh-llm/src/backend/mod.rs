mod llama;
mod vllm;

use crate::launch::{BinaryFlavor, InferenceServerProcess, ModelLaunchSpec};
use anyhow::Result;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Llama,
    Vllm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    pub requires_rpc_server: bool,
    pub supports_rpc_split: bool,
}

impl BackendKind {
    pub const ALL: [BackendKind; 2] = [BackendKind::Llama, BackendKind::Vllm];

    pub fn as_str(self) -> &'static str {
        backend_ops(self).as_str()
    }

    pub fn capabilities(self) -> BackendCapabilities {
        backend_capabilities(self)
    }
}

pub type BackendLaunchFuture<'a> =
    Pin<Box<dyn Future<Output = Result<InferenceServerProcess>> + Send + 'a>>;
pub type BackendControlFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

pub trait BackendOps: Send + Sync {
    fn as_str(&self) -> &'static str;
    fn process_label(&self) -> &'static str;
    fn health_path(&self) -> &'static str {
        "/health"
    }
    fn resolve_runtime_label(&self) -> Result<Option<String>> {
        Ok(None)
    }
    fn validate_model<'a>(&self, _model: &'a Path) -> Result<()> {
        Ok(())
    }
    fn start_server<'a>(
        &self,
        bin_dir: &'a Path,
        binary_flavor: Option<BinaryFlavor>,
        spec: ModelLaunchSpec<'a>,
    ) -> BackendLaunchFuture<'a>;
    fn kill_server_processes<'a>(&'a self) -> BackendControlFuture<'a>;
}

pub fn backend_ops(kind: BackendKind) -> &'static dyn BackendOps {
    match kind {
        BackendKind::Llama => &llama::LLAMA_BACKEND,
        BackendKind::Vllm => &vllm::VLLM_BACKEND,
    }
}

pub fn backend_capabilities(kind: BackendKind) -> BackendCapabilities {
    match kind {
        BackendKind::Llama => BackendCapabilities {
            requires_rpc_server: true,
            supports_rpc_split: true,
        },
        BackendKind::Vllm => BackendCapabilities {
            requires_rpc_server: false,
            supports_rpc_split: false,
        },
    }
}

pub async fn kill_all_server_processes() {
    for kind in BackendKind::ALL {
        backend_ops(kind).kill_server_processes().await;
    }
}

fn looks_like_hf_model_dir(model_path: &Path) -> bool {
    if !model_path.is_dir() || !model_path.join("config.json").exists() {
        return false;
    }

    model_path.join("model.safetensors").exists()
        || model_path.join("model.safetensors.index.json").exists()
        || model_path.join("pytorch_model.bin").exists()
        || model_path.join("pytorch_model.bin.index.json").exists()
}

pub fn detect_backend(model_path: &Path) -> BackendKind {
    if looks_like_hf_model_dir(model_path) {
        return BackendKind::Vllm;
    }

    BackendKind::Llama
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn backend_registry_returns_llama_ops() {
        let ops = backend_ops(BackendKind::Llama);
        assert_eq!(ops.as_str(), "llama");
        assert_eq!(ops.process_label(), "llama-server");
        assert_eq!(ops.health_path(), "/health");
    }

    #[test]
    fn backend_registry_returns_vllm_ops() {
        let ops = backend_ops(BackendKind::Vllm);
        assert_eq!(ops.as_str(), "vllm");
        assert_eq!(ops.process_label(), "vllm");
        assert_eq!(ops.health_path(), "/health");
    }

    #[test]
    fn detect_backend_uses_vllm_for_hf_model_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("mesh-llm-vllm-backend-{unique}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();

        assert_eq!(detect_backend(&dir), BackendKind::Vllm);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn detect_backend_does_not_use_vllm_for_config_only_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("mesh-llm-config-only-{unique}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();

        assert_eq!(detect_backend(&dir), BackendKind::Llama);

        let _ = std::fs::remove_dir_all(dir);
    }
}

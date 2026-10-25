#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("node build failed: {0}")]
    BuildFailed(String),
    #[error("node start or join failed: {0}")]
    JoinFailed(String),
    #[error("model discovery failed: {0}")]
    DiscoveryFailed(String),
    #[error("stream failed: {0}")]
    StreamFailed(String),
    #[error("node unavailable: {0}")]
    HostUnavailable(String),
    #[error("role does not permit inference: {0}")]
    ServingUnsupported(String),
    #[error("native runtime failed: {0}")]
    NativeRuntimeFailed(String),
    #[error("OpenAI request failed: {0}")]
    OpenAiRequestFailed(String),
}

pub(super) fn map_native_runtime_error(error: impl ToString) -> FfiError {
    FfiError::NativeRuntimeFailed(error.to_string())
}

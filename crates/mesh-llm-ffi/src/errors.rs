use mesh_llm_sdk::MeshApiError;

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("invalid invite token: {0}")]
    InvalidInviteToken(String),
    #[error("invalid owner keypair: {0}")]
    InvalidOwnerKeypair(String),
    #[error("client build failed: {0}")]
    BuildFailed(String),
    #[error("join failed: {0}")]
    JoinFailed(String),
    #[error("discovery failed: {0}")]
    DiscoveryFailed(String),
    #[error("stream failed: {0}")]
    StreamFailed(String),
    #[error("cancelled: {0}")]
    Cancelled(String),
    #[error("reconnect failed: {0}")]
    ReconnectFailed(String),
    #[error("host unavailable: {0}")]
    HostUnavailable(String),
    #[error("model management failed: {0}")]
    ModelManagementFailed(String),
    #[error("serving failed: {0}")]
    ServingFailed(String),
    #[error("serving is unsupported by this node: {0}")]
    ServingUnsupported(String),
    #[error("console failed: {0}")]
    ConsoleFailed(String),
    #[error("native runtime failed: {0}")]
    NativeRuntimeFailed(String),
}

pub(super) fn map_mesh_api_error(error: MeshApiError) -> FfiError {
    match error {
        MeshApiError::Client(error) => FfiError::BuildFailed(error.to_string()),
        MeshApiError::Discovery { message } => FfiError::DiscoveryFailed(message),
        MeshApiError::NoPublicMeshFound => {
            FfiError::HostUnavailable("no public mesh matched the requested criteria".to_string())
        }
        MeshApiError::InvalidInviteToken { message } => FfiError::InvalidInviteToken(message),
        MeshApiError::InvalidConfig { message } => FfiError::BuildFailed(message.to_string()),
        MeshApiError::ModelManagement { message } => FfiError::ModelManagementFailed(message),
        MeshApiError::Serving { message } => FfiError::ServingFailed(message),
        MeshApiError::Unsupported { feature } => FfiError::HostUnavailable(feature.to_string()),
    }
}

pub(super) fn map_model_error(error: MeshApiError) -> FfiError {
    match error {
        MeshApiError::ModelManagement { message } => FfiError::ModelManagementFailed(message),
        other => FfiError::ModelManagementFailed(other.to_string()),
    }
}

pub(super) fn map_serving_error(error: MeshApiError) -> FfiError {
    match error {
        MeshApiError::Unsupported { feature } => FfiError::ServingUnsupported(feature.to_string()),
        MeshApiError::Serving { message } => FfiError::ServingFailed(message),
        other => FfiError::ServingFailed(other.to_string()),
    }
}

pub(super) fn map_stream_error(error: MeshApiError) -> FfiError {
    match error {
        MeshApiError::Client(error) => FfiError::StreamFailed(error.to_string()),
        other => FfiError::StreamFailed(other.to_string()),
    }
}

pub(super) fn map_native_runtime_error(error: impl ToString) -> FfiError {
    FfiError::NativeRuntimeFailed(error.to_string())
}

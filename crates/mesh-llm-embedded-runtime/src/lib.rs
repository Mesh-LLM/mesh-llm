#![forbid(unsafe_code)]

pub use mesh_llm_host_runtime::initialize_host_runtime;
pub use mesh_llm_host_runtime::sdk::{
    EmbeddedChatMessage, EmbeddedMeshAdmissionConfig, EmbeddedMeshDiscoveryMode,
    EmbeddedMeshHttpConfig, EmbeddedMeshLogFormat, EmbeddedMeshNetworkConfig,
    EmbeddedMeshNodeBuilder, EmbeddedMeshNodeConfig, EmbeddedMeshNodeHandle, EmbeddedMeshNodeMode,
    EmbeddedMeshNodeStatus, EmbeddedMeshRequirementsConfig, EmbeddedMeshServingConfig,
    EmbeddedMeshStorageConfig, EmbeddedServeConfig, EmbeddedServeHandle, EmbeddedServeMode,
    EmbeddedServeStatus, EmbeddedServingController, EmbeddedTrustPolicy,
    SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION,
};
#[cfg(not(feature = "payments"))]
pub use mesh_llm_host_runtime::sdk::{start_embedded_node, start_embedded_serve};

/// Starts an embedded node with the payments engine installed.
#[cfg(feature = "payments")]
pub async fn start_embedded_node(
    config: EmbeddedMeshNodeConfig,
) -> anyhow::Result<EmbeddedServeHandle> {
    install_payments_engine();
    mesh_llm_host_runtime::sdk::start_embedded_node(config).await
}

/// Starts an embedded serve node with the payments engine installed.
#[cfg(feature = "payments")]
pub async fn start_embedded_serve(
    config: EmbeddedServeConfig,
) -> anyhow::Result<EmbeddedServeHandle> {
    install_payments_engine();
    mesh_llm_host_runtime::sdk::start_embedded_serve(config).await
}

/// This crate links the payments engine; the host only names the seam.
#[cfg(feature = "payments")]
fn install_payments_engine() {
    mesh_llm_host_runtime::install_payments_engine(std::sync::Arc::new(
        mesh_llm_payments::plugin_server::EngineProvider,
    ));
}

pub mod config {
    pub use mesh_llm_host_runtime::sdk::config::*;
}

pub mod native_runtime {
    pub use mesh_llm_host_runtime::sdk::native_runtime::*;
}

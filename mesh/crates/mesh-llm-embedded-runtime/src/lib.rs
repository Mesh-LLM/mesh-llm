#![forbid(unsafe_code)]

pub use mesh_llm_host_runtime::initialize_host_runtime;
pub use mesh_llm_host_runtime::sdk::{
    EmbeddedChatMessage, EmbeddedMeshAdmissionConfig, EmbeddedMeshDiscoveryMode,
    EmbeddedMeshHttpConfig, EmbeddedMeshLogFormat, EmbeddedMeshNetworkConfig,
    EmbeddedMeshNodeBuilder, EmbeddedMeshNodeConfig, EmbeddedMeshNodeHandle, EmbeddedMeshNodeMode,
    EmbeddedMeshNodeStatus, EmbeddedMeshRequirementsConfig, EmbeddedMeshServingConfig,
    EmbeddedMeshStorageConfig, EmbeddedOpenAiResponse, EmbeddedOpenAiStream, EmbeddedServeConfig,
    EmbeddedServeHandle, EmbeddedServeMode, EmbeddedServeStatus, EmbeddedServingController,
    EmbeddedSseEvent, EmbeddedTrustPolicy, MeshMembershipConfig, MeshMembershipStatus,
    SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION,
};
pub use mesh_llm_host_runtime::sdk::{start_embedded_node, start_embedded_serve};

pub mod config {
    pub use mesh_llm_host_runtime::sdk::config::*;
}

pub mod native_runtime {
    pub use mesh_llm_host_runtime::sdk::native_runtime::*;
}

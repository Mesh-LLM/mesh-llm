#![forbid(unsafe_code)]

mod discover;
mod node;

pub use discover::{
    create_auto_client, create_auto_node, discover_public_meshes, AutoConnectResult, AutoNodeResult,
};
pub use mesh_llm_api_client::events;
pub use mesh_llm_api_client::{
    ChatMessage, ChatRequest, ClientBuilder, ClientConfig, InviteToken, MeshApiError, MeshClient,
    Model, OwnerKeypair, PublicMesh, PublicMeshQuery, RequestId, ResponsesRequest, Status,
    MAX_RECONNECT_ATTEMPTS,
};
pub use mesh_llm_node::serving::ServingController;
pub use node::{
    CapabilityLevel, CleanupPolicy, CleanupResult, DeleteModelOptions, DeleteModelResult,
    DevicePolicy, DownloadId, DownloadOptions, DownloadedModel, InstalledModel, LoadModelOptions,
    MeshEvents, MeshInference, MeshModels, MeshNode, MeshNodeBuilder, MeshNodeConfig, MeshServing,
    MeshStatusApi, ModelCacheStatus, ModelCapabilities, ModelDetails, ModelKind, ModelSearchQuery,
    ModelSource, ModelSummary, PrunePolicy, PruneResult, ServedModel, ServingModelState,
    ServingStatus, UnloadModelOptions, UnloadTarget,
};

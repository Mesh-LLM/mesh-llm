use std::sync::LazyLock;

uniffi::setup_scaffolding!("mesh_ffi");

static SDK_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("mesh-llm-sdk")
        .build()
        .expect("create mesh-llm SDK runtime")
});

mod client;
mod console;
mod conversions;
mod errors;
mod events;
mod handles;
mod identity;
mod model_types;
mod native_runtime;
mod native_runtime_types;
mod node;
mod public_mesh;
mod request_types;
mod runtime_blocking;

pub use client::{create_auto_client, create_client};
pub use errors::FfiError;
pub use events::ClientEvent;
pub use handles::{ConsoleHandle, MeshClientHandle, MeshNodeHandle};
pub use identity::generate_owner_keypair_hex;
pub use model_types::{
    CapabilityLevel, CleanupPolicy, CleanupResult, DeleteModelOptions, DeleteModelResult,
    DevicePolicy, DownloadedModel, InstalledModel, LoadModelOptions, ModelCacheStatus,
    ModelCapabilities, ModelDetails, ModelKind, ModelSearchQuery, ModelSource, ModelSummary,
    PrunePolicy, PruneResult, ServedModel, ServingModelState, ServingStatus, UnloadModelOptions,
    UnloadTarget,
};
pub use native_runtime::{
    current_mesh_version, current_skippy_abi_version, install_native_runtime,
    installed_native_runtimes, prune_native_runtimes, remove_native_runtime,
};
pub use native_runtime_types::{
    EventListener, InstalledNativeRuntimeNative, NativeRuntimeDownloadProgressNative,
    NativeRuntimeInstallOptionsNative, NativeRuntimeInstallOutcomeNative,
    NativeRuntimeProgressListener, NativeRuntimePruneModeNative, NativeRuntimePruneResultNative,
    NativeRuntimeVerificationPolicyNative,
};
pub use node::{create_auto_node, create_node};
pub use public_mesh::discover_public_meshes;
pub use request_types::{
    ChatMessageNative, ChatRequestNative, ClientStatus, ConsoleOptionsNative, ModelNative,
    PublicMesh, PublicMeshQuery, ResponsesRequestNative,
};

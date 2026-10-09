use std::sync::LazyLock;

uniffi::setup_scaffolding!("mesh_ffi");

static SDK_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("mesh-llm-sdk")
        .build()
        .expect("create mesh-llm SDK runtime")
});

mod conversions;
mod errors;
mod events;
mod handles;
mod native_runtime;
mod native_runtime_types;
mod node;
mod request_types;
mod runtime_blocking;

pub use errors::FfiError;
pub use events::OpenAiStreamEventNative;
pub use handles::MeshNodeHandle;
pub use native_runtime::{
    current_mesh_version, current_skippy_abi_version, install_native_runtime,
    installed_native_runtimes, prune_native_runtimes, remove_native_runtime,
};
pub use native_runtime_types::{
    InstalledNativeRuntimeNative, NativeRuntimeDownloadProgressNative,
    NativeRuntimeInstallOptionsNative, NativeRuntimeInstallOutcomeNative,
    NativeRuntimeProgressListener, NativeRuntimePruneModeNative, NativeRuntimePruneResultNative,
    NativeRuntimeVerificationPolicyNative, OpenAiStreamListener,
};
pub use node::create_node;
pub use request_types::{ModelNative, NodeStatusNative, OpenAiResponseNative};

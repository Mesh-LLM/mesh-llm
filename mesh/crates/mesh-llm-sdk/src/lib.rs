#![forbid(unsafe_code)]

#[cfg(feature = "serving")]
pub mod embedded_node;

#[cfg(feature = "serving")]
pub use embedded_node::{MeshNode, MeshNodeBuilder, MeshNodeStatus, OpenAiClient};

#[cfg(feature = "serving")]
pub use mesh_llm_embedded_runtime::initialize_host_runtime;

#[cfg(feature = "console")]
pub mod console {
    pub use mesh_llm_console_server::{
        ConsoleServerHandle, ConsoleServerOptions, start_file_console,
    };
}

#[cfg(feature = "serving")]
pub mod native_runtime {
    pub use mesh_llm_embedded_runtime::native_runtime::{
        CURRENT_MESH_VERSION, current_runtime_release, default_manifest_url,
        default_release_manifest_url, discover_local_native_runtimes,
        discover_local_native_runtimes_with_filter, discover_native_runtime_bundle_dirs,
        mesh_native_runtime_catalog, mesh_native_runtime_install_options,
        mesh_native_runtime_manifest_options, native_runtime_versions_match_current_sdk,
    };
    pub use skippy_runtime_install::*;
}

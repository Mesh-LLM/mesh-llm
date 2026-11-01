#![forbid(unsafe_code)]

#[cfg(feature = "serving")]
pub mod embedded_node;

#[cfg(feature = "serving")]
pub mod documents;

#[cfg(feature = "serving")]
pub mod live_transcription;

/// Native discovery and inference types. Model metadata contains advertised
/// effort levels; inference targets retain their provider identity.
///
/// ```no_run
/// use mesh_llm_sdk::{MeshNode, inference::{
///     ChatCompletionRequest, ChatMessage, MessageContent, ReasoningEffort,
/// }};
/// use std::time::Duration;
///
/// async fn example(invite_token: String) -> anyhow::Result<()> {
///     let node = MeshNode::builder()
///         .client()
///         .join_token(invite_token)
///         .start()
///         .await?;
///     // Wait for a suitable local or remote model advertisement.
///     let model = tokio::time::timeout(Duration::from_secs(30), async {
///         loop {
///             if let Some(model) = node.models().await?.into_iter().find(|model| {
///                 model.thinking_efforts.as_ref().is_some_and(|efforts| {
///                     efforts.iter().any(|effort| effort == "high")
///                 })
///             }) {
///                 break Ok::<_, anyhow::Error>(model);
///             }
///             tokio::time::sleep(Duration::from_millis(100)).await;
///         }
///     }).await??;
///     let request = ChatCompletionRequest {
///         messages: vec![ChatMessage {
///             role: "user".into(),
///             content: Some(MessageContent::Text("Explain lock ordering.".into())),
///             extra: Default::default(),
///         }],
///         reasoning_effort: Some(ReasoningEffort::High),
///         max_tokens: Some(512),
///         ..Default::default()
///     };
///     let response = node.chat_completion(model.target, request).await?;
///     println!("{response}");
///     node.shutdown().await?;
///     Ok(())
/// }
/// ```
#[cfg(feature = "serving")]
pub mod inference {
    pub use mesh_llm_embedded_runtime::native::{
        ChatCompletionRequest, ChatCompletionStream, ChatMessage, MessageContent, Model,
        ModelTarget, ReasoningEffort,
    };
}

#[cfg(feature = "serving")]
pub mod models;

#[cfg(feature = "serving")]
pub use embedded_node::{
    MeshMembershipConfig, MeshMembershipStatus, MeshNode, MeshNodeBuilder, MeshNodeStatus,
    OpenAiClient,
};

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

/// Entry point for a host binary that the engine re-execs as a built-in
/// plugin (`<exe> --plugin <name>`), such as the blobstore.
#[cfg(feature = "serving")]
pub mod plugin {
    pub use mesh_llm_host_runtime::plugin::run_plugin_process;
}

/// Payment control and data types for driving the engine's payments over
/// its control API: policy, pricing, invoices and wallet status.
#[cfg(feature = "payments")]
pub mod payments {
    pub use mesh_llm_payments::{control, invoice, ledger, pricing, wallet};
}

/// Node identity keys on disk.
#[cfg(feature = "identity")]
pub use mesh_llm_identity as identity;

/// Install plugin archives into the local plugin store.
#[cfg(feature = "plugin-install")]
pub use mesh_llm_plugin_manager as plugin_install;

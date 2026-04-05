//! Shared runtime and protocol helpers for mesh-llm plugins.
//!
//! For simple tools-only plugins, implement [`Plugin::list_tools`] and
//! [`Plugin::call_tool`] instead of overriding raw RPC dispatch. Prompt and
//! resource plugins can likewise use the typed prompt/resource hooks before
//! falling back to [`Plugin::handle_rpc`] for custom MCP methods.

mod context;
mod error;
mod helpers;
mod io;
mod manifest;
mod runtime;

pub use async_trait::async_trait;
pub use context::PluginContext;
pub use error::{PluginError, PluginResult, PluginRpcResult, STARTUP_DISABLED_ERROR_CODE};
pub use helpers::{
    accept_bulk_transfer_message, bulk_transfer_message, bulk_transfer_sequence,
    cancel_task_result, channel_message, complete_result, empty_object_schema, get_prompt_result,
    get_task_payload_result, get_task_result, json_bytes, json_channel_message,
    json_reply_channel_message, json_response, json_schema_for, json_schema_tool, json_string,
    list_prompts, list_resource_templates, list_resources, list_tasks, list_tools,
    parse_get_prompt_request, parse_optional_json, parse_read_resource_request, parse_rpc_params,
    parse_tool_call_request, plugin_server_info, plugin_server_info_full, prompt, prompt_argument,
    read_resource_result, resource_template, structured_tool_result, task, text_resource,
    tool_error, tool_with_schema, BulkTransferSequence, CompletionFuture, CompletionRouter,
    JsonToolFuture, PromptFuture, PromptRouter, ResourceFuture, ResourceRouter, SubscriptionSet,
    TaskCancelFuture, TaskInfoFuture, TaskListFuture, TaskRecord, TaskResultFuture, TaskRouter,
    TaskStore, ToolCallRequest, ToolFuture, ToolRouter,
};
pub use io::{
    bind_side_stream, connect_from_env, read_envelope, send_bulk_transfer_message,
    send_channel_message, write_envelope, LocalListener, LocalStream,
};
pub use manifest::{
    capability, http_binding, http_delete, http_get, http_patch, http_post, http_put,
    mcp_completion, mcp_http_endpoint, mcp_prompt, mcp_resource, mcp_resource_template,
    mcp_stdio_endpoint, mcp_tcp_endpoint, mcp_tool, mcp_unix_socket_endpoint,
    openai_http_inference_endpoint, plugin_manifest, EndpointBuilder, HttpBindingBuilder,
    ManifestEntry, McpCompletionBuilder, McpPromptBuilder, McpResourceBuilder,
    McpResourceTemplateBuilder, McpToolBuilder, PluginManifestBuilder,
};
pub use runtime::{
    EnsureInferenceEndpointRequest, EnsureInferenceEndpointResponse, EnsureInferenceWorkerRequest,
    EnsureInferenceWorkerResponse, MeshVisibility, Plugin, PluginInitializeRequest, PluginMetadata,
    PluginRuntime, PluginStartupPolicy, PrepareMoeShardRequest, PrepareMoeShardResponse,
    SimplePlugin,
};

#[allow(dead_code)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/meshllm.plugin.v1.rs"));
}

pub const PROTOCOL_VERSION: u32 = 1;

#[macro_export]
macro_rules! plugin_manifest {
    ($($item:expr),* $(,)?) => {{
        let mut builder = $crate::plugin_manifest();
        $(
            builder = builder.item($item);
        )*
        builder.build()
    }};
}

#[macro_export]
macro_rules! plugin {
    (
        metadata: $metadata:expr,
        $(capabilities: [$($capability:expr),* $(,)?],)?
        $(mcp: [$($mcp:expr),* $(,)?],)?
        $(http: [$($http:expr),* $(,)?],)?
        $(endpoints: [$($endpoint:expr),* $(,)?],)?
        $(tool_router: $tool_router:expr,)?
        $(prompt_router: $prompt_router:expr,)?
        $(resource_router: $resource_router:expr,)?
        $(completion_router: $completion_router:expr,)?
        $(task_router: $task_router:expr,)?
    ) => {{
        let manifest = $crate::plugin_manifest![
            $($($capability),*,)?
            $($($mcp),*,)?
            $($($http),*,)?
            $($($endpoint),*,)?
        ];
        let metadata = $metadata
            .with_capabilities(manifest.capabilities.clone())
            .with_manifest(manifest);
        let plugin = $crate::SimplePlugin::new(metadata)
            $(.with_tool_router($tool_router))?
            $(.with_prompt_router($prompt_router))?
            $(.with_resource_router($resource_router))?
            $(.with_completion_router($completion_router))?
            $(.with_task_router($task_router))?;
        plugin
    }};
}

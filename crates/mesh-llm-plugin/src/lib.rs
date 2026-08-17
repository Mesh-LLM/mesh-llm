//! Shared runtime and protocol helpers for mesh-llm plugins.
//!
//! Plugins declare services in their manifest and implement typed service
//! handlers. MCP and HTTP are host-side projections over those services.

mod context;
mod dsl;
mod error;
mod helpers;
mod internal_rpc;
mod io;
mod manifest;
mod runtime;
mod simple_plugin;

pub use async_trait::async_trait;
pub use context::PluginContext;
pub use dsl::DeclarativePluginBuilder;
pub use error::{PluginError, PluginResult, PluginRpcResult, STARTUP_DISABLED_ERROR_CODE};
pub use helpers::{
    BulkTransferSequence, CompletionFuture, CompletionRouter, JsonOperationFuture, OperationFuture,
    OperationRequest, OperationRouter, PromptFuture, PromptRouter, ResourceFuture, ResourceRouter,
    SubscriptionSet, TaskCancelFuture, TaskInfoFuture, TaskListFuture, TaskRecord,
    TaskResultFuture, TaskRouter, TaskStore, accept_bulk_transfer_message, bulk_transfer_message,
    bulk_transfer_sequence, cancel_task_result, channel_message, complete_result,
    empty_object_schema, get_prompt_result, get_task_payload_result, get_task_result, json_bytes,
    json_channel_message, json_reply_channel_message, json_response, json_schema_for,
    json_schema_operation, json_string, list_prompts, list_resource_templates, list_resources,
    list_tasks, list_tools, operation_error, operation_with_schema, parse_get_prompt_request,
    parse_optional_json, parse_read_resource_request, parse_rpc_params, plugin_server_info,
    plugin_server_info_full, prompt, prompt_argument, read_resource_result, resource_template,
    structured_tool_result, task, text_resource,
};
pub use io::{
    LocalListener, LocalStream, bind_side_stream, connect_from_env, connect_side_stream,
    read_envelope, send_bulk_transfer_message, send_channel_message, write_envelope,
};
pub mod http {
    pub use crate::dsl::http::{delete, get, patch, post, put};
}
pub mod inference {
    pub use crate::dsl::inference::{openai_http, provider};
}
pub mod mesh {
    pub use crate::manifest::mesh_channel as channel;
}
pub mod events {
    pub use crate::manifest::{
        mesh_event_local_accepting as local_accepting, mesh_event_local_standby as local_standby,
        mesh_event_mesh_id_updated as mesh_id_updated, mesh_event_peer_down as peer_down,
        mesh_event_peer_up as peer_up, mesh_event_peer_updated as peer_updated,
    };
}
pub use manifest::{
    CompletionBuilder, EndpointBuilder, HttpBindingBuilder, ManifestEntry, OperationBuilder,
    PluginConfigObjectPropertyBuilder, PluginConfigSchemaBuilder, PluginConfigSettingBuilder,
    PluginManifestBuilder, PluginWebUiBuilder, PluginWebUiBundleBuilder,
    PluginWebUiConfigSectionBuilder, PluginWebUiPageBuilder, PromptBuilder, ResourceBuilder,
    ResourceTemplateBuilder, capability, completion, config_array, config_boolean, config_enum,
    config_float, config_integer, config_object, config_object_property, config_path,
    config_schema, config_setting, config_string, config_url, constraint_allowed_values,
    constraint_non_empty, constraint_positive, constraint_range, constraint_requires, http_binding,
    http_delete, http_get, http_patch, http_post, http_put, mcp_http_endpoint, mcp_stdio_endpoint,
    mcp_tcp_endpoint, mcp_unix_socket_endpoint, mesh_channel, mesh_event_local_accepting,
    mesh_event_local_standby, mesh_event_mesh_id_updated, mesh_event_peer_down, mesh_event_peer_up,
    mesh_event_peer_updated, mesh_event_subscription, openai_http_inference_endpoint, operation,
    package_manifest_json, plugin_manifest, prompt_service, resource, resource_template_service,
    web_ui, web_ui_bundle, web_ui_config_section, web_ui_page,
};
pub mod mcp {
    pub use crate::dsl::mcp::{
        completion, external_http, external_stdio, external_tcp, external_unix_socket, prompt,
        resource, resource_template, tool,
    };
}
pub use runtime::{
    InternalRpcPlugin, InternalRpcPluginBuilder, MeshVisibility, Plugin, PluginInitializeRequest,
    PluginMetadata, PluginRuntime, PluginStartupPolicy, SimplePlugin,
};

#[allow(dead_code)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/meshllm.plugin.v1.rs"));
}

pub const PROTOCOL_VERSION: u32 = 2;

#[macro_export]
macro_rules! plugin_manifest {
    ($($item:expr_2021),* $(,)?) => {{
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
        metadata: $metadata:expr_2021,
        $(startup_policy: $startup_policy:expr_2021,)?
        $(provides: [$($provide:expr_2021),* $(,)?],)?
        $(config: [$($config:expr_2021),* $(,)?],)?
        $(web_ui: [$($web_ui:expr_2021),* $(,)?],)?
        $(mesh: [$($mesh:expr_2021),* $(,)?],)?
        $(events: [$($event:expr_2021),* $(,)?],)?
        $(mcp: [$($mcp:expr_2021),* $(,)?],)?
        $(http: [$($http:expr_2021),* $(,)?],)?
        $(inference: [$($inference:expr_2021),* $(,)?],)?
        $(health: $health:expr_2021,)?
        $(on_initialized: $on_initialized:expr_2021,)?
        $(on_channel_message: $on_channel_message:expr_2021,)?
        $(on_mesh_event: $on_mesh_event:expr_2021,)?
    ) => {{
        let mut builder = $crate::DeclarativePluginBuilder::new($metadata);
        $(
            builder = builder.startup_policy($startup_policy);
        )?
        $(
            $(
                builder = builder.provide($provide);
            )*
        )?
        $(
            $(
                builder = builder.config_item($config);
            )*
        )?
        $(
            $(
                builder = builder.web_ui_item($web_ui);
            )*
        )?
        $(
            $(
                builder = builder.mesh_item($mesh);
            )*
        )?
        $(
            $(
                builder = builder.event_item($event);
            )*
        )?
        $(
            $(
                builder = builder.mcp_item($mcp);
            )*
        )?
        $(
            $(
                builder = builder.http_item($http);
            )*
        )?
        $(
            $(
                builder = builder.inference_item($inference);
            )*
        )?
        $(
            builder = builder.customize(move |plugin| plugin.with_health($health));
        )?
        $(
            builder = builder.customize(move |plugin| plugin.on_initialized($on_initialized));
        )?
        $(
            builder = builder.customize(move |plugin| plugin.on_channel_message($on_channel_message));
        )?
        $(
            builder = builder.customize(move |plugin| plugin.on_mesh_event($on_mesh_event));
        )?
        builder.build()
    }};
}

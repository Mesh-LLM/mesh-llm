//! Generic OpenAI-compatible inference endpoint plugin.
//!
//! Registers any external OpenAI-compatible server (vLLM, TGI, Ollama, etc.)
//! as an inference endpoint. The actual server does all the heavy lifting —
//! model loading, GPU management, batching. This plugin just makes it
//! available to mesh-llm's request routing.
//!
//! Configuration via `~/.mesh-llm/config.toml`:
//!
//! ```toml
//! [[plugin]]
//! name = "vllm"
//! url = "http://gpu-box:8000"
//! ```
//!
//! The `url` is the base URL of the OpenAI-compatible server. The plugin
//! reads it from the `MESH_LLM_PLUGIN_URL` environment variable, which
//! is set by the plugin host from the config `url` field.

use anyhow::Result;
use mesh_llm_plugin::{
    capability, plugin_server_info, PluginMetadata, PluginRuntime, PluginStartupPolicy,
};

/// Default base URL if none is configured (localhost:8000, common for vLLM).
const DEFAULT_BASE_URL: &str = "http://localhost:8000/v1";

/// Read the endpoint URL from the environment or fall back to the default.
fn endpoint_base_url() -> String {
    // MESH_LLM_PLUGIN_URL is set by the plugin host from config.toml `url` field.
    if let Some(url) = std::env::var("MESH_LLM_PLUGIN_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        // Normalize: ensure the URL ends with /v1 or similar for the OpenAI endpoint.
        // If the user gives us "http://host:8000", append "/v1".
        // If they give "http://host:8000/v1", use as-is.
        if url.ends_with("/v1") || url.ends_with("/v1/") {
            url.trim_end_matches('/').to_string()
        } else {
            format!("{}/v1", url.trim_end_matches('/'))
        }
    } else {
        DEFAULT_BASE_URL.to_string()
    }
}

fn build_plugin(name: String) -> mesh_llm_plugin::SimplePlugin {
    let base_url = endpoint_base_url();
    let health_url = base_url.clone();
    let display_name = name.clone();

    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            name,
            crate::VERSION,
            plugin_server_info(
                &format!("mesh-{display_name}"),
                crate::VERSION,
                &format!("{display_name} Inference Endpoint"),
                &format!("Registers {display_name} as an OpenAI-compatible inference endpoint."),
                Some(
                    "Exposes an external OpenAI-compatible inference server to mesh-llm.",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/openai_compatible"),
        ],
        inference: [
            mesh_llm_plugin::inference::openai_http(&display_name, base_url.clone())
                .managed_by_plugin(false),
        ],
        health: move |_context| {
            let health_url = health_url.clone();
            Box::pin(async move { Ok(format!("base_url={health_url}")) })
        },
    }
}

pub(crate) async fn run_plugin(name: String) -> Result<()> {
    PluginRuntime::run(build_plugin(name)).await
}

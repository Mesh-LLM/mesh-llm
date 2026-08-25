use anyhow::{Context, Result, bail};
use mesh_llm_sdk::MeshNode;
use serde_json::{Value, json};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    let provider_root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: apple_system <provider-runtime-bundle-or-parent>")?;
    let api_port = free_local_port()?;
    let console_port = free_local_port()?;
    let node = MeshNode::builder()
        .serve()
        .provider_runtime_root(provider_root)
        .api_port(api_port)
        .console_port(console_port)
        .startup_timeout(Duration::from_secs(30))
        .start()
        .await?;
    let openai = node.openai_client();
    let versioned_model = wait_for_apple_model(&openai).await?;
    let completion = openai
        .chat_completions(json!({
            "model": versioned_model,
            "messages": [{
                "role": "user",
                "content": "Reply with exactly: rust sdk apple ready"
            }],
            "temperature": 0,
            "max_tokens": 32
        }))
        .await?;
    let tool = openai
        .chat_completions(json!({
            "model": "apple/system",
            "messages": [{
                "role": "user",
                "content": "Use the tool with key: rust-sdk"
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "mesh_fixture_lookup",
                    "description": "Look up a fixture",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"]
                    }
                }
            }],
            "temperature": 0
        }))
        .await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "pass",
            "model": "apple/system",
            "versioned_model": versioned_model,
            "provider_discovery": "rust_sdk_typed_config",
            "completion_content": completion_content(&completion),
            "tool_executions": tool.get("mesh_tool_executions"),
        }))?
    );
    node.shutdown().await
}

async fn wait_for_apple_model(client: &mesh_llm_sdk::OpenAiClient) -> Result<String> {
    for _ in 0..300 {
        if let Ok(models) = client.models().await
            && let Some(model) = versioned_apple_model(&models)
        {
            return Ok(model);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    bail!("apple/system did not become available through the embedded Rust SDK host")
}

fn versioned_apple_model(models: &Value) -> Option<String> {
    models.get("data")?.as_array()?.iter().find_map(|model| {
        model
            .get("id")?
            .as_str()
            .filter(|id| id.starts_with("apple/system@"))
            .map(str::to_string)
    })
}

fn completion_content(completion: &Value) -> Option<&str> {
    completion
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
}

fn free_local_port() -> Result<u16> {
    Ok(TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

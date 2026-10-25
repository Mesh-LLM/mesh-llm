# mesh-llm-embedded-runtime

`mesh-llm-embedded-runtime` exposes the in-process full Mesh LLM node API for
applications that want a local OpenAI-compatible `/v1` endpoint without
spawning the `mesh-llm` CLI as a sidecar.

This crate is intentionally separate from the default `mesh-llm-sdk` facade so
client-only consumers do not pull in the full host runtime graph.

The embedded builder and configuration support `stage_transport_policy` through
`protocol::StageTransportPolicy`. `RequireProvider` restricts split activation
traffic to the registered application provider on every participant; the default
`PreferProvider` permits the existing fallback. See the
[SDK stage transport example](../mesh-llm-sdk/README.md#application-stage-transport)
for registration, readiness and enforcement details.

## Example

```rust,no_run
use mesh_llm_embedded_runtime::{
    EmbeddedMeshNodeConfig, EmbeddedMeshNodeMode, start_embedded_node,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let node = start_embedded_node(
        EmbeddedMeshNodeConfig::builder()
            .mode(EmbeddedMeshNodeMode::Serve)
            .model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
            .api_port(9337)
            .console_port(3131)
            .build(),
    )
    .await?;

    println!("OpenAI API: {}", node.api_base_url());
    println!("console: {}", node.console_url());

    node.stop().await?;
    Ok(())
}
```


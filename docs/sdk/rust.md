# Rust SDK

Use `mesh-llm-sdk` from crates.io for Rust client and serving applications.

## Install

Client-only applications can use the default features:

```toml
[dependencies]
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread", "sync"] }
mesh-llm-sdk = "0.72.1"
```

Serving applications need the `serving` feature:

```toml
[dependencies]
anyhow = "1"
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
mesh-llm-sdk = { version = "0.72.1", features = ["serving"] }
```

Add `console` with `serving` when the embedded node should serve packaged web
console assets.

## Client: Public Mesh

```rust
use mesh_llm_sdk::{
    ClientBuilder, OwnerKeypair, PublicMeshQuery, select_public_mesh,
};

let owner = OwnerKeypair::generate();
let public_mesh = select_public_mesh(PublicMeshQuery {
    model: Some("Qwen3".to_string()),
    ..Default::default()
})
.await?;

let mut client = ClientBuilder::from_public_mesh(owner, &public_mesh)?
    .with_direct_mesh_transport()
    .build()?;
client.join().await?;

let models = client.list_models().await?;
let model = models.first().expect("public mesh has models").id.clone();
let reply = chat_once(&client, model, "Say hello from the public mesh.").await?;
println!("{reply}");

client.disconnect().await;
```

## Client: Private Mesh

```rust
use mesh_llm_sdk::{ClientBuilder, InviteToken, OwnerKeypair};

let owner = OwnerKeypair::generate();
let invite = std::env::var("MESH_PRIVATE_INVITE")?.parse::<InviteToken>()?;

let mut client = ClientBuilder::new(owner, invite)
    .with_direct_mesh_transport()
    .build()?;
client.join().await?;

let models = client.list_models().await?;
let model = models.first().expect("private mesh has models").id.clone();
let reply = chat_once(&client, model, "Say hello from the private mesh.").await?;
println!("{reply}");

client.disconnect().await;
```

## Client Inference Helper

```rust
use mesh_llm_sdk::events::{Event, EventListener};
use mesh_llm_sdk::{ChatMessage, ChatRequest, MeshClient};
use std::sync::Arc;
use tokio::sync::mpsc;

struct Listener {
    tx: mpsc::UnboundedSender<Event>,
}

impl EventListener for Listener {
    fn on_event(&self, event: Event) {
        let _ = self.tx.send(event);
    }
}

async fn chat_once(client: &MeshClient, model: String, prompt: &str) -> anyhow::Result<String> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let request_id = client.chat(
        ChatRequest {
            model,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
        },
        Arc::new(Listener { tx }),
    ).0;

    let mut output = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            Event::TokenDelta { request_id: id, delta } if id == request_id => {
                output.push_str(&delta);
            }
            Event::Completed { request_id: id } if id == request_id => return Ok(output),
            Event::Failed { request_id: id, error } if id == request_id => anyhow::bail!(error),
            _ => {}
        }
    }
    anyhow::bail!("request ended before completion")
}
```

## Serving: Install Runtime

Serving applications should resolve or install the recommended native runtime
before starting a serving node:

```rust
use mesh_llm_sdk::native_runtime::{
    NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};

let runtime = install_native_runtime(NativeRuntimeInstallOptions {
    selection: RuntimeSelection::Recommended,
    ..Default::default()
})
.await?;
println!("runtime: {}", runtime.runtime.path.display());
```

For app-managed caches and progress:

```rust
let outcome = install_native_runtime(NativeRuntimeInstallOptions {
    selection: RuntimeSelection::Recommended,
    cache_dir: Some(app_cache_dir.join("mesh-llm-native-runtimes")),
    bundle_dirs: vec![app_resources.join("meshllm-native-runtime")],
    progress: Some(std::sync::Arc::new(|event| {
        update_progress(event.downloaded_bytes, event.total_bytes);
    })),
    ..Default::default()
})
.await?;
```

## Serving: Public Mesh

```rust
use mesh_llm_sdk::MeshNode;

let model_ref = "unsloth/Qwen3-0.6B-GGUF:Q4_K_M";
let public_invite = std::env::var("MESH_PUBLIC_INVITE")?;

let node = MeshNode::builder()
    .serve()
    .model(model_ref)
    .join_token(public_invite)
    .start()
    .await?;

let reply = embedded_chat_once(&node, model_ref, "Say hello from a public serving node.").await?;
println!("{reply}");
node.shutdown().await?;
```

## Serving: Private Mesh

```rust
let model_ref = "unsloth/Qwen3-0.6B-GGUF:Q4_K_M";
let private_invite = std::env::var("MESH_PRIVATE_INVITE")?;

let node = MeshNode::builder()
    .serve()
    .model(model_ref)
    .join_token(private_invite)
    .start()
    .await?;

let reply = embedded_chat_once(&node, model_ref, "Say hello from a private serving node.").await?;
println!("{reply}");
node.shutdown().await?;
```

## Serving Inference Helper

```rust
use mesh_llm_sdk::MeshNode;
use serde_json::{Value, json};

async fn embedded_chat_once(node: &MeshNode, model: &str, prompt: &str) -> anyhow::Result<String> {
    let response = node.openai_client().chat_completions(json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": 64,
        "temperature": 0,
    })).await?;

    Ok(response
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string())
}
```

## Console

Rust applications that package the built console can enable the optional
`console` feature and use the file-backed console server:

```rust
let console = mesh_llm_sdk::console::start_file_console(
    mesh_llm_sdk::console::ConsoleServerOptions {
        asset_dir: "/path/to/packaged/console".into(),
        port: 0,
        listen_all: false,
    },
).await?;
```

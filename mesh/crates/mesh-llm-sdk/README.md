# mesh-llm-sdk

Rust facade for the real embedded Mesh LLM node. `MeshNode::builder()` supports
client (`client()`), serve-only (`serve_only()`), and combined (`serve()`) roles.
The legacy thin client re-export is removed. The default feature is `serving`,
which includes the embedded runtime and native runtime management APIs.

See [Rust SDK usage](../../docs/sdk/rust.md) and
[native runtime guidance](../../docs/SDK.md#native-runtime-artifacts).

## Install the serving runtime

Client-only nodes need no native serving runtime. Before starting a serve-only
or combined node, check the cached runtime with
`native_runtime_versions_match_current_sdk`, then explicitly install a
compatible runtime when needed:

```rust,no_run
use mesh_llm_sdk::native_runtime::{
    NativeRuntimeInstallOptions, RuntimeSelection, current_runtime_release,
    current_skippy_abi_version, install_native_runtime,
    mesh_native_runtime_install_options, native_runtime_versions_match_current_sdk,
};
use mesh_llm_sdk::{MeshNode, initialize_host_runtime};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let outcome = install_native_runtime(NativeRuntimeInstallOptions {
        release_version: current_runtime_release().to_string(),
        skippy_abi_version: Some(current_skippy_abi_version()),
        selection: RuntimeSelection::Recommended,
        ..mesh_native_runtime_install_options()
    }).await?;
    let runtime = outcome.runtime;
    anyhow::ensure!(native_runtime_versions_match_current_sdk(
        &runtime.mesh_version,
        &runtime.manifest.runtime.skippy_abi,
    ));
    runtime.load_plan()?;
    initialize_host_runtime().await?;
    let node = MeshNode::builder().serve().start().await?;
    node.shutdown().await?;
    Ok(())
}
```

The installer resolves the
recommended runtime against the current host profile and returns a compatible
cached runtime when available. The load plan verifies its libraries before
startup. Embedded node startup only loads a compatible cached
runtime; it never downloads one. Install before starting a serving node when
the cache has no matching runtime.

## Serve into further meshes

A serving node can serve the models it has loaded into further meshes without
loading them again. Each membership gets its own node identity, owner
allowlist, peers and loopback OpenAI API port, and its peers never learn about
the node's other meshes. A membership never peers with the node or with its
other memberships, and refuses peers that announce a different mesh. Requests
from a membership's peers run on this node's model only. Requests to a
membership's API port run on this node's model when it serves it, and
otherwise on that mesh's peers, never on the peers of the node's other meshes.

```rust,no_run
use mesh_llm_sdk::{MeshMembershipConfig, MeshNode};

async fn share(node: &MeshNode, team_owner_ids: Vec<String>, team_tokens: Vec<String>)
    -> anyhow::Result<()> {
    let team = node
        .add_membership(MeshMembershipConfig {
            name: "team".into(),
            join_tokens: team_tokens, // every member's, this node's own included
            trusted_owners: team_owner_ids,
            api_port: 9338,
            node_key_path: None,
            enumerate_host: false,
        })
        .await?;
    // Other members join this node by `team.invite_token`.
    println!("{} {}", team.api_base_url, team.invite_token);
    node.remove_membership("team").await?;
    Ok(())
}
```

A membership needs an owner key (`owner_key`), since its peers admit it by
owner. Join tokens must be unsigned invite tokens; `add_membership` checks them
before it returns, and `memberships()` reports a failing join in `join_error`.
The invite token carries the node's addresses: the membership keeps its key and
UDP port across restarts when it can, and when the port is taken the token
changes and has to be shared again.

Adding a membership with a name already in use replaces it, which is how to
change its allowlist or tokens. A replacement that fails its checks (name,
port, tokens, key) leaves the old one running. One that fails while starting
has already stopped the old one; it starts the old one again, and the error
says whether that worked.

To keep the node's own API port to the local models only, start the node
without join tokens and with an `Allowlist` trust policy that lists no other
owners. A membership advertises what the node serves, not the model files on
disk or what the node's own mesh asks for. It carries no split-stage
transport, plugins or owner control, and it shares the node's payment
settings. Its allowlist is its own, but owner and certificate revocations in
the node's trust store apply to it too.

## Native model discovery and inference

Join a mesh, read the local and remote model catalog, choose an advertised
thinking effort, and call inference without a loopback HTTP request:

```rust,no_run
use mesh_llm_sdk::{MeshNode, inference::{
    ChatCompletionRequest, ChatMessage, MessageContent, ReasoningEffort,
}};

async fn infer(invite_token: String) -> anyhow::Result<serde_json::Value> {
    let node = MeshNode::builder()
        .client()
        .join_token(invite_token)
        .start()
        .await?;

    // Discovery is asynchronous. Give peers time to advertise a suitable model.
    let model = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            let models = node.models().await?;
            if let Some(model) = models.into_iter().find(|model| {
                model.thinking_efforts.as_ref().is_some_and(|efforts| {
                    efforts.iter().any(|effort| effort == "high")
                })
            }) {
                break Ok::<_, anyhow::Error>(model);
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }).await??;

    let response = node.chat_completion(model.target, ChatCompletionRequest {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(
                "Explain why a lock-order inversion can deadlock.".into(),
            )),
            extra: Default::default(),
        }],
        reasoning_effort: Some(ReasoningEffort::High),
        max_tokens: Some(512),
        ..Default::default()
    }).await?;
    node.shutdown().await?;
    Ok(response)
}
```

Each catalog entry retains its `target.provider_node_id` and exact
`target.serving_id`, plus the provider's descriptor. Providers of the same model
are separate entries, so one provider's thinking levels never authorize a request
to another provider. Local targets execute in process; remote targets use native
mesh transport. Unsupported native peers return an error rather than falling
back to HTTP. Startup readiness also uses the runtime's native control channel.

`thinking_efforts` contains only the model-specific strings observed by the
selected template's render probe, in their reported order. `None` means unknown,
including an older peer or disagreement between loaded instances; `Some([])`
means no tested effort visibly changed the template. Neither means thinking is
disabled. Advertisements update on load and unload. These observations do not
guarantee that generation obeys the controls.

Pass a `ReasoningEffort` matching an advertised string, or omit it for the
model's default. The exact target and current advertised controls are checked
again before execution. An unavailable target or changed controls produces an
error instead of silently selecting a different provider.

Use `node.chat_completion_stream(target, request).await?` for streaming, then
read deltas with `stream.next_chunk().await`:

```rust,no_run
use mesh_llm_sdk::{MeshNode, inference::{ChatCompletionRequest, ModelTarget}};

async fn stream_answer(
    node: &MeshNode,
    target: ModelTarget,
    request: ChatCompletionRequest,
) -> anyhow::Result<()> {
    let mut stream = node.chat_completion_stream(target, request).await?;
    while let Some(chunk) = stream.next_chunk().await {
        let chunk = chunk?;
        println!("{chunk}");
    }
    Ok(())
}
```

The request carries the selected effort exactly as in the unary example.
Provider, effort, and transport failures can arrive as errors from `next_chunk`.
Dropping the stream or shutting down the node cancels its producer. A retained
stream reports shutdown as an error rather than successful truncated output.
Native remote inference cannot consume a model requiring payment; use a provider
that permits unpaid inference or a separate supported payment integration.

The legacy `status()`, `openai_client()` and live-transcription APIs still use
HTTP. They are not used by the native discovery/inference path above.

## Live transcription

A node serving a speech-recognition model (Whisper) transcribes live audio.
The caller supplies 16-bit mono PCM at any sample rate, for example from a
microphone, and reads transcript events while the speaker talks. Any node can
host the session; the mesh relays it to the node serving the model.

```rust,no_run
use mesh_llm_sdk::MeshNode;
use mesh_llm_sdk::live_transcription::{LiveTranscriptionEvent, LiveTranscriptionOptions};

async fn transcribe(node: &MeshNode, microphone: tokio::sync::mpsc::Receiver<Vec<i16>>)
    -> anyhow::Result<()> {
    let live = node
        .start_live_transcription(LiveTranscriptionOptions {
            sample_rate: 48_000,
            ..Default::default() // turns end on a pause; `turn_detection: None` for push to talk
        })
        .await?;
    let (audio, mut events) = live.split();
    let mut microphone = microphone;
    tokio::spawn(async move {
        while let Some(frame) = microphone.recv().await {
            if audio.send_audio(&frame).await.is_err() {
                break;
            }
        }
        audio.close().await;
    });
    while let Some(event) = events.next_event().await {
        match event {
            LiveTranscriptionEvent::Delta { text, .. } => { /* append to the live line */ }
            LiveTranscriptionEvent::Completed { transcript, .. } => { /* the finished utterance */ }
            _ => {}
        }
    }
    Ok(())
}
```

Deltas are confirmed words and never change. They trail speech by about one to
two seconds; each turn's `Completed` transcript arrives as soon as the turn ends.
`LiveTranscription::stop` finishes the current turn, waits for its transcript,
and closes the session. Dropping the session, or its audio half after `split`,
closes it too. A lost connection arrives as a final `Error` event. Sessions
started from an `OpenAiClient` send its API key as a bearer token.

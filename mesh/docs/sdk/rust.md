# Rust SDK

`mesh-llm-sdk` exposes one embedded `MeshNode`. Choose its role with the builder:

| Builder | Consumes mesh inference | Serves local models |
|---|---:|---:|
| `.client()` | yes | no |
| `.serve_only()` | no | yes |
| `.serve()` | yes | yes |

```toml
[dependencies]
mesh-llm-sdk = { version = "0.78.0" }
```

Native discovery returns one entry per provider and serving ID. Its
`thinking_efforts` advertises only model-specific effort strings, preserving
unknown (`None`) separately from an observed empty list. Local inference runs
in process and remote inference uses mesh transport, without loopback HTTP.

```rust,no_run
use mesh_llm_sdk::{MeshNode, inference::{
    ChatCompletionRequest, ChatMessage, MessageContent, ReasoningEffort,
}};

async fn example(invite_token: String) -> anyhow::Result<()> {
    let node = MeshNode::builder()
        .client()
        .join_token(invite_token)
        .start()
        .await?;
    let model = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            if let Some(model) = node.models().await?.into_iter().find(|model| {
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
            content: Some(MessageContent::Text("Explain lock ordering.".into())),
            extra: Default::default(),
        }],
        reasoning_effort: Some(ReasoningEffort::High),
        max_tokens: Some(512),
        ..Default::default()
    }).await?;
    println!("{response}");
    node.shutdown().await?;
    Ok(())
}
```

Discovery is asynchronous; the example waits for a suitable advertisement.
Use `.auto_join_public_mesh()` instead of `.join_token(token)` for a public mesh.
For combined serving/client use, select `.serve()` and add
`.model("publisher/model:quant")`. Serving needs a compatible installed native
runtime; see [runtime artifacts](../SDK.md#native-runtime-artifacts).

The selected target preserves its provider identity. Controls are validated
against the current load before execution; there is no silent fallback to a
provider with different thinking levels. Missing thinking metadata is unknown,
not disabled. Advertised values are render-only observations, not a guarantee
that generation obeys them.

For native streaming, call `chat_completion_stream(target, request).await?`
and read deltas with `stream.next_chunk().await`. Provider and effort errors may
arrive when reading chunks. Dropping the stream cancels inference; node shutdown
is reported as an error on a retained stream. Native remote inference cannot
consume a provider that requires payment. Older peers that do not implement
native inference return an error, with no HTTP fallback.

Streaming deadlines bound progress, not total duration: `chat_completion()` must
finish within five minutes, while `chat_completion_stream()` fails only after
five minutes without a chunk, so a generation that keeps producing is never cut
off. The provider applies the same inactivity window to its local backend. A
single frame write blocked by flow control, and the provider's read of the
request frame, time out after fifteen seconds.

The legacy `OpenAiClient::request` and
`OpenAiClient::stream` remain available for HTTP integrations and preserve
OpenAI-compatible bodies, including tools and response-format fields. Legacy
`status()` and live transcription also use HTTP. A serve-only node rejects
client inference at the SDK boundary.

**Migration:** The old `MeshClient` re-export and `node` feature are removed from
this facade. Migrate callers to `MeshNode`; the embedded node owns mesh join,
readiness, and shutdown. The temporary local-only `thinking_controls` API is
replaced by `models()` metadata and native `chat_completion()`.

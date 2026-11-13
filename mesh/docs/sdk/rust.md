# Rust SDK

`mesh-llm-sdk` exposes one embedded `MeshNode`. Choose its role with the builder:

| Builder | Consumes mesh inference | Serves local models |
|---|---:|---:|
| `.client()` | yes | no |
| `.serve_only()` | no | yes |
| `.serve()` | yes | yes |

```toml
[dependencies]
mesh-llm-sdk = { version = "0.79.0" }
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
            if let Some(model) = node.inference_models().await?.into_iter().find(|model| {
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

Local native backend failures preserve `mesh_llm_sdk::inference::InferenceError` in the
`anyhow::Error` chain. Use `downcast_ref::<InferenceError>()` and its `body()` or
`status()` to distinguish context exhaustion, invalid requests and retryable
backend failures without parsing display text. Remote failures retain the peer's
error message; they do not carry a local backend error type. This Rust re-export
does not change the native request, chunk or cross-language error representation.

Backend-specific request options are retained in `ChatCompletionRequest::extra`.
For a local llama-compatible provider, `id_slot` selects a lane, `cache_prompt`
controls prompt reuse, and `chat_template_kwargs` preserves `enable_thinking` and
`preserve_thinking`. These are explicit extensions, not aliases for
`reasoning_effort`; bind them to the selected provider and check its support.
Image content parts, tool definitions/results and usage options use the existing
typed request fields. Native transport does not rename or discard extensions.

Streaming deadlines bound progress, not total duration: `chat_completion()` must
finish within five minutes, while `chat_completion_stream()` fails only after
five minutes without a chunk, so a generation that keeps producing is never cut
off. The provider applies the same inactivity window to its local backend. A
single frame write blocked by flow control, and the provider's read of the
request frame, time out after fifteen seconds.


## Discover every advertised workload

`node.models()` reads the primary mesh inventory directly in process, including
local and remote non-LLM models. `openai_client()?.models()` remains the LLM-only
compatibility list.

```rust,no_run
use mesh_llm_sdk::models::{MeshModelQuery, ModelWorkloadClass};
# async fn example(node: &mesh_llm_sdk::MeshNode) -> anyhow::Result<()> {
let page = node.models().list(MeshModelQuery::default()).await?;
let ocr = node.models().list(MeshModelQuery {
    workload_class: Some(ModelWorkloadClass::DocumentOcr),
    available: Some(true),
    ..Default::default()
}).await?;
let details = node.models().get("org/model@revision#profile").await?;
if let Some(cursor) = page.next_cursor {
    let following = node.models().list(MeshModelQuery {
        cursor: Some(cursor),
        ..Default::default()
    }).await?;
    let _ = following;
}
let _ = (ocr, details);
# Ok(())
# }
```

Filters combine with AND. Pages contain `data` and `next_cursor`; keep filters
unchanged when continuing and restart when a snapshot cursor expires. Model
metadata and provider observations are typed. A provider's runtime context limit
or thinking controls do not describe every replica. Missing metadata remains
unknown. See [Mesh model discovery](../MODEL_DISCOVERY.md) for complete shapes.

## Invoke document OCR

Upload bytes to a model discovered with the `DocumentOcr` workload:

```rust,no_run
use mesh_llm_sdk::documents::DocumentOcrOptions;
# async fn example(node: &mesh_llm_sdk::MeshNode, pdf: Vec<u8>) -> anyhow::Result<()> {
let mut options = DocumentOcrOptions::new("ocr-model");
options.filename = "scan.pdf".into();
options.content_type = "application/pdf".into();
options.response_format = Some("markdown".into());
let response = node.openai_client()?.document_ocr(pdf, options).await?;
// The default method rejects provider HTTP errors.
if response.status_code == 200 {
    let markdown = response.body;
    let _ = markdown;
}
# Ok(())
# }
```

Uploads must be nonempty and at most 64 MiB. `max_pages`, when supplied, must be
positive. Output formats are `json`, `text` and `markdown`. Validation/transport
failures and provider HTTP errors are errors. Use `document_ocr_raw` for
application status handling; it preserves provider status, content type and body.

`OpenAiClient::request` and `OpenAiClient::stream` preserve complete
OpenAI-compatible request bodies, including tools and response-format fields,
and reject non-success provider responses by default. For deliberate status
handling, use `request_raw` or `stream_raw`; `document_ocr_raw` provides the same
choice for multipart OCR. `OpenAiRequestError` retains `status_code`,
`content_type` and `body`. A serve-only node rejects `openai_client()` at the
SDK boundary.

**Migration:** The old `MeshClient` re-export and `node` feature are removed from
this facade. Migrate callers to `MeshNode`; the embedded node owns mesh join,
local HTTP routing, and shutdown. This changes the SDK API but not mesh wire
protocols.

Previously unchecked Rust `request`/`stream` callers that inspect provider HTTP
errors must migrate to `request_raw`/`stream_raw`. The checked default now
matches the other language SDKs; raw responses remain explicitly available.
The temporary local-only `thinking_controls` API is replaced by `models()`
metadata and native `chat_completion()`.

## Live transcription

```rust,no_run
use mesh_llm_sdk::live_transcription::LiveTranscriptionOptions;
# async fn example(node: &mesh_llm_sdk::MeshNode, pcm16_samples: &[i16]) -> anyhow::Result<()> {
let mut session = node.start_live_transcription(LiveTranscriptionOptions {
    model: Some("speech-model".into()),
    sample_rate: 16_000,
    turn_detection: None,
    step_ms: Some(400),
    tentative: true,
    ..Default::default()
}).await?;
session.send_audio(pcm16_samples).await?;
session.commit().await?;
if let Some(event) = session.next_event().await {
    println!("{event:?}");
}
let remaining_events = session.stop().await?;
println!("{remaining_events:?}");
# Ok(())
# }
```

Supply signed 16-bit mono samples, or use `send_audio_f32` for float samples.
`step_ms` accepts 200 to 5000 ms; `None` keeps the node's 1000 ms default.
Shorter steps run more transcription passes. `tentative` defaults to `false`.

Confirmed deltas are append-only. Recognition errors in confirmed text cannot
be corrected by later passes or the completed event. Shorter transcription
intervals do not guarantee better recognition accuracy.

Native transcription failures end the session after an error event. Start a
new session to retry. Request-validation errors may leave the session open.

Each `Tentative` event replaces the previous unconfirmed text for its `item_id`;
empty text clears it. Append `Delta` text to the confirmed transcript and let
`Completed` replace the whole turn, including tentative text. Read events
concurrently with microphone input. `stop()` consumes the session and returns
unread events; dropping the session on error closes its transport.

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

```rust
use mesh_llm_sdk::MeshNode;

# async fn example() -> anyhow::Result<()> {
let node = MeshNode::builder()
    .client()
    .auto_join_public_mesh()
    .start()
    .await?;
let client = node.openai_client()?;
let models = client.models().await?;
println!("{models}");
node.stop().await?;
# Ok(())
# }
```

For a private mesh, replace `.auto_join_public_mesh()` with `.join_token(token)`.
For serve-only or combined use, select `.serve_only()` or `.serve()` and add
`.model("publisher/model:quant")`. Serving needs a compatible installed native
runtime; see [runtime artifacts](../SDK.md#native-runtime-artifacts).

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

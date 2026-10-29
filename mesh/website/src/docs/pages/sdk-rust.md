---
title: Rust SDK
---

# Rust SDK

Use [`mesh-llm-sdk`](https://crates.io/crates/mesh-llm-sdk) to embed one `MeshNode`.
Its default `serving` feature includes the embedded node APIs. Select the role
with `.client()`, `.serve_only()`, or `.serve()` (combined).

## Install and join

```toml
[dependencies]
anyhow = "1"
mesh-llm-sdk = "{{ site.sdkVersion }}"
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use mesh_llm_sdk::MeshNode;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let node = MeshNode::builder()
        .client()
        .auto_join_public_mesh()
        .start()
        .await?;
    let llms = node.openai_client()?.models().await?;
    println!("{llms}");
    node.stop().await?;
    Ok(())
}
```

For a private mesh, replace `.auto_join_public_mesh()` with `.join_token(token)`.
Use `.owner_key(path)` when an owner keystore is required. Client mode does not
load a native model-serving runtime. To serve local models, install a matching
runtime explicitly, choose `.serve_only()` or `.serve()`, and add
`.model("publisher/model:quant")`. Node startup never downloads a runtime
implicitly. See [SDK overview](/docs/pages/sdk/).

The former thin client facade, `ClientBuilder`, and `client`/`node` feature split
have been removed from this SDK. Migrate callers to the embedded `MeshNode`.

## Discover all mesh-advertised models

`node.models()` queries the primary mesh inventory directly in process. It
includes local and remote LLM, speech, embedding, decision and OCR models.
`node.openai_client()?.models()` remains the LLM-only compatibility list.

```rust,no_run
use mesh_llm_sdk::models::{MeshModelQuery, ModelWorkloadClass};

let page = node.models().list(MeshModelQuery::default()).await?;
let ocr = node.models().list(MeshModelQuery {
    workload_class: Some(ModelWorkloadClass::DocumentOcr),
    available: Some(true),
    ..Default::default()
}).await?;
let details = node.models().get("org/model@revision#profile").await?;
if let Some(cursor) = page.next_cursor {
    let next = node.models().list(MeshModelQuery {
        cursor: Some(cursor),
        ..Default::default()
    }).await?;
}
```

Pages contain typed `data` and `next_cursor`. Filters combine with AND; keep
filters unchanged when continuing and restart if the snapshot expires. Metadata,
capabilities, runtime context limits and thinking/payment observations stay with
the provider that supplied them. Missing metadata means unknown. See
[Mesh model discovery](/docs/pages/model-discovery/) for the complete contract.

## Inference and OCR

Compatibility inference uses the node's HTTP listener. `request` and `stream`
preserve complete OpenAI-compatible bodies, including tools, multimodal content
and response-format fields, and check HTTP success by default. Use
`request_raw` or `stream_raw` when the application must inspect provider error
responses. `OpenAiRequestError` retains `status_code`, `content_type` and `body`.
Existing callers that relied on the former unchecked defaults must migrate to
the explicit raw methods. A serve-only node rejects `openai_client()` at the
SDK boundary.

Upload document bytes to an advertised OCR model with the dedicated multipart
method:

```rust,no_run
use mesh_llm_sdk::documents::DocumentOcrOptions;

let mut options = DocumentOcrOptions::new("ocr-model");
options.filename = "scan.pdf".into();
options.content_type = "application/pdf".into();
options.response_format = Some("markdown".into());
let response = node.openai_client()?.document_ocr(pdf_bytes, options).await?;
// Successful provider JSON, text, or Markdown remains intact.
println!("{} {}", response.status_code, response.body);
```

The file must be nonempty and at most 64 MiB. `max_pages`, when supplied, must
be positive; formats are `json`, `text` and `markdown`. Transport and validation
failures and provider HTTP errors return errors. `document_ocr_raw` preserves
provider status, content type and body for application status handling.

## Memberships and live transcription

`add_membership`, `remove_membership`, and `memberships` manage additional meshes
without loading local models again. Each membership has its own identity,
allowlist, peers and API port. The native model catalog above remains scoped to
the node's primary mesh.

`start_live_transcription` accepts `LiveTranscriptionOptions` and returns a
session for sending signed 16-bit mono PCM, committing turns, reading transcript
events, and stopping. The sample rate can be 8–96 kHz. Defaults use 16 kHz and
server-side turn detection. Disabling turn detection enables explicit commit.

Always call `stop`/`shutdown` as the application exits. `status`,
`api_base_url`, `console_url` and `invite_token` expose node lifecycle details.
Native catalog queries avoid HTTP; this does not mean the whole SDK lifecycle
or inference transport is HTTP-free.

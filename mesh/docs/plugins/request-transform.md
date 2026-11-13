# Request transforms (`request_transform.v1`)

> Draft contract. It may change before it is released.

A request transform lets an installed plugin rewrite an inference request
before the mesh routes it. With a transform, a plugin can turn content the
model cannot read, such as a PDF attachment, into content it can read, and
clients don't have to change anything. Exchange policies
([lifecycle contract](openai-exchange-lifecycle.md)) can only observe a
request or allow and deny it. Virtual models ([virtual models](virtual-models.md))
only see requests addressed to their own model name.

## Where it runs

The transform runs on the node the client sent the request to:

1. The host reads the request from `/v1/chat/completions`, `/v1/responses` or
   `/v1/messages`.
2. Core normalization converts Responses and Messages bodies to the Chat
   Completions shape. Request objects (`mesh://blob/...`) are resolved.
3. **Request transforms run** over that Chat-shaped body.
4. The host routes the request, to a local model, a virtual model or a mesh
   peer. The response is adapted back to the API the client called.

A request that a peer forwarded over the mesh already passed through the
transforms on its client's node, so the serving node doesn't run them again.
Other paths, such as embeddings, `count_tokens` and models, are never
transformed.

## Plugin contract

Declare the capability in the manifest and register a handler:

```rust
use mesh_llm_plugin::request_transform::{request_transform, RequestTransformOutput};

let plugin = SimplePlugin::new(/* ... */)
    .with_manifest(plugin_manifest![request_transform()])
    .with_request_transform(|input, _context| {
        Box::pin(async move {
            // input.path:        "/v1/chat/completions" (the normalized shape)
            // input.client_path: the path the client called, e.g. "/v1/messages"
            // input.body:        the Chat Completions request
            let mut body = input.body;
            // ... rewrite body["messages"] ...
            Ok(RequestTransformOutput::replaced(body))
        })
    });
```

The handler returns either `RequestTransformOutput::replaced(body)` or
`RequestTransformOutput::unchanged()`. A plugin that implements `Plugin`
directly overrides `transform_request` instead. On the wire this is
`InvokeServiceRequest` with `SERVICE_KIND_REQUEST_TRANSFORM`. It is
host-private and is never projected as an MCP tool or an HTTP route.

A plugin whose manifest declares `request_transform.v1` fails to start on a
host that does not support it. Without this check, an older host could serve
requests the plugin was installed to rewrite.

## Rules

- **Order:** every plugin that declares the capability runs, in plugin-name
  order. Each one sees the previous one's output.
- **Fail closed:** an error, a timeout (300 s, which leaves room for OCR) or an
  unavailable transform plugin rejects the request with `400`. The request
  never reaches a model untransformed. The error message goes back to the
  client.
- **Content only:** the replacement must be a JSON object with the same
  `model`, `stream` and output budget (`max_tokens`, `max_completion_tokens`,
  `max_output_tokens`, `n_predict`) as the request. Routing, the context check
  and the response format were already chosen from them.
- **Size:** the replacement may be at most 8 MiB once serialized, the limit a
  peer applies to a forwarded request.
- A plugin that declared a transform and is restarting or has failed is
  unavailable, so requests are rejected until it is back.
- **Input size:** the request reaches the plugin in one control frame, so a
  request whose normalized body (with request objects inlined) is over about
  16 MiB is rejected with `400` before the plugin is called.
- **Start eagerly:** do not set `lazy_start` on a transform plugin. A deferred
  plugin has not declared its capabilities yet, so like any other capability
  its transform is not seen, and requests pass untransformed, until something
  else starts it.
- The transform runs once per client request. It runs before routing, retries
  and virtual-model orchestration.

## Files in normalized requests

The host normalizes file attachments to Chat content parts:

| Client sends | Transform sees |
|---|---|
| Chat `{"type":"file","file":{"file_data"\|"file_id", "filename"}}` | unchanged |
| Responses `{"type":"input_file","file_data"\|"file_id"\|"file_url", "filename"}` | `{"type":"input_file","input_file":{...same keys}}` |
| Messages `{"type":"document","source":{"type":"base64","media_type","data"},"title"}` | `{"type":"input_file","input_file":{"file_data":"data:<media_type>;base64,<data>","mime_type","filename":<title>}}` |
| Messages document `source` of type `file` / `url` | `input_file` with `file_id` / `file_url` |
| Messages document `source` of type `text` | a text part |

A Messages document with a non-empty `context` or with `citations.enabled:
true` is rejected with `400`: neither survives normalization.

Messages documents inside `tool_result` content, which Claude Code's Read tool
sends for PDFs, are normalized the same way inside the resulting `tool`
message.

Parts made from a flat Responses `input_file` or a Messages `document` also
carry `"mesh_requires_transform": true`. These forms were rejected before
request transforms existed, and they still are when none converts them: if a
part with this flag is left after every transform has run (or no plugin
declares one), the request fails with `400`. A transform handles one by
replacing the part, for example with text. Chat `file` parts and nested
Responses `input_file` parts have no flag and reach the backend as before.

## Not covered yet

- The client-only `mesh-client` crate keeps its own Responses parser. It has
  no plugin host and still rejects flat `input_file` blocks.
- Transforms do not see `/v1/completions`, embeddings or `count_tokens`.

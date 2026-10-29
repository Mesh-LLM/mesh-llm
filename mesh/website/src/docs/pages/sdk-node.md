---
title: Node.js and Electron SDK
---

# Node.js and Electron SDK

Install `@mesh-llm/sdk` with `npm install @mesh-llm/sdk`. The SDK uses a native
N-API addon and one embedded `Node` with `client`, `serve`, and `combined`
roles. `serve` is serve-only; `combined` also permits inference.

## Join and infer

```js
const { Node } = require('@mesh-llm/sdk')
const node = Node.create({ mode: 'client', autoJoin: true })
await node.start()
try {
  const llms = await node.inference.listModels()
  const response = await node.inference.chatCompletions({
    model: llms[0].id,
    messages: [{ role: 'user', content: 'Hello from Node.js.' }]
  })
  console.log(response)
} finally {
  await node.stop()
}
```

Use `joinTokens: [token]` for a private mesh and `ownerKeyPath` for an existing
Mesh LLM owner keystore. Client mode loads no serving runtime.

## Discover advertised models

`node.models` queries the embedded node's mesh inventory directly, including
local and peer-advertised LLM, OCR, speech, embedding, decision and unknown models.
It does not use the OpenAI HTTP model list.

```js
const page = await node.models.list()
const ocr = await node.models.list({
  workloadClass: 'document_ocr',
  available: true,
  limit: 50
})
const model = await node.models.get('org/model@revision#profile')
for (const provider of model.providers) {
  console.log(provider.provider_id, provider.workload_class, provider.runtime)
}
```

Results use `MeshModelPage` with `data` and `next_cursor`. Query options are
camelCase; model fields retain the Mesh API's snake_case. Filters combine with
AND. Continue with the same filters and the returned cursor:

```js
const next = ocr.next_cursor
  ? await node.models.list({
      workloadClass: 'document_ocr',
      available: true,
      limit: 50,
      cursor: ocr.next_cursor
    })
  : null
```

A model's `operations` describe what at least one provider advertises. Runtime
limits, capabilities, thinking controls and payment offers stay under their
provider; they are not guarantees for every replica. Missing metadata means
unknown. `node.inference.listModels()` is the LLM-only compatibility view.

## Stream inference

Use `node.inference.streamChatCompletions(body)` or `streamResponses(body)`.
Each SSE event preserves the event name, complete JSON data and raw frame.
Tools, structured output, reasoning and usage fields remain available. Use
`event.json()` on a nonterminal SSE frame. Leaving iteration early cancels the
native request.

## Serve local models

Install or resolve a matching native serving runtime before starting a serving
node. Runtime installation is explicit:

```js
const { resolveNativeRuntime } = require('@mesh-llm/sdk')
await resolveNativeRuntime({
  artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
  allowDownload: false
})
const servingNode = Node.create({
  mode: 'combined',
  models: [process.env.MESH_SDK_MODEL_REF],
  joinTokens: [process.env.MESH_INVITE_TOKEN]
})
await servingNode.start()
```

Model loading comes from node configuration. Discovery `models.list/get`
queries advertisements; it does not download or load a model.

## Local development and Electron

Build the native addon with `npm run build:native` from `mesh/sdk/node`.
Package the matching addon under `native/<platform>-<arch>/` and a compatible
native runtime for applications that serve models. Resolve bundled runtimes
explicitly and put writable cache data in the application's data directory.

## Document OCR

Discover an OCR provider, then submit the PDF or image bytes through Mesh:

```js
const { readFile } = require('node:fs/promises')
const page = await node.models.list({ workloadClass: 'document_ocr', available: true })
if (!page.data.length) throw new Error('No OCR provider is available')
const result = await node.inference.documentOcr(await readFile('scan.pdf'), {
  model: page.data[0].id,
  filename: 'scan.pdf',
  contentType: 'application/pdf',
  responseFormat: 'json',
  maxPages: 10
})
console.log(result.json())
```

OCR uses binary multipart upload to `/v1/documents/ocr`, including when the
provider is remote. It does not send the document through chat completion.
The response preserves provider status, content type and body. OCR raises
`OpenAIRequestError` for non-2xx by default; pass a third argument
`{ raiseForStatus: false }` to inspect a raw error response. Use `body` for
plain text or Markdown responses and `json()` for JSON.

## Additional mesh memberships

```js
const team = await node.addMembership({
  name: 'team', apiPort: 9338,
  joinTokens: [inviteToken], trustedOwners: [ownerId]
})
const memberships = await node.memberships()
const removed = await node.removeMembership('team')
```

Each membership has its own identity, owner allowlist and peers. An empty
allowlist admits nobody. Adding an existing name replaces its membership under
the same Rust validation/recovery rules. Model discovery queries the node's
primary mesh; it does not combine these independent meshes.

## Full embedded-node configuration

`Node.fromConfig(config)` accepts the complete Rust `EmbeddedMeshNodeConfig`
JSON schema, including network relays, bind settings, publication, admission
and trust policy, storage, serving limits and startup timeout. The exported
TypeScript `EmbeddedMeshNodeConfig` type describes these fields.

```js
const node = Node.fromConfig({
  mode: 'client',
  network: { join_tokens: [inviteToken], enumerate_host: false },
  storage: { isolated_config: true },
  startup_timeout: { secs: 30, nanos: 0 }
})
await node.start()
```

Keys and enum values use the Rust schema's snake_case without language-specific
filtering. Omitted fields use the Rust defaults; unknown fields are rejected by
the core. In this schema, `mode: 'serve'` means combined serving and inference,
while `'serve_only'` means serving without SDK inference. The full-config
default is combined, exactly as in Rust. Existing `Node.create()` keeps its
client default and camelCase convenience options.

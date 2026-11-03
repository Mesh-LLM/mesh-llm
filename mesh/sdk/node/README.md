# MeshLLM Node SDK

The package exposes one real embedded `Node` with client, serve-only, and
combined roles. The legacy thin `Client` is removed. See the
[node usage guide](../../docs/sdk/node.md) for current constructors,
streaming, role selection, and native runtime requirements.

The `serve` role serves local models; `combined` also permits mesh inference.
Client mode starts no native serving runtime. Supply a Mesh LLM owner keystore
path through `ownerKeyPath`/`owner_key_path` if owner identity is required.

Discover every model advertised in the node's mesh through the native SDK:

```js
const page = await node.models.list()
const ocr = await node.models.list({ workloadClass: 'document_ocr', available: true })
const detail = await node.models.get('org/model@revision#profile')
const next = page.next_cursor
  ? await node.models.list({ cursor: page.next_cursor })
  : null
```

`models` includes LLMs, OCR, speech, embedding, decision, and unknown workloads,
both local and peer-advertised. Queries use camelCase options; result objects
retain the Mesh HTTP API's snake_case fields. Filters combine with AND; retain
the same filters when continuing a page. Runtime limits, capability evidence,
thinking controls and payment offers are observations under each `provider`.
They do not guarantee every replica has those properties. Missing facts remain
absent. `inference.listModels()` remains the LLM-only compatibility view.

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

OCR uses binary multipart upload to `/mesh/v1/documents/ocr`, including when the
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

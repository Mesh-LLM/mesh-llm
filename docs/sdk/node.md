# Node.js SDK

Use `@meshllm/sdk` from npm for Node.js and Electron applications.

## Install

```json
{
  "dependencies": {
    "@meshllm/sdk": "0.72.1"
  }
}
```

When building from this repository, build the native N-API addon first:

```bash
cd sdk/node
npm run build:native
```

## Client: Public Mesh

Node.js public discovery helpers are not currently exported by `@meshllm/sdk`.
Use a public invite token selected by your app or service.

```js
const { Client, generateOwnerKeypairHex } = require('@meshllm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_PUBLIC_INVITE
})

await client.start()
const models = await client.inference.listModels()
const result = await client.inference.chat({
  model: models[0].id,
  messages: [{ role: 'user', content: 'Say hello from a public mesh.' }]
})
console.log(result.content)
await client.stop()
```

## Client: Private Mesh

```js
const { Client, generateOwnerKeypairHex } = require('@meshllm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_PRIVATE_INVITE
})

await client.start()
const models = await client.inference.listModels()
const result = await client.inference.chat({
  model: models[0].id,
  messages: [{ role: 'user', content: 'Say hello from a private mesh.' }]
})
console.log(result.content)
await client.stop()
```

## Serving: Install Runtime

Install or resolve a native runtime before starting local serving:

```js
const { resolveNativeRuntime } = require('@meshllm/sdk')

await resolveNativeRuntime({
  artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
  allowDownload: process.env.MESH_SDK_RUNTIME_ALLOW_DOWNLOAD === '1',
  onProgress: (event) => console.log(event)
})
```

## Serving: Public Mesh

```js
const { Node, generateOwnerKeypairHex, resolveNativeRuntime } = require('@meshllm/sdk')

await resolveNativeRuntime({
  artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
  allowDownload: process.env.MESH_SDK_RUNTIME_ALLOW_DOWNLOAD === '1',
  onProgress: (event) => console.log(event)
})

const modelRef = process.env.MESH_SDK_MODEL_REF || 'Qwen2.5-3B-Instruct-Q4_K_M'
const node = Node.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_PUBLIC_INVITE,
  servingEnabled: true,
  cacheDir: process.env.MESH_SDK_CACHE_DIR,
  runtimeDir: process.env.MESH_SDK_RUNTIME_DIR
})

await node.start()
await node.models.download(modelRef)
const served = await node.serving.load(modelRef, { devicePolicy: 'auto' })
const result = await node.inference.chat({
  model: served.modelId,
  messages: [{ role: 'user', content: 'Say hello from a public serving node.' }]
})
console.log(result.content)
await node.serving.unloadModel(served.modelId)
await node.stop()
```

## Serving: Private Mesh

Private mesh serving uses the same lifecycle with `MESH_PRIVATE_INVITE`:

```js
const node = Node.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_PRIVATE_INVITE,
  servingEnabled: true,
  cacheDir: process.env.MESH_SDK_CACHE_DIR,
  runtimeDir: process.env.MESH_SDK_RUNTIME_DIR
})
```

## Console Assets

Published Node packages that advertise console support include the built web
console as package resources. Use the package helper to find those assets in
normal package usage:

```js
const { defaultConsoleAssetDir } = require('@meshllm/sdk')

const assetDir = defaultConsoleAssetDir()
```

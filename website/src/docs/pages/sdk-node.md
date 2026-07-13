---
title: Node.js and Electron SDK
---

# Node.js and Electron SDK

Use [`@meshllm/sdk`](https://www.npmjs.com/package/@meshllm/sdk) in Node.js services, desktop apps, and Electron applications. The package uses a native N-API addon and the same embedded serving path as the Swift and Kotlin SDKs.

## Install

```json
{
  "dependencies": {
    "@meshllm/sdk": "{{ site.sdkVersion }}"
  }
}
```

When building from the repository, build the addon first:

```bash
cd sdk/node
npm run build:native
```

## Connect as a client

```js
const { Client, generateOwnerKeypairHex } = require('@meshllm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_INVITE_TOKEN
})

await client.start()
try {
  const models = await client.inference.listModels()
  const result = await client.inference.chat({
    model: models[0].id,
    messages: [{ role: 'user', content: 'Say hello from Node.js.' }]
  })
  console.log(result.content)
} finally {
  await client.stop()
}
```

The current package expects an invite token selected by the app or service. Use the same `Client` lifecycle for public or private meshes; only the token source changes.

## Embed local serving

Serving needs a verified native runtime artifact. Bundle one with the app or explicitly allow the SDK to download a compatible release runtime:

```js
const { resolveNativeRuntime } = require('@meshllm/sdk')

const runtime = await resolveNativeRuntime({
  artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
  allowDownload: process.env.MESH_SDK_RUNTIME_ALLOW_DOWNLOAD === '1',
  onProgress: event => console.log(event)
})
console.log(`using ${runtime.nativeRuntimeId} from ${runtime.path}`)
```

Create a `Node`, download or locate the model, load it through the serving API, and unload it during shutdown:

```js
const {
  Node,
  generateOwnerKeypairHex,
  resolveNativeRuntime
} = require('@meshllm/sdk')

await resolveNativeRuntime({
  artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
  allowDownload: process.env.MESH_SDK_RUNTIME_ALLOW_DOWNLOAD === '1'
})

const modelRef = process.env.MESH_SDK_MODEL_REF || 'Qwen2.5-3B-Instruct-Q4_K_M'
const node = Node.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_INVITE_TOKEN,
  servingEnabled: true,
  cacheDir: process.env.MESH_SDK_CACHE_DIR,
  runtimeDir: process.env.MESH_SDK_RUNTIME_DIR
})

await node.start()
try {
  await node.models.download(modelRef)
  const served = await node.serving.load(modelRef, { devicePolicy: 'auto' })
  const result = await node.inference.chat({
    model: served.modelId,
    messages: [{ role: 'user', content: 'Say hello from local serving.' }]
  })
  console.log(result.content)
  await node.serving.unloadModel(served.modelId, { drainTimeoutMs: 1000 })
} finally {
  await node.stop()
}
```

Use `serving.unloadInstance(instanceId)` when the load call returns an instance id and the application manages multiple copies of a model.

## Electron packaging

Package these items for each target platform and architecture:

- `native/<platform>-<arch>/mesh_llm_nodejs.node`
- the matching `meshllm-native-runtime-*` directory
- optional `console/` assets when the app calls `node.startConsole()`

Resolve the runtime from the packaged artifact directory rather than relying on a writable global install. Keep the runtime cache and model cache in Electron's app-specific data directory.

## Optional console

When console assets are included in the npm package:

```js
const consoleHandle = await node.startConsole({ port: 0 })
console.log(consoleHandle.url)
```

For a repository checkout, pass an explicit asset directory such as `crates/mesh-llm-ui/dist`. Do not expose the console on all interfaces unless the app has its own authentication and network policy.

## Runtime and errors

The Node API exposes `Client` for remote inference and `Node` for serving, model management, status, reconnect, and console hosting. Handle runtime resolution failures, unavailable endpoints, model download failures, and serving errors as application errors. Persist the owner keypair instead of generating a new one each time the Electron window opens.

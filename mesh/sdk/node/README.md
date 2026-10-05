# MeshLLM Node SDK

Node.js bindings for MeshLLM client mode, model management, local serving, and
Electron-style desktop packaging.

The package uses a native N-API addon built from `mesh/crates/mesh-llm-nodejs`. It is
not a mock wrapper around the CLI. Local serving uses the same embedded runtime
path as the Swift and Kotlin SDKs.

## Install

```bash
npm install @mesh-llm/sdk
```

Release packages include prebuilt addons for macOS arm64, Linux arm64/x64, and
Windows x64. Intel Macs are not supported because they cannot run MeshLLM
inference.

## Build From Source

```bash
cd mesh/sdk/node
npm run build:native
```

The build copies the platform addon to:

```text
mesh/sdk/node/native/<platform>-<arch>/mesh_llm_nodejs.node
```

On Windows, the source binary is `target/release/mesh_llm_nodejs.dll` and the
packaged Node addon is renamed to `mesh_llm_nodejs.node`.

## Client Mode

```js
const { Client, generateOwnerKeypairHex } = require('@mesh-llm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: 'your-invite-token'
})

await client.start()
const models = await client.inference.listModels()

for await (const event of client.inference.streamChatCompletions({
  model: models[0].id,
  messages: [{ role: 'user', content: 'What is the weather in Sydney?' }],
  tools: [{
    type: 'function',
    function: {
      name: 'get_weather',
      parameters: { type: 'object', properties: { city: { type: 'string' } } }
    }
  }]
})) {
  if (event.type === 'sse' && !event.done) {
    const chunk = event.json()
    // Text, reasoning, and incremental tool-call arguments are preserved.
    console.log(chunk)
  }
}
await client.stop()
```

`chatCompletions()` and `responses()` return complete OpenAI-shaped JSON.
`streamChatCompletions()` and `streamResponses()` preserve each complete SSE
frame, including its optional event name, raw bytes as text, and arbitrary JSON
payload. Breaking out of the async iterator cancels the native request.

## Local Serving Mode

Resolve or install a native runtime before loading a local model. The SDK can
use a packaged artifact directory, or download the recommended runtime when
`MESH_SDK_RUNTIME_ALLOW_DOWNLOAD=1` is set:

```bash
MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-linux-x86_64-cuda \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
node mesh/sdk/node/example/local-inference.js
```

Use `Node` instead of `Client` when the app needs local model management or
serving:

```js
const {
  Node,
  generateOwnerKeypairHex,
  resolveNativeRuntime
} = require('@mesh-llm/sdk')

const runtime = await resolveNativeRuntime({
  artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
  allowDownload: process.env.MESH_SDK_RUNTIME_ALLOW_DOWNLOAD === '1'
})
console.log(`using ${runtime.nativeRuntimeId} from ${runtime.path}`)

const node = Node.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: 'local-electron-app',
  servingEnabled: true
})
```

In an Electron app, package both:

- `native/<platform>-<arch>/mesh_llm_nodejs.node`
- the selected `meshllm-native-runtime-*` runtime artifact directory
- optional `console/` assets when using `node.startConsole()`

Then pass the packaged artifact directory to `resolveNativeRuntime()` before
creating a serving-enabled node.

## Optional Console

Console assets are not embedded in the default native addon. A package that
includes the built console under `console/` can start the static console server
without asking the application for a raw path:

```js
const consoleHandle = await node.startConsole()
console.log(consoleHandle.url)
```

For development builds, pass an explicit asset directory:

```js
await node.startConsole({ assetDir: 'mesh/crates/mesh-llm-ui/dist', port: 3131 })
```

Release packages prepare that directory with:

```bash
scripts/package-sdk-console-assets.sh --sdk node
scripts/verify-sdk-console-assets.sh --sdk node
```

## Windows

Windows is supported through the same N-API addon shape:

- addon: `mesh_llm_nodejs.node`
- native runtime library: `meshllm_ffi.dll`
- target triple: `x86_64-pc-windows-msvc`
- runtime artifact names: `meshllm-native-windows-x86_64-cpu`,
  `meshllm-native-windows-x86_64-cuda`, `meshllm-native-windows-x86_64-rocm`,
  or `meshllm-native-windows-x86_64-vulkan`

The Windows release pipeline already builds CPU, CUDA, ROCm, and Vulkan runtime
bundles. The Node SDK should publish matching prebuilt addon/runtime packages
for Electron consumers instead of requiring local Visual Studio/CUDA/ROCm/Vulkan
toolchains.

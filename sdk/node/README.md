# MeshLLM Node SDK

Node.js bindings for MeshLLM client mode, model management, local serving, and
Electron-style desktop packaging.

The package uses a native N-API addon built from `crates/mesh-llm-nodejs`. It is
not a mock wrapper around the CLI. Local serving uses the same embedded runtime
path as the Swift and Kotlin SDKs.

## Build From Source

```bash
cd sdk/node
npm run build:native
```

The build copies the platform addon to:

```text
sdk/node/native/<platform>-<arch>/mesh_llm_nodejs.node
```

On Windows, the source binary is `target/release/mesh_llm_nodejs.dll` and the
packaged Node addon is renamed to `mesh_llm_nodejs.node`.

## Client Mode

```js
const { Client, generateOwnerKeypairHex } = require('@meshllm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: 'your-invite-token'
})

await client.start()
const models = await client.inference.listModels()
await client.stop()
```

## Local Serving Mode

Set `MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR` to a verified native runtime artifact
for the target machine:

```bash
MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-sdk/meshllm-native-linux-x86_64-cuda \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
node sdk/node/example/local-inference.js
```

Use `Node` instead of `Client` when the app needs local model management or
serving:

```js
const { Node, generateOwnerKeypairHex } = require('@meshllm/sdk')

const node = Node.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: 'local-electron-app',
  servingEnabled: true
})
```

In an Electron app, package both:

- `native/<platform>-<arch>/mesh_llm_nodejs.node`
- the selected `meshllm-native-*` runtime artifact directory
- optional `console/` assets when using `node.startConsole()`

Then set `MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR` to the packaged artifact
directory before creating a serving-enabled node.

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
await node.startConsole({ assetDir: 'crates/mesh-llm-ui/dist', port: 3131 })
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

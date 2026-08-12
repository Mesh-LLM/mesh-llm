# MeshLLM Node SDK

Node.js bindings for MeshLLM client mode, model management, local serving, and
Electron-style desktop packaging.

The package uses a native N-API addon built from `crates/mesh-llm-nodejs`. It is
not a mock wrapper around the CLI. Local serving uses the same embedded runtime
path as the Swift and Kotlin SDKs.

## Install

```bash
npm install @mesh-llm/sdk
```

Release packages include prebuilt addons for macOS arm64/x64, Linux arm64/x64,
and Windows x64.

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
const { Client, generateOwnerKeypairHex } = require('@mesh-llm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: 'your-invite-token'
})

await client.start()
const models = await client.inference.listModels()
await client.stop()
```

## Local Serving Mode

Resolve or install a native runtime before loading a local model. The SDK can
use a packaged artifact directory, or download the recommended runtime when
`MESH_SDK_RUNTIME_ALLOW_DOWNLOAD=1` is set:

```bash
MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-linux-x86_64-cuda \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
node sdk/node/example/local-inference.js
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
- the optional `@mesh-llm/apple-runtime-darwin-arm64` package when exposing
  `apple/system` on Golden Gate

Then pass the packaged artifact directory to `resolveNativeRuntime()` before
creating a serving-enabled node.

## Apple system model on macOS Golden Gate

The experimental provider host starts the packaged Apple sidecar and returns
the normal OpenAI-compatible loopback URL:

```js
const {
  ProviderHost,
  packagedAppleSystemProvider
} = require('@mesh-llm/sdk')

const host = await ProviderHost.start(packagedAppleSystemProvider())
console.log(host.apiBaseUrl) // http://127.0.0.1:<port>/v1
console.log(await host.status())
await host.stop()
```

Run `scripts/package-sdk-provider-runtime.sh --sdk node` before publishing the
platform package. It is an optional dependency of `@mesh-llm/sdk` and is
installed only on Darwin arm64. Electron packagers must keep the package's
`runtime/**` outside ASAR (for example with `asarUnpack`) because macOS must
execute and verify the real signed file. The Node process owns lifecycle;
`mesh-apple-runtime` is the one process that receives Apple accelerator access.

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

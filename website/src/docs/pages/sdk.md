---
title: SDKs
---

# Embed Mesh in your app

The Mesh SDKs let an application either connect to an existing mesh or embed a complete mesh node with local model serving. The public APIs are intentionally shaped the same way across Rust, Node.js, JVM/Android, and Swift.

## Choose a role

| Role | What it does | What it needs |
| --- | --- | --- |
| `Client` | Joins a private mesh or connects to a selected public mesh and runs inference. | SDK package, owner keypair, and invite token or public-mesh discovery. |
| `Node` | Includes the client role, model search/download, local model loading, serving, and optional console hosting. | SDK package plus a compatible native runtime artifact for local serving. |

Use `Client` when another machine already serves the model. Use `Node` when the application should own local model files, load/unload decisions, and serving lifecycle.

## Common lifecycle

Client-only applications usually follow this shape:

```text
load or create owner keypair
create Client with an invite token
start
list models
stream chat or responses
stop
```

Serving applications add runtime and model lifecycle:

```text
resolve a native runtime
create Node
start
download or locate a model
load the model through serving
run inference
unload the model or instance
stop
```

Persist the owner keypair in the host application's secure storage. Generate an ephemeral key only for demos; changing it on every launch creates a new mesh identity.

## Platform support

| SDK / target | Mesh inference | Model management | Local serving |
| --- | ---: | ---: | ---: |
| Rust on macOS/Linux | yes | yes | yes with `serving` and a compatible native runtime |
| Node.js on macOS/Linux/Windows | yes | yes | yes with a compatible native runtime |
| JVM on macOS/Linux | yes | yes | yes with a matching native runtime library |
| Android | yes | yes | not currently advertised |
| Swift on macOS | yes | yes | yes with a matching native runtime |
| Swift on Mac Catalyst | yes | yes | planned validation |
| Swift on iOS | yes | limited by app filesystem policy | no |

Targets without validated local serving should surface the typed `ServingUnsupported` error. Do not silently fall back to a fake local implementation.

## Native runtime artifacts

The language package provides the API and native bridge. Local inference also needs a release runtime artifact for the host platform and backend, such as Metal on Apple Silicon or CUDA on Linux. Runtime artifacts are selected and verified against the Mesh release and exact Skippy ABI; they are not compiled implicitly by npm, SwiftPM, Maven, or Cargo.

Serving apps can either:

- bundle the matching `meshllm-native-runtime-*` directory with the app, or
- allow the SDK runtime manager to download a verified artifact at startup.

For offline or packaged applications, pass the artifact directory directly to the language SDK resolver. For online development, opt into downloads explicitly with `allowDownload` or `MESH_SDK_RUNTIME_ALLOW_DOWNLOAD=1`.

## Pick a language guide

- [Rust SDK](/docs/pages/sdk-rust/) — crates.io client facade and embedded `MeshNode`.
- [Node.js and Electron SDK](/docs/pages/sdk-node/) — npm package, N-API addon, runtime packaging, and console assets.
- [Java, Kotlin, and Android SDK](/docs/pages/sdk-kotlin/) — GitHub Packages AAR, JVM serving, Android client mode, and coroutines.
- [Swift SDK](/docs/pages/sdk-swift/) — SwiftPM, XCFrameworks, async streams, macOS, iOS, and App Store notes.

## Shared rules

- Pass identity and join tokens explicitly; SDKs do not read credentials from global CLI config directories.
- Keep API and console ports isolated when embedding more than one node in a process or test suite.
- Treat model downloads as application work: show progress, choose an app-owned cache, and handle cancellation.
- Stop or reconnect nodes with the host application's lifecycle. On mobile, reconnect when returning to the foreground.
- Handle typed errors, especially invalid tokens, discovery failures, model-management failures, stream failures, and unsupported serving.
- Package console assets only when the app needs a local web console; the default native runtime should stay smaller without them.

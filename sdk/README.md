# SDK

This directory contains the language-specific Mesh SDK packages built on top of
the shared native node SDK.

Current SDKs:

- `swift/` for Apple platforms
- `kotlin/` for Android and JVM consumers
- `node/` for Node.js and Electron consumers

These SDK packages should stay thin. Shared node behavior belongs in the Rust
SDK crates:

- `crates/mesh-client/` for the low-level client implementation
- `crates/mesh-llm-api-client/` for the public Rust client-only SDK API
- `crates/mesh-llm-api-server/` for the public Rust node SDK API
- `crates/mesh-llm-node/` for embeddable model management and serving
  orchestration. Serving SDK calls should bind to in-process node
  controllers, not the local REST management API.
- `crates/mesh-llm-ffi/` for the UniFFI/native bridge used by Swift and Kotlin
- `crates/mesh-llm-nodejs/` for the N-API native bridge used by Node.js
- `crates/mesh-llm-console-server/` for serving packaged console assets from
  SDKs without embedding the React bundle into every native library

The public surface is split by role: `Client` consumes inference from an
existing mesh, while `Node` can also manage and serve local models. See
`docs/design/EMBEDDED_CLIENT_ADR.md` for the current SDK direction.

The customer-facing SDK usage guide lives in `docs/SDK.md`. SDK changes should
keep Rust, Swift, Kotlin, and Node aligned around real examples, polished
lifecycle, typed errors, and an honest platform support matrix.

## Optional Console Assets

The CLI embeds the built React console in the shipped binary. SDK packages keep
the default native runtime smaller and treat the console as optional package
resources instead. A console-enabled SDK package should include the built UI
`dist/` files, resolve that resource location internally, and call the native
`startConsole` hook with the resolved asset directory.

SDK users should not have to pass a raw directory in normal package usage. The
path-based native boundary exists so SwiftPM resources, Node package files, JVM
resources, and Android assets can all feed the same Rust console server.

Generated UniFFI bindings and Apple binary artifacts are build outputs, not
source. Do not check in `sdk/*/Generated`, generated `uniffi/mesh_ffi` Kotlin
sources, or `MeshLLMFFI.xcframework`; regenerate them from
`crates/mesh-llm-ffi/src/mesh_ffi.udl` in local builds and CI.

If you add another top-level SDK here, include a `README.md` in that SDK
directory explaining its packaging and public surface.

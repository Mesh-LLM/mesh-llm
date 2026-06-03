# SDK

This directory contains the language-specific Mesh SDK packages built on top of
the shared native node SDK.

Current SDKs:

- `swift/` for Apple platforms
- `kotlin/` for Android and JVM consumers
- `node/` for Node.js and Electron consumers

These SDK packages should stay thin. `mesh-llm-sdk` is the source of truth for
the public capability contract, and language packages bind that shape instead
of inventing separate Node/Swift/Kotlin semantics.

- `crates/mesh-llm-sdk/` for the canonical Rust SDK facade. Its public feature
  model is `client`, `node`, `serving`, and `console`.
- `crates/mesh-client/` for the low-level client implementation
- `crates/mesh-llm-api-client/` for the public Rust client-only SDK API
- `crates/mesh-llm-api-server/` for platform-neutral node/model APIs re-exported
  by `mesh-llm-sdk` when the `node` or `serving` features are enabled
- `crates/mesh-llm-node/` for embeddable model management and serving
  orchestration. Serving SDK calls should bind to in-process node
  controllers, not the local REST management API.
- `crates/mesh-llm-ffi/` for the UniFFI/native bridge used by Swift and Kotlin;
  it wraps `mesh-llm-sdk`
- `crates/mesh-llm-nodejs/` for the N-API native bridge used by Node.js; it wraps
  `mesh-llm-sdk`
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

Generated UniFFI source bindings are refreshed from
`crates/mesh-llm-ffi/src/mesh_ffi.udl`. Apple binary artifacts such as
`MeshLLMFFI.xcframework` are build outputs and should not be checked in.

If you add another top-level SDK here, include a `README.md` in that SDK
directory explaining its packaging and public surface.

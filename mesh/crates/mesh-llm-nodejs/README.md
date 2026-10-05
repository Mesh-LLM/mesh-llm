# mesh-llm-nodejs

`mesh-llm-nodejs` is the N-API native addon behind the
[`@mesh-llm/sdk` Node.js package](../../sdk/node/README.md). It adapts the
Rust [`mesh-llm-sdk`](../mesh-llm-sdk/README.md) to JavaScript; applications
should use the Node package rather than load this crate's library directly.

The addon exposes mesh identity, node lifecycle, model management, inference,
native-runtime installation, and optional local serving. Its
`embedded-runtime` feature is enabled by default and supplies the local
serving controller. The Node package owns the JavaScript API and platform
addon packaging.

Build and test the package from the workspace root:

```bash
just with-lld cargo test --locked -p mesh-llm-nodejs
cd mesh/sdk/node && npm run build:native && npm test
```

See the [Node SDK guide](../../sdk/node/README.md) for installation, examples,
and native-runtime packaging.

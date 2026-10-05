# mesh-llm-test-harness

`mesh-llm-test-harness` provides `FixtureMesh`, a process fixture for tests
that need a running `mesh-llm` server. It starts `mesh-llm serve` with a
selected model on a local port, waits for `/api/status`, and exposes the invite
token. Dropping the fixture terminates the child process.

The fixture locates the binary through `MESH_LLM_BIN` or
`target/release/mesh-llm`. It requires a model the binary can load. Build the
release binary with `just release-build` before running the ignored
end-to-end test:

```bash
just with-lld cargo test --locked -p mesh-llm-test-harness -- --ignored
```

The `spawn-fixture` helper binary starts the same fixture, prints
`INVITE_TOKEN=...`, and keeps it alive until standard input closes. The
crate's ordinary tests do not launch a model-serving process.

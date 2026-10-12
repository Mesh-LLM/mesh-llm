# Metered workflow fixtures

`metered_plugin.py` is a dependency-free test adapter, not a production SDK. The
host launches it through the normal Unix plugin-control handshake. Its manifest
registers an OpenAI-compatible HTTP endpoint with a `/adapter/v1` base path;
the normal endpoint health probe discovers `test` through `/models`. The fixture
asserts the rewritten inference path, request correlation ID and backend usage
opt-in, then serves CRLF SSE with authoritative usage.

The ignored `metered_live` tests exercise two authenticated QUIC nodes and the
actual v2 payer/provider, using simulated wallet settlement (no funds move).
The plugin test runs five fresh peer pairs (ten requests). They assert
first-deposit and trusted-repeat payment counts, completed buyer requests,
settled seller records, paid receivables, zero buyer reservations, zero seller
exposure and a bill derived from reported usage. The accepted QUIC connection
is retained until buyer completion; exchange timeouts include both ledger
snapshots and accounting diagnostics.
They do not establish real wallet-provider integration or public-mesh readiness.

Run after reading `docs/design/TESTING.md`, with Cargo commands serialized:

```bash
python3 -m unittest discover \
  -s crates/mesh-llm-host-runtime/src/network/payments/tests/fixtures \
  -p test_metered_plugin.py
cargo test -p mesh-llm-host-runtime --lib -- --include-ignored \
  network::payments::tests::metered_live::registered_plugin_first_and_repeat_settlement
```

For native qualification, build the supported release host/runtime lanes first,
then launch a task-owned `mesh-llm serve --local-model-only` against an existing
small GGUF. Point `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` at the freshly packaged
`dist/native-runtimes`, use separate runtime/cache paths and an unused API port.
Obtain the exact model ID from that endpoint's `/v1/models` response.

```bash
MESH_METERING_LIVE_PORT=<api-port> MESH_METERING_LIVE_MODEL=<model-id> \
  cargo test -p mesh-llm-host-runtime --lib -- --include-ignored \
  network::payments::tests::metered_live::native_serving_first_and_repeat_settlement
```

Also run the full host-runtime package suite and required check/Clippy gates.
Record `git rev-parse HEAD` in the same shell as qualification; an earlier
native build does not prove a later payer change works. Stop only the task-owned
server, clean disposable Cargo/native outputs after verification, and leave
existing shared models and other agents' runtimes untouched.

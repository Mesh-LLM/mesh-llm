# mesh-llm-host-runtime

`mesh-llm-host-runtime` composes the host-side mesh node runtime. It wires
model resolution, local serving, discovery, networking, runtime state, plugins,
the management API, and the shipped CLI entrypoint used by the `mesh-llm`
binary.

This crate is being split so reusable CLI, TUI, SDK, and embeddable runtime
surfaces can be published and consumed independently.

## Embedded lifecycle regression tests

From the workspace root, run the real SDK startup-cancellation and join-stop
scenarios (isolated child processes, loopback networking, no model required):

```sh
just with-lld cargo test -p mesh-llm-host-runtime --no-default-features --features dynamic-native-runtime --lib sdk::embedded_lifecycle_tests
```

The opt-in download and inference scenarios also require a compatible installed
native runtime. Set `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` when using a prepared
runtime bundle. Download `tinyllamas/stories15M-q4_0.gguf` from the Hugging Face
[`ggml-org/models` repository](https://huggingface.co/ggml-org/models/tree/main/tinyllamas)
and set `MESH_EMBEDDED_TEST_GGUF` to its absolute local path. The test verifies
SHA-256 `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`.

```sh
MESH_EMBEDDED_TEST_GGUF=/absolute/path/stories15M-q4_0.gguf \
  just with-lld cargo test -p mesh-llm-host-runtime --no-default-features --features dynamic-native-runtime --lib sdk::embedded_lifecycle_tests -- --include-ignored
```

The download fixture holds an actual HTTP transfer open until cancellation;
the join fixture observes an actual QUIC dial before stopping. Successful stop
must release listeners and permit restart on the same ports. The inference
scenario cancels the download, restarts serving in the same process, and checks
nonempty generated output through the OpenAI endpoint. Child deadlines turn
leaked runtime threads into failures rather than hanging the test runner.
These tests cover SDK lifecycle and native inference, not the Buzz UI or a
multi-machine community mesh.

SDK application stage providers are integrated at the stage bridge. The host
checks authenticated peer identity, admission, topology/run ownership and local
stage readiness before routing an inbound provider stream. Unhandled stage
connections retain the dedicated Iroh path. The provider never selects a local
TCP destination. See the [contract](../../docs/specs/APPLICATION_STAGE_TRANSPORT.md).

For split planning the host also reads optional provider link observations, gated
by current peer admission, matching protocol, sample count and freshness. It passes
directed local-to-peer latency edges to Skippy's pure planner; remote-to-remote
edges retain existing RTT estimates. These local observations require no wire
changes and do not trigger topology reloads on their own.

Embedded nodes may require their registered provider through
`StageTransportPolicy::RequireProvider`. The host validates registration, advertises
the policy through an existing capability feature, filters split participants,
verifies outgoing setup before stage loading and refuses default activation streams
in both directions. Each participant must enforce the policy. Required mode also
rejects identity-free remote stage endpoints and non-loopback local endpoints;
control, artifacts and ordinary mesh membership remain available. See the
[SDK example](../mesh-llm-sdk/README.md#application-stage-transport).

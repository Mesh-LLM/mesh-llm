# mesh-llm-host-runtime

`mesh-llm-host-runtime` composes the host-side mesh node runtime. It wires
model resolution, local serving, discovery, networking, runtime state, plugins,
the management API, and the shipped CLI entrypoint used by the `mesh-llm`
binary.

This crate is being split so reusable CLI, TUI, SDK, and embeddable runtime
surfaces can be published and consumed independently.

## Provider evidence

A plugin becomes a peer evidence verifier by declaring a namespace and handler in its manifest; see the [plugin contract](../mesh-llm-plugin/README.md#peer-evidence-verification). Mesh registers the verifier of a plugin running when it starts, so it refuses a verifier that starts lazily, and runs at most one verifier. It accepts the declaration only from an OS-authenticated plugin process, and a restart that declares a different verifier fails. A verifier that never initializes leaves the node running without one, like any other plugin that fails to start; once its first connection has initialized, the node keeps its declaration even if that connection drops before the node starts.

While a verifier is loaded, a host accepts third-party peer descriptions only from origin-signed records, and relays, lists in mesh discovery and routes inference only to providers whose signed evidence the verifier currently accepts, so an unverified provider reaches only its direct peers. Pure clients need no verdict. A provider that published no evidence under the verifier's namespace is ineligible without a verifier call. Verification runs in the background when peers change, a verifier restarts or a verdict expires; gossip and routing only read the cached verdicts, and a newly accepted provider is relayed at once. A verifier shutdown makes its providers ineligible until it is verified again. The checks cover a buyer's choice of seller: split placement and stage transport keep their usual membership checks and do not consult evidence, so the peers a node places split stages on need no verdict, and additional mesh memberships do not apply verifiers.

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
must release listeners and permit restart on the same ports. A separate
caller-pool regression holds the caller's blocking worker after the runtime
join and checks that the QUIC port is already free, so deferred socket cleanup
cannot make `stop()` report success early. The inference scenario cancels the
download, restarts serving in the same process, and checks nonempty generated
output through the OpenAI endpoint. Child deadlines turn
leaked runtime threads into failures rather than hanging the test runner.
These tests cover SDK lifecycle and native inference, not the Buzz UI or a
multi-machine community mesh.

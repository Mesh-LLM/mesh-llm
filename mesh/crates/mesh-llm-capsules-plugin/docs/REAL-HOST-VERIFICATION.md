# Real-host tests

`mesh/crates/mesh-llm-capsules-plugin/tests/interop.rs` and the other interop tests run the
compiled plugin against a stand-in host that speaks mesh-llm's plugin wire
protocol (length-prefixed protobuf `Envelope` frames over a local socket).
`tests/host_runtime_e2e.rs` closes the remaining gap: it runs the plugin under
a real `mesh-llm` process.

## Why these tests are ignored by default

`mesh-llm-host-runtime` is not a crates.io crate, and building mesh-llm needs
its full native runtime toolchain. The real-host tests are therefore
`#[ignore]`d and need `MESH_LLM_HOST_BIN` to point at a `mesh-llm` binary.
The nightly end-to-end workflow (`.github/workflows/e2e.yml`) runs the
two-node scenarios against a pinned mesh-llm release instead; see
CONTRIBUTING.md, "Tier 2".

## Running them

```sh
# A mesh-llm release binary, or one you built from the mesh-llm repository.
export MESH_LLM_HOST_BIN=/path/to/mesh-llm

cd mesh/crates/mesh-llm-capsules-plugin   # from the mesh-llm checkout
cargo build --bin capsules
cargo test --test host_runtime_e2e -- --ignored --test-threads=1
```

The admission endpoint these tests exercise is opt-in from 0.1.2: the tests set
`CAPSULES_BLOCKED_MODELS` themselves. A node with the variable unset,
empty, `none` or `off` registers no inference provider and no admission
capability at all.

What they check:

| Test | Checks |
| --- | --- |
| `denies_blocked_model_end_to_end_through_real_host` | a request for a blocked test model is refused (403) by the plugin, through the host |
| `allows_unblocked_advertised_model_end_to_end_through_real_host` | an allowed request reaches the plugin and succeeds |
| `malformed_body_fails_safe_at_the_real_host_before_reaching_the_plugin` | a body that does not parse is refused before it reaches the plugin |
| `allowed_exchange_emits_a_signed_chained_ledgered_capsule_and_publishes_lifecycle_event` | an allowed exchange is sealed, signed, chained and stored, verifies offline, and the host's `openai.exchange.v1` event for it reaches the plugin; a changed signature and a changed response digest are both caught |

## Mutant check

The `mutant-blocked-model-served` feature breaks the plugin's decision on purpose.
The same test must then fail, which shows the test exercises the plugin's
decision through the real host:

```sh
cargo test --test host_runtime_e2e --features mutant-blocked-model-served \
  -- --ignored --test-threads=1
# expected: denies_blocked_model_end_to_end_through_real_host FAILS (200, not 403)
```

The malformed-body case is refused by the host's own request parsing, so
`mutant-malformed-body-served` can only be seen through the stand-in host
(`tests/interop.rs`), not here.

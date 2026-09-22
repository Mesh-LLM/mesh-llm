# Synthetic graph-contract coverage

`just skippy-native-tests cpu` (or `metal`) includes 13 synthetic GGUF model
fixtures and 13 graph-contract tests. No downloaded model or weight allocation
is required. The fixtures use real architecture builders with sparse, zero-payload
GGUF files. They are planning fixtures, not inference or numerical-parity models.

The layer counts and activation widths come from this directory's
`family-certified.json` battery rows:

| Battery family | Architecture | Layers | Width |
| --- | --- | ---: | ---: |
| kimi-k3 | kimi-k3 | 8 | 1024 |
| arwkv7 | arwkv7 | 28 | 3584 |
| rwkv7 | rwkv7 | 32 | 2560 |
| llama | llama | 16 | 2048 |
| qwen3-dense | qwen3 | 28 | 1024 |
| deepseek2 | deepseek2 | 27 | 2048 |
| qwen3-moe | qwen3moe | 48 | 2048 |
| qwen2-moe | qwen2moe | 24 | 2048 |
| jamba2 | jamba | 28 | 2560 |
| granite-hybrid | granitehybrid | 32 | 768 |
| qwen35 | qwen35 | 24 | 1024 |
| lfm2 | lfm2 | 16 | 1024 |
| gemma3n | gemma3n | 35 | 2048 |

Other dimensions use deliberately small synthetic defaults, including vocabulary,
feed-forward widths and recurrent low-rank projections. These fixtures preserve
the selected model family's graph behavior, not every production GGUF parameter.
ARWKV7 uses one token-shift stream; RWKV7 uses two. Kimi includes hybrid attention
and cross-layer residual checkpoints.

Each fixture checks eight admitted profiles: token and raw-embedding variants of
single-sequence decode, two-sequence decode, eight-token prefill with all outputs,
and eight-token prefill with only the last output. It validates every interior
split and a three-stage chain at rounded thirds. Gemma3n cuts after layer 18
are required to reject foreign state ownership, matching the canary shared-KV
producer/consumer restriction in `scripts/family-certify.sh`; legal cuts must pass.
Profile names are checked
explicitly, so dropping raw-embedding or batching coverage cannot make tests pass.

For every chain, resident parameter identities and bindings must cover exactly the
whole-model plan's parameter set. Layer-local state accesses must belong to their
stage, and each causal profile must retain state reads and writes. Negative tests
reject a missing middle stage and stages planned for incompatible backends.

The Kimi and ARWKV fixtures reproduce the incomplete activation frontier when
`request_derived()` classifies input leaves before checking layer boundaries.
The tests pass with the boundary-ownership fix. The existing graph-level test also
checks that request-only auxiliary tensors do not become activation exports.

Implementation is carried in native core patch
`0021-test-skippy-synthetic-canary-graph-contracts.patch` and its expanded
`0022-test-skippy-wide-synthetic-graph-contracts.patch`. CTest fixture dependencies
ensure generation runs before each contract test. When updating these dimensions,
keep the table, CMake fixture arguments, and battery rows aligned. Extend this
matrix with other architecture behaviors as regressions are found; it does not
replace the full-registry real-model certification gate.

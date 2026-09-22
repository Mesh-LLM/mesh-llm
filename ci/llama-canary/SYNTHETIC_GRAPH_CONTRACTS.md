# Synthetic graph-contract coverage

`just skippy-native-tests cpu` (or `metal`) includes five synthetic GGUF model
fixtures and five graph-contract tests. No downloaded model or weight allocation
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

Other dimensions use deliberately small synthetic defaults, including vocabulary,
feed-forward widths and recurrent low-rank projections. These fixtures preserve
the selected model family's graph behavior, not every production GGUF parameter.
ARWKV7 uses one token-shift stream; RWKV7 uses two. Kimi includes hybrid attention
and cross-layer residual checkpoints.

Each fixture checks four admitted profiles: token decode, token prefill, raw
embedding decode and raw embedding prefill. It realizes and validates splits after
the first layer, halfway through the model, before the last layer, and a chain at
rounded thirds. A missing middle stage must be rejected. Profile names are checked
explicitly, so dropping raw-embedding coverage cannot make the tests pass.

The Kimi and ARWKV fixtures reproduce the incomplete activation frontier when
`request_derived()` classifies input leaves before checking layer boundaries.
The tests pass with the boundary-ownership fix. The existing graph-level test also
checks that request-only auxiliary tensors do not become activation exports.

Implementation is carried in native core patch
`0021-test-skippy-synthetic-canary-graph-contracts.patch`. CTest fixture dependencies
ensure generation runs before each contract test. When updating these dimensions,
keep the table, CMake fixture arguments, and battery rows aligned. Extend this
matrix with other architecture behaviors as regressions are found; it does not
replace the full-registry real-model certification gate.

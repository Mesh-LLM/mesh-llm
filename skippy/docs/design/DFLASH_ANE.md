# Qwen3.6 DFlash on Apple Neural Engine

## Scope

The paired Qwen3.6-35B-A3B artifact in the Mesh catalog is **DFlash**, not
DSpark. Its GGUF has no `markov_w1.weight` tensor. This design moves the
paired DFlash neural work to Core ML with `cpuAndNeuralEngine` while the target
continues on Metal. The host still selects tokens and manages speculative
acceptance. Core ML's compute plan must confirm ANE placement of the costly
operations; choosing `cpuAndNeuralEngine` alone does not establish that.

The ANE package is an optional, versioned derivative of the existing pinned
GGUF pair. Keep the GGUF drafter as the fallback. Never route another DFlash,
DFlash2, or DSpark checkpoint to this package by architecture name alone.

## Implementation status

The [Rust converter](../../evals/dflash_ane/README.md) builds the six-layer
Core ML graph from the paired GGUFs, including stateful feature injection,
block attention, an INT8 output head, and token readout. The macOS Skippy
runtime invokes Core ML through Rust `objc2` bindings when the compiled package
and target embeddings are supplied
through `SKIPPY_DFLASH_ANE_MODEL` and `SKIPPY_DFLASH_ANE_EMBEDDINGS`.
It verifies the source draft and embedding hashes before attaching. The
compiled package is about 854 MiB and the fp16 embedding file about 970 MiB;
these generated artifacts are not committed to the repository.

On this Mac, Core ML's compute plan placed every draft neural operation on
ANE and the final argmax on CPU. A three-prompt, 128-token run with the
512-slot ANE profile emitted the same 384 target tokens at 113.7 token/s
and 1.38x plain target decode. An adjacent Metal-drafter control run reached
101.1 token/s and 1.21x plain decode. A 640-token coding run crossed the
512-slot wrap, remained target-correct, and reached 1.79x plain decode versus
1.65x with the Metal drafter. These are exploratory single runs. Short
generations can still be slower than plain target decoding.

## Checkpoint contract

The current paired draft has six transformer layers, hidden width 2048,
32 query heads, eight KV heads, head width 128, and FFN width 6144. Its
feature fusion projects eight target layer inputs (layers 2, 7, 12, 17, 23,
28, 33, 38) through `fc.weight` and `enc.output_norm.weight`. The first five
draft layers use a 4096-token sliding attention window; the sixth uses full
attention. The trained block length is 16, though the catalog currently
limits proposals to six. The GGUF does not contain token embeddings or an
output head: llama.cpp borrows both from the target context. The ANE package
must bind copies of those two tensors from the exact target revision, or
validate their hashes before loading separately.

Conversion must preserve the existing two modes of the DFlash graph:

1. **Ingest:** target feature rows -> fusion projection and norm -> each
   layer's K and V projection, K norm, RoPE, and cache write at the exact
   target positions.
2. **Propose:** `[anchor, mask, ...]` embeddings -> six complete draft layers
   with noncausal block attention over committed context and block rows ->
   final norm -> borrowed target output projection -> proposal logits.

Neither mode is a standard autoregressive decoder. In particular, replacing
the feature injection with draft-token prefill would change the model.

## Runtime boundary

The stateful Core ML `draft` function ingests newly committed feature rows
and proposes the next block in one prediction. Prompt rows beyond one block
currently use extra calls to the same function. A context-only ingest
function sharing the state would reduce that prefill cost. A 4096-slot profile
uses 96 MiB of fp16 K/V state per lane; the measured 512-slot profile uses
12 MiB. The latter deliberately limits draft attention history at long
positions. The target still verifies every proposal, preserving exact output.

The existing Skippy target feature capture remains the source of the eight
taps. Transfer only committed rows into the ANE input. The host supplies the
anchor embedding, mask embedding, positions, RoPE values, and attention mask;
reads proposal logits; and takes the rowwise greedy argmax. The target still
performs exact verification and owns emitted tokens. Core ML placed dynamic
gather and argmax on CPU in the local probes, so a literal all-ANE execution
claim would be false. The goal is **all neural draft layers and the full
output projection on ANE, with no draft Metal kernels**.

Provide one state per Skippy lane and serialize predictions that use that
state. On a failed ANE prediction, invalidate that lane's ANE state and use
the existing GGUF drafter or plain target decoding; never reuse partly
updated state. Prefill, rejection, lane switching, and context truncation
must retain the exact committed-position contract of the current drafter.

Package conversion is offline. The installed node must neither require
Python nor download conversion tools. Identify the package by target and
draft hashes, conversion format version, context profile, proposal width,
quantization, and minimum OS version. Load it only on compatible Apple
Silicon with ANE placement verified; otherwise use the GGUF path.

## Measured budget and gates

An exploratory native benchmark on this Mac used three chat-formatted
prompts (code, math, explanation), 128 generated tokens each, context 4096,
and six draft tokens per window. All 384 speculative tokens matched plain
target output. Results are one run, not a production performance claim.

| Metric | Measured |
| --- | ---: |
| Plain decode | 8312 ms, 46.2 token/s |
| DFlash target verification | 4815 ms |
| Existing GGUF draft work | 1379 ms across 95 windows (14.5 ms/window) |
| Existing speculative decode | 62.0 token/s, 1.34x plain |
| Draft acceptance | 289/555 proposed tokens (52.1%) |

At the same acceptance and verification cost, a zero-cost drafter tops out
at 1.73x plain decode. To improve on the existing path, the complete ANE
draft prediction, feature transfer, and host readout must cost less than
14.5 ms per window. A separately compiled 248,320-vocabulary target head
took about 4.3 ms per seven-row prediction with INT8 per-channel weights on
ANE; that leaves roughly 10.2 ms for six layers and all remaining overhead.
The standalone head probe is not a fused-drafter measurement.

The same pair was 0.84x plain decoding on a shorter, 64-token coding prompt.
On the three 128-token prompts, speculative time to first token was roughly
65-70 ms versus 13-25 ms for plain decoding. An ANE deployment should keep
plain target decoding for the first token and use an acceptance and latency
gate for later windows. Moving the draft off Metal may also improve total
throughput with several concurrent lanes even when one lane's latency changes
little; measure both outcomes independently.

The INT8 head agreed with the fp16 head's top token on 109/112 synthetic
hidden rows. A 4-bit palettized head was faster in the same probe but agreed
on only 62/112 rows. Neither number predicts real DFlash acceptance. Start
with INT8 and compare actual target-feature traces. If full-vocabulary
logits dominate, evaluate a reduced head only after measuring token coverage
and exact speculative replay on held-out traces.

Release gates for this path:

1. Compare fusion output, injected K/V, each draft layer, final hidden rows,
   and logits against the current GGUF drafter on the same prompt and
   verification traces. Report numerical error and proposal agreement.
2. Replay complete greedy generation. Emitted target tokens must match plain
   target decoding exactly. Measure acceptance separately from correctness.
3. Inspect the compiled Core ML compute plan on each supported chip family.
   Confirm every costly draft projection, attention, and head operation is on
   ANE and no draft operation lands on Metal.
4. Benchmark representative chat, code, tool, and long-context workloads,
   including time to first token, steady-state latency, memory per lane,
   concurrent-lane throughput, and energy. Require a measured improvement
   over the GGUF drafter for the workload where ANE is enabled.

Apple documents stateful model predictions and warns that calls sharing one
`MLState` must be serialized: <https://developer.apple.com/documentation/CoreML/MLState>.
Its compute-plan tools report per-operation device placement:
<https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode>.
Anemll Forge has a related DFlash2 ANE prototype, but it targets a different
checkpoint and architecture; its conversion cannot be used unchanged:
<https://github.com/Anemll/anemll-forge/blob/main/DFLASH2_ANE_PLAN.md>.

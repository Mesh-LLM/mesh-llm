# Receipts without model-specific code (proposal)

Status: proposal for review. Nothing here is built.

Receipts ([RECEIPTS.md](RECEIPTS.md)) have model-specific code in two places
today:

1. **Leaf discovery** (native recorder). It finds the receipt tensors by GGUF
   weight names (`blk.N.attn_q.weight`, `ffn_gate`, ...), takes the residual
   from the `rms_norm` in front of `attn_q`, and finds the stage output by the
   node name `l_out-N`.
2. **Glue checks** (auditor). `glue.rs` re-implements the operations between
   the matmuls of one layer (two RMS norms, SwiGLU, residual adds) for the
   `llama`, `qwen2` and `qwen3` architectures only.

This proposal removes both. No code names an architecture, a tensor role or
a node name.

## Idea

Split every layer's computation for one position into three parts:

- **Weight matmuls.** Any `MUL_MAT` or `MUL_MAT_ID` whose weight operand is a
  model tensor of the layer, whatever its name. The row checks already verify
  these, and they need no model knowledge.
- **State operations.** Operations whose output at a position depends on other
  positions: they read the KV cache or a recurrent state. Attention, `ssm_conv`,
  `ssm_scan` and the gated delta net are examples. Find them by their inputs, not
  by op type: an op that reads a memory tensor (a KV or recurrent-state buffer
  of the context) is a state operation. Their inputs and outputs are committed,
  as q, k, v and the attention output are today. They stay unchecked, as
  attention is today.
- **Glue.** Everything else. Given the residual entering the layer, the
  matmul outputs and the state-op outputs, glue is a pure function of the
  position: norms, RoPE, biases, activations, gating, MoE routing, residual
  adds.

The client checks the glue by **evaluating it itself**. It builds the
layer's graph with its own pinned llama.cpp and Skippy graph builder from the
claimed GGUF. Then it runs the graph on the CPU, one position at a time, with
the opened matmul outputs and state-op outputs substituted for those nodes.
It compares what it computes with the opened matmul inputs and the opened
residual leaving the layer. Any architecture that llama.cpp can build is
covered, with ggml's own op implementations.

## Leaves

| Leaf | Found by | Today |
|---|---|---|
| Residual entering layer N | Skippy's block boundary records (entry tensor of layer N) | `rms_norm` input in front of `attn_q` |
| Stage output | Exit tensor of the last stage layer | node named `l_out-N` |
| State leaf of layer N | Inputs and outputs of the layer's state ops | attn_q, attn_k, attn_v outputs and attention output, by weight name |
| Opened matmuls | Every weight matmul of layer N | seven matmuls by weight name |

The block boundaries already exist. Model graph builders register a
canonical entry and exit tensor per layer, validated structurally
(`src/skippy/block_boundaries.h`), and stage splitting relies on them.

For attention, the state leaf holds q and k after RoPE instead of the
`attn_q` and `attn_k` outputs. The commitment stays the same size. The
`attn_q` and `attn_k` outputs are still opened by replay and row-checked, and
the glue evaluation links them to the committed post-RoPE values. This
changes the attention leaf, so the receipt wire version goes to 3.

## Auditor

A new native entry point, roughly:

```c
skippy_receipt_glue_eval(model_path, layer, position,
                         residual_in, matmul_outputs[], state_outputs[],
                         out matmul_inputs[], out residual_out)
```

- Load the claimed GGUF with mmap and the CPU backend, selecting layer N
  only. Only the small tensors the glue reads (norm weights, biases) are
  paged in. Matmul and state nodes are replaced by inputs, so their weights
  are never read.
- Build the layer graph with the normal model builder. Turn each weight-matmul
  and state-op node into an input that holds the opened value.
- Compute in f32 and return every weight-matmul input and the exit tensor.
- The Rust auditor compares them with the opening using the relative-error
  tolerance `glue.rs` uses now. `glue.rs` and its architecture list go away.

The node never supplies the glue program. A node that could send its own
program could make any values check out. The client's graph comes from its
own llama.cpp build and the GGUF it already holds for the row checks.

## What this covers and does not

- Covered: any decoder layer llama.cpp builds, including MoE (routing is glue;
  the selected experts' matmuls are opened and row-checked), biases, gated
  attention outputs, and hybrid or recurrent layers (their state ops become
  the state leaf).
- Still unchecked: what happens inside state ops (attention scores, SSM
  scans), the same as attention today. The output head and the input
  embedding are not part of any layer.
- Fused kernels on the node do not matter. The client computes unfused on the
  CPU and compares with a tolerance.
- Version skew: the client's llama.cpp must build the architecture in the same
  way as the node's. A model whose graph changes between llama.cpp versions
  can fail glue checks until both sides use the same pin. Receipts should
  carry the llama.cpp pin (already planned for milestone 2), so the client can
  tell skew from cheating.

## Costs

- The auditor needs the native runtime (libllama, CPU backend) as well as the
  GGUF. `skippy audit-receipt` already ships with it. A client that cannot
  load native code (a browser, a small SDK) would need a different path.
- The node commits about the same bytes per position. Replay and openings do
  not change.
- Per opened position, the client computes one layer without its matmuls:
  I estimate well under a millisecond on a CPU (not measured).

## Questions for review

1. Is it acceptable for the auditor to depend on libllama? The alternative is
   a Rust interpreter for a small op set, fed a layer program that the
   client's libllama exports. That keeps the evaluation in Rust but has the
   same dependency for building the program, plus a coverage gap for every op
   it lacks.
2. Is "an op that reads a memory tensor" the right definition of a state
   operation? It keeps attention, SSM, conv-state and delta-net ops out of
   the glue without listing them.

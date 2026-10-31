# Inference receipts

An inference receipt lets a client check that a split-serving node ran the
model it claims to run. This feature is experimental and off by default.

With receipts on, each stage commits to the residual stream entering every
layer it runs, and to each layer's token-mixing values: the inputs of its
attention (or other token-mixing op), the KV rows as stored in the cache, and
its outputs. At the end of a turn, each stage commits to these values with a
BLAKE3 Merkle root. The client then picks one layer and 16 token positions at
random. To open them, the node replays the turn in a scratch context and
reports every weight matmul of the challenged layer. The client checks that
the replay matches the committed values, and recomputes random rows of each
matmul from the claimed GGUF. Then it computes the challenged layer itself,
on the CPU with libllama, from the opened matmul and token-mixing outputs,
and compares the result with the opening. A node that serves a different
quantization, or a different model, fails these checks.

The receipt tensors are found from the structure of the graph llama.cpp
builds, so no code lists architectures or tensor names. The design is in
[RECEIPTS.md](../../mesh/docs/design/RECEIPTS.md) and
[RECEIPTS_GLUE.md](../../mesh/docs/design/RECEIPTS_GLUE.md), and the
remaining milestones are in
[RECEIPTS_PLAN.md](../../mesh/docs/design/RECEIPTS_PLAN.md).

Do not confuse inference receipts with the generation receipt in
`skippy-serving`. A generation receipt is a digest of the generated token ids
for local generation hooks. It does not prove which model computed them.

## Requirements

- Binary split serving (`skippy serve --stage-transport binary`). Single-stage
  serving and Mesh-launched stages do not commit receipts yet.
- Weight matmuls, dense or MoE, with F32, F16, BF16 or Q8_0 weights. The
  auditor stops with an error on other weight types, because no rounding
  profile exists for them yet.
- A graph the receipt classifier accepts. Each op must be in its per-ggml-op
  table, a layer may read nothing from another layer except the residual
  stream, and weights that token-mixing ops read directly (such as attention
  sinks) may be at most 1% of the layer's weights. Those are listed as
  unchecked in the audit. A stage whose graph is refused serves normally, and
  its turns get no receipt; `GET /receipts/{id}` returns the reason.
- A plain KV cache with one stream, flash attention (a transposed V cache is
  refused), and no MTP head on the stage. Recurrent and hybrid models (Mamba,
  Qwen3.5) are refused when the stage starts. A quantized KV cache gets no
  receipts yet: llama.cpp rotates keys before storing them in one, and the
  classifier cannot follow that rotation. Only an f16 cache has been tested.
- The auditor needs a local copy of the GGUF that the node claims, and loads
  the challenged layer of it on the CPU.

## Serve with receipts

Start every stage with `--receipts-listen`. Point each stage, except the last
one, at the receipts address of the next stage with `--receipts-downstream`:

```bash
skippy serve --stage-transport binary --worker-only --config plan/stage-1.json \
  --receipts-listen 127.0.0.1:9502 --receipts-claimed-model Qwen3-4B-Q8_0

skippy serve --stage-transport binary --config plan/stage-0.json \
  --bind-addr 127.0.0.1:9337 \
  --receipts-listen 127.0.0.1:9501 \
  --receipts-downstream http://127.0.0.1:9502 \
  --receipts-claimed-model Qwen3-4B-Q8_0
```

| Flag | Purpose |
|---|---|
| `--receipts-listen ADDR` | Turn receipts on and serve the receipts API on `ADDR` |
| `--receipts-downstream URL` | Receipts API of the next stage. Commits and openings for its layers go there |
| `--receipts-claimed-model NAME` | Model named in each receipt. The default is the stage model id |
| `--receipts-allow-remote` | Permit a `--receipts-listen` address that is not loopback |

The receipts API refuses a non-loopback address unless you also pass
`--receipts-allow-remote`. An opening contains activations that come from the
prompt, so anyone who can reach the API can learn about the prompts it served.
Expose it only to auditors that you trust with the prompts.

Each stage keeps leaf hashes and a replay log, not activations. Per position,
that is `2 · layers + 1` hashes (about 1.2 KB for 18 layers) plus the Merkle
tree above them. Per ubatch, the log holds the tokens, positions, output rows,
KV cells and n_kv, and, on a stage after the first, the stage input rows. The
stage input row is the largest part: one hidden-state row per token. Committed
turns are kept up to 1 GiB per stage. The oldest are removed first, and a
removed turn can no longer be opened.

A stage can only replay a sequence whose KV cells it can reproduce. A sequence
becomes unreceiptable, until it is cleared, when it shares cells with another
sequence (a copied prompt prefix), when its positions are shifted, when its
cells are loaded from saved state, or when the cache is compacted.

## Audit a response

`skippy audit-receipt` sends one chat request, gets its receipt, draws a
challenge, and checks the opening:

```bash
skippy audit-receipt \
  --api http://127.0.0.1:9337 \
  --receipts http://127.0.0.1:9501 \
  --claimed-model /models/Qwen3-4B-Q8_0.gguf \
  --prompt "Name three rivers in Europe."
```

Use `--layer N` to challenge a specific layer instead of a random one, and
`--max-tokens N` to change the response length (default 32). The command exits
with a non-zero status when the audit fails. Use `--output json` for a
machine-readable report.

Use `--timing-runs N` to repeat the glue evaluation N times and report the
fastest as the client cost.

An honest node that serves the claimed Q8_0 GGUF on an RTX 5060 Ti:

```text
receipt 3ecb44d7-1b6e-44cf-aa46-6f4338c4b191  model Qwen3-4B-Q8_0  positions 0..70
  stage 0  layers  0..18  root db15a72589558379…  leaves 2590  cache f16  (CUDA0)
  stage 1  layers 18..36  root b14a769991f7fbb3…  leaves 2590  cache f16  (CUDA0)
challenge: layer 5 (stage 0), positions [7, 9, 16, 18, 19, 20, 23, 25, 36, 38, 49, 50, 52, 59, 64, 69]
merkle paths: ok
  blk.5.attn_k.weight            q8_0   16/16  pass  max err 1.3e-4  q8_32_div_dh  ok
  blk.5.attn_output.weight       q8_0   16/16  pass  max err 3.0e-4  q8_32_div_dh  ok
  ...
  blk.5.ffn_up.weight            q8_0   16/16  pass  max err 2.7e-4  q8_32_div_dh  ok
glue (the layer computed here on the opened values):
  matmul inputs  112/112 pass  max err 1.8e-7  ok
  mixing inputs   16/16  pass  max err 3.0e-6  ok
  stored rows     32/32  pass  max err 0.0e0  ok
  residual out    16/16  pass  max err 0.0e0  ok
client cost: layer load 1.50 s, glue 0.37 ms per position (16 positions, first run 82 ms), row checks 765 ms
AUDIT PASS
```

A MoE layer (granite-3.1-3b-a800m) adds the router, which is checked on every
row, the experts each matmul used, and the router's selection:

```text
  blk.5.ffn_down_exps.weight     q8_0  128/128 pass  max err 3.6e-4  q8_32_div_dh  ok
  blk.5.ffn_gate_inp.weight      f32    16/16  pass  max err 2.7e-7  f16  ok  (router, all rows)
  ...
glue (the layer computed here on the opened values):
  matmul inputs  128/128 pass  max err 1.6e-7  ok
  expert ids      48/48  pass  max err 0.0e0  ok
  mixing inputs   16/16  pass  max err 3.2e-6  ok
  stored rows     32/32  pass  max err 0.0e0  ok
  selections      16/16  pass  max err 0.0e0  ok
  residual out    16/16  pass  max err 9.4e-8  ok
```

The same Qwen3-4B split serving Q6_K while the receipts claim Q8_0:

```text
challenge: layer 5 (stage 0), positions [...]
merkle paths: ok
  blk.5.attn_k.weight            q8_0    0/16  pass  max err 1.3e-2  q8_32_div_dh  FAIL
  ...
  blk.5.ffn_up.weight            q8_0    0/16  pass  max err 3.4e-2  q8_32_div_dh  FAIL
glue (the layer computed here on the opened values):
  ...
  residual out    16/16  pass  max err 0.0e0  ok
AUDIT FAIL
```

The Merkle paths and the glue checks still pass in the second case: the node
committed to what it computed, and its replay reproduced it. The row check
fails because those values do not come from the claimed weights. The row
tolerance is 1e-3; honest CUDA and CPU errors are about 1e-4 to 1e-3. The glue
tolerance is 1e-4, relative to the largest value of the reference column;
honest errors are about 1e-7 to 3e-6.

### What the glue checks bind

A replay recomputes the challenged layer, so its matmuls would pass the row
checks even if different weights served the turn. The glue check ties the
replayed matmuls to the committed values instead. The auditor builds the
challenged layer with its own llama.cpp from the claimed GGUF, and replaces
every weight matmul output, token-mixing output and selection with the opened
value, and the residual entering the layer with the opened one. Then it
computes the layer and compares:

- each weight matmul's input, and the experts of each MoE matmul;
- each token-mixing input, such as q after RoPE;
- each stored KV row. The opened row is in the cache's type; it may be no
  further from the auditor's f32 value than 1.5 times the auditor's own
  rounding of that value into the cache type, plus 1e-4 of the row's norm;
- each selection (a top-k must hold the highest scores, an argsort must be in
  order), against scores computed from the opened router output;
- the residual leaving the layer.

The node must also show that its replay reproduces every committed leaf of the
challenged positions. The served token-mixing values are committed, so the row
checks of the matmuls that produce them test the served values. What happens
inside attention or another token-mixing op is not checked. The audit first
checks that the opening has the plan of the auditor's graph: the same weights
by index and name, the same routers, and the same number of token-mixing
values and selections.

## HTTP API

Every stage serves these routes on its receipts address. Errors return HTTP 400
with `{"error": "..."}`, except for a receipt that does not exist (yet), which
returns HTTP 404. Stage 0 commits a turn just after it sends the response, so
a client that asks at once can get a 404 and should retry.

| Route | Served by | Purpose |
|---|---|---|
| `GET /receipts/{id}` | Stage 0 | The receipt for a response. `id` is the request's `x-request-id`. Stage 0 keeps the last 256 receipts |
| `POST /receipts/open` | Any stage | Opens a challenge `{receipt_id, layer, positions}`. A stage forwards a layer that it does not hold to its downstream stage |
| `POST /receipts/commit` | Stages after stage 0 | Commits a session's positions. The previous stage calls this at the end of a turn |

The receipt id is the request's `x-request-id`. A client can send its own UUID
in that header, or read the `x-request-id` header of the response.

## Cost and limits

- Receipts cost some decode and prefill throughput. Each ubatch copies the
  committed tensors, including the stored KV rows, to pinned host memory. A
  worker thread hashes them while the next ubatch computes, and a commit or
  opening waits for it to finish. Two-stage Qwen3-4B Q8_0 split on one RTX
  5060 Ti, release build, median of five runs: decode 71.7 → 69.5 tok/s (3%
  slower), prefill 2292 → 1661 tok/s (1.38x slower).

- An opening replays the challenged sequence from its first position, so its
  cost grows with the length of the conversation. Each replayed step stops
  after the challenged layer, so the later layers and the output head are
  skipped. The same two-stage split on the RTX 5060 Ti, challenging the last
  16 positions (the worst case):

  | Conversation (prompt + generated) | Serving | Open layer 5 (stage 0) | Open layer 30 (stage 1) |
  |---|---|---|---|
  | 86 + 256 | 3.9 s | 0.67 s | 1.27 s |
  | 88 + 1024 | 14.3 s | 2.02 s | 4.30 s |
  | 2859 + 32 | 2.1 s | 0.39 s | 0.66 s |
  | 2857 + 512 | 9.3 s | 1.32 s | 2.76 s |

- The auditor's cost per opening of 16 positions, release build on the CPU:
  loading the challenged layer takes 0.5 to 1.3 s, the first glue evaluation
  76 to 160 ms (it creates the glue context), and each later evaluation of
  the same layer 0.03 to 0.27 ms per position. The row checks take 20 ms to
  1 s, depending on the page cache and, for MoE, the full router check.

- A stage keeps the openings it made, up to 128 MiB, and answers a repeated
  challenge from them in milliseconds. They are removed with their turn.
- A receipt is replayed for at most 4 distinct challenges. A replay costs
  about as much as serving the turn did, so this bounds the work that one
  receipt can ask for.
- Replays run one at a time in a second context of the stage model, which
  needs its own KV cache (about 1.2 GB across the two stages above). The
  context is created by the first replay and freed after 60 seconds without
  one.
- Receipts are not signed and turns are not chained. The receipt is fetched
  after the response, not attached to it.
- The output head and what happens inside attention or another token-mixing
  op are not checked.
- On the last layer of the final stage, the FFN runs only for the output rows,
  so the FFN matmuls and glue checks cover fewer positions.
- The auditor's llama.cpp must build the layer as the node's does. A model
  whose graph changes between llama.cpp versions can fail glue checks until
  both sides use the same pin.
- The receipt tensors are graph outputs while receipts are on. This keeps them
  in memory and can disable some kernel fusions.

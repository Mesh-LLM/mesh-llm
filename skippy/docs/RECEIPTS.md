# Inference receipts

An inference receipt lets a client check that a split-serving node ran the
model it claims to run. This feature is experimental and off by default.

With receipts on, each stage commits to the residual stream entering every
layer it runs, and to each layer's attention projections (the attn_q, attn_k
and attn_v outputs and the attention output). At the end of a turn, each stage
commits to these values with a BLAKE3 Merkle root. The client then picks one
layer and 16 token positions at random. To open them, the node replays the turn
in a scratch context and reports every weight matmul of the challenged layer.
The client checks that the replay matches the committed values, recomputes
random rows of each matmul from the claimed GGUF, and checks the operations
that join the matmuls to the residual stream. A node that serves a different
quantization, or a different model, fails these checks.

The design is in [RECEIPTS.md](../../mesh/docs/design/RECEIPTS.md), and the
remaining milestones are in
[RECEIPTS_PLAN.md](../../mesh/docs/design/RECEIPTS_PLAN.md).

Do not confuse inference receipts with the generation receipt in
`skippy-serving`. A generation receipt is a digest of the generated token ids
for local generation hooks. It does not prove which model computed them.

## Requirements

- Binary split serving (`skippy serve --stage-transport binary`). Single-stage
  serving and Mesh-launched stages do not commit receipts yet.
- Dense weight matmuls with F32, F16, BF16 or Q8_0 weights. The auditor
  stops with an error on other weight types, because no rounding profile exists
  for them yet.
- A pre-norm layer with an RMS norm and a SwiGLU FFN. The auditor has glue
  checks for the `llama`, `qwen2` and `qwen3` architectures only.
- A unified KV cache with one stream, and no MTP head on the stage.
- The auditor needs a local copy of the GGUF that the node claims.

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

An honest node that serves the claimed Q8_0 GGUF:

```text
  stage 0  layers  0..18  root afff5e2d95865e6b…  leaves 1998  (CUDA0)
  stage 1  layers 18..36  root 941418e9e2c719d8…  leaves 1998  (CUDA0)
challenge: layer 3 (stage 0), positions [0, 1, 2, 5, 6, 15, 17, 19, 20, 23, 24, 31, 36, 41, 43, 45]
merkle paths: ok
  blk.3.attn_k.weight        q8_0  16/16 tokens pass  max err 1.3e-4  q8_32_div_dh  ok
  blk.3.attn_output.weight   q8_0  16/16 tokens pass  max err 2.4e-4  q8_32_div_dh  ok
  ...
  blk.3.ffn_up.weight        q8_0  16/16 tokens pass  max err 2.2e-4  q8_32_div_dh  ok
glue:
  attn_in = rms_norm(residual_in)    16/16 tokens pass  max err 9.1e-8  ok
  ffn_in = rms_norm(mid)             16/16 tokens pass  max err 1.3e-7  ok
  down_in = silu(gate) · up          16/16 tokens pass  max err 1.6e-7  ok
  residual_out = mid + down          16/16 tokens pass  max err 0.0e0  ok
AUDIT PASS
```

The same split serving Q6_K while the receipts claim Q8_0:

```text
challenge: layer 5 (stage 0), positions [...]
merkle paths: ok
  blk.5.attn_k.weight        q8_0   0/16 tokens pass  max err 9.1e-3  q8_32_div_dh  FAIL
  ...
  blk.5.ffn_up.weight        q8_0   0/16 tokens pass  max err 2.9e-2  q8_32_div_dh  FAIL
glue:
  ...
  residual_out = mid + down          16/16 tokens pass  max err 0.0e0  ok
AUDIT FAIL
```

The Merkle paths and the glue checks still pass in the second case: the node
committed to what it computed, and its replay reproduced it. The row check
fails because those values do not come from the claimed weights. The row
tolerance is 1e-3; honest CUDA and CPU errors are about 1e-4 to 1e-3. The glue
tolerance is 1e-4, relative to the largest value of the reference row; honest
errors are about 1e-7.

### What the glue checks bind

A replay recomputes the challenged layer, so its matmuls would pass the row
checks even if different weights served the turn. The glue checks tie the
replayed matmuls to the committed residual stream instead:

```text
attn_in      = rms_norm(residual_in) · attn_norm     (input of attn_q, attn_k, attn_v)
mid          = residual_in + attn_output(attn_out)
ffn_in       = rms_norm(mid) · ffn_norm              (input of ffn_gate, ffn_up)
down_in      = silu(ffn_gate) · ffn_up               (input of ffn_down)
residual_out = mid + ffn_down
```

The node must also show that its replay reproduces every committed leaf of the
challenged positions. The served values of attn_q, attn_k, attn_v and the
attention output are committed, so the row checks of those matmuls test the
served values. Attention itself is not checked.

## HTTP API

Every stage serves these routes on its receipts address. Errors return HTTP 400
with `{"error": "..."}`.

| Route | Served by | Purpose |
|---|---|---|
| `GET /receipts/{id}` | Stage 0 | The receipt for a response. `id` is the request's `x-request-id`. Stage 0 keeps the last 256 receipts |
| `POST /receipts/open` | Any stage | Opens a challenge `{receipt_id, layer, positions}`. A stage forwards a layer that it does not hold to its downstream stage |
| `POST /receipts/commit` | Stages after stage 0 | Commits a session's positions. The previous stage calls this at the end of a turn |

The receipt id is the request's `x-request-id`. A client can send its own UUID
in that header, or read the `x-request-id` header of the response.

## Cost and limits

- Receipts cost some decode and prefill throughput. Each ubatch copies the
  committed tensors to pinned host memory. A worker thread hashes them while
  the next ubatch computes, and a commit or opening waits for it to finish.
  Two-stage Qwen3-4B Q8_0 split on one GPU, release build, median of five
  runs:

  | GPU | Decode, off → on | Prefill, off → on |
  |---|---|---|
  | RTX 5090 | 151 → 147 tok/s (3% slower) | 3382 → 3056 tok/s (1.11x) |
  | RTX 5060 Ti | 71 → 69 tok/s (3% slower) | 2396 → 1868 tok/s (1.28x) |

- An opening replays the challenged sequence from its first position, so its
  cost grows with the length of the conversation. Replays run one at a time
  in a second context of the stage model, which needs its own KV cache.
- Receipts are not signed and turns are not chained. The receipt is fetched
  after the response, not attached to it.
- The output head, attention scores, and other non-matmul operations are not
  checked.
- On the last layer of the final stage, the FFN runs only for the output rows,
  so the FFN matmuls and glue checks cover fewer positions.
- The receipt tensors are graph outputs while receipts are on. This keeps them
  in memory and can disable some kernel fusions.

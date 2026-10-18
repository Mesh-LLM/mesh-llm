# Qwen3-MoE native MTP

Skippy runs a separately trained multi-token-prediction (MTP) head for
Qwen3-MoE checkpoints such as Qwen3-Coder-30B-A3B. It drafts with the existing
native MTP sidecar path, unsplit and on the final stage of a split. The target
verifies every draft.

## The head

The head is defined by `splitcoder.mtp.MTPHead` in `general-inference/models`:

```text
h_t      trunk hidden state at position t, after the trunk's final RMSNorm
e_{t+1}  token embedding of token t+1 (the trunk's table)
z        = proj([hnorm(h_t); enorm(e_{t+1})])          proj: 2d -> d, no bias
z'       = one dense Qwen3 decoder layer(z)            causal attention, SwiGLU FFN
logits   = lm_head(norm(z'))  for token t+2            the trunk's LM head
```

Deeper drafts chain on `z'`, the layer output before `norm`, with the drafted
token as the next embedding.

That is the shape the runtime already uses for GLM and Qwen3.5 NextN blocks, with
three differences. None of them needs retraining:

- The input order: GGUF `eh_proj` concatenates `[enorm(e); hnorm(h)]`, so the
  converter swaps `proj`'s two input halves.
- The FFN: the block's FFN is dense, unlike the trunk's MoE layers. Its width
  is read from the block's own tensors.
- The chained hidden state: the qwen3moe MTP graph exposes the pre-norm `z'` as
  its chained hidden state. GLM exposes its post-norm output.

The post-norm trunk hidden state matches what this runtime already feeds GLM and
Qwen3.5 heads.

## GGUF layout

The head is NextN block `n = num_hidden_layers` (48 for Qwen3-Coder-30B-A3B),
and `qwen3moe.nextn_predict_layers = 1`. Checkpoint keys map as follows:

| `mtp.safetensors` | GGUF |
| --- | --- |
| `hnorm.weight` | `blk.n.nextn.hnorm.weight` |
| `enorm.weight` | `blk.n.nextn.enorm.weight` |
| `proj.weight` | `blk.n.nextn.eh_proj.weight` (input halves swapped) |
| `norm.weight` | `blk.n.nextn.shared_head_norm.weight` |
| `layer.input_layernorm` | `blk.n.attn_norm` |
| `layer.self_attn.{q,k,v,o}_proj` | `blk.n.attn_{q,k,v,output}` |
| `layer.self_attn.{q,k}_norm` | `blk.n.attn_{q,k}_norm` |
| `layer.post_attention_layernorm` | `blk.n.ffn_norm` |
| `layer.mlp.{gate,up,down}_proj` | `blk.n.ffn_{gate,up,down}` |

The block reuses the trunk's `token_embd` and `output`.

## Conversion

Place `mtp.safetensors` and `mtp_config.json` beside the HF checkpoint. The
converter checks that the head's config matches the trunk. `--mtp` exports only
the head and the shared tensors, as a draft sidecar:

```bash
python3 convert_hf_to_gguf.py <checkpoint> --mtp --outtype bf16 --outfile mtp-head.gguf
```

Append the head to an existing quantized trunk instead of reconverting it:

```bash
skippy-quantize compose-mtp \
  --target-shard Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --mtp-gguf mtp-head.gguf --mtp-block 48 \
  --set-kv qwen3moe.nextn_predict_layers=1 \
  --output qwen3coder-q4km-mtp.gguf
```

`compose-mtp` skips shared tensors the target already carries.

## Serving

Unsplit, with the integrated GGUF:

```bash
skippy serve --model-path qwen3coder-q4km-mtp.gguf \
  --speculative-strategy native-mtp --mtp-max-tokens 3 --verify-max-tokens 3
```

Unsplit serving can also attach the sidecar to the plain trunk with
`--native-mtp-draft-model-path mtp-head.gguf`. A split needs the integrated
GGUF, because the final stage loads the NextN block with the trunk's last
layers. `skippy plan-split` enables native MTP on the stages when the GGUF
carries NextN tensors.

On a split, the final stage drafts. It is the only stage that has `h_t`, and it
already holds the LM head. The drafts return to stage 0 on the existing reply
path, a few bytes per token, and stage 0 builds the next verify window.

Shipping `h_t` back to stage 0 instead would cost 4 KB per token in f16. It
would also need a second LM head on stage 0, and save no network round trip.

Stage 0 asks for a draft depth on every decode and verify message. That depth
travels in spare bits of the stage-state header. Workers that predate the
field leave it unset and draft one token.

Two runtime changes keep a split drafting after a rejected window:

- **Drafts after the correction.** The final stage drafts from the first row
  that rejected its proposal, continuing from that row's correction token. Stage
  0 commits the accepted prefix plus the correction, so it can verify that draft
  in the very next window without a single-token decode first. A stage 0 opts in
  with a header flag. The reply marks such drafts, and older stage 0s never
  receive one.
- **Trim keeps the sidecar.** Trimming a session after a rejection used to clear
  the MTP sidecar's whole cache. The next decode then had no draft, and later
  drafts attended only to tokens decoded after the trim. The sidecar now keeps
  its rows below the trim point and reseeds from the last kept token.

## Cost of a verify pass

A verify pass over `k` tokens of an MoE model reads up to `k` times as many
experts as a one-token decode. On an M5 Max, a Qwen3-Coder-30B-A3B Q4_K_M
decode costs about 7 ms for one token. A verify pass costs about 10.5 ms for
two tokens, and llama.cpp alone takes 19 ms for three and 22 ms for four.

Unsplit, depth 1 pays off when about half the drafts are accepted, and deeper
drafts need more. On a split, every pass also pays the network round trip, so
each accepted draft saves a whole hop.

A sampled verify window detaches any backend (in-graph) sampler, because it
needs every row's prediction. A rejected window then rebuilds the sampler
chain. A session that has verified a window keeps sampling on the CPU until it
resets. Re-attaching the rebuilt chain made every rejection re-reserve the
scheduler twice, which doubled the cost of a two-token verify on Metal.

## Packages

`skippy-package-builder write-package` infers the native MTP declaration from
the GGUF. `--native-mtp-draft-depth N` records how many tokens a chaining head
should draft, with an adaptive verify window that starts at two:

```json
"mtp": {
  "type": "native-mtp",
  "prediction_depth": 3,
  "layer_indices": [48],
  "window_policy": {"default": "adaptive", "initial_window": 2, "min_window": 1, "max_window": 3}
}
```

Mesh selects the declared strategy under `strategy = "auto"`. It drafts
`prediction_depth` tokens unless `draft_max_tokens` is set explicitly.

## Verification

`skippy/evals/native-mtp-parity` compares the runtime head with `MTPHead` on the
same hidden states, tokens and weights. On Metal, with the head in f32, both
depths agree with the reference on every top-1 token, with KL below 5e-5. A
head whose `proj` halves are swapped fails the same check.

Greedy text with speculation can differ from greedy text without it. That holds
for every speculative mode, MTP or not. Batched verification is not
bit-identical to single-token decode, and for this MoE small differences can
flip expert routing. `batch_invariance_probe` measures that effect on the trunk
alone. `divergence.py` shows whether each text difference starts at a
numerical near-tie of the target or at a real error.

The synthetic CTest `skippy_qwen3moe_mtp` certifies the integrated stage program
and chained drafts for qwen3moe.

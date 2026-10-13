# Laya golden fixtures

PyTorch reference outputs for `convaiinnovations/laya-multilingual`, one
question per fixture: the rendered options, token ids, marker positions, raw
scorer and action-head logits, and the final `noul` / `choice` / `score`
answers.

Copied unchanged from `tests/laya/golden` in the upstream draft
[ggml-org/llama.cpp#29363](https://github.com/ggml-org/llama.cpp/pull/29363)
at head `6367bdd2da2d0107077ba98383bdfd9e4bf9f443`, which generated them with
`tests/laya/gen_fixtures.py` from the Python reference implementation.
`model_support/0006` ports that PR without its test tree; these files are the
part the parity check needs.

`cargo xtool automation laya parity` compares a mesh `/systemone` endpoint or a
`llama-laya-cli` build against them. Each fixture is allowed the error
upstream's own CPU runtime shows against the same goldens, plus 0.005; see
`allowance` in `tools/xtask/src/automation/laya/parity.rs`. Upstream CPU misses `noul_zh` by 0.0579,
twice the deviation the upstream PR reports, so that fixture is the one to
watch if the upstream graph or goldens change.

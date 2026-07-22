# Quantization Recipes

This directory contains tensor-type override recipes for `skippy-quantize`.
Recipe files are intentionally comment-free because `skippy-quantize
validate-tensor-types` parses each whitespace-separated token as a
`PATTERN=TYPE` entry.

`PATTERN` is passed to llama.cpp as a regular expression and matched with
`std::regex_search`. First match wins, so put narrow overrides before broad
ones.

## GLM-5.2

Use `glm-5.2-q2-k-mtp-q8.tensor-types.txt` with base `--quant Q2_K` to
recreate the lab's comfort-fit GLM-5.2 profile:

- GLM attention projection tensors that were sensitive in local testing stay
  at `Q8_0`.
- Shared expert tensors stay at `Q4_K`.
- Native NextN/MTP tensors containing `.nextn.` stay at `Q8_0`.
- Routed expert gate/up tensors follow the `Q2_K` default.
- Routed expert down tensors follow llama.cpp's `Q2_K` profile fallback, which
  currently keeps them at `Q3_K` where required.

Use `glm-5.2-q2-k-routed-down-mtp-q8.tensor-types.txt` for the Phase E
routed-down experiment. It changes only decoder-layer routed down projections
to `Q2_K`:

```text
^blk\.([0-9]|[1-6][0-9]|7[0-7])\.ffn_down_exps\.weight$=Q2_K
```

The range deliberately excludes `blk.78`, the GLM-5.2 native NextN/MTP block.
That lets us test the measured decode throughput win from q2_K routed-down
without lowering the MTP block's sparse MLP down projection.

Validate recipes before launching an expensive quantization job:

```bash
skippy-quantize validate-tensor-types recipes/quantization/glm-5.2-q2-k-mtp-q8.tensor-types.txt
skippy-quantize validate-tensor-types recipes/quantization/glm-5.2-q2-k-routed-down-mtp-q8.tensor-types.txt
```

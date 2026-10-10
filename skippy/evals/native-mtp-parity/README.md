# Native MTP parity

Checks a runtime's native MTP head against its PyTorch reference, and
explains greedy-text differences that speculation introduces.

These are developer evals for a real checkpoint. They are not CI: they need a
multi-gigabyte GGUF, the trained head, and PyTorch.

## Build

Prepare the patched checkout and package a local runtime (`just skippy`), then:

```bash
skippy/evals/native-mtp-parity/build.sh
```

The probes link against `target/debug/native-runtimes/*/lib`; set
`SKIPPY_RUNTIME_LIB` to use another runtime bundle.

## Head parity

`mtp_parity.py` runs the trunk over a fixed coding-agent context with tools,
dumps the hidden state that feeds the MTP head, and decodes the head at depth 1
and depth 2 (chained on the head's own pre-norm output). It feeds the same
hidden rows and tokens to `splitcoder.mtp.MTPHead`, using the embedding and
LM head dequantized from the same GGUF, and compares the logits.

```bash
PYTHONPATH=<llama.cpp>/gguf-py:<models>/split-coder/src python3 \
  skippy/evals/native-mtp-parity/mtp_parity.py \
  --gguf qwen3coder-q4km-mtp.gguf --head <dir with mtp.safetensors> \
  --tokenizer <HF checkpoint dir> --work /tmp/mtp-parity
```

It passes when both depths reach `--min-cos` (0.999) and `--min-top1` (0.97),
and a negative control fails: the same reference with `proj`'s two input halves
swapped, the conversion bug the check exists to catch.

## Greedy divergence

Batched verification is not bit-identical to single-token decode. For
Qwen3-MoE, small summation-order differences can flip expert routing, so
greedy text with any speculation can differ from greedy text without it.
`batch_invariance_probe` measures this on the trunk alone: it compares the
logits of the same tokens decoded one at a time and as one verify window.

`divergence.py` takes two bench output files for the same shape, with
speculation off and on. For each prompt whose text differs, it decodes the
shared prefix the way unspeculated generation does and reports the trunk's
logit margin between the two candidate tokens. A divergence at a near-tie is a
numerical flip. A large margin means speculation committed a token the target
did not choose, which is a bug.

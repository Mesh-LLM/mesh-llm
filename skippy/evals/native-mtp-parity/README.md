# Native MTP probes

These developer probes inspect a real checkpoint. They require a compatible
multi-gigabyte GGUF and a prepared native runtime; they are outside CI.

The former Python parity and divergence drivers are retired. No independent
PyTorch oracle, tokenizer, or automated comparison is provided here. Running a
probe alone does not establish head parity or validate a model conversion.

## Build

Prepare the patched checkout and package a local runtime with `just skippy`,
then build the probes with `just skippy-native-mtp-probes-build`. The probes
link against `target/debug/native-runtimes/*/lib`; set `SKIPPY_RUNTIME_LIB` to
select another runtime bundle.

## Inputs and outputs

Each probe consumes a `tokens.i32` file containing native-endian signed 32-bit
token IDs for the loaded model. Obtain these from the actual tokenizer or a
recorded run. The probes do not apply a chat template or tokenize text.

```bash
mkdir -p /tmp/mtp-probe
target/debug/native-mtp-probes/mtp_parity_probe \
  model.gguf tokens.i32 /tmp/mtp-probe
```

`mtp_parity_probe` requires at least three tokens and writes native-endian f32
rows to the existing output directory. Its final JSON line reports token count
`T`, embedding width `d`, and vocabulary size `V`.

| File | Shape | Contents |
| --- | --- | --- |
| `h.f32` | `T × d` | Trunk hidden states supplied to the head |
| `logits1.f32` | `(T - 1) × V` | Teacher-forced depth-one head logits |
| `z1.f32` | `(T - 1) × d` | Depth-one hidden states used for chaining |
| `logits2.f32` | `(T - 2) × V` | Chained depth-two head logits |

The other probes accept a prompt length and either a verification window width
or two candidate token IDs:

```bash
target/debug/native-mtp-probes/batch_invariance_probe \
  model.gguf tokens.i32 128 4
target/debug/native-mtp-probes/margin_probe \
  model.gguf tokens.i32 128 100 200
```

Use token files longer than the prompt, valid candidate IDs, and a positive
window width that fits the remaining tokens. These are low-level developer
probes; they do not validate arbitrary inputs.

`batch_invariance_probe` compares serial and batched trunk decode on the same
tokens. It reports window count, argmax flips, and the largest relative logit
difference. `margin_probe` decodes the prompt in batches and each subsequent
token individually, then reports the two candidates' logits and the argmax.

A small serial logit gap can explain why batched verification selects another
token. A large gap calls for inspecting the speculative verification window;
the serial replay does not observe the logits committed by that window.

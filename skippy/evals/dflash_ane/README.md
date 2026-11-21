# Qwen3.6 DFlash on ANE

The [Rust converter](../../crates/skippy-ane/README.md) turns
the existing paired Qwen3.6-35B-A3B DFlash GGUF into a stateful Core ML
drafter. Skippy runs its feature fusion, six draft
layers, attention, and output head on Apple Neural Engine. Core ML places the
final argmax on CPU; the target model and exact verification remain on Metal.
Conversion and serving need no Python or Core ML Tools. The converter uses a
Rust GGUF reader and MIL package writer, then calls Apple's Core ML compiler.
Serving needs the source draft GGUF, compiled model, embedding file, and
matching manifest. Skippy still opens the draft GGUF and checks its hash when
attaching the ANE drafter.

The converter accepts the exact draft GGUF used by the model catalog and the
target GGUF that will serve requests. They must be paired: the target supplies
the token embedding and output head omitted from the draft GGUF. It writes a
source-hash manifest, an `.mlpackage`, a compiled `.mlmodelc`, and an fp16
embedding file. Set aside about 4 GB of temporary disk space:
Conversion checks the compiled Core ML compute plan and fails unless all
convolution, matrix multiplication, and softmax operations prefer ANE. The
manifest records those counts.

```sh
just dflash-ane-convert \
  /path/to/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf \
  /path/to/Qwen3.6-35B-A3B.gguf \
  /path/to/qwen36-dflash-ane-512.mlpackage \
  /path/to/qwen36-dflash-embeddings.f16 \
  --context 512 --rows 7
```

The output uses a 512-slot rolling draft history. At longer positions it
discards older draft K/V, including for the draft's last full-attention layer;
this can change proposal acceptance but cannot change emitted target tokens.
Use a larger `--context` when maintaining more draft history is important.
The 512-slot profile performed best on this Mac in the measured workloads.

Select the compiled package for a Skippy or MeshLLM process that uses this
paired DFlash draft:

```sh
export SKIPPY_DFLASH_ANE_MODEL=/path/to/qwen36-dflash-ane-512.mlmodelc
export SKIPPY_DFLASH_ANE_EMBEDDINGS=/path/to/qwen36-dflash-embeddings.f16
```

The matching manifest must be beside the compiled model with the same stem
and `.json` extension. Skippy checks the draft and embedding hashes, the
model's input shape, and the macOS version before attaching. Without these
variables, the existing GGUF drafter remains in use. The ANE path requires
macOS 15 or newer and the package must be compiled on a compatible Mac.

The Core ML graph follows the stateful DFlash design explored in
[Anemll Forge](https://github.com/Anemll/anemll-forge/blob/main/DFLASH2_ANE_PLAN.md)
(MIT license), adapted to this six-layer DFlash checkpoint and its eight
feature taps. Its source is independent of the Qwen3.8 DFlash2 checkpoint.

# Native MTP coverage

The 2026-09-22 audit read the pinned GGUF metadata and tensor directories for
all 89 family-battery targets. Three selected artifacts contain integrated
NextN/MTP heads; the other 86 declare no native heads. This is a claim about
the exact selected artifacts, not every variant of each architecture.

| Family | Trunk layers | Native heads | Pinned repository revision |
| --- | ---: | ---: | --- |
| GLM-4.5-Air | 46 | 1 | [506d64a](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/tree/506d64aa8c5cfe9dbbf00bc7a15739438f83204d) |
| Nemotron | 52 | 1 | [f2d3fe3](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/f2d3fe3694501008786e81e5f20360cbf715496a) |
| MiMo2 | 48 | 3 | [e528cbf](https://huggingface.co/bartowski/MiMo-V2.5-GGUF/tree/e528cbf4b6197dd642b393acc069fb1cd529e085) |

The source of artifact paths and immutable identities is
[`family-certified.json`](family-certified.json), generated from the model
registry. MTP tensors can occur in a later shard than the model metadata; the
existing native artifact scan reads every shard. A search of all 89 pinned
repository file listings found no separately named MTP/NextN/assistant sidecar.
No duplicate draft-model download is needed for these integrated heads.

The existing staged single-step and chain lanes request one draft token. They
remain required, but cannot establish coverage of MiMo2's three heads. The
additional `native-mtp-heads` lane requests the metadata-declared head count,
requires that many proposals, and teacher-forces every proposal prefix through
target decoding and an independent MTP-disabled baseline. A rejected proposal
is valid; missing heads or changed target predictions fail. The report records
each head's proposal, target prediction, and baseline prediction.

The baseline opens after the integrated model closes, avoiding doubled model
residency. This lane tests native head execution and target-state isolation;
the existing split lanes retain responsibility for staged transport parity.

```bash
python3 scripts/plan-family-battery.py --inspect-gguf /path/to/model.gguf
target/debug/skippy-correctness native-mtp-heads \
  --model /path/to/model.gguf --layer-end 48 --n-gpu-layers 999 \
  --report-out /tmp/native-mtp-heads.json
```

Family planning rejects a cached GGUF head count that differs from the
registry declaration before compilation. Every native-head row requires the
new lane in its certification receipt; omitted or failed evidence cannot pass.

Native-head certification budgets include two additional startup allowances
for the integrated model and independent baseline loads, retaining the existing
absolute timeout cap. Dry-run planning reflects the declared native-head lane;
actual execution still requires the immutable metadata and tensor scans.

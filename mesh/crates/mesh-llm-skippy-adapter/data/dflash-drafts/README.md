# DFlash draft conversions

Some DFlash drafts are only published as safetensors, or only in a GGUF our
runtime cannot use. `manifest.json` lists the ones mesh-llm converts and hosts
under `meshllm/`. Nothing is uploaded there by hand: when the manifest changes
on `main`, `.github/workflows/dflash-drafts-publish.yml` converts each draft
that is not yet published and uploads it with the repository's `HF_TOKEN`.

A pairing may name a `meshllm/` draft only if this manifest produces it with
the same file, SHA-256 and size; the `dflash_drafts_tests` unit tests enforce
that. Prefer the publisher's own GGUF where one carries everything the
runtime needs.

## Entries

```json
{
  "name": "Laguna-XS-2.1-DFlash-Q8_0",
  "source": {
    "repo": "poolside/Laguna-XS-2.1-DFlash",
    "revision": "<40-character commit>",
    "files": [{ "path": "model.safetensors", "sha256": "<sha256>" }]
  },
  "target_model": {
    "repo": "poolside/Laguna-XS-2.1",
    "revision": "<40-character commit>",
    "files": [{ "path": "tokenizer.json", "sha256": "<sha256>" }]
  },
  "convert": { "converter": "dflash-laguna", "outtype": "q8_0" },
  "output": {
    "repo": "meshllm/Laguna-XS-2.1-DFlash-Q8_0-GGUF",
    "file": "Laguna-XS-2.1-DFlash-Q8_0.gguf",
    "sha256": "<sha256 of the converted file>",
    "bytes": 494735328
  }
}
```

- `source` and `target_model` list every file the converter reads, pinned by
  revision and SHA-256. The converter hashes each before reading it.
  `target_model` holds the target's `config.json`, `tokenizer.json` and
  `tokenizer_config.json`: DFlash drafts share the target's vocabulary.
- `convert` names the conversion `cargo xtool models dflash-draft-convert`
  performs. `dflash-laguna` writes Poolside's Laguna drafters
  (`DFlashLagunaForCausalLM`) the way the patched llama.cpp converter does,
  matrices as Q8_0. It refuses config keys and tokenizers it does not handle,
  so a new drafter needs converter support (checked against llama.cpp's
  output) before it gets an entry.
- `output` is the file CI uploads. It is uploaded only if the conversion
  hashes to `sha256` and is `bytes` long, and only if the file does not exist
  yet. A published file is never overwritten or deleted; one that hashes
  differently fails the run.

After the upload, the run's summary records the repo revision to pin in the
pairing index.

## Reproducing an output

The converter names the model after the source directory, so keep the
directory named after the source repo:

```sh
hf download poolside/Laguna-XS-2.1-DFlash config.json model.safetensors \
  --revision <revision> --local-dir work/source/Laguna-XS-2.1-DFlash
hf download poolside/Laguna-XS-2.1 config.json tokenizer.json tokenizer_config.json \
  --revision <revision> --local-dir work/target/Laguna-XS-2.1
cargo xtool models dflash-draft-convert \
  --manifest mesh/crates/mesh-llm-skippy-adapter/data/dflash-drafts/manifest.json \
  --draft Laguna-XS-2.1-DFlash-Q8_0 --source work/source/Laguna-XS-2.1-DFlash \
  --target-model work/target/Laguna-XS-2.1 --output Laguna-XS-2.1-DFlash-Q8_0.gguf
```

The command prints the SHA-256 and size it produced and fails if they differ
from the manifest, so it also gives a new entry its `output`. The conversion
is deterministic, and for Laguna-XS 2.1 it reproduces llama.cpp's
`convert_hf_to_gguf.py` output byte for byte.

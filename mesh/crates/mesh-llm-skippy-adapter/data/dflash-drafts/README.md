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
  "convert": { "llama_cpp_upstream": "<skippy/llama_cpp/upstream.txt>", "outtype": "q8_0" },
  "output": {
    "repo": "meshllm/Laguna-XS-2.1-DFlash-Q8_0-GGUF",
    "file": "Laguna-XS-2.1-DFlash-Q8_0.gguf",
    "sha256": "<sha256 of the converted file>",
    "bytes": 494735328
  }
}
```

- `source` and `target_model` list every file the converter reads, pinned by
  revision and SHA-256; nothing else is downloaded. `target_model` is the
  directory passed as `--target-model-dir` (DFlash drafts take the target's
  tokenizer and config) and may be omitted.
- `convert` names the llama.cpp pin the output was produced at and the
  `convert_hf_to_gguf.py --outtype`: `q8_0`, `bf16`, `f16` or `f32`. CI
  refuses to convert an entry whose pin differs from
  `skippy/llama_cpp/upstream.txt`; reconvert it and update `output`.
- `output` is the file CI uploads. It is uploaded only if the conversion
  hashes to `sha256` and is `bytes` long, and only if the file does not exist
  yet. A published file is never overwritten or deleted; one that hashes
  differently fails the run.

After the upload, the run's summary records the repo revision to pin in the
pairing index.

## Reproducing an output

The converter names the model after the source directory, so keep the
directory named after the source repo. With the pinned llama.cpp prepared and
the locked converter environment from `ci/dflash-drafts/convert`:

```sh
scripts/prepare-llama.sh pinned
uv sync --locked --project ci/dflash-drafts/convert
hf download poolside/Laguna-XS-2.1-DFlash config.json model.safetensors \
  --revision <revision> --local-dir work/source/Laguna-XS-2.1-DFlash
hf download poolside/Laguna-XS-2.1 config.json tokenizer.json tokenizer_config.json \
  --revision <revision> --local-dir work/target/Laguna-XS-2.1
uv run --locked --project ci/dflash-drafts/convert python .deps/llama.cpp/convert_hf_to_gguf.py \
  work/source/Laguna-XS-2.1-DFlash --target-model-dir work/target/Laguna-XS-2.1 \
  --outtype q8_0 --outfile Laguna-XS-2.1-DFlash-Q8_0.gguf
sha256sum Laguna-XS-2.1-DFlash-Q8_0.gguf
```

Check the downloaded files against the manifest before converting. The
conversion is deterministic: two runs from the same sources produce the same
bytes.

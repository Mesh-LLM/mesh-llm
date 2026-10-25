# Converter ownership

Mesh's native queue carries native runtime, model, graph and ABI changes. The Python converter additions formerly mixed into core patch `0004` and family patches `0001` (Inkling), `0003` (DiffusionGemma) and `0006` (Laya) are owned by `model-converters/` in the external research repository.

The extracted project preserves the complete patched conversion and GGUF Python packages, converter entrypoints, dependency constraints and resolved lock, templates/tokenizer fixtures, tests, documentation and licenses. Its provenance identifies upstream `8212c7802455255460ab8e18fc34754560031b34` and the original patched head `ff6779035febf91eed41cdd16757873e6a81d446`. Use a pinned, admitted research checkout, not the now-unmodified converter under a prepared Mesh `.deps` tree.

```sh
cd "${MESH_RESEARCH_ROOT:?}/model-converters"
just prepare
just offline-tests
uv run --locked --no-sync python convert_hf_to_gguf.py /absolute/checkpoint \
  --outtype f16 --outfile /absolute/output/model.gguf
```

Both extracted projects are published; fresh anonymous restore and native source admission passed for the [research descriptor](../../ci/python-research-source.json) and [SDK descriptor](../../ci/required-sdk-python/sdk-source.json). Hosted workflows and installed SDK/native/model qualification remain separate. The conversion command preserves the upstream interface; it does not download a checkpoint or qualify converted model accuracy merely by preparing dependencies.

The four owning patches were reconstructed without their Python diffs. Clean queue replay compares every non-Python path by Git mode and blob identity against the former full queue; all native C++ sources, installed headers, ABI constants, build files and native tests are unchanged. The upstream pin and ABI version remain unchanged. Upstream Python modules may still exist in the generated dependency checkout, but Mesh no longer modifies or advertises them as the source of these family converter additions.

Rust `skippy-quantize` delegates native conversion to `skippy-model` metadata, tokenizer, tensor-map and GGUF-writer owners. Native Inkling support overlaps parts of the extracted converter; complete family equivalence is not asserted. Keep external converter validation and native runtime/model qualification separate.

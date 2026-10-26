# Local configuration smoke

`manifest.tsv` preserves the original task-11 executed commands, evidence paths and statuses. Its Python command strings are historical data, not the current rerun interface. Do not replace those strings while retaining old PASS evidence. A new runtime run requires fresh output and new evidence; a coverage check validates declared evidence, not a new runtime pass.

Use the native owner with an existing normally built host and an operator-selected local model:

```sh
just with-lld cargo xtool automation manual-smoke run \
  --binary /absolute/mesh-llm \
  --fixture skippy/docs/manual-smoke/fixtures/model_fit/runtime-single-stage.toml \
  --model-path "$MESH_LLM_SMOKE_MODEL_PATH" \
  --native-runtime-root /absolute/native-runtimes --api-port 9437 --console-port 3231 --max-wait 600 \
  --output /absolute/fresh-smoke-output

just with-lld cargo xtool automation manual-smoke coverage
```

Use `--draft-path "$MESH_LLM_SMOKE_DRAFT_PATH"` for a speculative fixture and `--mmproj-path /absolute/projector.gguf` when required. Select available ports and preserve each fixture's purpose. The runtime owner rewrites the temporary config, probes status/models/chat, preserves partial evidence and owns server cleanup. Coverage accepts explicit `--root`, `--matrix`, `--manifest` and repeated `--required-evidence` overrides; defaults retain both historical task-11 receipts. Missing old evidence therefore fails honestly. New records must point to the new run's actual evidence, preserving PASS versus rejection and blocked distinctions.

The native owner does not acquire models, build products or certify native performance. Use the repository's existing Just build/release products appropriate to the intended runtime qualification.

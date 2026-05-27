# New Model Split Onboarding

Use this checklist for new-model requests before adding a family to the
customer-facing Skippy support matrix. The goal is to make split onboarding
repeatable without accidentally advertising a model before the current
llama.cpp pin, package writer, staged runtime, cache policy, and serving path
have real evidence.

Issue #630, Cohere Command A+, is the first model tracked with this flow.

## Evidence Levels

| Level | Meaning | Promotion effect |
| --- | --- | --- |
| Candidate identified | A source model and at least one plausible GGUF or package artifact are known. | No support claim. |
| Artifact inspected | GGUF metadata or package manifest gives architecture, layer count, activation width, quant, shard layout, and tokenizer sidecars. | May plan a package job. |
| Package validated | `skippy-model-package` writes, validates, and preflights a package with no unresolved manifest, artifact, sidecar, materialization, missing, duplicate, or checksum diagnostics. | May test staged serving. |
| Runtime smoke passed | A package-backed model starts and answers through the OpenAI-compatible surface. | May collect serving evidence. |
| Family certified | Split correctness, dtype matrix, state handoff/cache policy, and required multimodal sidebands pass. | May promote to reviewed support. |
| Reviewed support | `docs/skippy/FAMILY_STATUS.md` and reviewed topology records are updated from evidence. | User-visible support claim. |

Do not skip from candidate to reviewed support. A big or popular model is still
only a candidate until the evidence exists.

## Decision Tree

1. Start from the source model and record the exact Hugging Face repo, revision,
   license, architecture, model type, modality, and file layout.
2. If the source repo is safetensors-only, find or produce a GGUF before any
   Skippy split claim. The staged runtime cannot certify a safetensors source
   directly.
3. If the GGUF is sharded, inspect the first shard and keep the full shard set
   available for package writing or runtime loading.
4. If the model is MoE, recurrent, hybrid, multimodal, or uses custom
   llama.cpp support, treat it as high risk until the pinned llama.cpp tree can
   load and inspect it.
5. Run package validation before runtime certification for models too large for
   a local full-model parity pass.
6. Promote only after the evidence maps cleanly to both
   `docs/skippy/FAMILY_STATUS.md` and
   `crates/skippy-topology/capabilities/reviewed-family-capabilities.json`.

## Standard Checks

Validate the parity manifest and current llama.cpp family inventory:

```bash
python3 scripts/skippy-llama-parity.py validate
python3 scripts/skippy-llama-parity.py inventory --priority p0
```

List a candidate package job without spending HF Jobs credits:

```bash
cargo run -p model-package --bin queue-unsloth-layer-packages -- \
  --author DevQuasar \
  --search command-a-plus \
  --max-jobs 1 \
  --max-per-family 1 \
  --quant-preference Q4_K_M,Q3_K_M \
  --dry-run
```

For a selected GGUF repo and quant, use the normal package CLI in dry-run mode:

```bash
mesh-llm models package DevQuasar/CohereLabs.command-a-plus-05-2026-bf16-GGUF:Q4_K_M --dry-run
```

After a package exists, validate the package and then run the certification
surface that matches the evidence you need:

```bash
skippy-model-package validate-package /path/to/model.gguf /path/to/package
skippy-model-package preflight /path/to/package --stages 2 --verify-sha256
mesh-llm models certify hf://namespace/repo@revision --package-only --report-out cert.json
mesh-llm models certify hf://namespace/repo@revision --api-base http://127.0.0.1:9337 --json
```

Full family promotion still uses the family certification runbook:

```bash
just family-certify FAMILY /path/to/model.gguf ...
```

## Cohere Command A+ Starter Record

Current request:

- Issue: `#630`
- Source model: `CohereLabs/command-a-plus-05-2026-bf16`
- Current source shape: safetensors-only, `model_type=cohere2_vision`,
  `pipeline_tag=image-text-to-text`, 9 safetensors shards.
- Candidate GGUF: `DevQuasar/CohereLabs.command-a-plus-05-2026-bf16-GGUF`
- Candidate GGUF shape: third-party Q3_K_M and Q4_K_M sharded GGUF artifacts.
- Known risk: the candidate GGUF model card marks support as experimental and
  points at a custom llama.cpp branch for `cohere2-moe-support`.

Current certification status:

| Gate | Command A+ candidate status |
| --- | --- |
| Pinned mesh-llm llama.cpp inspect/load | Not run |
| Package write and `validate-package` | Not run |
| `/v1/models` runtime smoke | Not run |
| `/v1/chat/completions` runtime smoke | Not run |
| `/v1/responses` runtime smoke | Not run |
| Text split evidence | Not run |
| Vision/projector/media sideband evidence | Not run |
| MoE and cache-policy review | Not run |

Treat this as a candidate, not support. Do not add Command A+ to
`FAMILY_STATUS.md`, `reviewed-family-capabilities.json`, or the catalog until:

1. The pinned mesh-llm llama.cpp tree can inspect/load the selected GGUF.
2. Package writing and `validate-package` pass for the selected quant.
3. The package-backed runtime smoke passes through `/v1/models`,
   `/v1/chat/completions`, and `/v1/responses`.
4. Text split evidence passes for the selected topology.
5. Vision/projector/media sideband evidence passes before any multimodal
   support claim.
6. MoE/cache policy is explicitly reviewed instead of inheriting the older
   Command-R7B `cohere2` evidence.

The existing `Cohere2` support row covers a Command-R7B-style text GGUF. It is
not proof that Command A+ / `cohere2_vision` / `cohere2_moe` can be split by
the current runtime.

## Promotion Rule

Every promotion PR must state:

- exact source and package artifact refs;
- selected quant and shard layout;
- package validation result;
- runtime certification result;
- family certification result or explicit package-only rationale;
- cache/state policy decision;
- whether multimodal support was tested or intentionally left unsupported.

If any of those are unknown, keep the model in candidate/onboarding state.

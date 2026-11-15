# agentx-fleet-bench

Reproduce AgentX llama-vs-skippy benchmarks across parallel fleet machines, driven by xtask, with
everything pinned and reviewable.

## When to use this skill

- Run, reproduce, or harvest an AgentX comparison (llama vs skippy) or smoke probe across one or
  more fleet machines.
- Diagnose a fleet benchmark run (bounced jobs, missing results, stranded evidence).
- Publish fleet benchmark results to MLflow or a PR.

## Surface

All verbs live under `cargo xtool automation agentx`:

| Verb | Purpose |
|---|---|
| `pin` / `comparison` | Single-machine provenance pin and the 6-arm comparison (existing) |
| `fleet --input manifest.json` | Validate a manifest, pre-flight every kit URL, generate one jy spec per run, submit all in parallel, stamp a run-root |
| `fleet-status --run-root DIR` | Per-job status table for a run-root |
| `fleet-harvest --run-root DIR` | Fetch evidence for terminal jobs, extract matrices, print the results table |
| `mlflow-export` / `mlflow-publish` | Publication (existing) |

The fleet worker script is repo-owned at `skippy/evals/agentx/fleet-worker.sh` and embedded by
SHA; never edit a copy in a run directory. Models are pinned by repo/rev/sha256 in the manifest.
Kits are pinned by CDN URL + sha256 (chunks reassembled and verified on the worker).

## Manifest

```json
{
  "campaign": "agentx-prNNNN-YYYYMMDD",
  "product_source_commit": "<full40>",
  "mode": "smoke",
  "kit": {
    "cuda_chunks": {"prefix": "cuda5-chunk-...-", "count": 22, "sha256": "...", "url_base": "https://dist.../agentx-prNNNN"},
    "llama_chunks": {"prefix": "llama-chunk-...-", "count": 9, "sha256": "...", "url_base": "..."},
    "metal_kit": {"key": "metalkit9-....tgz", "sha256": "...", "url_base": "..."}
  },
  "runs": [
    {"model": "qwen3.5-4b", "backend": "cuda", "probe": {"conversations": 4, "turns": 4, "block_chars": 6300}}
  ]
}
```

Model references resolve to repo/rev/file/sha entries that live in
`skippy/evals/kv-validation/models.json` plus the manifest's per-run overrides.

## Hard rules (learned expensively)

1. **Placement is by hardware only.** Targets are `os: linux|macos` plus worker-side guards
   (`NO_AARCH64_HOST`, memory floor for Metal). Never target nodes by name.
2. **`jy result` returns nothing until a job TERMINATES.** A running job with no results is
   healthy. Never cancel for missing results.
3. **Artifacts ship only on clean worker exit.** For stranded evidence use the scavenge pattern:
   a pinned-hardware job that tars the dead run dir into `$JIANYANG_ARTIFACT_DIR` and exits 0.
4. **Guard-exits requeue; real errors diagnose.** `NO_PYTHON312`/`NO_AARCH64_HOST`/placement
   bounces: `jy requeue <job>` (submit dedupes identical specs). Anything else: read
   `jy logs <job> | tail -30` first.
5. **`fleet` pre-flights every kit URL** (range GET) before submitting. Freshly uploaded chunks
   can take minutes to propagate to every CDN edge; never submit right after an upload without
   the pre-flight.
6. **A/B experiments must target the intended budget knob.** `AGENTX_SKIPPY_PREFIX_CACHE_BUDGET`
   (exact-state catalog cap) is distinct from `AGENTX_SKIPPY_PREFIX_CACHE_RAM` (resident
   checkpoint RAM).
7. The prefix-reuse regression probe is `skippy/evals/agentx/prefix-reuse-probe.py` (exit 1 on
   any post-first-turn miss).
8. `jy` sessions expire; empty CLI output means `jy login` and retry.
9. Watch long runs with a paseo heartbeat that carries these rules; 10-minute cadence for smoke,
   30 for full campaigns.
10. Harvest order: evidence -> results table -> PR comment -> MLflow publish, in that order.

## Reading results

- Smoke reports per-turn `tok/s` and `cached%` plus a `steady_tok_s` summary per engine. Skippy
  `cached%` should track llama's on every model; a gap with matching cache means decode-bound, a
  gap with low cache means retention is still failing.
- Comparison campaigns interleave arms (`llama, skippy, skippy, llama, llama, skippy`) at
  c1/c2/c4 with 69 requests and 18,713 verified output tokens per cell. Judge skippy by
  `output_tok_s`, `cache_read_tokens.p50`, and `ttft.p50` per cell.

---
pretty_name: MeshLLM Performance History
license: apache-2.0
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/runs/**/*.jsonl
---

# MeshLLM Performance History

Append-only, machine-readable results from MeshLLM's trusted nightly competitive benchmark. Each immutable run shard contains one normalized row per backend, model, concurrency, hardware/config cohort, and source revision.

The dataset contains performance metrics and content-addressed provenance only. It does not contain prompts, completions, model weights, credentials, local filesystem paths, or raw benchmark logs. GitHub Actions retains the corresponding raw evidence separately.

The schema is versioned in `schema.json`. Dataset Viewer converts the JSONL shards to Parquet for SQL and charting. Regression reports compare only exact cohort keys and require at least three prior complete runs before classifying throughput or TTFT drift.

Normalize saved benchmark artifacts and write a regression report with the typed CLI:

```sh
cargo xtool ci-ops performance-history --artifact ./competitive-artifact \
  --output ./history/run.jsonl --report ./history/report.md --baseline ./data/runs
```

The report is informational by default. Add `--gate` to return failure for correctness failures or statistically significant throughput/TTFT regression. This command consumes saved evidence and does not run a benchmark. Competitive schema1 cohorts remain separate from schema3 agentic replay history.

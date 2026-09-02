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

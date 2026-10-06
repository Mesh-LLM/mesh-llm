# KV restart replay

The native caller measures fill, one first request after restart, and warm replays using two owned `mesh-llm serve` sessions on the same private state/cache/config. Supply an admitted host and local GGUF, with the host's default runtime loader authority. Adjacent package location is provenance, not proof of the loaded runtime. No model download or build occurs here.

Write a JSON input with absolute `binary` and `model` paths. The original default workload is:

```json
{
  "schema_version": 1,
  "binary": "/absolute/mesh-llm",
  "model": "/absolute/model.gguf",
  "turns": 4,
  "turn_target_tokens": 4750,
  "system_tokens": 500,
  "restore_repeats": 3,
  "max_output_tokens": 256,
  "request_timeout_secs": 900.0,
  "ready_timeout_secs": 900,
  "worker_timeout_secs": 7200,
  "timeout_secs": 30000,
  "serve_extra_args": []
}
```

Run through the repository-owned wrapper:

```bash
just automation-run automation waiting-prefix kv-restart-run \
  --input /absolute/input.json --output-directory /absolute/fresh-output
```

TCP9337 must be free. Occupancy refuses before child launch or output/state mutation and leaves the existing listener alone. The whole budget is explicit and shared; request/readiness caps do not reset it. Endpoint, tuning, identity/config/state and invite/credential overrides are refused; explicit cache-policy extras such as `--kv-cache-disk 32GiB` remain available.

`run.json`, `requests.jsonl` and `report.md` retain measured rows, cohort counts, honest nullable percentiles and lifecycle/provenance. Wall-clock observations use nullable Unix UTC seconds, while deadline/readiness/latency use monotonic clocks. A failed final interrupt/cancellation/deadline produces `terminal_complete: false`, nonzero status and retained rows. Completed individual rows do not imply a completed run. Restore is a cold-prefill reference on builds without durable KV; the caller does not claim a speedup or restoration merely because an inert fixture passes.

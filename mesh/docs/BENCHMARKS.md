# Benchmarks

These numbers are a quick reality check, not a universal promise.

## Example results

GLM-4.7-Flash-Q4_K_M (17GB), tested on an M4 Max and a Mac mini M4 over Wi-Fi:

| Configuration | tok/s |
|---|---|
| Solo (no mesh) | 68 |
| 2-node split (85/15) | 21 |
| 3-node split (62/31/8) | 12-13 |

Cross-network from Sydney to Queensland at roughly 20ms RTT measured 10-25 tok/s. In those runs, the overhead was dominated by per-token RPC latency.

## Notable implementation win

Stock llama.cpp RPC transfers about 16.88GB on connect.

This fork uses local GGUF loading on peers, which cuts that to:

- 0 bytes transferred
- about 9 seconds to connect

For deeper design and performance notes, see [design/DESIGN.md](design/DESIGN.md).

## Guardrail Evidence Scaffolding

Use the guardrail corpus runner for reliability evidence and keep the output in
`.sisyphus/evidence/`. The default corpus run uses `20 trials`, records the
expected server mode, and falls back to a deterministic fake backend when the
runtime is not available.

For benchmarks that should exercise the real guardrail middleware, enable the
server-side mode first and confirm `/api/status` reports it:

```bash
mesh-llm serve --model MiniMax-M2.5-Q4_K_M --mesh-guardrails metrics
# or switch a running node without restart:
mesh-llm runtime guardrails --mode metrics --port 3131
curl -s localhost:3131/api/status | jq '.runtime.openai_guardrails'
```

```bash
python3 scripts/run-openai-guardrail-corpus.py \
  --base-url http://127.0.0.1:9337/v1 \
  --model MiniMax-M2.5-Q4_K_M \
  --guardrail-mode metrics \
  --trials 20 \
  --out .sisyphus/evidence/openai-guardrail-corpus.json
```

Use `scripts/run-llama-benchy-openai.sh` for throughput evidence once the
server mode is active. Keep the corpus JSON and the benchy markdown together in
the same evidence folder so the reliability and throughput stories stay paired.

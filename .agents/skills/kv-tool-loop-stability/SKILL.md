---
name: kv-tool-loop-stability
description: Use this skill when certifying mesh-llm KV/cache stability under repeated OpenAI tool-call loops, same-prefix cache reuse, suffix-prefill limits, or native Skippy slot/decode/eviction failures.
metadata:
  short-description: Certify KV/tool-loop stability
---

# KV Tool-Loop Stability

Use this skill when changing Skippy KV slot cleanup, prefix-cache lookup,
OpenAI tool-loop behavior, agent harnesses, or any runtime path related to
`llama_decode failed`, `failed to find a memory slot`, low same-prefix cache
reuse, or proactive eviction failures.

## Workflow

1. Attach to an existing OpenAI-compatible `/v1` endpoint. This harness does
   not start nodes, load models, join meshes, or change routing policy.
2. Prefer a direct model when reproducing Skippy KV/cache issues. Use `auto`
   only when intentionally validating routed behavior.
3. Run `--print-plan` first and confirm the models, attempts,
   `pressure_turns`, timeout, cache thresholds, output directory, and native
   logs.
4. Pass the active Skippy native log when available. The harness checkpoints
   native logs at run start and scans only appended bytes.
5. Preserve the evidence directory: `manifest.json`, `results.jsonl`,
   `summary.json`, `summary.md`, and `transcripts/*.jsonl`.

## Commands

Preview the run without touching the endpoint:

```bash
scripts/qa-kv-tool-loop-stability.py \
  --base-url http://127.0.0.1:9337/v1 \
  --models Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m \
  --attempts 5 \
  --pressure-turns 8 \
  --timeout 180 \
  --min-cached-tokens 2048 \
  --suffix-prefill-limit 256 \
  --native-log ~/.mesh-llm/runtime/<pid>/logs/skippy-native.log \
  --output-dir target/kv-tool-loop-stability/local \
  --print-plan
```

Run the certification:

```bash
scripts/qa-kv-tool-loop-stability.py \
  --base-url http://127.0.0.1:9337/v1 \
  --models Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m \
  --attempts 5 \
  --pressure-turns 8 \
  --timeout 180 \
  --min-cached-tokens 2048 \
  --suffix-prefill-limit 256 \
  --native-log ~/.mesh-llm/runtime/<pid>/logs/skippy-native.log \
  --output-dir target/kv-tool-loop-stability/local
```

## Reporting Rules

- Report the model list, attempts, pressure turns, timeout, cache thresholds,
  success rate, native log paths, and output directory.
- Include the summary verdict and failing phase details from `summary.md` or
  `summary.json`.
- Do not paste full prompts, auth headers, huge stable prefixes, or private
  endpoint data.
- If no native log is available, say that native-log scanning was not run.

## Validation

When changing this harness, run:

```bash
python3 -m unittest scripts.tests.test_qa_kv_tool_loop_stability
python3 -m py_compile scripts/qa-kv-tool-loop-stability.py scripts/tests/test_qa_kv_tool_loop_stability.py
```

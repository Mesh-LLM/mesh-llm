# Skippy OpenAI Prompt Cache

Skippy OpenAI serving has an automatic prompt-prefix cache for normal text
generation. The cache is intended to provide the same operator-facing outcome
as llama-server prompt reuse: repeated prompts, and prompts with a long shared
prefix, avoid recomputing the whole prefix on later requests.

## llama-server Reference Behavior

llama-server keeps prompt state in serving slots. With `cache_prompt` enabled,
each request compares its prompt tokens with the selected slot and reuses the
longest common prefix already resident in that slot. With `--cache-ram`,
llama-server can also save idle slot state into a RAM prompt cache and reload
the closest reusable prompt state later. No client cache key is required for
the common case.

Important llama-server properties:

- `cache_prompt` controls per-request prompt reuse.
- `--cache-ram N` enables RAM prompt-cache storage.
- idle slots may be saved to the prompt cache and cleared.
- later requests choose reusable state by prompt similarity or cache lookup.
- reported timings include cached versus processed prompt tokens.

## Skippy Behavior

Skippy uses the stage KV integration rather than llama-server slots. When a
stage has `kv_cache.mode = "lookup-record"`, Skippy records prompt-prefix
state after prefill and probes that state before later OpenAI chat/completion
requests. A client `prompt_cache_key` is optional; ordinary requests without a
key share the default namespace and can hit automatically.

Skippy records exact prompt-prefix identities and a small shared-prefix grid.
The exact path covers identical prompts. The shared-prefix path covers prompts
whose common prefix falls on the configured stride, such as a stable system
prompt or tool schema followed by different user tails. This is not arbitrary
llama-server LCP slot selection; it is deterministic prefix identity probing.

Skippy reports cache state in OpenAI usage and telemetry:

- `usage.prompt_tokens_details.cached_tokens` is the OpenAI-compatible cached
  prompt token count.
- `stage.openai_generation_summary` emits `skippy.kv.status`, one of
  `disabled`, `miss`, or `hit`.
- The same summary emits `skippy.kv.cached_prompt_tokens`,
  `skippy.kv.matched_prefix_tokens`, `skippy.kv.suffix_prefill_tokens`, and
  `skippy.kv.hit_kind`. The hit kind is `none` for disabled or missed cache
  lookups.

## mesh-llm Defaults

mesh-llm wires Skippy prefix cache through family policy. For supported model
families, generated `StageConfig` values receive a bounded cache config with
`mode = "lookup-record"` and a production payload such as `resident-kv` or
`kv-recurrent`. This applies to normal mesh-llm embedded Skippy serving without
requiring users to send `prompt_cache_key`.

The default is conditional, not universal:

- unsupported or unknown families may leave `kv_cache` unset.
- raw `skippy-server serve-openai` leaves cache off unless the stage config or
  `SKIPPY_KV_CACHE`/`SKIPPY_PREFIX_CACHE` enables it.
- operators can disable cache with `model_fit.prompt_cache = false` or
  `model_fit.prefix_cache.enabled = false`.
- recurrent-state families use `kv-recurrent` rather than `resident-kv` when
  the family policy requires it.

## Benchmarking

Use `evals/skippy-openai-cache-matrix.py` to compare cold and warm behavior
across native llama-server and Skippy OpenAI endpoints. The script records four
rows:

- cold native: llama-server with request cache disabled.
- cold Skippy: Skippy endpoint started with prefix cache disabled.
- warm native: llama-server with request cache enabled.
- warm Skippy: Skippy endpoint started with prefix cache enabled.

The report is intended to prove cache behavior before comparing timing. Each
row includes a verdict, observed cache statuses, prompt tokens, cacheable
prefix tokens, cached tokens, suffix/uncached prefix tokens, and cache
efficiency. For Skippy OpenAI chat generation, the cacheable prefix excludes
the final current token that drives decode, so `cacheable = prompt_tokens - 1`.

Example:

```bash
python3 evals/skippy-openai-cache-matrix.py \
  --llama-cold-base-url http://127.0.0.1:8081 \
  --llama-warm-base-url http://127.0.0.1:8082 \
  --skippy-cold-base-url http://127.0.0.1:9337/v1 \
  --skippy-warm-base-url http://127.0.0.1:9447/v1 \
  --model Qwen/Qwen3-0.6B:Q4_K_M \
  --output-dir target/skippy-openai-cache-matrix/local
```

Start the cold native endpoint with `--cache-ram 0`. Start the warm native
endpoint with `--cache-ram N` or the llama-server default prompt-cache setting.

The default benchmark pattern is `exact`, so the warmup and measured request
use the same prompt. To exercise Skippy's shared-prefix grid against
llama-server's LCP reuse, add `--pattern shared-prefix`. By default, the script
exits non-zero when either warm row reports zero cached tokens; use
`--allow-missing-warm-cache` only for exploratory timing runs where the endpoint
does not expose cached-token counts.

The cold Skippy endpoint should use a config with no `kv_cache` or with
`kv_cache.mode = "disabled"`. The warm Skippy endpoint should use mesh-llm
family defaults or an explicit `lookup-record` prefix cache.

---
title: KV Caching
description: Reuse prompt work in memory or on disk and tune KV memory use
---

# KV Caching

KV caching lets Mesh reuse the model state created while reading a prompt.
When a later request starts with the same tokens, Mesh can restore that state
and skip some or all of the repeated prefill work. This lowers time to first
token for repeated system prompts, long documents, and multi-turn chats.

Applications do not need a cache-specific API. Keep using the
[OpenAI-compatible API](/docs/pages/openai-compatible-api/) and send the full
conversation or prompt on every request. Mesh identifies matching prefixes and
reports reused prompt tokens in `usage.prompt_tokens_details.cached_tokens`.
Callers that already use OpenAI's `prompt_cache_key` may keep sending it; Mesh
trims the value and uses it as a cache and routing namespace. Keep it stable for
the requests that should share cached prefixes. It does not force a cache hit,
and requests without it share the default namespace. Do not put secrets in the
key because Mesh records it verbatim in request telemetry.

For the Responses API, an explicit `prompt_cache_key` wins. When it is absent,
Mesh uses `previous_response_id`, then the conversation ID, as the cache key.

## Default experience

With no cache settings:

- Mesh enables the in-memory prefix cache with family-aware limits. Prefixes of
  at least 256 tokens are eligible, and Mesh selects the stored state format
  from the model architecture.
- With no explicit K/V dtype, the resolver selects Q8_0 for both caches when
  the model is smaller than 50 GiB and Q4_0 for both caches at 50 GiB or above.
  It checks GGUF architecture and head dimensions before applying that default
  and falls back to F16 when metadata proves the quantized layout is invalid.
  An explicit incompatible dtype fails instead of changing silently.
- KV offload and unified-cache behavior remain automatic.
- The durable disk cache is off, so a process restart starts with an empty
  prompt cache and Mesh writes no prompt state to disk.

These defaults apply across the supported CPU, Metal, CUDA, and ROCm runtimes.

## The controls at a glance

| Goal | Setting | Default | Where to set it |
|---|---|---|---|
| Pin key/value cache formats | `model_fit.cache_type_k`, `model_fit.cache_type_v` | `auto` | config file |
| Control KV device offload | `model_fit.kv_offload` | `auto` | config file |
| Control unified KV allocation | `model_fit.kv_unified` | `auto` | config file |
| Control the attention kernel required by quantized V | `model_fit.flash_attention` | derived from V dtype | config file |
| Cap retained idle native sessions | `model_fit.cache_idle_slots` | lane count | config file |
| Disable all prompt-prefix reuse | `model_fit.prompt_cache` | `auto` | config file |
| Tune or disable in-memory prefix reuse | `model_fit.prefix_cache.*` | family defaults | config file |
| Persist prompt state across restarts | `runtime.kv_cache.disk.*` | `off` | config, environment, or `serve` flags |
| Inspect or remove disk entries | `mesh-llm kv-cache ...` | n/a | CLI |

Model-level cache controls do not currently have CLI equivalents. Use
`~/.mesh-llm/config.toml`, or pass a different file with `mesh-llm serve
--config PATH`. The disk tier has CLI overrides for one-off runs.

## Choose the KV representation

Configure the representation directly when you need deterministic behavior:

```toml
[defaults.model_fit]
cache_type_k = "q8_0"
cache_type_v = "q8_0"
kv_offload = "auto"
kv_unified = "auto"
flash_attention = "enabled"
```

The relevant controls are:

| Setting | Loadable embedded-runtime values | Runtime effect |
|---|---|---|
| `cache_type_k` | `auto`, `f16`, `q8_0`, `q4_0` | Storage and compute dtype for attention keys |
| `cache_type_v` | `auto`, `f16`, `q8_0`, `q4_0` | Storage and compute dtype for attention values |
| `kv_offload` | `auto`, `true`, `false` | Whether KV tensors may reside on the selected accelerator rather than host memory |
| `kv_unified` | `auto`, `true`, `false` | Whether runtime slots use the backend unified KV allocation |
| `flash_attention` | `auto`, `enabled`, `disabled` | Selects the fused attention path; a quantized V cache requires the enabled path |

Q8_0 and Q4_0 encode values in 32-element blocks. A model whose KV head
dimension cannot satisfy that block layout cannot use the corresponding
quantized cache. Automatic selection can detect that from GGUF metadata and
fall back to F16. Explicit dtype selection bypasses that fallback so invalid
combinations fail during model load. A backend Flash Attention capability
failure is only known when the runtime loads, so metadata validation alone
cannot prove that every quantized combination will start.

K and V may use different dtypes. For these technical fields, per-model values
override global values, which override family defaults and finally the built-in
size rule.

The config validator currently recognizes additional GGML dtype labels that
the pinned embedded runtime does not load. The table above lists the values
accepted by `skippy_runtime::parse_cache_type`; use those values for a serving
configuration. `auto` is consumed by the resolver and does not reach that
parser.

You can override one model without changing the others:

```toml
[[models]]
model = "org/model-GGUF"

[models.model_fit]
cache_type_k = "f16"
cache_type_v = "f16"
kv_offload = false
```

## Tune in-memory prefix reuse

The automatic prefix cache is usually the right choice. To disable it for a
model, set:

```toml
[defaults.model_fit]
prompt_cache = false
```

To keep it enabled but set explicit bounds:

```toml
[defaults.model_fit.prefix_cache]
enabled = true
payload_mode = "auto"
min_tokens = 512
max_entries = 256
max_bytes = 8589934592
shared_stride_tokens = 128
shared_record_limit = 4
```

`payload_mode = "auto"` stores resident KV for known dense models and KV plus
recurrent state for known recurrent or hybrid models. Unknown architectures do
not cache automatically. The size fields are byte counts; `max_bytes = 0`
means no explicit byte cap. Setting `prompt_cache = false` disables prefix
caching and conflicts with an explicitly enabled `prefix_cache` block.

Per-model `[[models]]` values override `[defaults]`; explicit cache dtypes
override policy-derived dtypes; unresolved values fall back through family
policy and built-ins. Model cache changes apply when the model reloads.

The OpenAI request field `prompt_cache_retention` accepts `in_memory` and
`24h`. Mesh records it as telemetry, but neither value currently enforces a
cache lifetime. Use the runtime limits above and the disk maintenance commands
below to control retention.

`cache_idle_slots` limits how many reset native sessions remain available for
reuse. Unset means the runtime lane count is the bound, `0` drops every reset
lane, and a positive value adds a lower cap. `cache_ram_mib` is reserved; any
positive value currently fails model loading, so there is no configurable
host-RAM L2 tier.

Cache matches require the exact token prefix and exact runtime identity. Mesh
does not use fuzzy or semantic prompt matching.

## Enable durable disk caching

Disk caching preserves reusable prompt state across process restarts. It is
node-local and disabled until you opt in.

For an automatically sized cache:

```toml
[runtime.kv_cache.disk]
mode = "auto"
directory = "/var/lib/mesh-llm/kv-cache"
minimum_free_mib = 16384
```

For a fixed 32 GiB cap:

```toml
[runtime.kv_cache.disk]
mode = "fixed"
directory = "/var/lib/mesh-llm/kv-cache"
budget_mib = 32768
minimum_free_mib = 16384
```

The directory must be absolute. If omitted, it is
`$MESH_LLM_HOME/kv-cache`, or `~/.mesh-llm/kv-cache`. Auto mode uses at most
20% of the filesystem capacity basis, never consumes the configured free-space
reserve, and is capped at 64 GiB.

The same settings can be supplied for one run:

```bash
mesh-llm serve --kv-cache-disk auto

mesh-llm serve \
  --kv-cache-disk 32GiB \
  --kv-cache-disk-dir /var/lib/mesh-llm/kv-cache \
  --kv-cache-min-free 16GiB
```

Environment equivalents are `MESH_LLM_KV_CACHE_DISK`,
`MESH_LLM_KV_CACHE_DISK_DIR`, and `MESH_LLM_KV_CACHE_MIN_FREE`. CLI values win
over environment values, which win over the config file, independently for
each field.

Invalid disk settings stop startup with a configuration error. Once a valid
configuration is running, storage trouble fails open: Mesh logs the problem
and serves the request with cold prefill.

## Inspect and maintain the disk cache

```bash
mesh-llm kv-cache status
mesh-llm kv-cache status --json

mesh-llm kv-cache prune --target 16GiB --yes
mesh-llm kv-cache clear --yes
```

Human-readable `status` shows the effective state, configured mode, root, and
used/budget bytes. `status --json` also includes the free-space reserve, entry
counts, degradation reason, activity, reconciliation, and per-model inventory.
`prune` and `clear` only remove inactive entries. Both can be limited to an
exact numeric model identity with `--model-identity ID`.

Budget and minimum-free changes apply live. Changing the mode or directory
requires a node restart.

## Advanced environment controls

These environment variables are intended for incident response and controlled
experiments:

- Setting `SKIPPY_KV_CACHE=off` or `SKIPPY_PREFIX_CACHE=off` disables worker
  prefix-cache storage even when the stage plan enables it. Other
  `SKIPPY_KV_CACHE_*` tuning variables only construct cache settings when the
  stage plan did not supply them.
- `MESH_LLM_DISABLE_PREFIX_AFFINITY` disables routing toward a peer with known
  prefix state. `MESH_LLM_DISABLE_STICKY_ROUTING` disables sticky routing, and
  `MESH_LLM_PREFIX_ONLY=1` uses the request's prefix hash as the deterministic
  fallback when neither cache evidence nor a session route applies. These
  change peer selection; they do not disable or resize cache storage.

## CacheGen status

CacheGen is an experimental compressed representation for saved KV state. The
implementation targets CPU, Metal, CUDA, and ROCm, but it is still undergoing
backend qualification, including completion and verification of the ROCm path,
and is not selected by the normal serving config or CLI today. Use the stable
resident/exact-state cache paths described above for production operation; do
not add an invented `cachegen` setting to the config.

For every field and allowed value, see the [Config Reference](/docs/pages/config-reference/).
For disk storage details, failure modes, and recovery procedures, see the
[KV-cache disk operator guide](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/skippy/KV_CACHE_DISK.md).

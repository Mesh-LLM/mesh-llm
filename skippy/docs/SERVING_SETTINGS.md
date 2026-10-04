# Standalone serving settings

`skippy serve --help` lists all supported operator controls, grouped by purpose.
The same settings can be supplied in a TOML file with `--settings serve.toml`.
A serving settings file is distinct from `--config stage.json`, which contains
an admitted stage plan. Stage identities, tensor closures, source digests and
execution contracts remain owned by the planner.

## Precedence and values

Settings resolve in this order:

1. Explicit command-line options.
2. `SKIPPY_SERVE_<OPTION_NAME>` environment variables (uppercase, `_` for `-`).
3. Values in the serving settings file.
4. Existing stage/model policies and built-in defaults.

For example, `SKIPPY_SERVE_THREADS=8` overrides `[execution].threads`, while
`--threads 4` overrides both. Environment values use the CLI representation;
structured environment values must be JSON. Existing native-runtime and disk
cache environment variables remain fallbacks below these surfaces.

Booleans accept either a bare flag or an explicit value with `=`:
`--mlock`, `--mlock=true`, `--mlock=false`. This also applies to existing boolean
flags, so file settings can be disabled explicitly. Unknown keys, incorrect
sections and invalid values are rejected. Conflicting model sources are errors;
override the same source option when changing the model from a settings file.
With a prepared `--config`, model preparation options (`ctx_size`,
`n_gpu_layers`, `quant`, `checkpoint_imatrix`, `mmproj`, `hash_cache`) retain
their existing conflicts; those values belong in the prepared stage plan.

Use snake_case option names in TOML, under the section shown below. CLI-only
`--settings`, `--print-effective-config` and `--help` cannot appear in the file.
`version = 1` is optional and identifies the current file format.

Paths in the file are relative to its directory. Model repository references
remain repository references; explicit local paths and `.gguf` filenames are
resolved relative to the settings file. The CLI keeps paths relative to the
working directory. `runtime_bundle` accepts an array of directories.

JSON-valued CLI options accept inline JSON or `@file.json`. In TOML, use native
arrays/tables or `@file.json`. Text-valued content options (`system_prompt`,
`chat_template`, `grammar`) accept literal text or `@file.txt`. Content-file
paths in TOML are relative to the settings file. Ordinary strings such as model
IDs and template markers are literal strings, not content files.

## Example

```toml
version = 1

[model]
model = "Qwen3-0.6B-Q4_K_M"
mmap = true
mlock = false

[execution]
n_gpu_layers = -1
threads = 4
threads_batch = 8
batch_size = 512
microbatch_size = 256

[kv]
ctx_size = 8192
kv_cache_type_k = "f16"
kv_cache_type_v = "f16"
flash_attention = "auto"

[cache]
prefix_cache = "auto"
prefix_cache_budget = "auto"
prefix_cache_ram = "off"
kv_cache_disk = "off"

[scheduling]
generation_concurrency = 2
continuous_batching = true
prefill_chunk_size = 64

[sampling]
temperature = 0.6
top_p = 0.95
stop = ["END"]

[chat]
system_prompt = "Answer clearly and concisely."
reasoning = "auto"
reasoning_budget = "auto"

[api]
bind_addr = "127.0.0.1:9337"
default_max_tokens = 2048
guardrails = "disabled"
compact = true
compact_trigger_percent = 90
compact_target_percent = 80

[diagnostics]
telemetry_level = "summary"
startup_timeout_secs = 60
```

```sh
skippy serve --settings serve.toml --temperature 0.2 --mlock=false
skippy serve --settings serve.toml --print-effective-config
```

`--print-effective-config` prints the resolved stage, frontend, operator values,
source attribution and constraints, then exits without opening a model or cache
store. Local model acquisition/verification and memory planning can still run;
prepared stage configurations do not require loading a native runtime. The
report is explicitly **before model load**: loaded-model cache payload selection,
backend capability checks and package request profiles can resolve later.
The `options` map includes CLI defaults; `null` marks an automatic or omitted
operator value whose resolved behavior is represented by the stage/frontend or
selected after loading. `overrides` contains only explicitly supplied settings.

## Mode and resource constraints

Local and split-stage serving consume the same operator controls. Sampling,
chat, guardrails, compaction and generation scheduling require a public API;
workers reject those controls. Native loading, KV/cache policy, CPU threads and
continuous batching also apply to workers. Network simulation and activation
codec settings require `--stage-transport binary`. A complete speculative JSON
plan can be loaded with `--speculative-config`; individual speculative flags
override its fields.

Live K/V precision is independent of weight quantization. Supported explicit
values are `f16`, `q8_0`, and `q4_0`. Quantized V requires compatible Flash
Attention and model geometry; incompatible explicit choices fail at runtime.
Precision and KV offload are supplied to local memory planning before context
selection. Unified KV is mandatory, so the effective-config report states this
constraint rather than advertising a switch to disable it.

`prefix_cache = "auto"` follows the model's prefix-cache policy; `on` enables
lookup/record, `off` disables reuse, and `record` only records state. Automatic
payload selection preserves recurrent/indexer state when required. Explicit
payload and experimental codec compatibility resolve against the loaded model.

Cache sizes use whole IEC units (`KiB`, `MiB`, `GiB`, `TiB`). The prefix retention
budget accepts `auto` or `unbounded`; the exact-state ceiling accepts `auto` or
`unbounded` as well. The RAM tier accepts `off` or a fixed size and currently
requires an active disk tier. Resident prefix retention shares the native KV
pool with active generations; this budget does not allocate another KV pool.
Exact-state snapshots are indivisible: one retained snapshot may exceed its
ceiling. `prefix_cache_exact_budget` overrides the legacy
`SKIPPY_KV_CACHE_EXACT_MAX_BYTES` setting.

Legacy `SKIPPY_KV_CACHE=off` and `SKIPPY_PREFIX_CACHE=off` remain emergency kill
switches. An explicit request to enable prefix caching while either is active
fails with an explanation; unset the kill switch to enable the cache.

## Option reference

Each setting below has a named CLI flag and a TOML key. New controls are listed
with their runtime purpose; existing controls follow in the same sections.

### [model]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--mmap` | `mmap` | Map model weights into memory |
| `--mlock` | `mlock` | Lock model weight pages in RAM |
| `--repack` | `repack` | Repack model tensors when supported |
| `--direct-io` | `direct_io` | Load weights using direct I/O; takes precedence over mmap/mlock |
| `--check-tensors` | `check_tensors` | Validate tensor data while loading |
| `--no-host-buffer` | `no_host_buffer` | Bypass the model host buffer |
| `--model` | `model` | See `skippy serve --help` for values and defaults. |
| `--model-path` | `model_path` | See `skippy serve --help` for values and defaults. |
| `--quant` | `quant` | See `skippy serve --help` for values and defaults. |
| `--checkpoint-imatrix` | `checkpoint_imatrix` | See `skippy serve --help` for values and defaults. |
| `--hash-cache` | `hash_cache` | See `skippy serve --help` for values and defaults. |

### [execution]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--device` | `device` | Backend device name from skippy doctor |
| `--split-mode` | `split_mode` | Multi-GPU split mode: auto, none, layer, row, tensor |
| `--glm-dsa-policy` | `glm_dsa_policy` | GLM DSA execution policy: auto or v1 |
| `--main-gpu` | `main_gpu` | GPU index for an unsplit model |
| `--batch-size` | `batch_size` | Native batch token limit |
| `--microbatch-size` | `microbatch_size` | Native microbatch token limit |
| `--threads` | `threads` | CPU threads for generation |
| `--threads-batch` | `threads_batch` | CPU threads for prompt processing |
| `--op-offload` | `op_offload` | Offload supported host operations to the backend |
| `--n-gpu-layers` | `n_gpu_layers` | See `skippy serve --help` for values and defaults. |

### [kv]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--kv-cache-type-k` | `kv_cache_type_k` | Live key-cache precision: f16, q8_0, q4_0 |
| `--kv-cache-type-v` | `kv_cache_type_v` | Live value-cache precision: f16, q8_0, q4_0 |
| `--flash-attention` | `flash_attention` | Flash Attention: auto, enabled, disabled |
| `--kv-cache-offload` | `kv_cache_offload` | Offload live KV memory to the backend |
| `--swa-full` | `swa_full` | Use a full cache window for sliding-window attention |
| `--cache-idle-slots` | `cache_idle_slots` | Maximum retained idle sessions; zero retains none |
| `--ctx-size` | `ctx_size` | See `skippy serve --help` for values and defaults. |

### [cache]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--prefix-cache` | `prefix_cache` | Prompt reuse: auto, on, off, record |
| `--prefix-cache-payload` | `prefix_cache_payload` | State payload: auto, resident-kv, kv-recurrent, full-state |
| `--prefix-cache-codec` | `prefix_cache_codec` | Durable representation: native or experimental cachegen |
| `--prefix-cache-budget` | `prefix_cache_budget` | Retention target: auto, unbounded, or IEC size; exact snapshots may exceed it |
| `--prefix-cache-ram` | `prefix_cache_ram` | Host-RAM tier: off or IEC size; requires disk caching |
| `--prefix-cache-max-entries` | `prefix_cache_max_entries` | Maximum retained prefix entries |
| `--prefix-cache-min-tokens` | `prefix_cache_min_tokens` | Minimum prefix length to record or reuse |
| `--prefix-cache-stride-tokens` | `prefix_cache_stride_tokens` | Shared-prefix checkpoint spacing in tokens |
| `--prefix-cache-record-limit` | `prefix_cache_record_limit` | Maximum shared-prefix checkpoints per recording |
| `--prefix-cache-exact-budget` | `prefix_cache_exact_budget` | Exact-state retention ceiling: auto, unbounded, or IEC size; one snapshot may exceed it |
| `--kv-cache-disk` | `kv_cache_disk` | See `skippy serve --help` for values and defaults. |
| `--kv-cache-disk-dir` | `kv_cache_disk_dir` | See `skippy serve --help` for values and defaults. |
| `--kv-cache-min-free` | `kv_cache_min_free` | See `skippy serve --help` for values and defaults. |

### [scheduling]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--continuous-batching` | `continuous_batching` | Allow the iteration scheduler to serve concurrent lanes |
| `--pipeline-decode-groups` | `pipeline_decode_groups` | Number of decode dispatch groups |
| `--generation-signal-window` | `generation_signal_window` | Generation signal observation window in tokens |
| `--generation-concurrency` | `generation_concurrency` | See `skippy serve --help` for values and defaults. |
| `--adaptive-generation-concurrency` | `adaptive_generation_concurrency` | See `skippy serve --help` for values and defaults. |
| `--adaptive-generation-min-concurrency` | `adaptive_generation_min_concurrency` | See `skippy serve --help` for values and defaults. |
| `--generation-queue-capacity` | `generation_queue_capacity` | See `skippy serve --help` for values and defaults. |
| `--generation-admission-timeout-secs` | `generation_admission_timeout_secs` | See `skippy serve --help` for values and defaults. |
| `--prefill-chunk-size` | `prefill_chunk_size` | See `skippy serve --help` for values and defaults. |
| `--prefill-chunk-policy` | `prefill_chunk_policy` | See `skippy serve --help` for values and defaults. |
| `--prefill-chunk-schedule` | `prefill_chunk_schedule` | See `skippy serve --help` for values and defaults. |
| `--prefill-adaptive-start` | `prefill_adaptive_start` | See `skippy serve --help` for values and defaults. |
| `--prefill-adaptive-step` | `prefill_adaptive_step` | See `skippy serve --help` for values and defaults. |
| `--prefill-adaptive-max` | `prefill_adaptive_max` | See `skippy serve --help` for values and defaults. |
| `--prefill-adaptive-target-ms` | `prefill_adaptive_target_ms` | See `skippy serve --help` for values and defaults. |

### [sampling]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--temperature` | `temperature` | Default sampling temperature |
| `--top-p` | `top_p` | Default nucleus sampling probability |
| `--min-p` | `min_p` | Default minimum token probability |
| `--typical-p` | `typical_p` | Default typical sampling probability |
| `--top-nsigma` | `top_nsigma` | Default top-n-sigma sampling bound |
| `--presence-penalty` | `presence_penalty` | Default presence penalty |
| `--frequency-penalty` | `frequency_penalty` | Default frequency penalty |
| `--repeat-penalty` | `repeat_penalty` | Default repetition penalty |
| `--dynatemp-range` | `dynatemp_range` | Dynamic temperature range |
| `--dynatemp-exponent` | `dynatemp_exponent` | Dynamic temperature exponent |
| `--mirostat-entropy` | `mirostat_entropy` | Mirostat target entropy |
| `--mirostat-learning-rate` | `mirostat_learning_rate` | Mirostat learning rate |
| `--top-k` | `top_k` | Default top-k token count |
| `--repeat-last-n` | `repeat_last_n` | Repetition history length; -1 uses the runtime default |
| `--mirostat-mode` | `mirostat_mode` | Mirostat mode: 0, 1, or 2 |
| `--seed` | `seed` | Default sampling seed |
| `--request-max-tokens` | `request_max_tokens` | Operator default output limit, overridden by an explicit request |
| `--ignore-eos` | `ignore_eos` | Ignore end-of-sequence tokens by default |
| `--stop` | `stop` | JSON array of default stop strings; @file loads JSON |
| `--logit-bias` | `logit_bias` | JSON map of token IDs to biases; @file loads JSON |
| `--samplers` | `samplers` | JSON array defining the sampler order; @file loads JSON |
| `--sampler-sequence` | `sampler_sequence` | Compact sampler sequence |
| `--dry-multiplier` | `dry_multiplier` | DRY repetition penalty multiplier |
| `--dry-base` | `dry_base` | DRY exponential base |
| `--xtc-probability` | `xtc_probability` | XTC sampling probability |
| `--xtc-threshold` | `xtc_threshold` | XTC probability threshold |
| `--dry-allowed-length` | `dry_allowed_length` | DRY allowed repetition length |
| `--dry-penalty-last-n` | `dry_penalty_last_n` | DRY history length; -1 uses the runtime default |
| `--dry-sequence-breakers` | `dry_sequence_breakers` | JSON array of DRY sequence breakers; @file loads JSON |

### [chat]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--reasoning-format` | `reasoning_format` | Reasoning format: auto, none, deepseek, deepseek-legacy, hidden |
| `--reasoning` | `reasoning` | Reasoning default: auto, enabled, disabled |
| `--reasoning-budget` | `reasoning_budget` | JSON: "auto", "unrestricted", {"tokens":N}, or {"effort":"high"} |
| `--chat-template-kwargs` | `chat_template_kwargs` | JSON template arguments; @file loads JSON |
| `--prefill-assistant` | `prefill_assistant` | JSON assistant string or message object; @file loads JSON |
| `--json-schema` | `json_schema` | Default structured-output JSON schema; @file loads JSON |
| `--system-prompt` | `system_prompt` | Default system prompt; @file loads UTF-8 text |
| `--chat-template` | `chat_template` | Chat template; @file loads UTF-8 text |
| `--grammar` | `grammar` | Default grammar; @file loads UTF-8 text |
| `--jinja` | `jinja` | Use the Jinja template engine |
| `--skip-chat-parsing` | `skip_chat_parsing` | Skip structured chat-output parsing |

### [speculative]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--speculative-strategy` | `speculative_strategy` | Speculation: auto, disabled, draft-model, native-mtp, ngram, mtp-ngram |
| `--draft-device` | `draft_device` | Backend device for the draft model |
| `--draft-cache-type-k` | `draft_cache_type_k` | Draft key-cache precision: f16, q8_0, q4_0 |
| `--draft-cache-type-v` | `draft_cache_type_v` | Draft value-cache precision: f16, q8_0, q4_0 |
| `--ngram-kind` | `ngram_kind` | N-gram proposer: cache or suffix |
| `--native-mtp` | `native_mtp` | Enable native multi-token prediction |
| `--mtp-suppress-cooldown-drafts` | `mtp_suppress_cooldown_drafts` | Suppress drafts during MTP rejection cooldown |
| `--ngram-fallback-draft` | `ngram_fallback_draft` | Use the draft model after an N-gram miss; requires pipelined verification |
| `--draft-threads` | `draft_threads` | CPU threads for the draft model |
| `--mtp-max-tokens` | `mtp_max_tokens` | Maximum native MTP proposal length |
| `--mtp-min-tokens` | `mtp_min_tokens` | Minimum native MTP proposal length |
| `--mtp-reject-cooldown-tokens` | `mtp_reject_cooldown_tokens` | Cooldown after native MTP rejection |
| `--mtp-suppress-cooldown-draft-limit` | `mtp_suppress_cooldown_draft_limit` | Maximum suppressed cooldown drafts |
| `--ngram-min` | `ngram_min` | Minimum matching N-gram length |
| `--ngram-max` | `ngram_max` | Maximum matching N-gram length |
| `--ngram-max-tokens` | `ngram_max_tokens` | Maximum N-gram proposal length |
| `--ngram-extension-max-tokens` | `ngram_extension_max_tokens` | Maximum N-gram extension after an MTP prefix |
| `--verify-min-tokens` | `verify_min_tokens` | Minimum verification window |
| `--verify-max-tokens` | `verify_max_tokens` | Maximum verification window |
| `--verify-pipeline-depth` | `verify_pipeline_depth` | Maximum outstanding verification windows |
| `--verify-runahead-max-tokens` | `verify_runahead_max_tokens` | Outstanding speculative-token budget; zero uses pipeline depth |
| `--draft-acceptance-threshold` | `draft_acceptance_threshold` | Minimum accepted fraction of a draft window |
| `--draft-split-probability` | `draft_split_probability` | Probability of splitting a draft proposal |
| `--speculative-config` | `speculative_config` | See `skippy serve --help` for values and defaults. |
| `--draft-model-path` | `draft_model_path` | See `skippy serve --help` for values and defaults. |
| `--speculative-window` | `speculative_window` | See `skippy serve --help` for values and defaults. |
| `--adaptive-speculative-window` | `adaptive_speculative_window` | See `skippy serve --help` for values and defaults. |
| `--draft-n-gpu-layers` | `draft_n_gpu_layers` | See `skippy serve --help` for values and defaults. |
| `--native-mtp-draft-model-path` | `native_mtp_draft_model_path` | See `skippy serve --help` for values and defaults. |

### [media]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--projector-gpu` | `projector_gpu` | Run the multimodal projector on the GPU |
| `--media-marker` | `media_marker` | Media marker used by the model template |
| `--image-min-tokens` | `image_min_tokens` | Minimum image token budget |
| `--image-max-tokens` | `image_max_tokens` | Maximum image token budget |
| `--media-batch-max-tokens` | `media_batch_max_tokens` | Maximum media processing batch token count |
| `--mmproj` | `mmproj` | See `skippy serve --help` for values and defaults. |

### [api]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--guardrails-tool-retries` | `guardrails_tool_retries` | Maximum retries for invalid tool output |
| `--guardrails-structured-retries` | `guardrails_structured_retries` | Maximum retries for invalid structured output |
| `--compact-context-limit` | `compact_context_limit` | Context token limit used by compaction |
| `--compact-trigger-percent` | `compact_trigger_percent` | Context utilisation percent that triggers compaction |
| `--compact-target-percent` | `compact_target_percent` | Target utilisation percent after compaction |
| `--guardrails-retry-exhaustion` | `guardrails_retry_exhaustion` | Retry exhaustion: error or pass-last-text |
| `--guardrails-reserved-tool-prefix` | `guardrails_reserved_tool_prefix` | Tool-name prefix reserved by guardrails |
| `--guardrails-all-models` | `guardrails_all_models` | Apply guardrails to every model |
| `--compact` | `compact` | Automatically compact chat history |
| `--compact-drop-reasoning` | `compact_drop_reasoning` | Allow compaction to drop historical reasoning |
| `--guardrails-small-model-threshold` | `guardrails_small_model_threshold` | Small-model threshold in billions of parameters |
| `--bind-addr` | `bind_addr` | See `skippy serve --help` for values and defaults. |
| `--model-id` | `model_id` | See `skippy serve --help` for values and defaults. |
| `--default-max-tokens` | `default_max_tokens` | See `skippy serve --help` for values and defaults. |
| `--guardrails` | `guardrails` | See `skippy serve --help` for values and defaults. |
| `--prompt` | `prompt` | See `skippy serve --help` for values and defaults. |

### [distributed]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--activation-codec` | `activation_codec` | Stage activation codec: raw-f32-v1, f16-rne-v1, bf16-rne-v1, s8-row-f32-rne-v1 |
| `--activation-codec-policy` | `activation_codec_policy` | Stage activation policy: fixed or auto-lossless-v1 |
| `--config` | `config` | See `skippy serve --help` for values and defaults. |
| `--topology` | `topology` | See `skippy serve --help` for values and defaults. |
| `--stage-transport` | `stage_transport` | See `skippy serve --help` for values and defaults. |
| `--worker-only` | `worker_only` | See `skippy serve --help` for values and defaults. |
| `--max-inflight` | `max_inflight` | See `skippy serve --help` for values and defaults. |
| `--reply-credit-limit` | `reply_credit_limit` | See `skippy serve --help` for values and defaults. |
| `--async-prefill-forward` | `async_prefill_forward` | See `skippy serve --help` for values and defaults. |
| `--no-async-prefill-forward` | `no_async_prefill_forward` | See `skippy serve --help` for values and defaults. |
| `--downstream-connect-timeout-secs` | `downstream_connect_timeout_secs` | See `skippy serve --help` for values and defaults. |

### [runtime]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--runtime-bundle` | `runtime_bundle` | See `skippy serve --help` for values and defaults. |
| `--runtime-cache` | `runtime_cache` | See `skippy serve --help` for values and defaults. |
| `--runtime-release` | `runtime_release` | See `skippy serve --help` for values and defaults. |
| `--runtime-selection` | `runtime_selection` | See `skippy serve --help` for values and defaults. |

### [diagnostics]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--output` | `output` | See `skippy serve --help` for values and defaults. |
| `--debug` | `debug` | See `skippy serve --help` for values and defaults. |
| `--metrics-otlp-grpc` | `metrics_otlp_grpc` | See `skippy serve --help` for values and defaults. |
| `--telemetry-queue-capacity` | `telemetry_queue_capacity` | See `skippy serve --help` for values and defaults. |
| `--telemetry-level` | `telemetry_level` | See `skippy serve --help` for values and defaults. |
| `--startup-timeout-secs` | `startup_timeout_secs` | See `skippy serve --help` for values and defaults. |

### [network]

| CLI flag | TOML key | Purpose |
|---|---|---|
| `--downstream-wire-delay-ms` | `downstream_wire_delay_ms` | See `skippy serve --help` for values and defaults. |
| `--downstream-wire-mbps` | `downstream_wire_mbps` | See `skippy serve --help` for values and defaults. |
| `--downstream-wire-jitter-ms` | `downstream_wire_jitter_ms` | See `skippy serve --help` for values and defaults. |
| `--downstream-wire-stall-ms` | `downstream_wire_stall_ms` | See `skippy serve --help` for values and defaults. |
| `--downstream-wire-stall-p` | `downstream_wire_stall_p` | See `skippy serve --help` for values and defaults. |

# Anonymous usage analytics

mesh-llm reports a small amount of anonymous usage data to the maintainers so
we can see which platforms to support, which commands matter, and where nodes
fall over. Official release builds have an analytics key compiled in, so for
them it is on by default, disclosed on first run, and takes one command to
turn off permanently. A build with no key from either source — compiled in or
set at run time — reports nothing at all.

This page is the complete description of what is collected. If something
happens that is not on this page, it is a bug — please
[open an issue](https://github.com/Mesh-LLM/mesh-llm/issues).

## Turning it off

```bash
mesh-llm analytics disable
```

That writes `[analytics] enabled = false` to `~/.mesh-llm/config.toml` and
survives restarts and upgrades. Any of these also turns reporting off:

| Control | Effect |
|---|---|
| `mesh-llm analytics disable` | Permanent opt-out, written to the config file |
| `[analytics] enabled = false` | The same thing, edited by hand |
| `MESH_LLM_ANALYTICS=0` | Off for one invocation or one shell |
| `DO_NOT_TRACK=1` | Off, honoring the [console opt-out convention](https://consoledonottrack.com) |
| `CI=true` (or another CI variable) | Off automatically; CI runs are not users |
| A build with no analytics key | Off; source builds and forks report nothing unless `MESH_LLM_POSTHOG_KEY` is set at run time |

These settings are read when a process starts. A node that is already running
with reporting on keeps reporting until it restarts: `mesh-llm analytics
disable` writes the config file, it does not reach into a running process.

To see what your machine is doing right now:

```bash
mesh-llm analytics status
```

## What is collected

Every event carries:

| Property | Example | Why |
|---|---|---|
| `distinct_id` | `9f1c…` | A random UUID generated on first run |
| `mesh_llm_version` | `0.76.0` | Which versions are actually in the field |
| `build_channel` | `release`, `prerelease`, `development` | Whether pre-releases get real use |
| `os`, `arch` | `macos`, `aarch64` | Which platforms to prioritize |
| `exec_env` | `plain`, `container`, `ci`, `service` | Whether a run came from someone's machine or from throwaway infrastructure. A container gets a fresh home directory every run, so it looks like a brand new install each time; this is what separates those from real adoption. Detected from `/.dockerenv`, `/run/.containerenv`, and the `container`, `KUBERNETES_SERVICE_HOST`, `INVOCATION_ID`, and CI provider variables |

And one of these events:

| Event | Extra properties | What it answers |
|---|---|---|
| `install_first_run` | none | How many installs there are |
| `install_updated` | `from_version` (the version previously recorded, or `redacted`), `trigger` (`self_update` when mesh-llm's own updater restarted the process, otherwise `external`) | Whether people actually upgrade, how fast a release spreads, and whether anyone is stranded on an old one. Sent once when a run sees a different version than the last run recorded |
| `cli_command` | `family` (`models`, `runtime`, `diagnostics`, …), `outcome` (`completed`, `failed`, …) | Which commands are used, and which fail |
| `serve_started` | `surface`, `auto`, `headless`, `publish`, `discover`, `joined_explicitly`, `model_requested`, `auto_update`, `post_update_restart` — all booleans | How nodes are started. `auto_update` is whether the node manages its own upgrades; `post_update_restart` marks the second half of a self-update, since updating restarts the process and would otherwise count as a second node start |
| `serve_stopped` | `session_length` (bucketed: `under_1m`, `1m-15m`, …), `succeeded` | Whether nodes stay up |
| `model_loaded` | `model` (catalog name, or `redacted`; always `redacted` when `source` is `direct_gguf`, and for a model served under a configured alias such as `server.alias`), `source` (`direct_gguf`, `layer_package`, `whisper_file`, `split_layer_package` for stage 0 of a split from a Hugging Face package, `split_local_source` for stage 0 of a split from anything else, which is always `redacted`), and `system_one_backend` (`laya` or `openjev`, only when the model serves [System One](/docs/pages/system-one-api/) reads; read from the model architecture, never the file name) | Which models actually get run |
| `model_download` | `model` (catalog name, or `redacted`), `succeeded` | Which models people try to get, including ones they fail to |
| `hardware_profile` | see below | What hardware mesh-llm runs on |
| `inference_request` | see below | How fast real requests are served: prefill, decode, context length, prompt cache, speculative decoding, and lanes |
| `inference_route` | see below | Where requests go (this node, a peer, an endpoint), whether they succeed, and the link to the peer |
| `mesh_links` | see below | How this node's peers are connected and how far away they are |

`hardware_profile` is sent once per `serve` process, after the native runtime
has loaded, so it describes the devices and memory the node uses. It carries:

| Property | Example | Notes |
|---|---|---|
| `gpu_model` | `apple-m1-pro`, `nvidia-geforce-rtx-4090`, `none`, `unreported` | The device name, lowercased and hyphenated, and only when a naming probe actually produced it. `none` means no GPU was found. `unreported` means one exists but nothing named it — including when the survey holds a placeholder such as `GPU 0`, which is never published as though it were a real device |
| `gpu_count` | `1`, `3-4`, `33+` | Bucketed |
| `vram_gb` | `24`, `36` | GPU memory available to mesh-llm, in whole GiB |
| `system_ram_gb` | `16`, `64` | System RAM in whole GiB, when the platform reports it |
| `unified_memory` | `true` | Whether this is a unified-memory SoC |
| `backend_metal`, `backend_cuda`, `backend_rocm`, `backend_vulkan` | `true` / `false` | Which backends this machine can run |

GPU counts are bucketed (`3-4`, `9-16`, `33+`). Memory is rounded to the
nearest whole GiB: ranges with shared edges (`8-16`, `16-32`) were ambiguous,
and because the VRAM figure is a budget that moves slightly between runs, the
same machine could land in one range on one start and the next range on the
following one. Earlier versions reported `vram_total` and `system_ram` as
ranges.

`inference_request` is sent for a generation this node served. At most 30
are sent per minute per process; when some were skipped, the next one
carries a `weight` saying how many requests it stands for. It carries:

| Property | Example | Notes |
|---|---|---|
| `model` | `Qwen3-8B-Q4_K_M`, `redacted` | Named only when `model_loaded` named it: an exact catalog name for a model loaded from a catalog or repository reference. Local files and configured aliases are `redacted` |
| `weight` | `1`, `12` | Requests this event represents |
| `serving` | `local`, `split` | Whole model on this node, or stage 0 of a split |
| `downstream_lanes`, `concurrency_limit` | `4`, `2` | Persistent lanes to the next split stage, and generations admitted at once |
| `context_size`, `prompt_tokens`, `completion_tokens`, `context_tokens`, `context_fill` | `32768`, `3000`, `200`, `3200`, `0.0977` | Token counts only, never tokens |
| `cached_prompt_tokens`, `prefill_tokens`, `cache_hit_kind` | `2000`, `1000`, `prefix` | How much of the prompt the KV cache saved |
| `queue_wait_ms`, `restore_ms`, `prefill_ms`, `decode_ms`, `ttft_ms`, `total_ms` | `512.4` | Timings |
| `prefill_tps`, `decode_tps` | `2000`, `50` | Tokens per second |
| `finish_reason` | `stop`, `length` | Why generation ended |
| `spec_strategy`, `spec_allowed` | `dflash`, `ngram-suffix`, `disabled` | The speculative decoding strategy, and whether the speculation gate let this request use it |
| `spec_proposed_tokens`, `spec_accepted_tokens`, `spec_acceptance`, `spec_windows`, `spec_accepted_per_window` | `300`, `150`, `0.5`, `50`, `3` | Draft tokens proposed and accepted, overall and split into draft-model/N-gram (`spec_draft_*`) and native MTP/split DFlash (`spec_mtp_*`) |
| `activation_codec`, `activation_prefill_bytes`, `activation_decode_bytes` | `f16` | Split only: bytes sent to the next stage |

`inference_route` is sent for each routing attempt this node fronted, under
the same per-minute limit and `weight` rule. It carries `model` (as above),
`target` (`local`, `remote`, `endpoint`), `outcome` (`success`, `timeout`,
`unavailable`, `context_overflow`, `rejected`), `queue_wait_ms`, `attempt_ms`,
`completion_tokens`, `end_to_end_tps`, and for a remote target `link` and
`rtt_ms`.

`mesh_links` is sent five minutes after a node starts and hourly after that.
It carries `peers`, and for each link class (`lan`, `wan`, `relay`,
`unknown`) the number of peers (`lan_peers`, …) and their median and maximum
round-trip time (`lan_rtt_ms_median`, `lan_rtt_ms_max`, …). A peer with no observed path counts under `unknown` and carries no round-trip time.

A link is `lan` when it is direct to a private, link-local, or loopback
address, `wan` when it is direct to a public address, and `relay` when it
goes through a relay. The address decides the class and is never reported.

Processes mesh-llm starts for its own purposes report nothing at all: the
plugin services `serve` spawns, and the bundle-verification run the updater
performs before installing a release. They exist to serve a user action that
is already being reported, so counting them again would inflate installs,
commands and node starts.

## What is never collected

- **Prompts, completions, and any model input or output.** None of it is read
  by the analytics code, and the API cannot carry free text.
- **Model contents or file paths.** `--gguf /home/you/private.gguf` reports
  `model_requested: true` on `serve_started`, and a `model_loaded` carrying
  `model: redacted`, `source: direct_gguf`. The file name is never reported,
  not even the stem without its directories — a name you chose for a local
  file is yours, and `acme-merger-finetune` says as much as the path does.
  Elsewhere, a model name that is not catalog-shaped is reported as
  `redacted`.
- **Your IP address or location.** Every event sets `$geoip_disable` and a
  null `$ip`, and the project discards client IP data at ingestion.
- **Anything that identifies your mesh peers.** No peer IDs, addresses, mesh
  names, or invite tokens. Only how many peers there are, how each link is
  carried, and how long a round trip takes.
- **Hostnames, usernames, MAC addresses, or hardware serials.** The install
  identifier is random and derived from nothing. The hardware survey can
  report a hostname and per-GPU serials, PCI addresses, and vendor UUIDs;
  none of them are read by the analytics code, and the hostname is not even
  requested from the probe.
- **Your API keys, tokens, or config values**, other than the single
  `analytics.enabled` flag.

The reporting code accepts only a fixed set of event names and a closed set of
property values. Free-form text cannot be attached to an event, so a prompt
cannot leak through this path even by mistake. Text that does appear — a model
name — passes a grammar that rejects paths, whitespace, and anything
over-long, and becomes `redacted` when it fails. Names sourced from a local
file are redacted ahead of that grammar rather than relying on it, because a
bare file stem would satisfy it.

## How it differs from `[telemetry]`

They are unrelated. Analytics uses the maintainers' endpoint. Telemetry uses
the endpoint you configure, which may be local or remote.

| | `[analytics]` | `[telemetry]` |
|---|---|---|
| Goes to | The mesh-llm maintainers | An OTLP endpoint **you** configure |
| Default | On, disclosed on first run | Off |
| Contents | The fixed event list above | Operational metrics for your own dashboards |
| Purpose | Product decisions | Running your own deployment |

Turning analytics off does not affect `[telemetry]`, and vice versa.

## The install identifier

A random v4 UUID in `~/.mesh-llm/analytics-id`, created on first run. It is
derived from nothing about your machine, and it is deliberately **not** your
mesh node identity — that key is published to the public mesh, and reusing it
would tie usage data to a publicly visible node.

Delete the file to reset it. You will be counted as a new install.

## Self-hosting the endpoint

`MESH_LLM_POSTHOG_HOST` points reporting at your own PostHog instance, and
`MESH_LLM_POSTHOG_KEY` sets the project key. Both are mainly useful for
verifying what your build actually sends:

```bash
MESH_LLM_POSTHOG_HOST=http://127.0.0.1:8000 \
MESH_LLM_POSTHOG_KEY=phc_your_key \
MESH_LLM_ANALYTICS=1 \
  mesh-llm gpus
```

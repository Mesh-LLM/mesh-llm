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

And one of these events:

| Event | Extra properties | What it answers |
|---|---|---|
| `install_first_run` | none | How many installs there are |
| `cli_command` | `family` (`models`, `runtime`, `diagnostics`, …), `outcome` (`completed`, `failed`, …) | Which commands are used, and which fail |
| `serve_started` | `surface`, `auto`, `headless`, `publish`, `discover`, `joined_explicitly`, `model_requested` — all booleans | How nodes are started |
| `serve_stopped` | `session_length` (bucketed: `under_1m`, `1m-15m`, …), `succeeded` | Whether nodes stay up |
| `model_loaded` | `model` (catalog name, or `redacted`; always `redacted` when `source` is `direct_gguf`), `source` (`direct_gguf`, `layer_package`) | Which models actually get run |
| `model_download` | `model` (catalog name, or `redacted`), `succeeded` | Which models people try to get, including ones they fail to |
| `hardware_profile` | see below | What hardware mesh-llm runs on |

`hardware_profile` is sent once per `serve` process and carries:

| Property | Example | Notes |
|---|---|---|
| `gpu_model` | `apple-m1-pro`, `nvidia-geforce-rtx-4090` | The device name, lowercased and hyphenated |
| `gpu_count` | `1`, `3-4`, `33+` | Bucketed |
| `vram_total` | `8-16`, `32-64` | Bucketed gigabytes across all GPUs |
| `system_ram` | `16-32`, `64-128` | Bucketed gigabytes of system RAM, when the platform reports it |
| `unified_memory` | `true` | Whether this is a unified-memory SoC |
| `backend_metal`, `backend_cuda`, `backend_rocm`, `backend_vulkan` | `true` / `false` | Which backends this machine can run |

Counts and sizes are bucketed rather than exact (`3-4`, `9-16`, `33+`),
because an exact VRAM figure or GPU count at the tail can identify a single
deployment.

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
- **Anything about your mesh peers.** No peer IDs, addresses, mesh names,
  invite tokens, or topology.
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

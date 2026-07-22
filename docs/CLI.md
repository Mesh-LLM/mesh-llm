# CLI User Guide

This is a practical user guide to the `mesh-llm` CLI.
It explains what to run for common tasks, then documents each command and switch.

Catalog id definition: a catalog id is the model id shown in `mesh-llm models recommended` (for example `Qwen3-0.6B-Q4_K_M`).

## Get help

```bash
mesh-llm --help
mesh-llm <command> --help
mesh-llm setup --help
mesh-llm uninstall --help
mesh-llm doctor --help
mesh-llm models --help
mesh-llm models <subcommand> --help
```

## Start here (common tasks)

If you want to:

1. Finish a fresh install:

```bash
mesh-llm setup
```

2. Start serving on this machine:

```bash
mesh-llm serve --model Qwen3-0.6B-Q4_K_M
```

3. Join the public mesh:

```bash
mesh-llm serve --auto
```

4. Find a model you can run:

```bash
mesh-llm models search gemma --gguf
mesh-llm models search smoll --mlx
```

5. Inspect a model before downloading:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

6. Download a model:

```bash
mesh-llm models download unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

7. Check what is already installed:

```bash
mesh-llm models installed
```

8. Remove the executable and setup-owned files:

```bash
mesh-llm uninstall --dry-run
mesh-llm uninstall --yes
```

## Runtime entrypoints (`serve` / `client`)

If you want to start serving, join a mesh, or run as an API-only client, start here.

Examples:

```bash
mesh-llm setup
mesh-llm serve
mesh-llm serve --model Qwen3-0.6B-Q4_K_M
mesh-llm client --auto
```

### `setup`

Use this to finish a fresh install after the executable is on your `PATH`.

`mesh-llm setup` downloads and configures the native runtime, can install and
enable the background service on supported macOS and Linux machines, and only
shows the GitHub star prompt when it is interactive and eligible. The star
prompt defaults to Yes, and `--yes` or `--no-interactive` skip it without
starring anything. Default output is concise; use `--verbose` when you want
service paths, commands, log locations, and detailed setup status.

Usage:

```bash
mesh-llm setup
mesh-llm setup --service
mesh-llm setup --no-service --skip-runtime
mesh-llm setup --yes
mesh-llm setup --verbose
```

Switches:

- `--yes`: automatically answer yes to setup prompts. This accepts the service prompt and skips the GitHub star prompt.
- `--no-interactive`: run without prompting. When service is not requested, setup prints guidance to rerun with `--service`.
- `--service`: install and enable the background service.
- `--no-service`: skip installing and enabling the background service.
- `--skip-runtime`: skip downloading or configuring the native runtime.
- `--verbose`: print detailed service paths, commands, log locations, and setup status.

On Windows, `--service` is unsupported.

### `uninstall`

Use this to remove a Mesh executable install and setup-owned service/runtime files from a machine.

By default, uninstall stops tracked `mesh-llm` processes, disables and removes
the per-user service when present, removes setup-owned service helper files,
removes the native-runtime cache, and removes the executable last. It preserves
`~/.mesh-llm` configuration and identity data unless you explicitly pass
`--purge-config`.

Usage:

```bash
mesh-llm uninstall --dry-run
mesh-llm uninstall --yes
mesh-llm uninstall --yes --keep-cache
mesh-llm uninstall --yes --purge-config
mesh-llm uninstall --verbose --dry-run
```

Switches:

- `--dry-run`: print the cleanup plan without changing the machine.
- `--yes`: run without a confirmation prompt.
- `--json`: print dry-run plans and outcomes as JSON.
- `--verbose`: print detailed cleanup steps and removed paths.
- `--keep-cache`: preserve downloaded native runtimes.
- `--keep-service-files`: preserve setup-owned service helper files.
- `--purge-config`: remove `~/.mesh-llm` configuration and identity data.
- `--keep-config`: explicitly preserve configuration and identity data.
- `--binary-path <PATH>`: remove a specific executable path.

If the setup service configuration directory contains unrelated files,
uninstall leaves that directory in place and reports a warning instead of
recursively deleting it. Default text output is concise; use `--verbose` when
you want the full cleanup plan or exact removed paths.

### `doctor`

Use this only when troubleshooting a failed install or runtime problem. It gathers local status, runtime diagnostics, and logs.

Usage:

```bash
mesh-llm doctor
```

Switches:

- `--json`: machine-readable output.

### Common runtime options

- `--join <TOKEN>`: join a specific mesh using an invite token (repeatable).
- `--discover [NAME]`: discover a mesh and join it. With a name, joins the mesh matching that name. Without a name, behaves like `--auto`.
- `--mesh-discovery-mode <nostr|mdns>`: choose the discovery provider. `nostr`
  is the default public/WAN-capable mode. `mdns` browses LAN DNS-SD records,
  requires a supplied matching invite token for join material and LAN detail
  proof, and disables public iroh relays plus raw STUN startup probing. LAN
  detail endpoints are only advertised when the management API is reachable
  from LAN peers, for example with `--listen-all`.
- `--auto`: auto-join the best discovered mesh.
- `--model <MODEL>`: model to serve (catalog id from `models recommended`, HF ref/URL, or path).
- `--gguf <GGUF>`: serve a specific local GGUF file directly (repeatable).
- `--port <PORT>`: API port (default `9337`).
- `--client`: API-only mode (no GPU/model serving).
- `--console <CONSOLE>`: console/API management port (default `3131`).
- `--headless`: disable the embedded web UI; keep the management API on the `--console` port.
- `--bind-ip <IP>`: bind mesh QUIC to a specific local IP address. In default
  Nostr discovery mode the invite can still include relay/public candidates.
  In `--mesh-discovery-mode mdns`, only LAN/direct candidates are advertised.
  Use this on multi-interface hosts where Docker/CNI bridge addresses overlap
  across nodes.
- `--bind-port <PORT>`: bind mesh QUIC to a fixed UDP port, usually paired
  with `--bind-ip` for firewall or NAT rules.
- `--swarm-capture <DIR>`: write passive local mesh membership and connection
  diagnostics as JSONL. See [SWARM_CAPTURE.md](SWARM_CAPTURE.md) for the full
  debug-capture workflow.
- `--publish`: publish your mesh for discovery.
- `--require-release-attestation`: when creating a requirement-aware mesh,
  require peers to present a trusted release attestation.
- `--release-signer-key <KEY>`: allow a release signer key in the creation-time
  attestation policy (repeatable). Use with `--require-release-attestation`.
- `--mesh-name <MESH_NAME>`: friendly mesh name in discovery.
- `--region <REGION>`: region hint for discovery.
- `--name <NAME>`: display name for this node.
- `--max-vram <MAX_VRAM>`: cap VRAM used for planning and fit decisions.
- `--llama-flavor <LLAMA_FLAVOR>`: force backend binary flavor (`cpu|cuda|rocm|vulkan|metal`).
- `--config <CONFIG>`: explicit config file path. The file applies on future
  starts or owner-control reloads, not to already running sessions.
- `--owner-key <OWNER_KEY>`: keystore used to attest this runtime node.
- `--owner-required`: fail startup if owner attestation cannot be loaded.
- `--node-label <NODE_LABEL>`: attach a human label to this runtime node certificate.
- `--trust-policy <TRUST_POLICY>`: override peer ownership trust policy.
- `--trust-owner <TRUST_OWNER>`: add trusted owner IDs on top of the local trust store.
- `--mesh-guardrails <MODE>`: server-side mesh guardrail mode for hosted
  Skippy backends (`disabled`, `metrics`, or `enforce`; default `disabled`).
  This controls `GuardrailPolicy.mode`; request-level `mesh_guardrails` flags
  cannot upgrade a disabled server.

Config file semantics:

- `mesh-llm serve` reads `~/.mesh-llm/config.toml` by default.
- Precedence is request values, then per-model config, then `[defaults.*]`, then
  family or topology policy, then built-in runtime defaults.
- Request defaults only fill absent or null request fields at the OpenAI
  frontend boundary. Explicit request values win, and the defaults never flow
  into `StageConfig`, runtime load structs, protobuf, or lower runtime.
- Staged-only controls stay staged-only. Activation wire dtype, prefill
  controls, speculative draft controls, and manual stage layer ranges only
  execute in staged mode.
- Unsupported or deferred rows are documented as rejected, not silent no-ops.

## Commands

### `models`

Start with `models` when you’re working with models: finding them, checking details, downloading them, or checking update state.

Subcommands:

- `recommended`
- `installed`
- `cleanup`
- `prune`
- `search`
- `show`
- `download`
- `package`
- `certify`
- `updates`
- `delete`

### `models recommended`

Run this when you want the official built-in model IDs (catalog IDs) and sizes.

Switches:

- `--json`: machine-readable output.

### `models installed`

Run this when you want to see what’s already on your machine.

Switches:

- `--json`: machine-readable output.

### `models cleanup`

Run this when you want to remove stale managed model-cache entries that are no
longer usable or referenced.

Use `mesh-llm models cleanup --help` for the current safety and confirmation
switches before deleting anything.

### `models prune`

Run this when you want to clean derived Skippy materialized stage cache. The
default mode is a preview; pass the confirmation switch shown by
`mesh-llm models prune --help` to apply the cleanup.

This command treats materialized stages as derived cache and preserves active or
pinned stage artifacts.

### `models search`

Use this to find something you can actually download and run (GGUF or MLX).

Usage:

```bash
mesh-llm models search gemma --gguf
mesh-llm models search smoll --mlx --limit 5
mesh-llm models search qwen --catalog
```

Switches:

- `--gguf`: GGUF-only search (default).
- `--mlx`: MLX-only search.
- `--catalog`: search only built-in catalog.
- `--limit <LIMIT>`: max results (default `20`).
- `--json`: machine-readable output.

### `models show`

Use this when you want to sanity-check one exact model ref before you download or serve it.

Usage:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
mesh-llm models show mlx-community/SmolLM-135M-8bit
```

Switches:

- `--json`: machine-readable output.

### `models download`

Use this when you’re ready to download one specific resolved model.

Usage:

```bash
mesh-llm models download unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
mesh-llm models download mlx-community/SmolLM-135M-8bit
```

Switches:

- `--draft`: also download the recommended draft model (if available).
- `--direct`: download the exact Hugging Face GGUF file directly, bypassing catalog layer-package resolution.
- `--json`: machine-readable output.

### `models package`

Use this to plan or submit a Hugging Face Job that splits a source GGUF into a
Skippy layer-package repository. This is spend-bearing, so the command defaults
to dry-run behavior and requires `--confirm` before it submits jobs.

Usage:

```bash
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --dry-run
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --confirm --follow
mesh-llm models package --status <job-id>
mesh-llm models package --logs <job-id>
mesh-llm models package --list
```

Switches:

- `--target <REPO>`: destination Hugging Face package repo.
- `--model-id <MODEL_ID>`: OpenAI-facing package model id.
- `--flavor <FLAVOR>`: package flavor, default `auto`.
- `--timeout <DURATION>`: HF Jobs timeout, default `1h` unless size estimates raise it.
- `--mesh-llm-ref <REF>`: mesh-llm git ref used inside the job, default `main`.
- `--dry-run`: print the resolved plan, selected hardware, timeout, and maximum cost without side effects.
- `--confirm`: submit the job.
- `--follow`: wait for submitted job progress.
- `--json`: machine-readable output.
- `--status <JOB_ID>`: inspect a job.
- `--logs <JOB_ID>`: fetch job logs.
- `--cancel <JOB_ID>`: cancel a job.
- `--list`: list known jobs.
- `--update-script`: refresh the bucket script before a confirmed submission.

Keep source refs in colon-selector form such as
`unsloth/Qwen3-8B-GGUF:Q4_K_M`. Do not use the deprecated separate `--quant`
form in generated job inputs.

### `models certify`

Use this when you want a repeatable Skippy layer-package confidence report
before treating a split package as ready for a release or rollout.

Choose exactly one mode: use `--package-only` for package integrity and local
stage materialization, or pass `--api-base` to also prove an already running
OpenAI-compatible mesh endpoint. Runtime certification checks the model list and
requires real text-bearing responses from both chat completions and Responses
API smoke requests, not only successful HTTP status codes.

Usage:

```bash
mesh-llm models certify hf://meshllm/Qwen3-8B-Q4_K_M-layers --package-only --report-out cert.json
mesh-llm models certify unsloth/Qwen3-8B-GGUF:Q4_K_M --api-base http://127.0.0.1:9337 --json
```

Switches:

- `--package-only`: verify package resolution, artifact integrity, and local stage materialization without claiming runtime OpenAI readiness.
- `--api-base <URL>`: run `/v1/models`, `/v1/chat/completions`, and `/v1/responses` smoke gates against an already running mesh-llm API. The URL must be an `http` or `https` base URL.
- `--report-out <PATH>`: write the JSON certification report to disk.
- `--prompt <PROMPT>`: prompt for runtime smoke gates.
- `--max-tokens <N>`: max tokens for runtime smoke gates. Must be greater than zero when runtime gates are enabled.
- `--json`: print the certification report.

### `models updates`

Use this when you want to check for new upstream revisions or refresh cached repo metadata.

Usage:

```bash
mesh-llm models updates --check
mesh-llm models updates --all
mesh-llm models updates unsloth/gemma-4-31B-it-GGUF
```

Switches:

- `--all`: operate on all cached HF repos.
- `--check`: check only; do not refresh cache.
- `--json`: machine-readable output.

### `models delete`

Use this when you need to remove a specific managed model entry. Run
`mesh-llm models delete --help` first; deletion commands intentionally keep
confirmation behavior close to the CLI implementation so operators see the
current safety prompts.

### `download`

Use this to quickly download by built-in catalog ID or shorthand.

Usage:

```bash
mesh-llm download
mesh-llm download 32b
mesh-llm download Qwen3-0.6B-Q4_K_M --draft
```

Switches:

- `--draft`: download recommended draft model too.

### `update`

Use this to update mesh-llm and exit.

Switches:
- `--version <VERSION>`: install a specific release tag or version, for example `v0.60.0`.
- `--flavor <FLAVOR>`: install or switch to a specific release bundle flavor (`cpu`, `cuda`, `rocm`, `vulkan`, or `metal`).
- `--detect-flavor`: re-detect the best host backend flavor before selecting the release bundle. Cannot be combined with `--flavor`.
- `--auto-update`: available on most commands; when set, mesh-llm checks for a newer bundled release before proceeding.


### `release-attestation inspect`

Use this to inspect the embedded release attestation on the packaged `mesh-llm` executable.

Usage:

```bash
cargo run -p xtask -- release-attestation inspect --binary /path/to/mesh-bundle/mesh-llm --public-key-file /path/to/release-signing-public-key.json
```

Switches:

- `--binary <PATH>`: packaged `mesh-llm` executable to inspect.
- `--public-key-file <PATH>`: release-signing trust root required to validate an embedded stamped binary.
- `--json`: machine-readable output.

The command reports `missing`, `valid`, or `invalid`. It applies only to the
shipped executable, not SDK, XCFramework, or other native artifacts. Local and
dev builds are unstamped by default, so `missing` is normal there. A
post-download mutation can turn a stamped binary `invalid`, but the default
startup path still allows it because this is provenance and admission
hardening, not runtime integrity proof. Bare `inspect --binary ...` is only
enough to classify an unstamped binary as `missing`; if an embedded attestation
is present, the command requires `--public-key-file` and otherwise reports
`invalid` with an explicit error.

### `gpus`

Use this to inspect local GPU identity and capacity, including per-device VRAM, unified-memory state, and cached benchmark-derived bandwidth when present. `mesh-llm gpus detect` refreshes the raw hardware fingerprint, bandwidth, and compute hints used by local planning.

### `benchmark tune`

Use this to benchmark model-serving throughput for already-downloaded local models. It resolves local targets, plans safe startup settings, then starts isolated trial `mesh-llm serve` children from temporary configs and reports per-candidate decode tok/s.

The recommendation is tolerance-aware: benchmark tune reports the raw highest-throughput trial, computes the Pareto frontier for decode tok/s versus `ctx_size`, then recommends the largest context window whose decode throughput is within the configured tolerance of the raw best.

Examples:

```bash
mesh-llm benchmark tune --model /models/qwen3-8b.gguf
mesh-llm benchmark tune --models /models/qwen3-8b.gguf,/models/mixtral.gguf --json
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --ctx-sizes 4096,8192,16384 --batch-sizes 1024,2048 --ubatch-sizes 256,512
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --mmap-values auto,true,false --mlock-values true,false
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --flash-attention on,off
mesh-llm benchmark tune --model /models/qwen3-mtp.gguf --speculative-types auto
mesh-llm benchmark tune --model /models/qwen3-mtp.gguf --speculative-types mtp --debug-telemetry --json
mesh-llm benchmark tune --model /models/qwen3-mtp.gguf --speculative-types mtp,mtp-ngram,disabled --spec-draft-max-tokens 4,8,16 --spec-ngram-min 2,3 --spec-ngram-max 3,4
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --throughput-tolerance-pct 2.5
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --apply
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --apply --replace-existing
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --launch-args
```

Switches:

- `--model <MODEL>`: benchmark one exact local model that is already downloaded.
- `--models <MODELS>`: benchmark multiple exact local models, separated by commas.
- `--json`: machine-readable benchmark tune report.
- `--apply`: persist the recommended settings to the local config file (`~/.mesh-llm/config.toml`).
- `--replace-existing`: when persisting, overwrite existing writable recommendation fields instead of preserving current values.
- `--launch-args`: print the exact `mesh-llm serve` arguments generated for the recommendation path instead of running benchmark output/apply mode.
- `--ctx-sizes <TOKENS>`: comma-separated context sizes to benchmark. If omitted, tune derives a small context ladder up to the planned context.
- `--batch-sizes <VALUES>` / `--ubatch-sizes <VALUES>`: comma-separated batch and micro-batch values to benchmark. Candidates where `ubatch > batch` are skipped.
- `--mmap-values <VALUES>`: comma-separated mmap values to benchmark independently: `auto`, `enabled`/`true`, or `disabled`/`false`. If omitted, benchmark tune tries all three.
- `--mlock-values <VALUES>`: comma-separated mlock values to benchmark independently: `enabled`/`true` or `disabled`/`false`. If omitted, benchmark tune tries `false` and also tries `true` only when the mlock probe says the evaluated budget can be locked.
- `--flash-attention <VALUES>`: comma-separated flash attention values to benchmark independently: `on`/`enabled`/`true` or `off`/`disabled`/`false`. When omitted, flash attention is not varied during the sweep. When supplied (e.g. `--flash-attention on,off`), trial count doubles and the recommendation applies the best flash attention setting.
- `--speculative-types <VALUES>`: comma-separated speculative decoding types to benchmark: `auto`, `mtp`, `mtp-ngram`, `draft`, or `disabled`. If omitted, `auto` tries native MTP and the bounded MTP + request-local N-gram cache composite for MTP-looking targets, then discovered draft candidates and a disabled baseline.
- `--no-speculative-tune`: skip speculative sweeps and benchmark only the disabled speculative baseline.
- `--spec-draft-models <PATHS>`: comma-separated local draft GGUF paths for `draft` speculation trials. Tune also considers configured `draft_model` values and obvious local sibling draft/EAGLE GGUF files.
- `--spec-draft-max-tokens <TOKENS>` / `--spec-draft-min-tokens <TOKENS>`: comma-separated draft-token window candidates for MTP and draft speculation.
- `--spec-ngram-min <TOKENS>` / `--spec-ngram-max <TOKENS>`: comma-separated cache match-window candidates for `mtp-ngram` speculation.
- `--throughput-tolerance-pct <PCT>`: treat candidates within this percent of the raw best decode tok/s as throughput-equivalent, then prefer the largest `ctx_size` among them, default `10.0`.
- `--max-tokens <TOKENS>`: generated tokens per measured request, default `128`.
- `--startup-timeout-secs <SECONDS>` / `--request-timeout-secs <SECONDS>`: per-trial startup and HTTP request limits, both default `600`.
- `--debug-telemetry`: run each isolated trial with Skippy debug telemetry mirrored into the trial log. Use this to prove speculative decoding activity; MTP summaries appear as `stage.openai_decode` telemetry lines with `llama_stage.native_mtp.*` attributes.
- `--prompt <TEXT>`: prompt sent during measured chat-completion requests.

Benchmark trials keep lifecycle timing stats in JSON under `benchmarks[].trials[].timings`: `setup_ms`, `readiness_ms`, `request_ms`, `shutdown_ms`, `total_ms`, and `readiness_attempts`. The legacy `elapsed_ms` field remains the measured chat-completion request duration used for decode tok/s.

### `load`

Use this to load a model into an already-running local mesh-llm runtime.

Usage:

```bash
mesh-llm load Qwen3-0.6B-Q4_K_M
```

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `unload`

Use this to unload a model from a running local runtime.

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `status`

Use this to inspect model status from a running local runtime.

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `runtime guardrails`

Use this to switch mesh guardrail mode on a running local runtime without
restarting it. The command updates the server-side shared guardrail policy used
by hosted Skippy backends and future runtime-loaded/replacement models.

Usage:

```bash
mesh-llm runtime guardrails --mode metrics --port 3131
mesh-llm runtime guardrails --mode enforce --port 3131 --json
```

Switches:

- `--mode <MODE>`: `disabled`, `metrics`, or `enforce`.
- `--port <PORT>`: target management/API port (default `3131`).
- `--json`: machine-readable response with `mode`, `updated_models`, and the
  current `status` payload.

Equivalent REST call:

```bash
curl -s -X POST localhost:3131/api/runtime/mesh-guardrails \
  -H 'Content-Type: application/json' \
  -d '{"mode":"metrics"}' | jq .
```

Verify the active posture through `/api/status`:

```bash
curl -s localhost:3131/api/status | jq '.runtime.openai_guardrails'
```

### `discover`

Use this to discover meshes through the selected discovery provider and
optionally select one automatically. The default provider is Nostr; pass the
global `--mesh-discovery-mode mdns` before the command for LAN mDNS discovery.

Switches:

- `--name <NAME>`: filter by mesh name (case-insensitive exact match).
- `--model <MODEL>`: filter discovered meshes by model name substring.
- `--min-vram <MIN_VRAM>`: filter by minimum VRAM (GB).
- `--region <REGION>`: filter by region.
- `--auto`: print best invite token (useful for piping).
- `--relay <RELAY>`: custom Nostr relay URL(s). Only valid with
  `--mesh-discovery-mode nostr`.

### `goose`

Use this to launch Goose already wired to mesh-llm’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--port <PORT>`: mesh-llm API port (default `9337`).

### `claude`

Use this to launch Claude Code already wired to mesh-llm’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--port <PORT>`: mesh-llm API port (default `9337`).

### `pi`

Use this to launch Pi already wired to mesh-llm’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--host <HOST|HOST:PORT|URL>`: Pi target host or URL (default `127.0.0.1:9337`).
- `--write`: write the mesh provider config without launching Pi.

### `opencode`

Use this to launch OpenCode already wired to mesh-llm’s OpenAI-compatible endpoint.

It injects a temporary OpenCode config through `OPENCODE_CONFIG_CONTENT` at launch time, so it does not edit persistent OpenCode config files unless you explicitly pass `--write`.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--host <HOST|HOST:PORT|URL>`: OpenCode target host or URL (default `127.0.0.1:9337`). Bare host forms assume `http`, default inference port `9337`, and default management port `3131`.
- `--write`: write a merged `~/.config/opencode/opencode.json` that preserves unrelated root keys and sibling providers. If only `opencode.jsonc` exists, mesh-llm errors and tells you to rename or migrate it to `opencode.json` first.

### `skills`

Use this to install Agent Skills exposed by installed mesh plugins.

Usage:

```bash
mesh-llm skills install
mesh-llm skills install --agent goose --agent codex
mesh-llm skills install --all --dry-run
```

By default, the installer writes only to detected agents. Plugin packages expose
skills by shipping `skills/<name>/SKILL.md` under the plugin archive root.
Agent launch commands (`goose`, `pi`, `opencode`, and `claude`) install
available plugin skills for that agent before starting the session.

Switches:

- `--agent <AGENT>`: install to a specific agent (`goose`, `pi`, `codex`, `opencode`, `claude`); repeatable.
- `--all`: install to every supported target location even if the agent is not detected.
- `--dry-run`: print planned writes without changing files.
- `--force`: overwrite an existing user-owned skill directory with the same name.

### `stop`

Use this to stop local `mesh-llm` instances tracked in the runtime root.


### `blackboard`

Use this external plugin command to post/search/read shared mesh notes after
installing and configuring the blackboard plugin.

Usage:

```bash
mesh-llm plugins install blackboard
mesh-llm blackboard
mesh-llm blackboard "STATUS: testing gguf resolution"
mesh-llm blackboard --search "gemma"
```

Switches:

- `--search <SEARCH>`: search blackboard entries.
- `--from <FROM>`: filter by author.
- `--since <SINCE>`: last N hours.
- `--limit <LIMIT>`: max rows (default `20`).
- `--port <PORT>`: target management/API port (default `3131`).

### `plugins` (alias: `plugin`)

Use this to install, inspect, and manage native plugins. `plugin` remains an
alias for scripts that used the older singular spelling.

Subcommands:

- `plugins install <REFERENCE>`: install from a catalog name, GitHub
  `owner/repository`, or GitHub URL.
- `plugins install --archive <PATH> --name <NAME> [--version <VERSION>]`:
  install a local `.tar.gz` or `.zip` release archive. `--name` is required;
  `--version` defaults to `dev`. These flags conflict with `<REFERENCE>`.
- `plugins update <NAME>`: install the latest compatible release for an
  installed plugin.
- `plugins enable <NAME>`: mark an installed plugin runnable by mesh-llm.
- `plugins disable <NAME>`: keep an installed plugin on disk but prevent host
  startup from launching it.
- `plugins delete <NAME>`: remove the installed archive contents and local
  metadata.
- `plugins info <NAME>`: show source, version, target, path, and latest known
  status for an installed or configured plugin.
- `plugins search [QUERY]`: search the configured plugin catalog.
- `plugins list`: list installed, auto-registered, and configured plugins.

Local archives are for authoring and validation. Rebuild and reinstall them;
`plugins update` remains a GitHub release workflow.

Plugin installation and enablement are separate from the optional web UI
projection. A plugin that declares a web UI can be shown or hidden with its
`web_ui_enabled` config field or the console toggle without changing its
process state.


### `auth`

Use this to manage owner identity and keystore files.

Subcommands:

- `auth init`: generate/save owner keypair.
- `auth status`: show identity/keystore status.

`auth init` switches:

- `--owner-key <OWNER_KEY>`: keystore path.
- `--force`: overwrite existing keystore.
- `--no-passphrase`: leave keys unencrypted.
- `--keychain`: store random unlock passphrase in OS keychain.

`auth status` switches:

- `--owner-key <OWNER_KEY>`: keystore path.

`auth sign-node` / `auth renew-node` / `auth rotate-node` switches:

- `--owner-key <OWNER_KEY>`: keystore path.
- `--node-label <NODE_LABEL>`: attach a human label to the signed node certificate.

`auth rotate-owner` switches:

- `--owner-key <OWNER_KEY>`: keystore path.

## Model reference formats

Supported for `models show`, `models download`, and `serve --model`:

1. Catalog id (an id from `mesh-llm models recommended`):

```bash
mesh-llm models show Qwen3-0.6B-Q4_K_M
```

2. HF repo or GGUF selector:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

3. HF URL:

```bash
mesh-llm models show https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
```

4. Revision pin:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL@main
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL@<commit-sha>
mesh-llm models show mlx-community/SmolLM-135M-8bit@<commit-sha>
mesh-llm models show https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/main
```

For MLX, use repo shorthand (not `/model`):

```bash
mesh-llm models show mlx-community/SmolLM-135M-8bit
mesh-llm models download mlx-community/SmolLM-135M-8bit
```

5. Skippy layer package ref:

```bash
mesh-llm models show hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@<commit-sha>
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@<commit-sha> --split
```

Prefer immutable `hf://namespace/repo@revision` refs for production split runs.

## Model resolution behavior

Resolution order:

1. exact catalog id
2. exact HF ref
3. HF URL
4. bare-name discovery

GGUF behavior:

1. GGUF search uses Hub `gguf` pre-filter.
2. Excludes sidecars like `mmproj*.gguf`.
3. Split GGUF uses first shard (`-00001-of-...`) for selection/display.
4. `repo` with no selector uses fit-aware ranking against local VRAM.
5. `repo:SELECTOR` resolves exact quant/variant.

MLX behavior:

1. MLX search uses Hub `mlx` pre-filter.
2. Model must include weight files (`model.safetensors` or split first shard).
3. `model.safetensors.index.json` by itself is not treated as a model artifact.
4. Display reference stays repo shorthand.

## Machine-readable output (`--json`)

All `models` subcommands support `--json`.

Examples:

```bash
mesh-llm models search smoll --mlx --limit 1 --json | jq .
mesh-llm models show mlx-community/SmolLM-135M-8bit --json | jq .
mesh-llm models download Qwen3-0.6B-Q4_K_M --json | jq .
mesh-llm models installed --json | jq .
mesh-llm models recommended --json | jq .
mesh-llm models updates --check --json | jq .
```

Shape summary:

- `search --json`: `{ filter, query, machine, results[] }`
- `show --json`: resolved model + `variants[]`
- `download --json`: requested/resolved refs + local `path`
- `installed --json`: `{ cache_dir, results[] }`
- `recommended --json`: `{ source, results[] }`
- `updates --json`: check/update results
- `package --json`: package job plan/status/log/list output

Automation tips:

1. Prefer explicit refs in scripts.
2. Pin `@<commit-sha>` when reproducibility matters.
3. Parse stable keys such as `type`, `ref`, `fit`, `path`, and `results`.

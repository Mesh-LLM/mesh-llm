# Skippy WAN Docker Lab

This lab runs a four-stage CPU-backed Skippy chain in Docker with Linux
`tc netem` shaping between stages and `metrics-server` enabled.

The lab is intentionally host-cache first: `up.sh` ensures the Hugging Face
layer package is present and complete in the host HF cache before Docker
containers are brought up. Containers then mount that cache read-only.

## Layout

- `metrics` runs `metrics-server` on HTTP `:18080` and OTLP/gRPC `:14317`.
- `stage0` owns the first layer range, exposes OpenAI on host port `9337`, and
  forwards binary activation traffic to `stage1`.
- `stage1`, `stage2`, and `stage3` run the remaining layer ranges.
- Every stage container has `NET_ADMIN` and applies `tc netem` to the Docker
  interface used for stage-to-stage traffic.

The shaping is Linux-level traffic control, not Skippy's artificial
`--downstream-wire-delay-ms` or `--downstream-wire-mbps` flags.

## Configure

Create the local env file if you want to override defaults:

```bash
cp skippy/evals/wan-lab/.env.example skippy/evals/wan-lab/.env
```

The default package is:

```text
hf://meshllm/gemma-4-26B-A4B-it-UD-Q4_K_M-layers
```

`HF_HOME` defaults to `${HOME}/.cache/huggingface`. The actual Hub cache root
is `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`, then `${HF_HOME}/hub`. The
launcher passes that Hub root to the native package owner and mounts the same
directory read-only at `/hf-cache`; containers use `HF_HUB_CACHE=/hf-cache`.
No `hf` CLI or Python package client is needed for native cache admission or
package acquisition.

To calibrate WAN latency from a target such as `100.90.121.70`:

```bash
scripts/skippy-wan-calibrate.sh 100.90.121.70 skippy/evals/wan-lab/.env.link
```

The script writes `WAN_RTT_MS` and `WAN_DELAY_MS`. If the target has an `iperf3`
server, it also records `WAN_RATE_MBIT`; otherwise fill bandwidth manually if
you want rate limiting as well as latency.

## Run

Use the launcher, not raw `docker compose`, so the model invariant is enforced:

```bash
just --command bash skippy/evals/wan-lab/up.sh
```

Before compose starts, the launcher uses the existing source-built native
package recipes. To inspect an exact requested ref without network acquisition:

```bash
just skippy-package-reference "$MODEL_PACKAGE_REF"
just skippy-layer-package-cache --reference "$MODEL_PACKAGE_REF" --cache-root "$HUB_ROOT"
just skippy-layer-package-inspect --package "$SNAPSHOT_PATH" \
  --expected-layer-count "$LAYER_COUNT" --expected-activation-width "$ACTIVATION_WIDTH"
```

`HUB_ROOT` is the Hub cache directory, not its `HF_HOME` parent. Cache lookup
requires the requested ref's exact immutable commit and admitted snapshot;
it makes no HTTP requests and does not fall back to another cached revision.
The inspector checks the complete declared package closure and source geometry.
Use the package's actual layer count and activation width; do not substitute
example numbers.

Explicit acquisition uses the same native owner:

```bash
just skippy-layer-package-fetch --reference "$MODEL_PACKAGE_REF" --cache-root "$HUB_ROOT" \
  --expected-layer-count "$LAYER_COUNT" --expected-activation-width "$ACTIVATION_WIDTH" \
  --timeout-secs 3600
```

The launcher reuses a fully verified requested cache entry or acquires the
package at one resolved immutable commit. It verifies artifact identities,
byte sizes, and complete package geometry before starting containers, then
exports that exact commit in `MODEL_PACKAGE_REF`. Fetch uses an owned native
worker with bounded execution/capture and cleanup; failure prevents serving.
It preserves valid Hugging Face blob symlinks inside the admitted cache.

Stage artifact selection uses the native `plan-layer-package-artifacts`
command inside the image, with the manifest, stage index/count, and declared
layer range. The planner shares the package manifest validators and includes
required shared/head/tail/projector artifacts. Planning declares files; it does
not prove that they were downloaded. Containers read the host's admitted cache
and refuse incomplete stage inputs before launch. Acquisition and fixture
validation are separate from actual model inference or WAN benchmark proof.

## Interactive Prompt

With the lab running in one terminal, attach a `skippy prompt` REPL to
stage0 from another terminal:

```bash
skippy/evals/wan-lab/prompt.sh
```

Extra REPL flags are passed through to `skippy prompt`, for example:

```bash
skippy/evals/wan-lab/prompt.sh --max-new-tokens 64 --no-think
```

The prompt helper does not start containers or download the model. It attaches
to the existing `stage0` container and sends requests to its OpenAI endpoint.

The OpenAI-compatible endpoint is:

```bash
curl -s http://127.0.0.1:9337/v1/models | jq
```

Example request:

```bash
curl -s http://127.0.0.1:9337/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M",
    "messages": [{"role": "user", "content": "Write one short sentence."}],
    "max_tokens": 16
  }' | jq
```

## Metrics

Stage telemetry is emitted to `metrics-server` using OTLP/gRPC:

```bash
curl -s http://127.0.0.1:18080/v1/runs/skippy-docker-wan/status | jq
curl -s -X POST http://127.0.0.1:18080/v1/runs/skippy-docker-wan/finalize | jq
curl -s http://127.0.0.1:18080/v1/runs/skippy-docker-wan/report.json | jq
```

The DuckDB file is stored in the `metrics_data` Docker volume.

## Inspect Traffic Control

```bash
docker compose \
  --env-file skippy/evals/wan-lab/.env \
  --env-file skippy/evals/wan-lab/.env.link \
  -f skippy/evals/wan-lab/docker-compose.yml \
  exec stage0 tc -s qdisc
```

## Notes

- The actual host Hub cache root is mounted read-only at `/hf-cache`.
- The default load mode is `layer-package` with tensor filtering enabled.
- CPU execution is forced with `n_gpu_layers: 0`.
- Use `WAN_ENABLE=0` to disable shaping without changing the compose topology.
- Docker Desktop runs Linux containers inside a Linux VM; `tc netem` is applied
  inside that VM.

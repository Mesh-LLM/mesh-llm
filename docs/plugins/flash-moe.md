# Flash-MoE Plugin

The external `flash-moe` plugin connects mesh-llm to a Flash-MoE OpenAI-compatible HTTP server. The plugin now lives at [Mesh-LLM/flash-moe](https://github.com/Mesh-LLM/flash-moe).

Use it for the SSD expert streaming roadmap path: a giant MoE model fits on one node's local NVMe, but not in RAM. The plugin owns Flash-MoE process lifecycle when configured for managed mode. Mesh-llm only launches the plugin through the plugin API, consumes its declared endpoint, and routes requests. Flash-MoE owns model execution.

This is intentionally a single-node backend adapter. It does not change the mesh protocol, Skippy stage protocol, model-package format, or llama.cpp patch queue.

## Prerequisites

Flash-MoE is an external backend. Mesh-llm does not vendor, install, or build the Flash-MoE binary, model conversion tooling, or SSD-streaming artifacts.

Install the mesh-llm plugin:

```bash
mesh-llm plugins install flash-moe
```

You can also install directly from GitHub:

```bash
mesh-llm plugins install Mesh-LLM/flash-moe
```

Install or build Flash-MoE separately from [danveloper/flash-moe](https://github.com/danveloper/flash-moe), prepare the model files with its tooling, then either:

- pass plugin args for managed process mode; or
- set `url` to an already-running Flash-MoE `/v1` endpoint.

If upstream release artifacts are not available for your platform, use a source build or deployment-managed binary and pass that binary to the plugin with `--backend-command`.

## Managed Process Mode

Pass the Flash-MoE `infer` binary and normal model arguments to the plugin in
`args`.

```toml
[[plugin]]
name = "flash-moe"
args = [
  "--backend-command", "/opt/flash-moe/metal_infer/infer",
  "--",
  "--model", "/models/qwen3.5-397b/model.gguf",
  "--weights", "/models/qwen3.5-397b/experts.bin",
  "--manifest", "/models/qwen3.5-397b/manifest.json",
  "--vocab", "/models/qwen3.5-397b/vocab.json"
]
```

The plugin allocates a local port and appends:

```text
--serve <port>
```

Do not pass `--serve` in backend args. Keeping the port plugin-owned prevents collisions between plugins, external backends, and local llama.cpp serving.

When the plugin starts, it registers an OpenAI-compatible inference endpoint like:

```text
http://127.0.0.1:<port>/v1
```

The host probes `GET /v1/models` through the normal plugin endpoint health path, so Flash-MoE models appear and disappear the same way other plugin-backed models do.

## Existing Endpoint Mode

If Flash-MoE is already running, attach the endpoint instead of letting mesh-llm spawn it:

```toml
[[plugin]]
name = "flash-moe"
url = "http://127.0.0.1:8000/v1"
```

This mode leaves process lifecycle outside mesh-llm and only registers the endpoint.

## Config Rules

- Configure either `url` or plugin args with `--backend-command`, not both.
- `--serve` is plugin-owned and must not appear in backend args.
- Model paths and weights stay local to the node running Flash-MoE.
- No HuggingFace token or private credential is required by the plugin itself.

## Scope

Included:

- external `flash-moe` plugin entrypoint
- managed Flash-MoE process launch
- existing HTTP endpoint attachment
- OpenAI-compatible endpoint registration
- plugin health and lifecycle checks
- model discovery via the existing `/v1/models` probe path

Not included:

- mesh-distributed SSD expert streaming
- Skippy package slicing changes
- Flash-MoE binary vendoring
- Flash-MoE installation or source-build automation
- Flash-MoE model conversion or download automation
- changes to public mesh protocol fields

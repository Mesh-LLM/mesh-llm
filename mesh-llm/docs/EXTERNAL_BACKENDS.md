# External Inference Backends

mesh-llm can route requests to any OpenAI-compatible inference server —
vLLM, TGI, Ollama, or anything else that speaks the `/v1/chat/completions`
and `/v1/models` API. The external server does all the heavy lifting
(model loading, GPU management, batching, quantization). mesh-llm just
makes it available through the mesh.

## Quick start

Add the backend to `~/.mesh-llm/config.toml`:

```toml
[[plugin]]
name = "vllm"
url = "http://gpu-box:8000"
```

Then start mesh-llm normally:

```bash
mesh-llm serve
```

mesh-llm's plugin system probes `/v1/models` on the backend, discovers what
models it serves, and makes them available through the local API. Requests
for those models are routed to the backend. Health checks run automatically —
if the backend goes down, its models are withdrawn; when it comes back,
they reappear.

## How it works

This uses the same inference endpoint plugin system that powers Lemonade
integration. A plugin entry with a `url` field (and no `command`) is
automatically treated as an OpenAI-compatible inference endpoint:

1. mesh-llm spawns the built-in `openai-endpoint` plugin.
2. The plugin registers the URL as an inference endpoint.
3. The plugin health check system periodically probes `GET /v1/models`
   and discovers which models the backend serves.
4. When a request arrives for one of those models, the proxy routes it
   to the backend URL.
5. The response streams back to the client unmodified.

## What mesh-llm does NOT do

- Does not start, stop, restart, or manage the external server.
- Does not download models for the external server.
- Does not configure batching, quantization, or tensor parallelism.
- Does not translate between API formats.

The external server is a black box. mesh-llm sends HTTP requests and relays
responses verbatim.

## Examples

### vLLM

```bash
# Start vLLM (your responsibility)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

```toml
# ~/.mesh-llm/config.toml
[[plugin]]
name = "vllm"
url = "http://localhost:8000"
```

```bash
mesh-llm serve
```

### Ollama

```bash
# Start Ollama (your responsibility)
ollama serve
```

```toml
# ~/.mesh-llm/config.toml
[[plugin]]
name = "ollama"
url = "http://localhost:11434"
```

```bash
mesh-llm serve
```

### TGI (Text Generation Inference)

```bash
# Start TGI (your responsibility)
text-generation-launcher --model-id meta-llama/Llama-3.1-8B-Instruct --port 8000
```

```toml
# ~/.mesh-llm/config.toml
[[plugin]]
name = "tgi"
url = "http://localhost:8000"
```

### Remote server

```toml
# ~/.mesh-llm/config.toml
[[plugin]]
name = "cloud-gpu"
url = "http://gpu-server.internal:8000"
```

### Multiple backends

```toml
# ~/.mesh-llm/config.toml
[[plugin]]
name = "vllm-large"
url = "http://gpu-box-1:8000"

[[plugin]]
name = "vllm-small"
url = "http://gpu-box-2:8000"
```

Each backend's models appear independently. The proxy routes by model name.

### Alongside local llama.cpp models

```toml
# ~/.mesh-llm/config.toml

# Local model served by mesh-llm's built-in llama.cpp
[[models]]
model = "Qwen3-8B-Q4_K_M"

# External vLLM server for a bigger model
[[plugin]]
name = "vllm"
url = "http://gpu-box:8000"
```

Both are available through the same API at `http://localhost:9337/v1`.

## Plugin name

The `name` field in the config is just an identifier — call it whatever you
want. It shows up in `mesh-llm plugin list` and in logs. Common choices:

- `vllm`, `ollama`, `tgi` — when the backend type is the identity
- `gpu-box`, `cloud-inference` — when the machine is the identity
- `llama-70b` — when the model is the identity

## Health checks

The plugin system probes `GET /v1/models` on the backend periodically. If
the backend is unreachable or returns an error, its models are removed from
the routing table. When it recovers, they come back automatically.

No manual intervention is needed to handle backend restarts.

## Testing locally

A mock server is included for development and testing:

```bash
python3 tools/mock-vllm.py 8000
```

Then configure it:

```toml
[[plugin]]
name = "mock"
url = "http://localhost:8000"
```

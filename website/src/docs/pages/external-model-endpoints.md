---
title: Share Ollama, vLLM or LM Studio
---

# Share Ollama, vLLM or LM Studio

**Start the server → install the plugin → set its URL → run `mesh-llm serve`.**
You do not need a second model download, a GGUF, or a separate `share` command.
These examples assume Mesh and the provider run on the same machine and that
Mesh's default ports (9337 API, 3131 console) are free. Start with a fresh/default
private-mesh configuration; existing discovery/publication settings still apply.
Do not add `--auto` or `--publish` for this private recipe.

### 1. Start your provider

If your server already works, leave it running and use its existing **HTTP** base URL.
This forwarding path does not support a direct `https://` upstream, even when
the HTTPS model-health probe succeeds. See the gateway limitation below.
Otherwise choose one recipe below. Install the provider first using its own
documentation; model downloads and hardware requirements belong to that provider.

| Provider | Start it | Plugin `url` |
| --- | --- | --- |
| Ollama | Start the Ollama app, or run `ollama serve`; in another terminal run `ollama pull llama3.2:1b` (or use a model you already have). | `http://127.0.0.1:11434/v1` |
| vLLM | On a supported machine: `vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000` (or substitute your model). | `http://127.0.0.1:8000/v1` |
| LM Studio | Download/select a chat model, load it, then open **Developer → Start server**. With the CLI installed, `lms server start` starts the API server. | `http://127.0.0.1:1234/v1` |
| Other OpenAI-compatible servers | Start your existing TGI, SGLang, Lemonade, or other server using its own instructions. Copy its HTTP OpenAI base URL, including any prefix before `/v1`. | The server's actual base URL; do not assume port 8000. |

The small model names above are examples, not required models. For Ollama use
`/v1`, **not** its native `/api` API. For LM Studio use the server's displayed
port if it differs from 1234; Just-In-Time loading may list downloaded models
that are not loaded yet.

Check the provider directly before adding Mesh (Ollama example; change the URL
for your provider):

```bash
curl --fail --silent --show-error http://127.0.0.1:11434/v1/models
```

Expect a JSON `data` array containing at least one model `id`. For this simple
recipe the endpoint must be reachable without upstream API-key authentication.
Keep an unauthenticated server loopback-only; do not disable authentication on
an existing shared service. See the API-key FAQ below.

### 2. Install the compatible adapter

Check compatibility first: **adapter 0.1.2 is incompatible with Mesh 0.76.0**
(protocol 2 versus 3). For Mesh 0.76.0, use a protocol-3 adapter build.
The installer selects a platform archive, not a negotiated protocol match.
Once a compatible release is available:

```bash
mesh-llm plugins install openai-endpoint
```

If the compatible release is not yet published, follow the [adapter source-build instructions](https://github.com/Mesh-LLM/openai-endpoint/blob/ea568baff71037badb8e5c7e479c33c0082412b9/README.md#build-from-source)
instead. Do not repeatedly reinstall 0.1.2 on Mesh 0.76.0.

### 3. Set the URL once

Create `~/.mesh-llm/config.toml` if absent, or edit your existing file. For Ollama:

```toml
[runtime]
mode = "on_demand"

[[plugin]]
name = "openai-endpoint"
url = "http://127.0.0.1:11434/v1"
```

For vLLM or LM Studio, change **only the URL** to the value in the table.
Merge into an existing `[runtime]` section and existing `openai-endpoint` entry;
do not append duplicate tables. No `[[models]]` entry is needed for the provider.
`on_demand` avoids eagerly loading configured native models; it does not disable
native-runtime initialization or future native serving. Explicit `--model` or
`--gguf` arguments still request loading, so omit them here.

### 4. Serve and check

```bash
mesh-llm serve
```

Leave it running. In another terminal:

```bash
curl --fail --silent --show-error http://127.0.0.1:9337/v1/models
```

Copy an exact `id` from **Mesh's** response into `model` below:

```bash
curl --fail --silent --show-error http://127.0.0.1:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"REPLACE_WITH_LISTED_ID","messages":[{"role":"user","content":"Say hello."}],"max_tokens":64,"stream":false}'
```

Success is a completion response with `choices`, not merely a healthy plugin
process. Your application now uses `http://127.0.0.1:9337/v1` as its OpenAI base
URL. The provider keeps managing its own models, GPU memory and process.

## Use it from another machine

On a second machine with Mesh installed, join using the invite token printed by
the serving node:

```bash
mesh-llm client --join YOUR_INVITE_TOKEN
```

On that second machine, repeat the two curl checks above against its own port
9337. It needs neither the endpoint plugin nor a copy of the provider's model.
Keep the invite private.

Only the serving node needs to reach the upstream URL. The second machine does
**not** connect to your Ollama/vLLM/LM Studio HTTP port directly. If testing two
Mesh processes on one machine instead, use separate profiles and distinct API
and console ports; do not start a second process over a running instance.

## FAQ and troubleshooting

### Does Mesh start or stop Ollama, vLLM or LM Studio?

No. Mesh manages the small adapter process, not your model server. The adapter
registers the URL and Mesh forwards inference to it directly. Keep your provider
running; stopping Mesh does not stop that provider.

### Why not `mesh-llm client` on the provider machine?

Use `serve` on the machine sharing the endpoint. `client` is for consuming a
mesh, not advertising this machine as an inference host. The consumer machine
can use `client --join` without installing the plugin.

### Is “OpenAI-compatible” an OpenAI account requirement?

No. It describes the HTTP API format. These local recipes need no OpenAI account
or OpenAI API key. Use `openai-endpoint` as the install/config name, regardless of its display title.

### What about upstream API keys?

This adapter reads a URL only; it has no provider API-key/header configuration.
Mesh's endpoint health/model probe makes its own request to `/v1/models` without
an injected upstream bearer token. Setting `OPENAI_API_KEY` in your app does not
configure that probe. Do not put credentials in the URL or assume caller
Authorization headers will authenticate mesh-wide forwarding.

Use the loopback-only unauthenticated recipe for a local server you control.
An HTTPS or protected provider needs a separately secured gateway exposing
**loopback HTTP to Mesh** and handling upstream TLS/authentication for both
model discovery and inference; that setup is outside this quick
start and must be validated separately. Do not expose an unauthenticated gateway
to the LAN or Internet to work around this limitation. Do not disable TLS or
authentication on an existing shared provider.

### Why is the plugin healthy but my model missing?

Plugin-process health and upstream readiness are separate. Check in this order:

1. `mesh-llm plugins info openai-endpoint`: correct install/version and executable?
2. Provider `BASE_URL/models`: reachable from the serving node, HTTP 200, nonempty
   `data` with model IDs? A 401/403 indicates authentication; a 404 often means a
   wrong base path; connection refused usually means wrong port or stopped server.
3. Mesh `http://127.0.0.1:9337/v1/models`: allow time for health checks/discovery.
4. Completion: use the exact listed ID and a chat-capable model. A model listing
   alone does not prove generation; loading or a missing chat template can fail.
   An `https://` upstream may pass discovery but fail generation: this host
   forwarding path requires HTTP (see the secured gateway limitation above).

Do not point the plugin at Mesh's own 9337 endpoint: that creates a routing loop.
`plugins info` describes the installed package, which may not be the executable
in use when config contains a `command` override. Check that absolute path and
the startup handshake too; remove the override when switching back to a
compatible released package.

Restart your own Mesh instance after changing the plugin URL. If the provider is
in Docker or on another host, `127.0.0.1` means the Mesh process's own network
namespace, not that other host/container.

### Why does startup say “uses protocol 2, host uses 3”?

The adapter and host releases are incompatible, even if installation succeeded.
Use a protocol-3 build for Mesh 0.76.0; retain adapter 0.1.2 for protocol-2 hosts.
Do not bypass the handshake. See the [adapter compatibility notes](https://github.com/Mesh-LLM/openai-endpoint/blob/ea568baff71037badb8e5c7e479c33c0082412b9/README.md#compatibility).

### Does this split my provider's model across Mesh GPUs?

No. It shares an existing inference endpoint. vLLM/Ollama/LM Studio retain their
own execution and parallelism; attaching their URL does not convert their models
into Mesh layer packages or distribute their weights.

### Are streaming, tools, vision and every provider certified?

No blanket guarantee. Features depend on the provider/model and Mesh's routing
path. The compatibility repair was validated on Mesh 0.76.0 with local archive
installation and non-streaming HTTP-fixture requests through two processes on
one Mac. The recipes are based on provider documentation, not a claim of live
certification for every named provider, platform, streaming or authentication mode.

## Provider references

- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [LM Studio server](https://lmstudio.ai/docs/developer/core/server) and [OpenAI compatibility](https://lmstudio.ai/docs/developer/openai-compat)
- [Lemonade API reference](https://github.com/lemonade-sdk/lemonade/tree/main/docs/api)

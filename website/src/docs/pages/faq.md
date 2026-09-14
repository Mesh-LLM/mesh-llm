# FAQ

## Is Mesh a model provider?

No. Mesh runs models through machines you control, then exposes them through a local OpenAI-compatible API.

## Do I need multiple machines?

No. Start with one machine. Add machines later when you want more capacity, more models, or an API-only client laptop.

## Does Mesh require a local model?

No. Bare `mesh-llm serve` can run as a healthy idle daemon or route only to
plugin and remote endpoints. Load a local model later if the node is
worker-capable. See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/).

## Can I keep using Ollama, vLLM or LM Studio?

Yes. Use the `openai-endpoint` plugin: start your existing server, install a
host-compatible adapter, set its URL in one `[[plugin]]` entry, then run
`mesh-llm serve`. No second model download and no separate `share` command.
The [provider quick start](/docs/pages/external-model-endpoints/) has the exact
Ollama, vLLM and LM Studio recipes, including the 0.1.2/0.76.0 compatibility warning.

## Does the plugin launch my model server or split its weights?

Neither. You keep managing your provider and its models. The plugin registers
its endpoint; Mesh discovers model IDs and routes inference to it. It does not
turn an Ollama/vLLM/LM Studio model into a distributed Mesh layer package.

## Do I install the endpoint plugin on every machine?

No. Install it on the node sharing the provider and use `serve` there. A consuming
machine only needs `mesh-llm client --join YOUR_INVITE_TOKEN`; applications use
that machine's `http://127.0.0.1:9337/v1` endpoint.

## Can the endpoint plugin inject my provider API key?

Not with its current URL-only configuration. Model discovery/health requests
also need upstream authorization; an application's `OPENAI_API_KEY` does not
configure those requests. Use a local loopback-only server for the simple recipe;
do not remove authentication from an existing shared service. See the
[API-key limitations](/docs/pages/external-model-endpoints/#what-about-upstream-api-keys).

## What is the difference between client and on-demand mode?

`client` is routing-only and disables local model loading. `on_demand` starts
without eagerly loading configured models but retains the ability to load one
later. Explicit `--model` and `--gguf` arguments remain eager in `on_demand`.

## Does pausing inference unload the model?

No. Activity policy pauses admission or reduces priority. The model stays
loaded and can resume without a reload. Use unload or drain when you intend to
remove a model process.

## Why is `/v1/models` empty when the daemon is healthy?

The endpoint lists currently available local, plugin, and remote routes. A
`ready_idle` daemon has its durable surfaces ready but has no route to list
yet, so an empty `data` array is valid.

## What URL do tools use?

Use:

```text
http://localhost:9337/v1
```

The console is separate:

```text
http://localhost:3131
```

## What model should I start with?

Use the [model picker](/docs/pages/choose-a-model/). If you are unsure, start smaller. A model that loads and responds is more useful than a larger model that fails during setup.

## What is a layer package?

A layer package is a prepared model artifact Mesh can use for multi-machine serving. You do not need layer packages for the first run.

## Should I use the public mesh first?

Use a private mesh first if you are testing your install. Use the public mesh when you specifically want public discovery behavior.

## Can I use existing agent tools?

Yes. Use the [Coding agents](/docs/pages/agents/) page after console chat works.

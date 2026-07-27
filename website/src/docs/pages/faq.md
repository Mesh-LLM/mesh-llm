# FAQ

## Is Mesh a model provider?

No. Mesh runs models through machines you control, then exposes them through a local OpenAI-compatible API.

## Do I need multiple machines?

No. Start with one machine. Add machines later when you want more capacity, more models, or an API-only client laptop.

## Does Mesh require a local model?

No. Bare `mesh-llm serve` can run as a healthy idle daemon or route only to
plugin and remote endpoints. Load a local model later if the node is
worker-capable. See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/).

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

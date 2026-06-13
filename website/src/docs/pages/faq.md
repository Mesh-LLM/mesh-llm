# FAQ

## Is Mesh a model provider?

No. Mesh runs models through machines you control, then exposes them through a local OpenAI-compatible API.

## Do I need multiple machines?

No. Start with one machine. Add machines later when you want more capacity, more models, or an API-only client laptop.

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

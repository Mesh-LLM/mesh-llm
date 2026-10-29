# Node.js SDK

Install the package with `npm install @mesh-llm/sdk`. When developing from this
checkout, run `npm run build:native` in `mesh/sdk/node` before using the SDK.

The npm package exposes one embedded `Node` with `client`, `serve`, and
`combined` roles. `serve` is serve-only.

```js
const { Node } = require('@mesh-llm/sdk')
const node = Node.create({ mode: 'client', autoJoin: true })
await node.start()
const models = await node.inference.listModels()
const response = await node.inference.chatCompletions({
  model: models[0].id,
  messages: [{ role: 'user', content: 'Hello' }]
})
await node.stop()
```

Use `joinTokens: [token]` for a private mesh. To serve local models, choose
`mode: 'serve'` or `'combined'` and set `models: [modelRef]`. Serving requires a
matching native runtime. `ownerKeyPath` accepts a Mesh LLM keystore path. The
old `Client` and owner keypair hex constructor are removed.

Stream with `node.inference.streamChatCompletions(body)` or
`streamResponses(body)`. See [runtime artifacts](../SDK.md#native-runtime-artifacts).

## Thinking effort levels

`listModels()` returns model metadata in the `/v1/models` shape, including the
effort strings observed in the loaded chat template. The native addon currently
queries this metadata through the embedded node's HTTP client:

```js
const models = await node.inference.listModels()
const chosen = models.find((model) => model.metadata?.thinking_efforts?.includes('high'))
if (chosen) {
  const response = await node.inference.chatCompletions({
    model: chosen.id,
    messages: [{ role: 'user', content: 'Explain lock ordering.' }],
    reasoning_effort: 'high'
  })
}
```

`metadata.thinking_efforts` is absent when the provider is older or has not
probed the model, and `[]` when the template exposes no effort levels. The list
is an observation of the template, not a guarantee that generation obeys it.

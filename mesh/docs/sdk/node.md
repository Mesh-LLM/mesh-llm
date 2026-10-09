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

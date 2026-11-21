'use strict'
const assert = require('node:assert/strict')
const test = require('node:test')
const { decodeStatus, StatusEvents } = require('../status')

function snapshot(state, download = null) {
  return JSON.stringify({ state, serving_models: ['chat'],
    models: [{ id: 'chat#profile', providers: [] }],
    local_models: [{ identity: { model_name: 'chat', weights_hash: 'digest' } }],
    capacities: [{ model: 'chat', context_per_lane: [4096, 8192], last_success_unix_ms: 123 }], download })
}

test('native status retains lifecycle, download bytes, catalog, identity and lane capacity', () => {
  const status = decodeStatus(snapshot({ phase: 'downloading' }, { file: 'weights', downloaded_bytes: 10, total_bytes: null }))
  assert.deepEqual(status.download, { file: 'weights', downloadedBytes: 10, totalBytes: null })
  assert.equal(status.models[0].id, 'chat#profile')
  assert.equal(status.localModels[0].identity.weights_hash, 'digest')
  assert.deepEqual(status.capacities[0], { model: 'chat', contextPerLane: [4096, 8192], lastSuccessUnixMs: 123 })
})

test('initial, ready and failed snapshots arrive in order and closure releases subscription', async () => {
  const states = ['loading', 'downloading', 'ready', 'failed'].map(phase => ({ phase, ...(phase === 'failed' ? { error: 'runtime failed' } : {}) }))
  const queue = states.map(state => snapshot(state))
  let closed = 0
  const events = new StatusEvents({ nextEventJson: async () => queue.shift() ?? null, close: async () => { closed++ } })
  const seen = []
  for await (const status of events) seen.push(status.state)
  assert.deepEqual(seen, states)
  assert.equal(closed, 1)
})

test('breaking iteration and native errors close the subscription', async () => {
  let closed = 0
  const events = new StatusEvents({ nextEventJson: async () => snapshot({ phase: 'ready' }), close: async () => { closed++ } })
  for await (const status of events) { assert.equal(status.state.phase, 'ready'); break }
  assert.equal(closed, 1)
  const failure = new Error('native source closed')
  const broken = new StatusEvents({ nextEventJson: async () => { throw failure }, close: async () => { closed++ } })
  await assert.rejects(async () => { for await (const _ of broken) {} }, error => error === failure)
  assert.equal(closed, 2)
})

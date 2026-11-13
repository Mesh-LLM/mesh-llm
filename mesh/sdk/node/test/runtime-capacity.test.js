'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')

test('capacity reads forward exact IDs, changes, unavailable state and native errors', async () => {
  const module = { exports: {} }
  const directory = path.resolve(__dirname, '..')
  const fakeProcess = { env: { MESHLLM_NODE_NATIVE_PATH: '/fixture/native.node' },
    platform: process.platform, arch: process.arch,
    dlopen(nativeModule) { nativeModule.exports = {} } }
  vm.runInNewContext(fs.readFileSync(path.join(directory, 'index.js'), 'utf8'), {
    module, exports: module.exports, __dirname: directory,
    require: id => id.startsWith('.') ? require(path.resolve(directory, id)) : require(id),
    process: fakeProcess
  })
  const initial = { executionLanes: 2n, contextTokensPerLane: (1n << 63n) - 1n,
    contextPoolTokens: (1n << 64n) - 2n, capturedAtUnixMs: (1n << 64n) - 1n }
  const smaller = { executionLanes: 1n, contextTokensPerLane: 4096n,
    contextPoolTokens: 4096n, capturedAtUnixMs: 123n }
  let current = initial
  const failure = new Error('node is not running')
  const node = new module.exports.Node({ async runtimeCapacity(model) {
    assert.equal(model, 'org/model@revision#profile')
    if (current === failure) throw failure
    return current
  } })
  assert.deepEqual(await node.runtimeCapacity('org/model@revision#profile'), initial)
  current = smaller
  assert.deepEqual(await node.runtimeCapacity('org/model@revision#profile'), smaller)
  current = null
  assert.equal(await node.runtimeCapacity('org/model@revision#profile'), null)
  current = failure
  await assert.rejects(node.runtimeCapacity('org/model@revision#profile'), error => error === failure)
})

'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')

function nodeClass(factory) {
  const module = { exports: {} }
  const directory = path.resolve(__dirname, '..')
  const fakeProcess = { env: { MESHLLM_NODE_NATIVE_PATH: '/fixture/native.node' },
    platform: process.platform, arch: process.arch,
    dlopen(nativeModule) { nativeModule.exports = { Node: { createConfig: factory } } } }
  vm.runInNewContext(fs.readFileSync(path.join(directory, 'index.js'), 'utf8'), {
    module, exports: module.exports, __dirname: directory,
    require: id => id.startsWith('.') ? require(path.resolve(directory, id)) : require(id),
    process: fakeProcess
  })
  return module.exports.Node
}

test('full config constructor forwards all nested Rust configuration without filtering', () => {
  const config = { mode: 'client', network: { mesh_name: 'team', iroh_relay_auth: { relay: 'value' },
    enumerate_host: false }, admission: { trust_policy: 'allowlist', trusted_owners: ['owner'],
    mesh_requirements: { min_protocol_version: 1 } },
  storage: { isolated_config: true }, startup_timeout: { secs: 10, nanos: 0 }, future_field: true }
  const handle = {}
  const Node = nodeClass(source => { assert.deepEqual(JSON.parse(source), config); return handle })
  assert.equal(Node.fromConfig(config)._handle, handle)
})

test('full config constructor preserves native validation failures', () => {
  const failure = new Error('unknown field future_field')
  const Node = nodeClass(() => { throw failure })
  assert.throws(() => Node.fromConfig({ future_field: true }), error => error === failure)
})

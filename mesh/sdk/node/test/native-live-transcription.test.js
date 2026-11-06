'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const fs = require('node:fs')
const path = require('node:path')
const crypto = require('node:crypto')
const { setTimeout: delay } = require('node:timers/promises')

const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex')
const write = (root, name, value) => fs.writeFileSync(path.join(root, name), JSON.stringify(value, null, 2))

async function bounded(promise, milliseconds, operation) {
  let timer
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${operation} timed out`)), milliseconds)
    })])
  } finally { clearTimeout(timer) }
}

function loadedIdentity(evidence, phase) {
  const expected = fs.realpathSync(process.env.MESHLLM_NODE_NATIVE_PATH)
  assert.equal(digest(fs.readFileSync(expected)), process.env.MESH_SDK_NATIVE_SHA256)
  // Persist only mapped image names, never the report's environment or headers.
  const images = process.report.getReport().sharedObjects
    .filter(file => path.basename(file) === path.basename(expected) || /^lib(ggml|llama|mtmd)/.test(path.basename(file)))
    .map(file => fs.realpathSync(file))
  assert.ok(images.includes(expected), 'the expected addon is actually mapped')
  const selected = images.filter(file => file === expected || /^lib(ggml|llama|mtmd)/.test(path.basename(file)))
  const mapped = selected.map(file => ({ path: file, sha256: digest(fs.readFileSync(file)) }))
  if (phase === 'model-loaded') {
    const bundle = fs.realpathSync(process.env.MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR)
    const manifest = fs.realpathSync(process.env.MESH_SDK_RUNTIME_MANIFEST)
    const relative = path.relative(bundle, manifest)
    assert.ok(relative && !relative.startsWith('..') && !path.isAbsolute(relative), 'manifest must belong to the selected runtime bundle')
    const manifestBytes = fs.readFileSync(manifest)
    assert.match(process.env.MESH_SDK_RUNTIME_MANIFEST_SHA256 || '', /^[a-f0-9]{64}$/)
    assert.equal(digest(manifestBytes), process.env.MESH_SDK_RUNTIME_MANIFEST_SHA256)
    const hashes = Object.values(JSON.parse(manifestBytes).runtime.files)
    assert.ok(mapped.some(file => path.basename(file.path).startsWith('libllama.')))
    for (const file of mapped.filter(file => file.path !== expected)) assert.ok(hashes.includes(file.sha256), file.path)
  }
  write(evidence, `${phase}-identity.json`, { pid: process.pid, addon: expected, mapped,
    symbolAddressIdentity: 'not exposed by this JavaScript API' })
}

function assertSpeech(events) {
  const deltas = events.filter(event => event.type === 'delta').map(event => event.text).join('')
  const completed = events.filter(event => event.type === 'completed')
  assert.equal(completed.length, 1)
  assert.equal(deltas, completed[0].transcript)
  assert.ok(events.some(event => event.type === 'tentative' && event.text.trim()))
  assert.ok(!events.some(event => event.type === 'error'))
  assert.ok(events.findIndex(event => event.type === 'completed') > events.findLastIndex(event => event.type === 'delta'))
  const words = (completed[0].transcript.toLowerCase().match(/\b\w+\b/g) || []).join(' ')
  assert.match(words, /fellow americans.*ask not.*can do for you.*ask what you can do for your country/)
  for (const phrase of ['fellow americans', 'can do for you', 'what you can do', 'do for your country']) {
    assert.equal([...words.matchAll(new RegExp(`\\b${phrase}\\b`, 'g'))].length, 1, phrase)
  }
}

async function speechSession(node, model, pcm, evidence) {
  const options = { model, sampleRate: 16000, language: 'en', turnDetection: null, stepMs: 400, tentative: true }
  const session = await bounded(node.startLiveTranscription(options), 35000, 'start speech')
  const events = []
  const started = performance.now()
  const record = event => {
    events.push(event)
    fs.appendFileSync(path.join(evidence, 'speech.jsonl'), JSON.stringify({ elapsedMs: performance.now() - started, event }) + '\n')
  }
  const reader = (async () => { for (;;) {
    const event = await session.nextEvent()
    if (event == null) return
    record(event)
  } })()
  try {
    for (let offset = 0; offset < pcm.length; offset += 3200) {
      const chunk = pcm.subarray(offset, offset + 3200)
      const samples = Array.from({ length: chunk.length / 2 }, (_, index) => chunk.readInt16LE(index * 2))
      await bounded(session.sendAudio(samples), 10000, 'send audio')
      await delay(100)
    }
    write(evidence, 'sent-audio.json', { samples: pcm.length / 2, sends: 110, sha256: digest(pcm), options })
    const tail = await bounded(session.stop(), 20000, 'stop speech')
    await bounded(reader, 5000, 'event reader')
    for (const event of tail) record(event)
    write(evidence, 'transcript.json', { transcript: events.filter(event => event.type === 'completed').map(event => event.transcript).join('') })
    assertSpeech(events)
    await assert.rejects(() => bounded(session.stop(), 2000, 'repeat stop'), /closed/i)
    assert.equal(await bounded(session.nextEvent(), 2000, 'ended reader'), null)
  } finally {
    await bounded(session.close(), 3000, 'close speech')
    await bounded(reader, 3000, 'closed event reader')
    await bounded(session.close(), 3000, 'repeat close')
  }
}

async function closeBlockedRead(node, model) {
  const session = await bounded(node.startLiveTranscription({ model }), 35000, 'start blocked session')
  let settled = false
  const pending = session.nextEvent().finally(() => { settled = true })
  try {
    await delay(200)
    assert.equal(settled, false)
    // JavaScript exposes no AbortSignal here. Native close is its cancellation.
    await bounded(session.close(), 2000, 'close blocked read')
    assert.equal(await bounded(pending, 2000, 'unblocked read'), null)
    assert.equal(await bounded(session.nextEvent(), 2000, 'repeat ended read'), null)
  } finally { await bounded(session.close(), 3000, 'repeat blocked close') }
}

test('actual native speech, discovery, option validation and cancellation', {
  skip: process.env.MESH_SDK_LIVE_SPEECH !== '1', timeout: 150000
}, async () => {
  assert.equal(process.env.MESH_LLM_NO_RUNTIME_DOWNLOAD, '1', 'native runtime downloads must be disabled')
  assert.ok(path.isAbsolute(process.env.MESH_SDK_STATE_ROOT), 'state root must be absolute')
  const state = path.resolve(process.env.MESH_SDK_STATE_ROOT)
  const actualState = fs.realpathSync(state)
  for (const name of ['MESH_LLM_CONFIG', 'MESH_LLM_RUNTIME_ROOT', 'MESH_LLM_DATA_DIR',
    'MESH_LLM_PLUGIN_DIR', 'MESH_LLM_NODE_KEY_PATH', 'MESH_LLM_NATIVE_RUNTIME_CACHE_DIR',
    'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_RUNTIME_DIR', 'HF_HUB_CACHE', 'HF_XET_CACHE']) {
    const value = process.env[name]
    assert.ok(value && path.isAbsolute(value), `${name} must be explicit and absolute`)
    const relative = path.relative(state, value)
    assert.ok(relative && !relative.startsWith('..') && !path.isAbsolute(relative), `${name} must be isolated`)
    if (fs.existsSync(value)) {
      assert.ok(fs.statSync(value).isDirectory(), `${name} must start absent or empty`)
      assert.deepEqual(fs.readdirSync(value), [], `${name} must start empty`)
      const actual = path.relative(actualState, fs.realpathSync(value))
      assert.ok(actual && !actual.startsWith('..') && !path.isAbsolute(actual), `${name} must resolve inside its isolated root`)
    }
  }
  const inputs = process.env.MESH_SPEECH_INPUTS
  const evidence = process.env.MESH_SPEECH_EVIDENCE
  fs.mkdirSync(evidence, { recursive: true })
  const pcm = fs.readFileSync(path.join(inputs, 'jfk.s16le'))
  assert.equal(pcm.length, 352000)
  assert.equal(digest(pcm), 'a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9')
  const model = path.join(inputs, 'ggml-tiny.bin')
  assert.equal(digest(fs.readFileSync(model)), 'be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21')
  const { Node, currentMeshVersion } = require('../index')
  loadedIdentity(evidence, 'before-start')
  write(evidence, 'version.json', { version: currentMeshVersion() })
  const ports = Object.fromEntries(['API', 'CONSOLE', 'BIND'].map(kind => {
    const port = Number(process.env[`MESH_SDK_${kind}_PORT`])
    assert.ok(Number.isInteger(port) && port > 1024 && port < 65536, `explicit ${kind} port required`)
    return [kind, port]
  }))
  assert.equal(new Set(Object.values(ports)).size, 3, 'owned ports must be distinct')
  const node = Node.fromConfig({ mode: 'serve',
    http: { api_port: ports.API, console_port: ports.CONSOLE, console_ui: false },
    serving: { models: [model] }, storage: { isolated_config: true },
    network: { auto_join: false, publish: false, enumerate_host: true, bind_ip: '127.0.0.1', bind_port: ports.BIND,
      disable_iroh_relays: true, discovery_mode: 'mdns' }, startup_timeout: { secs: 60, nanos: 0 } })
  try {
    await bounded(node.start(), 75000, 'node start')
    assert.equal((await node.status()).running, true)
    const deadline = performance.now() + 45000
    let modelId
    for (;;) {
      const page = await bounded(node.models.list({ workloadClass: 'speech_recognition', available: true }), 10000, 'native discovery')
      const loaded = page.data.find(model => model.kind === 'model' && model.availability === 'available' &&
        model.operations.includes('live_transcription') && model.providers.some(provider => provider.local && provider.available && provider.runtime?.ready))
      if (loaded) {
        const exact = await bounded(node.models.get(loaded.id), 10000, 'native model lookup')
        assert.equal(exact.id, loaded.id)
        assert.equal(exact.kind, 'model')
        assert.equal(exact.workload_class, 'speech_recognition')
        assert.equal(exact.availability, 'available')
        assert.deepEqual(exact.operations, loaded.operations)
        assert.ok(exact.providers.some(provider => provider.local && provider.available && provider.runtime?.ready))
        modelId = exact.id
        write(evidence, 'native-discovery.json', { kind: exact.kind, workloadClass: exact.workload_class,
          availability: exact.availability, operations: exact.operations, localReady: true, modelId })
        break
      }
      assert.ok(performance.now() < deadline, 'loaded native Whisper must be advertised')
      await delay(100)
    }
    loadedIdentity(evidence, 'model-loaded')
    for (const stepMs of [199, 5001]) await assert.rejects(
      () => bounded(node.startLiveTranscription({ model: modelId, stepMs }), 10000, 'invalid cadence'), /200.*5000/)
    await speechSession(node, modelId, pcm, evidence)
    await closeBlockedRead(node, modelId)
  } finally {
    await bounded(node.stop(), 30000, 'node stop')
    const stopped = !(await bounded(node.status(), 3000, 'stopped node status')).running
    write(evidence, 'cleanup.json', { pid: process.pid, stopped, ports })
    assert.equal(stopped, true)
    const runtime = process.env.MESH_LLM_RUNTIME_ROOT
    if (fs.existsSync(runtime)) for (const file of fs.readdirSync(runtime, { recursive: true })) {
      if (path.basename(file) === 'skippy-native.log') {
        fs.copyFileSync(path.join(runtime, file), path.join(evidence, file.replaceAll(path.sep, '_')))
      }
    }
  }
})

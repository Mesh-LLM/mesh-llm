'use strict'
const assert = require('node:assert/strict')
const test = require('node:test')
const { startLiveTranscription } = require('../live-transcription')

test('live options preserve defaults and explicit push-to-talk while audio and events stay lossless', async () => {
  const calls = []
  const event = { type: 'completed', item_id: 'turn', transcript: 'hello' }
  const handle = {
    async startLiveTranscriptionJson(source) { calls.push(JSON.parse(source)); return this },
    async sendAudio(samples) { calls.push(samples) },
    async sendAudioF32(samples) { calls.push(samples) },
    async commit() { calls.push('commit') },
    async nextEventJson() { return JSON.stringify(event) },
    async stopJson() { return JSON.stringify([event]) },
    async close() { calls.push('close') }
  }
  await startLiveTranscription(handle)
  assert.deepEqual(calls.shift(), {})
  const live = await startLiveTranscription(handle, { sampleRate: 48000, turnDetection: null })
  assert.deepEqual(calls.shift(), { sample_rate: 48000, turn_detection: null })
  await live.sendAudio(new Int16Array([1, -2]))
  await live.sendAudioF32(new Float32Array([0.5]))
  await live.commit()
  assert.deepEqual(calls, [[1, -2], [0.5], 'commit'])
  assert.deepEqual(await live.nextEvent(), event)
  assert.deepEqual(await live.stop(), [event])
})

test('breaking event iteration closes the session and native errors propagate', async () => {
  let closed = false
  const live = await startLiveTranscription({ async startLiveTranscriptionJson() { return {
    async nextEventJson() { return '{"type":"delta","item_id":"a","text":"word"}' },
    async close() { closed = true },
    async sendAudio() { throw new Error('closed') }
  } } })
  for await (const event of live) { assert.equal(event.type, 'delta'); break }
  assert.equal(closed, true)
  await assert.rejects(live.sendAudio([]), /closed/)
})

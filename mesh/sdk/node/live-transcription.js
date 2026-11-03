'use strict'

/** A live session uses signed 16-bit mono PCM at its configured sample rate. */
class LiveTranscription {
  constructor(handle) { this._handle = handle }
  sendAudio(samples) { return this._handle.sendAudio(Array.from(samples)) }
  sendAudioF32(samples) { return this._handle.sendAudioF32(Array.from(samples)) }
  commit() { return this._handle.commit() }
  async nextEvent() {
    const source = await this._handle.nextEventJson()
    return source == null ? null : JSON.parse(source)
  }
  async stop() { return JSON.parse(await this._handle.stopJson()) }
  close() { return this._handle.close() }
  async *events() {
    try {
      for (;;) {
        const event = await this.nextEvent()
        if (event == null) return
        yield event
      }
    } finally { await this.close() }
  }
  [Symbol.asyncIterator]() { return this.events() }
}

async function startLiveTranscription(handle, options = {}) {
  const { sampleRate, turnDetection, ...rest } = options
  const wire = { ...rest, sample_rate: sampleRate }
  if (turnDetection !== undefined) {
    wire.turn_detection = turnDetection === null ? null : {
      threshold: turnDetection.threshold,
      prefix_padding_ms: turnDetection.prefixPaddingMs,
      silence_duration_ms: turnDetection.silenceDurationMs
    }
  }
  return new LiveTranscription(await handle.startLiveTranscriptionJson(JSON.stringify(wire)))
}

module.exports = { LiveTranscription, startLiveTranscription }

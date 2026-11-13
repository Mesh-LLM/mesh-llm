'use strict'

function decodeStatus(source) {
  const value = JSON.parse(source)
  return {
    state: value.state,
    servingModels: value.serving_models,
    models: value.models,
    localModels: value.local_models,
    capacities: value.capacities.map(item => ({
      model: item.model,
      contextPerLane: item.context_per_lane,
      lastSuccessUnixMs: item.last_success_unix_ms
    })),
    download: value.download == null ? null : {
      file: value.download.file,
      downloadedBytes: value.download.downloaded_bytes,
      totalBytes: value.download.total_bytes
    }
  }
}

/** One node generation. Close when finished; stopping the node also closes it. */
class StatusEvents {
  constructor(handle) { this._handle = handle }
  async nextEvent() {
    const source = await this._handle.nextEventJson()
    return source == null ? null : decodeStatus(source)
  }
  close() { return this._handle.close() }
  async *events() {
    try {
      for (;;) {
        const status = await this.nextEvent()
        if (status == null) return
        yield status
      }
    } finally { await this.close() }
  }
  [Symbol.asyncIterator]() { return this.events() }
}

module.exports = { decodeStatus, StatusEvents }

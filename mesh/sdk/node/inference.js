'use strict'

const MAX_STREAM_EVENTS = 256

class OpenAIRequestError extends Error {
  constructor(statusCode, body, message) {
    super(message || body || 'OpenAI-compatible request failed')
    this.name = 'OpenAIRequestError'
    this.statusCode = statusCode == null ? null : statusCode
    this.body = body == null ? null : body
  }
}

class AsyncEventQueue {
  constructor() {
    this._values = []
    this._waiters = []
    this._overflow = null
  }

  push(value) {
    if (this._overflow) return false
    const waiter = this._waiters.shift()
    if (waiter) {
      waiter(value)
      return true
    }
    if (this._values.length >= MAX_STREAM_EVENTS) {
      this._values = []
      this._overflow = new Error(`OpenAI stream consumer fell behind by ${MAX_STREAM_EVENTS} events`)
      return false
    }
    this._values.push(value)
    return true
  }

  next() {
    if (this._overflow) return Promise.reject(this._overflow)
    const value = this._values.shift()
    if (value) return Promise.resolve(value)
    return new Promise(resolve => this._waiters.push(resolve))
  }
}

class Inference {
  constructor(handle) {
    this._handle = handle
  }

  async listModels() {
    return parse(await this._handle.listModelsJson())
  }

  async request(path, body, options = {}) {
    const response = parse(await this._handle.openaiRequestJson(path, JSON.stringify(body)))
    const result = {
      statusCode: response.statusCode,
      contentType: response.contentType,
      body: response.body,
      json() {
        return parse(response.body)
      }
    }
    if (options.raiseForStatus !== false && (result.statusCode < 200 || result.statusCode >= 300)) {
      throw new OpenAIRequestError(result.statusCode, result.body)
    }
    return result
  }

  async chatCompletions(body) {
    return (await this.request('/v1/chat/completions', { ...body, stream: false })).json()
  }

  async responses(body) {
    return (await this.request('/v1/responses', { ...body, stream: false })).json()
  }

  async *stream(path, body) {
    const queue = new AsyncEventQueue()
    const requestId = await this._handle.openaiStream(
      path,
      JSON.stringify({ ...body, stream: true }),
      (error, eventJson) => {
        if (error) {
          return queue.push({
            type: 'failed',
            statusCode: null,
            body: null,
            error: String(error)
          })
        }
        return queue.push(parse(eventJson))
      }
    )
    let finished = false
    try {
      while (true) {
        const event = await queue.next()
        if (event.type === 'completed') {
          finished = true
          return
        }
        if (event.type === 'failed') {
          finished = true
          throw new OpenAIRequestError(event.statusCode, event.body, event.error)
        }
        if (event.type === 'sse') {
          event.done = event.data === '[DONE]'
          event.json = function () {
            return this.done ? null : parse(this.data)
          }
        }
        yield event
      }
    } finally {
      if (!finished) await this._handle.cancel(requestId)
    }
  }

  streamChatCompletions(body) {
    return this.stream('/v1/chat/completions', body)
  }

  streamResponses(body) {
    return this.stream('/v1/responses', body)
  }

  cancel(requestId) {
    return this._handle.cancel(requestId)
  }
}

function parse(json) {
  return JSON.parse(json)
}

module.exports = { Inference, OpenAIRequestError }

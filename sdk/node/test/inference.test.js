'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const { Inference, OpenAIRequestError } = require('../inference')

test('buffered requests preserve rich agent payloads and responses', async () => {
  const calls = []
  const handle = {
    async openaiRequestJson(path, body) {
      calls.push({ path, body: JSON.parse(body) })
      return JSON.stringify({
        statusCode: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          choices: [{
            message: {
              role: 'assistant',
              tool_calls: [{ id: 'call-1', function: { name: 'weather', arguments: '{"city":"Sydney"}' } }]
            }
          }],
          usage: { total_tokens: 12 }
        })
      })
    }
  }
  const inference = new Inference(handle)
  const body = {
    model: 'model',
    messages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }] }],
    tools: [{ type: 'function', function: { name: 'weather', parameters: { type: 'object' } } }],
    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: { type: 'object' } } }
  }

  const result = await inference.chatCompletions(body)

  assert.deepEqual(calls, [{ path: '/v1/chat/completions', body: { ...body, stream: false } }])
  assert.equal(result.choices[0].message.tool_calls[0].function.arguments, '{"city":"Sydney"}')
  assert.equal(result.usage.total_tokens, 12)
})

test('responses is the canonical rich Responses API', async () => {
  let request
  const handle = {
    async openaiRequestJson(path, body) {
      request = { path, body: JSON.parse(body) }
      return JSON.stringify({
        statusCode: 200,
        contentType: 'application/json',
        body: JSON.stringify({ output: [{ type: 'function_call', name: 'weather' }] })
      })
    }
  }

  const result = await new Inference(handle).responses({ model: 'model', input: 'weather' })

  assert.deepEqual(request, {
    path: '/v1/responses',
    body: { model: 'model', input: 'weather', stream: false }
  })
  assert.equal(result.output[0].type, 'function_call')
})

test('stream preserves fragmented tool-call SSE data, raw frames, and done', async () => {
  const toolDelta = {
    choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] } }]
  }
  let request
  const handle = {
    async openaiStream(path, body, callback) {
      request = { path, body: JSON.parse(body) }
      queueMicrotask(() => {
        callback(JSON.stringify({ type: 'started', requestId: 'req-1', statusCode: 200, contentType: 'text/event-stream' }))
        callback(JSON.stringify({ type: 'sse', requestId: 'req-1', event: null, data: JSON.stringify(toolDelta), raw: `data: ${JSON.stringify(toolDelta)}\n\n` }))
        callback(JSON.stringify({ type: 'sse', requestId: 'req-1', event: null, data: '[DONE]', raw: 'data: [DONE]\n\n' }))
        callback(JSON.stringify({ type: 'completed', requestId: 'req-1' }))
      })
      return 'req-1'
    },
    async cancel() {
      assert.fail('completed stream must not be cancelled')
    }
  }

  const events = []
  for await (const event of new Inference(handle).streamChatCompletions({ model: 'model' })) {
    events.push(event)
  }

  assert.deepEqual(request, { path: '/v1/chat/completions', body: { model: 'model', stream: true } })
  assert.equal(events[1].json().choices[0].delta.tool_calls[0].function.arguments, '{"city":')
  assert.match(events[1].raw, /^data: /)
  assert.equal(events[2].done, true)
  assert.equal(events[2].json(), null)
})

test('Responses streams preserve named event types', async () => {
  const handle = {
    async openaiStream(_path, _body, callback) {
      queueMicrotask(() => {
        callback(JSON.stringify({ type: 'sse', requestId: 'req-2', event: 'response.function_call_arguments.delta', data: '{"delta":"{\\"city\\":"}', raw: 'event: response.function_call_arguments.delta\ndata: {}\n\n' }))
        callback(JSON.stringify({ type: 'completed', requestId: 'req-2' }))
      })
      return 'req-2'
    },
    async cancel() {}
  }

  const events = []
  for await (const event of new Inference(handle).streamResponses({ model: 'model', input: 'weather' })) {
    events.push(event)
  }
  assert.equal(events[0].event, 'response.function_call_arguments.delta')
})

test('stream failures expose status, body, and error text', async () => {
  const handle = {
    async openaiStream(_path, _body, callback) {
      queueMicrotask(() => callback(JSON.stringify({ type: 'failed', requestId: 'req-3', statusCode: 429, error: 'rate limited', body: '{"error":"slow down"}' })))
      return 'req-3'
    },
    async cancel() {}
  }

  await assert.rejects(
    async () => {
      for await (const _event of new Inference(handle).stream('/v1/chat/completions', {})) {}
    },
    error => error instanceof OpenAIRequestError && error.statusCode === 429 && error.body === '{"error":"slow down"}' && error.message === 'rate limited'
  )
})

test('closing a stream early cancels the native request', async () => {
  const cancelled = []
  const handle = {
    async openaiStream(_path, _body, callback) {
      queueMicrotask(() => callback(JSON.stringify({ type: 'started', requestId: 'req-4', statusCode: 200, contentType: 'text/event-stream' })))
      return 'req-4'
    },
    async cancel(requestId) {
      cancelled.push(requestId)
    }
  }

  for await (const _event of new Inference(handle).stream('/v1/chat/completions', {})) break
  assert.deepEqual(cancelled, ['req-4'])
})

test('stream buffer overflow fails explicitly and cancels the native request', async () => {
  const cancelled = []
  const handle = {
    async openaiStream(_path, _body, callback) {
      for (let index = 0; index < 300; index += 1) {
        callback(JSON.stringify({ type: 'sse', requestId: 'req-fast', event: null, data: '{}', raw: 'data: {}\n\n' }))
      }
      return 'req-fast'
    },
    async cancel(requestId) {
      cancelled.push(requestId)
    }
  }

  await assert.rejects(
    async () => {
      for await (const _event of new Inference(handle).stream('/v1/chat/completions', {})) {}
    },
    /consumer fell behind/
  )
  assert.deepEqual(cancelled, ['req-fast'])
})

'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const { Models } = require('../models')

test('discovery preserves OCR providers and translates filters without invoking inference', async () => {
  let query
  const page = {
    data: [{
      id: 'shared', display_name: 'Shared', kind: 'model',
      availability: 'available', operations: ['document_ocr'],
      providers: [{ provider_id: 'peer-a', local: false, available: true,
        workload_class: 'document_ocr', operations: ['document_ocr'],
        metadata: { architecture: 'ocr' }, runtime: { model_name: 'shared', ready: true } }]
    }], next_cursor: 'opaque-next-page'
  }
  const models = new Models({
    async meshModelsJson(source) { query = JSON.parse(source); return JSON.stringify(page) },
    async listModelsJson() { assert.fail('discovery must not use the LLM list') }
  })
  assert.deepEqual(await models.list({ workloadClass: 'document_ocr', available: false, limit: 3, cursor: 'opaque-input' }), page)
  assert.deepEqual(query, { workload_class: 'document_ocr', available: false, limit: 3, cursor: 'opaque-input' })
})

test('default query includes all workloads and passes through native-normalized fields', async () => {
  // Native deserialization maps unrecognized peer workloads and kinds to
  // "unknown" before this surface sees them, so a realistic page carries the
  // normalized values; unknown extra fields still pass through untouched.
  const page = { data: [{ id: 'future', kind: 'unknown', workload_class: 'unknown', providers: [], extra: true }], next_cursor: null }
  const models = new Models({ async meshModelsJson(source) {
    assert.deepEqual(JSON.parse(source), {})
    return JSON.stringify(page)
  } })
  assert.deepEqual(await models.list(), page)
})

test('lookup forwards exact IDs and native lookup failures', async () => {
  const id = 'org/model@revision#profile'
  const failure = new Error('model not found')
  const models = new Models({ async meshModelJson(value) { assert.equal(value, id); throw failure } })
  await assert.rejects(models.get(id), error => error === failure)
})

test('query rejection propagates and unknown filter names reach native validation', async () => {
  const failure = new Error('unknown field avaiable')
  const models = new Models({ async meshModelsJson(source) {
    assert.deepEqual(JSON.parse(source), { avaiable: true })
    throw failure
  } })
  await assert.rejects(models.list({ avaiable: true }), error => error === failure)
})

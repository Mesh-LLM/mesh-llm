'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const { Inference, OpenAIRequestError } = require('../inference')

test('OCR sends exact binary bytes and options through native document inference', async () => {
  const backing = Uint8Array.from([8, 0, 255, 13, 10, 9])
  const source = backing.subarray(1, 5)
  const inference = new Inference({ async documentOcrJson(file, options) {
    assert.deepEqual([...file], [0, 255, 13, 10])
    assert.deepEqual(JSON.parse(options), { model: 'ocr', filename: 'scan.pdf',
      content_type: 'application/pdf', response_format: 'json', max_pages: 2 })
    return JSON.stringify({ statusCode: 200, contentType: 'application/json',
      body: '{"pages":[{"text":"Hello"}]}' })
  } })
  const result = await inference.documentOcr(source, { model: 'ocr', filename: 'scan.pdf',
    contentType: 'application/pdf', responseFormat: 'json', maxPages: 2 })
  assert.equal(result.statusCode, 200)
  assert.equal(result.json().pages[0].text, 'Hello')
})

test('OCR preserves provider errors and raw plain text responses', async () => {
  const inference = new Inference({ async documentOcrJson() {
    return JSON.stringify({ statusCode: 422, contentType: 'text/plain', body: 'bad PDF' })
  } })
  await assert.rejects(inference.documentOcr(new Uint8Array(), { model: 'ocr' }),
    error => error instanceof OpenAIRequestError && error.statusCode === 422 && error.body === 'bad PDF')
  const response = await inference.documentOcr(new Uint8Array(), { model: 'ocr' }, { raiseForStatus: false })
  assert.equal(response.body, 'bad PDF')
  assert.equal(response.contentType, 'text/plain')
})

test('OCR native transport failures propagate unchanged', async () => {
  const failure = new Error('node is stopped')
  const inference = new Inference({ async documentOcrJson() { throw failure } })
  await assert.rejects(inference.documentOcr(new Uint8Array(), { model: 'ocr' }), error => error === failure)
})

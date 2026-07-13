import { afterEach, describe, expect, it, vi } from 'vitest'

import { createRafBatcher, hasBlobContent, parseApiErrorBody } from '@/lib/streaming'

afterEach(() => vi.restoreAllMocks())

describe('createRafBatcher', () => {
  it('publishes updates without waiting for animation frames in a hidden tab', () => {
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    const requestFrame = vi.spyOn(window, 'requestAnimationFrame')
    const onUpdate = vi.fn()

    createRafBatcher(onUpdate).push('background response')

    expect(onUpdate).toHaveBeenCalledWith('background response')
    expect(requestFrame).not.toHaveBeenCalled()
  })

  it('cancels a pending animation frame when the tab becomes hidden', () => {
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    const requestFrame = vi.spyOn(window, 'requestAnimationFrame')
    const cancelFrame = vi.spyOn(window, 'cancelAnimationFrame')
    const onUpdate = vi.fn()
    const batcher = createRafBatcher(onUpdate)

    batcher.push('visible response')
    const frame = requestFrame.mock.results[0].value
    visibility.mockReturnValue('hidden')
    batcher.push('background response')

    expect(cancelFrame).toHaveBeenCalledWith(frame)
    expect(onUpdate).toHaveBeenCalledWith('background response')
  })
})

// ---------------------------------------------------------------------------
// hasBlobContent
// ---------------------------------------------------------------------------

describe('hasBlobContent', () => {
  it('returns false for non-array inputs', () => {
    expect(hasBlobContent(null)).toBe(false)
    expect(hasBlobContent(undefined)).toBe(false)
    expect(hasBlobContent('string')).toBe(false)
    expect(hasBlobContent({})).toBe(false)
  })

  it('returns false when no message has blob URLs', () => {
    const input = [
      { role: 'user', content: 'hello' },
      {
        role: 'user',
        content: [{ type: 'input_text', text: 'hello' }]
      }
    ]
    expect(hasBlobContent(input)).toBe(false)
  })

  it('returns false for empty array', () => {
    expect(hasBlobContent([])).toBe(false)
  })

  it('returns true when an image_url block contains a mesh://blob/ URL', () => {
    const input = [
      {
        role: 'user',
        content: [
          { type: 'input_text', text: 'describe this' },
          {
            type: 'input_image',
            image_url: 'mesh://blob/abc123'
          }
        ]
      }
    ]
    expect(hasBlobContent(input)).toBe(true)
  })

  it('returns true for audio blob URLs', () => {
    const input = [
      {
        role: 'user',
        content: [
          {
            type: 'input_audio',
            audio_url: 'mesh://blob/def456'
          }
        ]
      }
    ]
    expect(hasBlobContent(input)).toBe(true)
  })

  it('returns true when only one of several messages has blob content', () => {
    const input = [
      { role: 'user', content: 'plain text message' },
      {
        role: 'user',
        content: [{ type: 'input_image', image_url: 'mesh://blob/xyz' }]
      }
    ]
    expect(hasBlobContent(input)).toBe(true)
  })

  it('does not match non-blob URLs', () => {
    const input = [
      {
        role: 'user',
        content: [
          {
            type: 'input_image',
            image_url: 'https://example.com/image.png'
          }
        ]
      }
    ]
    expect(hasBlobContent(input)).toBe(false)
  })

  it('does not match data: URLs', () => {
    const input = [
      {
        role: 'user',
        content: [
          {
            type: 'input_image',
            image_url: 'data:image/png;base64,abc123'
          }
        ]
      }
    ]
    expect(hasBlobContent(input)).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// parseApiErrorBody
// ---------------------------------------------------------------------------

function makeResponse(body: string, status = 500): Response {
  return new Response(body, {
    status,
    headers: { 'Content-Type': 'application/json' }
  })
}

describe('parseApiErrorBody', () => {
  it('returns HTTP <status> for empty body', async () => {
    const res = makeResponse('', 500)
    expect(await parseApiErrorBody(res)).toBe('HTTP 500')
  })

  it('extracts message from {"error":{"message":"..."}}', async () => {
    const res = makeResponse(JSON.stringify({ error: { message: 'image input is not supported' } }), 500)
    expect(await parseApiErrorBody(res)).toBe('image input is not supported')
  })

  it('extracts message from {"error":"..."}', async () => {
    const res = makeResponse(JSON.stringify({ error: 'model not loaded' }), 503)
    expect(await parseApiErrorBody(res)).toBe('model not loaded')
  })

  it('returns raw body text for short non-JSON responses', async () => {
    const res = makeResponse('Service Unavailable', 503)
    expect(await parseApiErrorBody(res)).toBe('Service Unavailable')
  })

  it('returns HTTP status for long non-JSON body (>=500 chars)', async () => {
    const longBody = 'x'.repeat(500)
    const res = makeResponse(longBody, 502)
    expect(await parseApiErrorBody(res)).toBe('HTTP 502')
  })

  it('returns HTTP status for long JSON body without recognised error shape', async () => {
    const longPayload = JSON.stringify({ detail: 'x'.repeat(500) })
    const res = makeResponse(longPayload, 500)
    expect(await parseApiErrorBody(res)).toBe('HTTP 500')
  })

  it('returns raw body for short JSON without recognised error shape', async () => {
    const body = JSON.stringify({ detail: 'bad request' })
    const res = makeResponse(body, 400)
    expect(await parseApiErrorBody(res)).toBe(body)
  })

  it('returns HTTP status for unreadable body (consumed stream)', async () => {
    const res = makeResponse('{"error":"oops"}', 500)
    // Exhaust the body so reading it again throws
    await res.text()
    expect(await parseApiErrorBody(res)).toBe('HTTP 500')
  })
})

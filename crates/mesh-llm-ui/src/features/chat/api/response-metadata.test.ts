import { describe, expect, it } from 'vitest'
import {
  clientNonceFromHeaders,
  extractThreadMessageMetadata,
  mergeThreadMessageMetadata,
  responseMetadataToThreadMessage,
  threadMessageMetadataEquals
} from '@/features/chat/api/response-metadata'
import type { ThreadMessage } from '@/features/app-tabs/types'

describe('responseMetadataToThreadMessage', () => {
  it('formats backend completion metadata for assistant response rows', () => {
    expect(
      responseMetadataToThreadMessage({
        messageId: 'assistant-1',
        model: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M',
        usage: { input_tokens: 12, output_tokens: 44, total_tokens: 56 },
        timings: { decode_time_ms: 1522, ttft_ms: 1117, total_time_ms: 2639 },
        servedBy: 'carrack'
      })
    ).toEqual({
      model: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M',
      route: 'carrack',
      routeNode: 'carrack',
      tokens: '44 tok',
      tokPerSec: '28.9 tok/s',
      ttft: '1117ms'
    })
  })

  // MoA blocks until a worker wins, then emits the full reduced answer in a
  // single SSE event. decode_time_ms is ~0 in that case; divide by total
  // wall time instead so tok/s reflects the user's actual wait.
  it('falls back to total_time_ms when decode interval is too small to be real', () => {
    expect(
      responseMetadataToThreadMessage({
        messageId: 'moa-1',
        model: 'mesh',
        usage: { input_tokens: 0, output_tokens: 9, total_tokens: 9 },
        timings: { decode_time_ms: 0.04, ttft_ms: 1389, total_time_ms: 1390 },
        servedBy: 'reducer'
      }).tokPerSec
    ).toBe('6.5 tok/s')
  })
})

describe('mergeThreadMessageMetadata', () => {
  it('prefers backend model metadata while keeping submitted model as fallback', () => {
    const message: ThreadMessage = {
      id: 'assistant-1',
      messageRole: 'assistant',
      timestamp: '2026-05-06T00:00:00.000Z',
      body: 'Hello'
    }

    expect(
      mergeThreadMessageMetadata(
        message,
        { model: 'backend-model', tokens: '27 tok', tokPerSec: '15.3 tok/s', ttft: '1116ms' },
        'submitted-model'
      )
    ).toMatchObject({ model: 'backend-model', tokens: '27 tok', tokPerSec: '15.3 tok/s', ttft: '1116ms' })

    expect(mergeThreadMessageMetadata(message, undefined, 'submitted-model')).toMatchObject({
      model: 'submitted-model'
    })
  })
})

describe('threadMessageMetadataEquals', () => {
  it('detects metadata-only differences for persistence checks', () => {
    const base: ThreadMessage = {
      id: 'assistant-1',
      messageRole: 'assistant',
      timestamp: '2026-05-06T00:00:00.000Z',
      body: 'Hello',
      tokens: '27 tok'
    }

    expect(threadMessageMetadataEquals(base, { ...base })).toBe(true)
    expect(threadMessageMetadataEquals(base, { ...base, tokens: '44 tok' })).toBe(false)
  })
})

describe('clientNonceFromHeaders', () => {
  it('keeps the client nonce the serving frontend echoed', () => {
    const headers = new Headers({ 'x-capsule-client-nonce': '14af686f-86e5-4baa-bb6b-d3dd3c81cfec' })
    expect(clientNonceFromHeaders(headers)).toEqual({ clientNonce: '14af686f-86e5-4baa-bb6b-d3dd3c81cfec' })
  })

  it('leaves an absent or empty header absent', () => {
    expect(clientNonceFromHeaders(new Headers())).toEqual({})
    expect(clientNonceFromHeaders(new Headers({ 'x-capsule-client-nonce': '  ' }))).toEqual({})
  })

  it('the nonce is one of the fields a thread message keeps (extract -> merge round trip)', () => {
    const metadata = responseMetadataToThreadMessage({ messageId: 'm1', clientNonce: 'nonce-1' })
    const message: ThreadMessage = { id: 'm1', messageRole: 'assistant', timestamp: 't', body: 'b', ...metadata }
    const kept = extractThreadMessageMetadata(message)
    expect(kept).toMatchObject({ clientNonce: 'nonce-1' })
    const reloaded = mergeThreadMessageMetadata({ id: 'm1', messageRole: 'assistant', timestamp: 't', body: 'b' }, kept)
    expect(reloaded.clientNonce).toBe('nonce-1')
  })
})

// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogArtifactId, LogRequestId } from '@/features/logs/api/ids'
import type { LogArtifact } from '@/features/logs/api/schemas'
import {
  classifyLogArtifacts,
  decodeEventStream,
  decodeLogArtifactContent,
  LOG_PAYLOAD_RENDER_LIMIT_BYTES
} from '@/features/logs/lib/log-payload-content'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')
const ARTIFACT_ID = LogArtifactId.parse('00000000-0000-4000-8000-000000000011')

type ArtifactOptions = {
  readonly bytes?: number
  readonly contentBase64?: string
  readonly kind?: string
  readonly mediaKind?: string
  readonly occurredAt?: string
}

function availableArtifact(options: ArtifactOptions = {}): LogArtifact {
  return {
    artifactId: ARTIFACT_ID,
    requestId: REQUEST_ID,
    occurredAt: options.occurredAt ?? '2026-08-04T12:00:00Z',
    kind: options.kind ?? 'request_body',
    mediaKind: options.mediaKind ?? 'application/json',
    checksum: 'sha256:0123456789abcdef',
    bytes: options.bytes ?? 32,
    version: 1,
    redacted: true,
    truncated: false,
    contentState: 'available',
    contentBase64: options.contentBase64 ?? btoa('{}')
  }
}

function stateArtifact(
  contentState: Exclude<LogArtifact['contentState'], 'available'>,
  kind = 'response_body'
): LogArtifact {
  const base = availableArtifact({ kind })
  return { ...base, contentState, contentBase64: undefined }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('classifyLogArtifacts', () => {
  it('classifies artifacts by complete kind tokens', () => {
    // Given
    const request = availableArtifact({ kind: 'request_headers' })
    const response = stateArtifact('missing', 'response_body')
    const error = stateArtifact('corrupt', 'provider-error-trace')
    const unclassified = stateArtifact('unavailable', 'prerequest-summary')

    // When
    const classified = classifyLogArtifacts([request, response, error, unclassified])

    // Then
    expect(classified.request.artifacts).toEqual([request])
    expect(classified.response.artifacts).toEqual([response])
    expect(classified.error.artifacts).toEqual([error])
    expect(classified.unclassified.artifacts).toEqual([unclassified])
  })

  it('selects a body artifact as primary while preserving occurrence order', () => {
    // Given
    const laterAvailable = availableArtifact({
      kind: 'request_headers',
      occurredAt: '2026-08-04T12:00:02Z'
    })
    const earlierBody = stateArtifact('missing', 'request_body')

    // When
    const group = classifyLogArtifacts([laterAvailable, earlierBody]).request

    // Then
    expect(group.artifacts).toEqual([earlierBody, laterAvailable])
    expect(group.primary).toBe(earlierBody)
  })

  it('selects the earliest body by instant and preserves equal-instant input order', () => {
    // Given
    const later = availableArtifact({
      kind: 'request_later_body',
      occurredAt: '2026-08-04T10:00:00-02:00'
    })
    const earlier = availableArtifact({
      kind: 'request_earlier_body',
      occurredAt: '2026-08-04T11:00:00Z'
    })
    const tied = availableArtifact({
      kind: 'request_tied_body',
      occurredAt: '2026-08-04T13:00:00+01:00'
    })

    // When
    const group = classifyLogArtifacts([later, earlier, tied]).request

    // Then
    expect(group.artifacts).toEqual([earlier, later, tied])
    expect(group.primary).toBe(earlier)
  })
})

describe('decodeLogArtifactContent', () => {
  it('returns raw and pretty representations for valid JSON', () => {
    // Given
    const rawText = '{"model":"Qwen3","tokens":3}'

    // When / Then
    expect(decodeLogArtifactContent(availableArtifact({ contentBase64: btoa(rawText) }))).toEqual({
      state: 'json',
      text: rawText,
      prettyText: '{\n  "model": "Qwen3",\n  "tokens": 3\n}'
    })
  })

  it('keeps malformed JSON as inert plaintext', () => {
    // Given
    const malformed = '{"model":<img src=x onerror=alert(1)>}'

    // When / Then
    expect(decodeLogArtifactContent(availableArtifact({ contentBase64: btoa(malformed) }))).toEqual({
      state: 'malformed-json',
      text: malformed
    })
  })

  it('accepts scalar JSON when the media type declares JSON', () => {
    // Given / When / Then
    expect(decodeLogArtifactContent(availableArtifact({ contentBase64: btoa('false') }))).toEqual({
      state: 'json',
      text: 'false',
      prettyText: 'false'
    })
  })

  it('decodes normalized event streams into ordered independently typed frames', () => {
    // Given
    const streamText =
      ': heartbeat\r\n\r\n' +
      'event: token\r\nid: frame:1\r\ndata: {"delta":\r\ndata: {"content":"hello:world"}}\r\n\r\n' +
      'id: frame-2\rdata: plain: text\rdata:  leading space\r\r' +
      'data: {"broken":\n\n' +
      'event: done\r\ndata: [DONE]\r\n\r\n'

    // When
    const result = decodeLogArtifactContent(
      availableArtifact({
        contentBase64: btoa(streamText),
        mediaKind: 'Text/Event-Stream; charset=UTF-8'
      })
    )

    // Then
    expect(result).toEqual({
      state: 'event-stream',
      frames: [
        {
          event: 'token',
          id: 'frame:1',
          data: {
            state: 'json',
            text: '{"delta":\n{"content":"hello:world"}}',
            prettyText: '{\n  "delta": {\n    "content": "hello:world"\n  }\n}'
          }
        },
        {
          id: 'frame-2',
          data: { state: 'text', text: 'plain: text\n leading space' }
        },
        {
          data: { state: 'text', text: '{"broken":' }
        },
        {
          event: 'done',
          data: { state: 'text', text: '[DONE]' }
        }
      ]
    })
  })

  it('caps event-stream frame previews and exposes truncation', () => {
    const stream = Array.from({ length: 300 }, (_, index) => `id: ${index}\ndata: frame-${index}`).join('\n\n')

    const frames = decodeEventStream(stream)

    expect(frames).toHaveLength(256)
    expect(frames.at(-1)).toMatchObject({
      event: 'preview-truncated',
      truncated: true,
      data: { state: 'text', text: expect.stringContaining('safe preview limit') }
    })
  })

  it('caps aggregate event-stream preview allocation and exposes truncation', () => {
    const frames = decodeEventStream(`data: ${'x'.repeat(1_048_577)}`)

    expect(frames).toEqual([
      {
        event: 'preview-truncated',
        truncated: true,
        data: { state: 'text', text: expect.stringContaining('safe preview limit') }
      }
    ])
  })

  it('rejects invalid base64', () => {
    expect(decodeLogArtifactContent(availableArtifact({ contentBase64: 'not base64!' }))).toEqual({
      state: 'decode-error',
      reason: 'base64'
    })
  })

  it('rejects non-canonical base64 padding before decoding', () => {
    // Given
    const atobSpy = vi.spyOn(globalThis, 'atob')

    // When
    const result = decodeLogArtifactContent(availableArtifact({ contentBase64: 'AB==' }))

    // Then
    expect(result).toEqual({ state: 'decode-error', reason: 'base64' })
    expect(atobSpy).not.toHaveBeenCalled()
  })

  it.each([
    ['TQ', 'M'],
    ['TWE', 'Ma']
  ])('accepts valid unpadded base64 %s', (contentBase64, text) => {
    expect(
      decodeLogArtifactContent(
        availableArtifact({
          contentBase64,
          mediaKind: 'text/plain',
          bytes: text.length
        })
      )
    ).toEqual({ state: 'text', text })
  })

  it.each(['TR', 'TWF'])('rejects non-canonical unpadded base64 %s', (contentBase64) => {
    expect(decodeLogArtifactContent(availableArtifact({ contentBase64 }))).toEqual({
      state: 'decode-error',
      reason: 'base64'
    })
  })

  it('contains base64 decoder exceptions', () => {
    // Given
    vi.spyOn(globalThis, 'atob').mockImplementation(() => {
      throw new DOMException('Invalid character', 'InvalidCharacterError')
    })

    // When / Then
    expect(decodeLogArtifactContent(availableArtifact({ contentBase64: 'AAAA' }))).toEqual({
      state: 'decode-error',
      reason: 'base64'
    })
  })

  it('rejects invalid UTF-8 for declared text', () => {
    const invalidUtf8 = btoa(String.fromCharCode(0xc3, 0x28))

    expect(
      decodeLogArtifactContent(
        availableArtifact({
          contentBase64: invalidUtf8,
          mediaKind: 'text/plain'
        })
      )
    ).toEqual({ state: 'decode-error', reason: 'utf8' })
  })

  it('rejects invalid UTF-8 before classifying a declared event stream', () => {
    const invalidUtf8 = btoa(String.fromCharCode(0xc3, 0x28))

    expect(
      decodeLogArtifactContent(
        availableArtifact({
          contentBase64: invalidUtf8,
          mediaKind: 'text/event-stream'
        })
      )
    ).toEqual({ state: 'decode-error', reason: 'utf8' })
  })

  it('keeps binary content non-renderable', () => {
    const result = decodeLogArtifactContent(
      availableArtifact({
        contentBase64: btoa('binary'),
        mediaKind: 'application/octet-stream'
      })
    )

    expect(result).toEqual({
      state: 'binary',
      bytes: new Uint8Array([98, 105, 110, 97, 114, 121])
    })
  })

  it('returns hostile markup only as plaintext', () => {
    const markup = '<img src=x onerror=alert(1)><script>alert(2)</script>'

    expect(
      decodeLogArtifactContent(
        availableArtifact({
          contentBase64: btoa(markup),
          mediaKind: 'text/plain'
        })
      )
    ).toEqual({ state: 'text', text: markup })
  })

  it.each([
    ['unavailable', { state: 'unavailable' }],
    ['missing', { state: 'missing' }],
    ['corrupt', { state: 'corrupt' }]
  ] as const)('preserves the explicit %s state', (contentState, expected) => {
    expect(decodeLogArtifactContent(stateArtifact(contentState))).toEqual(expected)
  })

  it('rejects an oversized declared body before base64 decoding', () => {
    // Given
    const atobSpy = vi.spyOn(globalThis, 'atob')

    // When
    const result = decodeLogArtifactContent(
      availableArtifact({
        bytes: LOG_PAYLOAD_RENDER_LIMIT_BYTES + 1,
        contentBase64: btoa('{}')
      })
    )

    // Then
    expect(result).toEqual({ state: 'too-large' })
    expect(atobSpy).not.toHaveBeenCalled()
  })

  it('rejects encoded content whose decoded size exceeds the ceiling before atob', () => {
    // Given
    const encodedLength = 4 * Math.ceil((LOG_PAYLOAD_RENDER_LIMIT_BYTES + 1) / 3)
    const oversizedContent = `${'A'.repeat(encodedLength - 1)}=`
    const atobSpy = vi.spyOn(globalThis, 'atob').mockReturnValue('')

    // When
    const result = decodeLogArtifactContent(
      availableArtifact({
        bytes: LOG_PAYLOAD_RENDER_LIMIT_BYTES,
        contentBase64: oversizedContent
      })
    )

    // Then
    expect(result).toEqual({ state: 'too-large' })
    expect(atobSpy).not.toHaveBeenCalled()
  })

  it('allows content exactly at the backend ceiling', () => {
    // Given
    const encodedLength = 4 * Math.ceil(LOG_PAYLOAD_RENDER_LIMIT_BYTES / 3)
    const boundaryContent = `${'A'.repeat(encodedLength - 2)}==`
    const atobSpy = vi.spyOn(globalThis, 'atob').mockReturnValue('')

    // When
    const result = decodeLogArtifactContent(
      availableArtifact({
        bytes: LOG_PAYLOAD_RENDER_LIMIT_BYTES,
        contentBase64: boundaryContent,
        mediaKind: 'application/octet-stream'
      })
    )

    // Then
    expect(result).toEqual({ state: 'binary', bytes: new Uint8Array() })
    expect(atobSpy).toHaveBeenCalledOnce()
  })
})

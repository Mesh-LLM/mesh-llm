import { describe, expect, it } from 'vitest'
import { LogArtifactId, LogRequestId } from '@/features/logs/api/ids'
import type { LogArtifact } from '@/features/logs/api/schemas'
import { deriveArtifactCounts } from '@/features/logs/lib/log-artifact-counts'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')
const ARTIFACT_ID = LogArtifactId.parse('00000000-0000-4000-8000-000000000011')

function artifact({
  contentState,
  bytes = 0,
  version = 1,
  redacted = false,
  truncated = false
}: {
  readonly contentState: LogArtifact['contentState']
  readonly bytes?: number
  readonly version?: number
  readonly redacted?: boolean
  readonly truncated?: boolean
}): LogArtifact {
  const base = {
    artifactId: ARTIFACT_ID,
    requestId: REQUEST_ID,
    occurredAt: '2026-08-04T12:00:00Z',
    kind: 'request_body',
    mediaKind: 'application/json',
    checksum: undefined,
    bytes,
    version,
    redacted,
    truncated
  }
  if (contentState === 'available') return { ...base, contentState, contentBase64: undefined }
  if (contentState === 'unavailable') {
    return { ...base, contentState, unavailableReason: 'artifact_capture_disabled', contentBase64: undefined }
  }
  return { ...base, contentState, contentBase64: undefined }
}

describe('deriveArtifactCounts', () => {
  it('returns zeroed counts for an empty list', () => {
    expect(deriveArtifactCounts([])).toEqual({
      total: 0,
      contentStates: '',
      redacted: 0,
      truncated: 0,
      bytes: 0,
      versions: []
    })
  })

  it('derives a single item across all count dimensions', () => {
    const counts = deriveArtifactCounts([artifact({ contentState: 'available', bytes: 648, version: 1 })])
    expect(counts).toEqual({
      total: 1,
      contentStates: '1 available',
      redacted: 0,
      truncated: 0,
      bytes: 648,
      versions: [1]
    })
  })

  it('groups mixed content states in fixed order joined by a middle dot', () => {
    const items = [
      artifact({ contentState: 'available' }),
      artifact({ contentState: 'available' }),
      artifact({ contentState: 'unavailable' }),
      artifact({ contentState: 'missing' }),
      artifact({ contentState: 'corrupt' })
    ]
    expect(deriveArtifactCounts(items).contentStates).toBe('2 available · 1 unavailable · 1 missing · 1 corrupt')
  })

  it('omits content states with no items', () => {
    const counts = deriveArtifactCounts([artifact({ contentState: 'available' })])
    expect(counts.contentStates).toBe('1 available')
  })

  it('counts redacted and truncated subsets independently', () => {
    const items = [
      artifact({ contentState: 'available', redacted: true }),
      artifact({ contentState: 'available', truncated: true }),
      artifact({ contentState: 'available', redacted: true, truncated: true }),
      artifact({ contentState: 'available' })
    ]
    const counts = deriveArtifactCounts(items)
    expect(counts.redacted).toBe(2)
    expect(counts.truncated).toBe(2)
  })

  it('sums stored bytes across items', () => {
    const items = [
      artifact({ contentState: 'available', bytes: 1_000 }),
      artifact({ contentState: 'available', bytes: 992 })
    ]
    expect(deriveArtifactCounts(items).bytes).toBe(1_992)
  })

  it('dedupes and sorts versions ascending', () => {
    const items = [
      artifact({ contentState: 'available', version: 3 }),
      artifact({ contentState: 'available', version: 1 }),
      artifact({ contentState: 'available', version: 2 }),
      artifact({ contentState: 'available', version: 1 })
    ]
    expect(deriveArtifactCounts(items).versions).toEqual([1, 2, 3])
  })

  it('formats counts with locale grouping', () => {
    const items = Array.from({ length: 1_234 }, (_, index) =>
      artifact({
        contentState: index % 2 === 0 ? 'available' : 'missing',
        bytes: 1_000,
        version: (index % 3) + 1
      })
    )
    const counts = deriveArtifactCounts(items)
    const numberFormat = new Intl.NumberFormat()
    expect(counts.total.toLocaleString()).toBe(numberFormat.format(1_234))
    expect(counts.contentStates).toBe('617 available · 617 missing')
    expect(counts.bytes.toLocaleString()).toBe(numberFormat.format(1_234_000))
  })
})

import { describe, expect, it } from 'vitest'
import { EVIDENCE_SOURCE_LABEL, evidenceSource } from '@/features/capsules/lib/evidence-source'

describe('evidenceSource -- look finding 6: a replayed sample never reads "Live"', () => {
  it('a fixture replay run is sample data even though the API answers', () => {
    expect(evidenceSource(true, 'freeze-candidate')).toBe('sample')
    expect(EVIDENCE_SOURCE_LABEL.sample).toBe('Sample data')
  })

  it('no replay run: live when the API answers, local otherwise', () => {
    expect(evidenceSource(true, undefined)).toBe('live')
    expect(evidenceSource(true, '')).toBe('live')
    expect(evidenceSource(false, null)).toBe('local')
  })
})

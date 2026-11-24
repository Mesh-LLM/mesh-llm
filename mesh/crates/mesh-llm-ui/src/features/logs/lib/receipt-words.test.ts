import { createHash } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import wording from '@/features/logs/lib/receipt-wording-en.json'
import { groupThousands, receiptWord, selfReportedLine, tokenWords } from '@/features/logs/lib/receipt-words'

describe('receipt wording copy', () => {
  it('matches the digest of the capsule-viewer pack entries it was copied from', () => {
    const canonical = JSON.stringify(
      Object.fromEntries(Object.entries(wording.entries).sort(([a], [b]) => a.localeCompare(b)))
    )
    expect(createHash('sha256').update(canonical).digest('hex')).toBe(wording.sha256)
  })

  it('names the source commit it was copied at', () => {
    expect(wording.source.commit).toMatch(/^[0-9a-f]{40}$/)
  })
})

describe('receiptWord', () => {
  it('fills the named values and leaves unknown placeholders visible', () => {
    expect(receiptWord('verify.self_reported', { verifier: 'a log', checks: 'x: none' })).toBe(
      'Reported by a log: x: none. This view shows that report; it did not re-check anything itself.'
    )
    expect(receiptWord('verify.self_reported')).toContain('{verifier}')
  })
})

describe('groupThousands', () => {
  it('groups by hand with a comma', () => {
    expect(groupThousands(0)).toBe('0')
    expect(groupThousands(999)).toBe('999')
    expect(groupThousands(2000)).toBe('2,000')
    expect(groupThousands(1234567)).toBe('1,234,567')
    expect(groupThousands(-1234)).toBe('-1,234')
  })
})

describe('tokenWords', () => {
  it('reads a recorded token as words', () => {
    expect(tokenWords('attempt_failed')).toBe('Attempt failed')
    expect(tokenWords('stream-error')).toBe('Stream error')
    expect(tokenWords('ok')).toBe('Ok')
  })
})

describe('selfReportedLine', () => {
  it('says not available for a check that did not load and counts retries from attempts', () => {
    expect(selfReportedLine({ errorsRecorded: undefined, attemptCount: 0 })).toContain(
      'errors recorded: not available; retries: 0'
    )
    expect(selfReportedLine({ errorsRecorded: 1200, attemptCount: 3 })).toContain('errors recorded: 1,200; retries: 2')
  })
})

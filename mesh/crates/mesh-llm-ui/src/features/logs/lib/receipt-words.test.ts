import { createHash } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import wording from '@/features/logs/lib/receipt-wording-en.json'
import { groupThousands, receiptWord, selfReportedLine, tokenWords } from '@/features/logs/lib/receipt-words'

describe('receipt wording copy', () => {
  it('matches the digest of the capsule-viewer pack entries it was copied from', () => {
    const canonical = JSON.stringify(
      Object.fromEntries(Object.entries(wording.entries).sort(([a], [b]) => (a < b ? -1 : 1)))
    )
    expect(createHash('sha256').update(canonical).digest('hex')).toBe(wording.sha256)
  })

  it('names the source commit it was copied at', () => {
    expect(wording.source.release).toBe(
      'capsule-viewer 0.25.0 (GitHub Release go/v0.25.0; npm @action-state-group/capsule-viewer@0.25.0)'
    )
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
    expect(groupThousands(Number.NaN)).toBe('NaN')
    expect(groupThousands(1.5)).toBe('1.5')
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
  const loaded = { failureEvents: 0, failedAttempts: 0, errorFiles: 0, attemptCount: 1, historyPartial: false }

  it('counts each record type on its own, so one failure is not summed into several errors', () => {
    expect(selfReportedLine({ ...loaded, failureEvents: 2, failedAttempts: 1, attemptCount: 3 })).toContain(
      'failure events: 2; failed attempts: 1; error files kept: none; retries: 2'
    )
  })

  it('says not available for a check that did not load', () => {
    expect(selfReportedLine({ ...loaded, failureEvents: undefined, attemptCount: undefined })).toContain(
      'failure events: not available; failed attempts: none; error files kept: none; retries: not available'
    )
  })

  it('qualifies every count when only the first page of records loaded', () => {
    expect(selfReportedLine({ ...loaded, failureEvents: 1200, historyPartial: true })).toContain(
      'failure events: at least 1,200; failed attempts: none in the loaded records; error files kept: none in the loaded records; retries: none in the loaded records'
    )
  })
})

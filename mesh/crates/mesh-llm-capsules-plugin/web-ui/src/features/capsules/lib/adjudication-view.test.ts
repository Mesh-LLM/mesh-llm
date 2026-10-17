import { describe, expect, it } from 'vitest'
import type { DeliveredAdjudication, VerdictRecordJson } from '@/features/capsules/api/sidecarTypes'
import {
  deliveredVerdictLine,
  issuedVerdictLine,
  parseVerdict,
  verdictRecordFacts,
  verdictSentence,
  verdictSignatureText
} from '@/features/capsules/lib/adjudication-view'

const NODE_A = 'a70d3967bea3b22f'.repeat(4)
const NODE_B = '2e981e80899b3424'.repeat(4)

function delivered(overrides: Partial<DeliveredAdjudication>): DeliveredAdjudication {
  return {
    verdict: 'corroborated',
    verdict_capsule_id: 'c'.repeat(64),
    referee_node_id: NODE_A,
    received_at: '2026-09-28T15:00:00Z',
    about_this_node: false,
    ...overrides
  }
}

describe('adjudication view -- a referee’s verdict in plain words', () => {
  it('reads the sealed verdict and nothing else', () => {
    expect(parseVerdict('corroborated')).toEqual({ kind: 'corroborated' })
    expect(parseVerdict(`contradicted:${NODE_B}`)).toEqual({ kind: 'contradicted', party: NODE_B })
    expect(parseVerdict('inconclusive')).toEqual({ kind: 'inconclusive' })
    expect(parseVerdict('not_comparable')).toEqual({ kind: 'not_comparable' })
    expect(parseVerdict('contradicted:')).toBeNull()
    expect(parseVerdict('maybe')).toBeNull()
    expect(parseVerdict(null)).toBeNull()
  })

  it('on a judged node: says whose answer was found contradicted, never a full id', () => {
    expect(deliveredVerdictLine(delivered({}))).toBe('A referee (node a70d3967be…) found this answer corroborated.')
    expect(deliveredVerdictLine(delivered({ verdict: `contradicted:${NODE_B}`, about_this_node: true }))).toBe(
      'A referee (node a70d3967be…) found your answer contradicted.'
    )
    expect(deliveredVerdictLine(delivered({ verdict: `contradicted:${NODE_B}`, about_this_node: false }))).toBe(
      'A referee (node a70d3967be…) found the other answer (node 2e981e8089…) contradicted.'
    )
    expect(deliveredVerdictLine(delivered({ verdict: 'garbled' }))).toBeNull()
    for (const line of [delivered({}), delivered({ verdict: `contradicted:${NODE_B}` })].map(deliveredVerdictLine)) {
      expect(line).not.toMatch(/[0-9a-f]{64}/)
    }
  })

  it('on the referee node: the verdict it issued', () => {
    expect(issuedVerdictLine({ verdict: 'corroborated', verdict_capsule_id: 'v', halves: ['a', 'b'] })).toBe(
      'You judged the two answers: corroborated.'
    )
    expect(issuedVerdictLine({ verdict: `contradicted:${NODE_B}`, verdict_capsule_id: 'v', halves: ['a', 'b'] })).toBe(
      "You judged the two answers: node 2e981e8089…'s answer contradicted."
    )
  })

  it('reads the verdict record’s own sealed block, and says when its signature does not check', () => {
    const record: VerdictRecordJson = {
      capsule: {
        model_attestation: {
          compute_attestation: {
            adjudication: {
              verdict: `contradicted:${NODE_B}`,
              half_a_capsule_id: 'a'.repeat(64),
              half_b_capsule_id: 'b'.repeat(64),
              referee_capsule_id: 'd'.repeat(64)
            }
          }
        }
      },
      signed_by_key_id: 'k'.repeat(64),
      verify_ok: true
    }
    expect(verdictRecordFacts(record)).toEqual({
      verdict: { kind: 'contradicted', party: NODE_B },
      halves: ['a'.repeat(64), 'b'.repeat(64)],
      refereeCapsuleId: 'd'.repeat(64)
    })
    expect(verdictSignatureText({ ...record, recorded_as: 'received' })).toBe(
      'The referee’s signature checks on this node, and it was delivered to this node and its log records it.'
    )
    expect(verdictSignatureText({ ...record, verify_ok: false, signature_error: 'bad signature' })).toBe(
      'The referee’s signature doesn’t check on this node (bad signature). Treat this verdict as unconfirmed.'
    )
    expect(verdictSignatureText({ ...record, verify_ok: false })).toMatch(/couldn’t confirm this verdict.*unconfirmed/)
    expect(verdictSignatureText({ ...record, verify_ok: false, legacy: true })).toMatch(/^Legacy, unverified: .*unconfirmed\.$/)
    expect(verdictSignatureText({ ...record, verify_ok: true, legacy: true })).toMatch(/^Legacy, unverified/)
    expect(verdictRecordFacts({ capsule: {}, signed_by_key_id: null, verify_ok: false })).toEqual({
      verdict: null,
      halves: [],
      refereeCapsuleId: null
    })
  })

  it('inconclusive and not comparable are said as such, never as a disagreement', () => {
    expect(deliveredVerdictLine(delivered({ verdict: 'inconclusive' }))).toBe(
      'A referee (node a70d3967be…) couldn’t decide between the two answers.'
    )
    expect(deliveredVerdictLine(delivered({ verdict: 'not_comparable' }))).toBe(
      'A referee (node a70d3967be…) found the two answers can’t be compared: they were sampled.'
    )
    expect(issuedVerdictLine({ verdict: 'not_comparable', verdict_capsule_id: 'v', halves: ['a', 'b'] })).toBe(
      'You judged the two answers: not comparable, because they were sampled.'
    )
    expect(verdictSentence({ kind: 'not_comparable' })).toMatch(/never a disagreement/)
    for (const wire of ['inconclusive', 'not_comparable']) {
      expect(deliveredVerdictLine(delivered({ verdict: wire }))).not.toMatch(/contradicted|wrong/)
    }
  })
})

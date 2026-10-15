import { describe, expect, it } from 'vitest'
import type { PaneBRow } from '@/features/capsules/api/sidecarTypes'
import { HARNESS_PANE_B_PAYLOAD } from '@/features/capsules/lib/peer-fixtures'
import { dealingsLines } from '@/features/capsules/lib/peer-routing-view'
import { alarmSignal, peerAttention } from '@/features/capsules/lib/peer-row-view'

describe('dealingsLines -- the drill says what the Peers row says', () => {
  it('your dealings: counts with their denominators, disputes said as none when none were judged', () => {
    const [clean, alarmed] = HARNESS_PANE_B_PAYLOAD.rows
    expect(dealingsLines(clean)).toEqual([
      '24 exchanges with you · they confirmed 16 of 24',
      'Same request & answer: 16 · 0 differ',
      'Disputes judged: 8 of 24 · 8 corroborated'
    ])
    expect(dealingsLines(alarmed)[1]).toMatch(/· 1 differ$/)
    const unjudged: PaneBRow = { ...clean, verdicts: { state: 'NOT_CHECKED' } }
    expect(dealingsLines(unjudged)[2]).toBe('Disputes judged: none')
  })

  it('a half the door refused for contradicting your record is a disagreement, never "0 differ"', () => {
    const row: PaneBRow = {
      ...HARNESS_PANE_B_PAYLOAD.rows[0],
      exchange_count: 3,
      confirmed_siblings: [
        {
          mine: { state: 'present-unverified', capsule_id: 'a'.repeat(64) },
          theirs: {
            state: 'present-unverified',
            capsule_id: null,
            evidence_outcome: 'claims_refused',
            evidence_outcome_date: '2026-09-28T08:00:00Z'
          }
        }
      ]
    }
    expect(dealingsLines(row).slice(0, 2)).toEqual([
      '3 exchanges with you · they confirmed 0 of 3',
      'Same request & answer: 0 · 1 differ'
    ])
  })
})

describe('referee-signed verdicts -- the drill, the attention chip and the alarm agree with the counts', () => {
  const bucket = (n: number) => ({ count: n, verdict_capsule_ids: Array.from({ length: n }, (_, i) => `v${i}`.padEnd(64, '0')) })

  it('reads the referee-signed counts, never the older not-checked field', () => {
    const row: PaneBRow = {
      ...HARNESS_PANE_B_PAYLOAD.rows[0],
      exchange_count: 5,
      verdicts: { state: 'NOT_CHECKED' },
      referee_verdicts: { corroborated: bucket(1), contradicted: bucket(2), inconclusive: bucket(0), not_comparable: bucket(3) }
    }
    expect(dealingsLines(row)[2]).toBe('Disputes judged: 3 of 5 · 1 corroborated · 2 contradicted')
    expect(peerAttention(row).find((item) => item.key === 'differingAnswers')?.label).toBe('2 differing answers')
    expect(alarmSignal(row)).toEqual({ present: true, text: 'A referee found their answer contradicted', tone: 'bad' })
  })

  it('a not-comparable ruling alone is not a judgment: disputes judged none, no alarm', () => {
    const row: PaneBRow = {
      ...HARNESS_PANE_B_PAYLOAD.rows[0],
      verdicts: { state: 'NOT_CHECKED' },
      referee_verdicts: { corroborated: bucket(0), contradicted: bucket(0), inconclusive: bucket(0), not_comparable: bucket(2) }
    }
    expect(dealingsLines(row)[2]).toBe('Disputes judged: none')
    expect(alarmSignal(row).present).toBe(false)
  })
})

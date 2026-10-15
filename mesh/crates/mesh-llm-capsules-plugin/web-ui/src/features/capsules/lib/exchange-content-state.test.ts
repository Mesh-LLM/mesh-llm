// Toggle ① content (v3 §3).
import { describe, expect, it } from 'vitest'
import {
  deriveContentToggleState,
  theirContentAction,
  theirContentText,
  yourContentFixedText,
  type TheirContentState,
  type TheirContentStateKind,
  type YourContentState
} from '@/features/capsules/lib/exchange-content-state'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'

function paneCRow(overrides: Partial<PaneCRow> = {}): PaneCRow {
  return {
    exchange_key: 'exch-1',
    role_tag: 'ASKED',
    header_state: 'ok',
    properties: null,
    has_issue: false,
    mine: { state: 'present', capsule_id: 'mine-1' },
    theirs: { state: 'present', capsule_id: 'theirs-1' },
    unilateral: false,
    timestamp: '2026-09-08T00:00:00Z',
    ...overrides
  }
}

describe('deriveContentToggleState — L-F: content follows the role, never the row', () => {
  it('Case A (ASKED) -- your side is populated, their side never is', () => {
    const state = deriveContentToggleState(paneCRow({ role_tag: 'ASKED' }))
    expect(state.your.kind).toBe('populated')
    expect(state.their.kind).not.toBe('not_visible_holds')
  })

  it('Case B (SERVED) -- your side streamed-not-retained, their side not-visible-requester-holds, regardless of what the record carries', () => {
    const state = deriveContentToggleState(
      paneCRow({
        role_tag: 'SERVED',
        mine: { state: 'present', capsule_id: 'mine-1', text: 'should never surface' },
        theirs: { state: 'present', capsule_id: 'theirs-1' }
      })
    )
    expect(state.your).toEqual({ kind: 'streamed_not_retained', date: null })
    expect(state.their).toEqual({ kind: 'not_visible_holds', date: null })
  })

  it('ADVERSARIAL -- a SERVED row with a present theirs.state (a corroborated outcome) still renders not_visible_holds, never a populated/content-bearing their side (L-F/L-G)', () => {
    const state = deriveContentToggleState(
      paneCRow({ role_tag: 'SERVED', theirs: { state: 'present', capsule_id: 'theirs-1' } })
    )
    expect(state.their.kind).toBe('not_visible_holds')
  })

  it('deletion (L-D): mine.deleted flips your side to populated_deleted, carrying its date', () => {
    const state = deriveContentToggleState(
      paneCRow({ mine: { state: 'present', capsule_id: 'mine-1', deleted: true, deleted_date: '5 Sep' } })
    )
    expect(state.your).toEqual({ kind: 'populated_deleted', date: '5 Sep' })
  })

  it('a present theirs.state (CLOSED/CONTRADICTED outcome) never leaks into the their-content side as populated -- theirs.state answers a different axis (L-G)', () => {
    const state = deriveContentToggleState(
      paneCRow({
        theirs: { state: 'present', capsule_id: 'theirs-1' },
        properties: { outcome_corroboration: { state: 'PASS' } }
      })
    )
    expect(state.their.kind).toBe('not_asked')
  })

  it.each<[string, TheirContentStateKind]>([
    ['recorded_absence', 'recorded_absence'],
    ['unanswered', 'unanswered'],
    ['signed_refusal', 'signed_refusal'],
    ['not_asked', 'not_asked']
  ])('evidence_outcome %s maps to their-content kind %s', (outcome, expectedKind) => {
    const state = deriveContentToggleState(
      paneCRow({
        theirs: {
          state: 'absent',
          capsule_id: null,
          evidence_outcome: outcome as never,
          evidence_outcome_date: '4 Sep'
        }
      })
    )
    expect(state.their.kind).toBe(expectedKind)
  })

  it('ADVERSARIAL — L-C: no evidence_outcome carried at all degrades to not_asked, never an invented ask-log', () => {
    const state = deriveContentToggleState(paneCRow({ theirs: { state: 'absent', capsule_id: null } }))
    expect(state.their.kind).toBe('not_asked')
  })

  it('every their-content kind is reachable', () => {
    const outcomes = ['recorded_absence', 'unanswered', 'signed_refusal', undefined] as const
    const reached = new Set<TheirContentStateKind>()
    for (const outcome of outcomes) {
      reached.add(
        deriveContentToggleState(paneCRow({ theirs: { state: 'absent', capsule_id: null, evidence_outcome: outcome } }))
          .their.kind
      )
    }
    reached.add(deriveContentToggleState(paneCRow({ role_tag: 'SERVED' })).their.kind)
    expect(reached).toEqual(
      new Set<TheirContentStateKind>([
        'recorded_absence',
        'unanswered',
        'signed_refusal',
        'not_asked',
        'not_visible_holds'
      ])
    )
  })
})

function yourState(kind: YourContentState['kind'], date: string | null = null): YourContentState {
  return { kind, date }
}

function theirState(kind: TheirContentStateKind, date: string | null = null): TheirContentState {
  return { kind, date }
}

describe('yourContentFixedText', () => {
  it('populated has no fixed text -- the caller renders the real quoted content instead', () => {
    expect(yourContentFixedText(yourState('populated'))).toBeNull()
  })

  it('renders the exact copy from v3 §3 for the deleted and streamed-not-retained states', () => {
    expect(yourContentFixedText(yourState('populated_deleted', '5 Sep'))).toBe(
      'Content deleted 5 Sep · record still verifies'
    )
    expect(yourContentFixedText(yourState('streamed_not_retained'))).toBe(
      'No content. You streamed this response and did not retain it.'
    )
  })

  it('never invents a date when none is carried', () => {
    expect(yourContentFixedText(yourState('populated_deleted'))).toContain('date unavailable')
  })
})

describe("theirContentText — v3 §3's exact three sub-states + not-visible-holds", () => {
  it('renders the exact copy from v3 §3', () => {
    expect(theirContentText(theirState('not_visible_holds'))).toBe(
      'Their content — not visible to you. The requester holds it.'
    )
    expect(theirContentText(theirState('recorded_absence', '4 Sep'))).toBe(
      'They state they hold no payload for this exchange — signed 4 Sep.'
    )
    expect(theirContentText(theirState('not_asked'))).toBe('Not compared. They would be expected to hold none.')
    expect(theirContentText(theirState('unanswered', '3 Sep'))).toBe('Asked 3 Sep. No reply yet.')
  })

  it('every their-content sub-state renders visibly distinct text', () => {
    const texts = (['not_visible_holds', 'recorded_absence', 'not_asked', 'unanswered', 'signed_refusal'] as const).map(
      (kind) => theirContentText(theirState(kind, '4 Sep'))
    )
    expect(new Set(texts).size).toBe(texts.length)
  })
})

describe('theirContentAction', () => {
  it('only not_asked carries an action', () => {
    expect(theirContentAction(theirState('not_asked'))).toBe('Ask them to state it')
    for (const kind of ['not_visible_holds', 'recorded_absence', 'unanswered', 'signed_refusal'] as const) {
      expect(theirContentAction(theirState(kind))).toBeNull()
    }
  })
})

describe('LOAD-BEARING — L-H: the your-empty voice and the their-empty voice are never the same string', () => {
  it('streamed_not_retained (a fact you know) never equals any their-content absence claim', () => {
    const yourVoice = yourContentFixedText(yourState('streamed_not_retained'))
    for (const kind of ['recorded_absence', 'not_asked', 'unanswered', 'signed_refusal'] as const) {
      expect(theirContentText(theirState(kind, '4 Sep'))).not.toBe(yourVoice)
    }
  })

  it('ADVERSARIAL — the your-empty string is first-person fact voice ("you streamed"); no their-content string ever uses first-person fact voice', () => {
    const yourVoice = yourContentFixedText(yourState('streamed_not_retained')) ?? ''
    expect(yourVoice.toLowerCase()).toContain('you streamed')
    for (const kind of ['recorded_absence', 'not_asked', 'unanswered', 'signed_refusal'] as const) {
      expect(theirContentText(theirState(kind, '4 Sep')).toLowerCase()).not.toContain('you streamed')
    }
  })

  it('ADVERSARIAL — every their-content absence state is claim-attributed ("they"/"asked"), never asserted as our own fact', () => {
    for (const kind of ['recorded_absence', 'not_asked', 'unanswered'] as const) {
      const text = theirContentText(theirState(kind, '4 Sep')).toLowerCase()
      expect(/they|asked/.test(text)).toBe(true)
    }
  })
})

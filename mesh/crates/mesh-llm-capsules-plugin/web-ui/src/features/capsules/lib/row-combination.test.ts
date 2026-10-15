import { describe, expect, it } from 'vitest'
import type { RightCellStateKind } from '@/features/capsules/lib/exchange-row-state'
import { rowCombinationText, WITNESS_HOLDS_IT, type RowCombinationInput } from '@/features/capsules/lib/row-combination'

const ALL_OK = { content: '✓', sig: '✓', inclusion: '✓', registered: '–' } as const

function input(overrides: Partial<RowCombinationInput>): RowCombinationInput {
  return { state: 'closed', roleTag: 'ASKED', yoursHeld: true, marks: ALL_OK, refusedAtTheDoor: false, late: false, ...overrides }
}

describe('rowCombinationText -- what the badge and the chips say together', () => {
  it('confirmed: in a checkpoint or newer than it (A2/A3), and says when a witness holds it (A18)', () => {
    expect(rowCombinationText(input({}))).toMatch(/^Both records agree and check out/)
    expect(rowCombinationText(input({ marks: { ...ALL_OK, inclusion: '–' } }))).toMatch(/next one will cover it/)
    expect(rowCombinationText(input({ marks: { ...ALL_OK, registered: '✓' } }))).toContain(WITNESS_HOLDS_IT.trim())
  })

  it('a failed check on your own copy wins over any badge (A15)', () => {
    for (const state of ['closed', 'contradicted', 'open_not_held'] as RightCellStateKind[]) {
      expect(rowCombinationText(input({ state, marks: { ...ALL_OK, content: '✗' } }))).toMatch(/^Your own copy fails its checks/)
      expect(rowCombinationText(input({ state, marks: { ...ALL_OK, sig: '✗' } }))).toMatch(/^Your own copy fails its checks/)
    }
  })

  it('disagreement: refused at the door names another server or model (A6); otherwise the record disagrees (A4/A5)', () => {
    expect(rowCombinationText(input({ state: 'contradicted', refusedAtTheDoor: true }))).toMatch(/another server or another model/)
    expect(rowCombinationText(input({ state: 'contradicted' }))).toMatch(/disagrees with it, or doesn’t match its own id/)
  })

  it('not arrived: on its way, then late (A7/A8), never telling you to do something the row doesn’t offer', () => {
    expect(rowCombinationText(input({ state: 'open_not_held' }))).toMatch(/normally arrives/)
    const late = rowCombinationText(input({ state: 'open_not_held', late: true }))
    expect(late).toMatch(/is late/)
    expect(late).not.toMatch(/\bAsk\b/)
  })

  it('only their record, nothing loaded, served locally, unknown other side (A17/A16/A10/A11)', () => {
    expect(rowCombinationText(input({ yoursHeld: false }))).toMatch(/^Only their record exists/)
    expect(rowCombinationText(input({ marks: { content: '–', sig: '–', inclusion: '–', registered: '–' } }))).toMatch(
      /isn’t loaded on this page/
    )
    expect(rowCombinationText(input({ state: 'open_not_asked', roleTag: 'SERVED' }))).toMatch(/You served this yourself/)
    const paid = rowCombinationText(input({ state: 'open_not_asked', roleTag: 'SERVED', servedForPay: true }))
    expect(paid).toMatch(/You served this for pay to another node/)
    expect(paid).not.toMatch(/yourself/)
    expect(rowCombinationText(input({ state: 'open_not_asked', roleTag: 'ASKED' }))).toMatch(/no one to ask/)
  })

  it('every state has its own text, at most two sentences', () => {
    const states: RightCellStateKind[] = [
      'closed', 'contradicted', 'open_refused', 'open_absent', 'open_asked', 'open_not_held', 'open_not_given', 'open_not_asked'
    ]
    const texts = states.map((state) => rowCombinationText(input({ state })))
    expect(new Set(texts).size).toBe(states.length)
    for (const text of texts) expect(text.split(/(?<=[.!?])\s+/).length).toBeLessThanOrEqual(2)
  })
})

import { describe, expect, it } from 'vitest'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import type { RightCellStateKind } from '@/features/capsules/lib/exchange-row-state'
import type { RecomputedIdentity } from '@/features/capsules/lib/recompute-identity'
import { buildChecksRows } from '@/features/capsules/lib/security-checks-view'
import {
  ENTRY_ROW_CHIP_ORDER,
  entryRowChipMark,
  entryRowChipPropertyKey
} from '@/features/capsules/lib/entry-row-chips'

function rowWithProperties(properties: PaneCRow['properties']): PaneCRow {
  return {
    exchange_key: 'exch-1',
    role_tag: 'ASKED',
    header_state: 'ok',
    properties,
    has_issue: false,
    mine: { state: 'present', capsule_id: 'mine-1' },
    theirs: { state: 'absent', capsule_id: null },
    unilateral: false,
    timestamp: null
  }
}

describe('entryRowChipPropertyKey', () => {
  it('names the exact property each chip is a link into, verbatim design §3A', () => {
    expect(entryRowChipPropertyKey('content')).toBe('content_binding')
    expect(entryRowChipPropertyKey('sig')).toBe('producer_signature')
    expect(entryRowChipPropertyKey('inclusion')).toBe('local_inclusion')
    expect(entryRowChipPropertyKey('registered')).toBe('external_registration')
    expect(entryRowChipPropertyKey('theirs')).toBe('outcome_corroboration')
  })

  it('orders the strip content · sig · inclusion · registered · theirs', () => {
    expect(ENTRY_ROW_CHIP_ORDER).toEqual(['content', 'sig', 'inclusion', 'registered', 'theirs'])
  })
})

const NOT_RUN = { idMatch: null, signatureOk: null } as RecomputedIdentity
const RECOMPUTED_OK = { idMatch: true, signatureOk: true } as RecomputedIdentity

function marks(row: PaneCRow, identity: RecomputedIdentity, gateKind?: RightCellStateKind) {
  const checks = buildChecksRows(row, identity, undefined, { gateKind })
  return ENTRY_ROW_CHIP_ORDER.map((chip) => entryRowChipMark(checks, chip)).join(' ')
}

describe('entryRowChipMark -- reads the SAME check results as the checks panel (look finding 1)', () => {
  it('PASS -> ✓, FAIL -> ✗, from the panel rows', () => {
    const row = rowWithProperties({ local_inclusion: { state: 'PASS' }, external_registration: { state: 'FAIL' } })
    expect(marks(row, NOT_RUN)).toBe('– – ✓ ✗ –')
  })

  it.each(['NOT_PRESENT', 'NOT_CHECKED', 'INCONCLUSIVE'])(
    '%s -> the neutral dash, never a fabricated pass/fail',
    (state) => {
      expect(marks(rowWithProperties({ local_inclusion: { state } }), NOT_RUN)).toBe('– – – – –')
    }
  )

  it('REGRESSION: the native pane (`properties: null`) still shows the in-browser recompute -- never all dashes while the panel says established', () => {
    const row = rowWithProperties(null)
    expect(marks(row, RECOMPUTED_OK)).toBe('✓ ✓ – – –')
    const panel = buildChecksRows(row, RECOMPUTED_OK)
    expect(panel.find((r) => r.key === 'content_binding')?.yours?.state).toBe('PASS')
    expect(panel.find((r) => r.key === 'producer_signature')?.yours?.state).toBe('PASS')
  })

  it('the theirs chip follows the ONE gate: CLOSED -> ✓, CONTRADICTED -> ✗, OPEN -> –', () => {
    const row = rowWithProperties(null)
    expect(marks(row, RECOMPUTED_OK, 'closed')).toBe('✓ ✓ – – ✓')
    expect(marks(row, RECOMPUTED_OK, 'contradicted')).toBe('✓ ✓ – – ✗')
    expect(marks(row, RECOMPUTED_OK, 'open_not_held')).toBe('✓ ✓ – – –')
  })
})

import { describe, expect, it } from 'vitest'
import { exchangeEvidenceBundle, exchangeRowsToCsv } from '@/features/capsules/lib/exchange-export'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'

function row(overrides: Partial<ExchangeLedgerRow>): ExchangeLedgerRow {
  return {
    exchangeKey: 'exch-1',
    timestamp: '2026-09-08T00:00:00Z',
    roleTag: 'ASKED',
    counterparty: 'node:aa11bb22',
    confirmed: true,
    localOnly: false,
    hasIssue: false,
    checksText: '—',
    rightCellState: { kind: 'closed', date: null },
    contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'not_asked', date: null } },
    sessionId: null,
    twinBracketId: null,
    raw: {} as PaneCRow,
    ...overrides
  }
}

describe('exchangeRowsToCsv', () => {
  it('is a distinct view export — the header names exactly the table columns', () => {
    const csv = exchangeRowsToCsv([row({})])
    const [header] = csv.split('\n')
    expect(header).toBe('Time,Exchange ID,Counterparty,Your role,Confirmed,Checks')
  })

  it('never renders Checks as a count in the CSV either', () => {
    const csv = exchangeRowsToCsv([row({ checksText: 'checkpoint signature' })])
    expect(csv).toContain('checkpoint signature')
    expect(csv).not.toMatch(/\d\/\d/)
  })

  it('quotes a field containing a comma', () => {
    const csv = exchangeRowsToCsv([row({ checksText: 'checkpoint signature, continuity' })])
    expect(csv).toContain('"checkpoint signature, continuity"')
  })
})

describe('exchangeEvidenceBundle', () => {
  it('is the raw Pane C records, not a re-shaping of the table — a different shape than the CSV', () => {
    const record: PaneCRow = {
      exchange_key: 'exch-1',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: { checkpoint_signature: { state: 'FAIL' } },
      has_issue: true,
      mine: { state: 'present', capsule_id: 'mine-1' },
      theirs: { state: 'present', capsule_id: 'theirs-1' },
      unilateral: false,
      timestamp: '2026-09-08T00:00:00Z'
    }
    const bundle = JSON.parse(exchangeEvidenceBundle([record]))
    expect(bundle.schema).toBe('mesh-ledger-exchanges-evidence/1')
    expect(bundle.exchanges[0].properties.checkpoint_signature.state).toBe('FAIL')
  })
})

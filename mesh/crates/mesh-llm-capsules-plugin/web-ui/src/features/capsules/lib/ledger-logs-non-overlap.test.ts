// NON-OVERLAP INVARIANT: Logs and
// Ledger reuse the same table chrome but must never read as the same tab.
// No Ledger column may show routing/retry/status/category (Logs concepts);
// no Logs column may show an evidence property (Ledger concepts);
// `exchange_id` is the only token allowed to be shared between them.
//
// The Logs tab is the console's, not this plugin's, so its column keys are
// pinned here: the `id`/`accessorKey` of every column
// `buildLogEventLedgerColumns()` returned in mesh-llm-ui
// `src/features/logs/components/LogEventLedgerColumns.tsx` at
// upstream Mesh-LLM/mesh-llm main at commit d635e6156c87a2e7e7d0c0c7173a2b388fb225b2. Re-pin when
// the console's Logs columns change.
import { describe, expect, it } from 'vitest'
import { buildExchangeColumns } from '@/features/capsules/components/ExchangeColumns'

const LOGS_COLUMN_KEYS: ReadonlySet<string> = new Set(['occurredAt', 'category', 'state', 'origin', 'event', 'context'])

const SHARED_ALLOWED = new Set(['exchange_id'])

function columnKeys(columns: ReadonlyArray<{ id?: string; accessorKey?: string }>): Set<string> {
  return new Set(
    columns.map((column) => {
      const key = column.id ?? column.accessorKey
      if (!key) throw new Error('column has neither an id nor an accessorKey — cannot compute its key')
      return key
    })
  )
}

describe('Ledger/Logs column non-overlap invariant', () => {
  it('the Ledger exchange columns and the Logs event columns are disjoint except exchange_id', () => {
    const ledgerKeys = columnKeys(buildExchangeColumns())
    const logsKeys = LOGS_COLUMN_KEYS

    const overlap = [...ledgerKeys].filter((key) => logsKeys.has(key))
    expect(overlap.every((key) => SHARED_ALLOWED.has(key))).toBe(true)
  })

  it('fails the invariant if a routing/status-shaped column is added to the Ledger set (proves the test is a real guard)', () => {
    const ledgerKeys = columnKeys(buildExchangeColumns())
    const logsKeys = LOGS_COLUMN_KEYS

    // Simulate the violation the spec calls out by name: a Ledger column
    // reusing one of Logs' own routing/status-shaped keys.
    const mutatedLedgerKeys = new Set(ledgerKeys)
    mutatedLedgerKeys.add('state')

    const overlap = [...mutatedLedgerKeys].filter((key) => logsKeys.has(key))
    expect(overlap.every((key) => SHARED_ALLOWED.has(key))).toBe(false)
  })

  it('the Ledger set carries no routing/retry/status/category key, and the Logs set carries no evidence-property key', () => {
    const ledgerKeys = columnKeys(buildExchangeColumns())
    const logsKeys = LOGS_COLUMN_KEYS
    const forbiddenOnLedger = ['routing', 'retry', 'status', 'category']
    const forbiddenOnLogs = ['checks', 'confirmed', 'counterparty']

    for (const key of forbiddenOnLedger) expect(ledgerKeys.has(key)).toBe(false)
    for (const key of forbiddenOnLogs) expect(logsKeys.has(key)).toBe(false)
  })
})

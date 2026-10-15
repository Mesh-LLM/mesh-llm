// The settlement wording gate. Settlement on this page shows
// only this node's own payment-lifecycle records and the host's metering:
// invoices issued and settled per leg, the final amount as the host recorded
// it, per-peer counts. It never adds amounts up, never shows a balance, never
// says "unpaid" for an exchange with no payment records, and carries no
// price, priced or pricing wording, nor any other word the gate below
// bans. This test fails on any such
// phrase in the settlement copy, in the settlement sources' shipped code
// (comments left out), or in what the views render for every state.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { PayerBook, SettlementEntry } from '@/features/capsules/api/sidecarTypes'
import {
  peerSettlementText,
  settlementCloseCounts,
  settlementCloseLine,
  settlementEntryViews,
  settlementRowView,
  unjoinedSettlementText,
  providerSettlementText
} from '@/features/capsules/lib/settlement-view'
import * as COPY from '@/features/capsules/lib/tooltip-copy'

const BANNED_SETTLEMENT =
  /\b(unpaid|balances?|debts?|owed?|owes|owing|pric(e|es|ed|ing)|price list|rates?|fees?|tariffs?|totals?|sum(s|med)?|relay(s|ed|ing)?|subscriptions?|markup|margin|invoice total)\b/i

function offenders(label: string, text: string): string[] {
  const match = text.match(BANNED_SETTLEMENT)
  return match ? [`${label}: "${match[0]}" in ${JSON.stringify(text.slice(0, 160))}`] : []
}

/** The settlement copy in `tooltip-copy.ts`, every string of it. */
function settlementCopy(): Array<[string, string]> {
  const out: Array<[string, string]> = []
  const walk = (label: string, value: unknown) => {
    if (typeof value === 'string') out.push([label, value])
    else if (value && typeof value === 'object')
      for (const [key, inner] of Object.entries(value)) walk(`${label}.${key}`, inner)
  }
  for (const [name, value] of Object.entries(COPY)) {
    if (/^(SETTLEMENT_|PEER_PAYMENTS_|CLOSE_CARD_)/.test(name)) walk(name, value)
  }
  return out
}

function withoutComments(code: string): string {
  return code.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`])\/\/.*$/gm, '$1')
}

const HERE = resolve(__dirname, '..')
const SOURCES = ['lib/settlement-view.ts', 'components/SettlementRow.tsx']

const STATES = ['settled', 'no_settlement_seen', 'outcome_not_reported', 'terms_only', 'unmatched_settlement', 'something_new']

function book(state: string, entries: SettlementEntry[] = []): PayerBook {
  return {
    observed_by: 'payer',
    state,
    terms_digests: ['a'.repeat(64)],
    entries,
    provider_book: 'not_available'
  }
}

const ENTRIES: SettlementEntry[] = [
  'terms_accepted',
  'input_invoice_issued',
  'input_settlement_observed',
  'output_invoice_issued',
  'output_settlement_observed',
  'final_accounted'
].map((phase, index) => ({
  capsule_id: `c${index}`,
  timestamp: null,
  phase,
  source: ['payer_asserted', 'provider_asserted', 'wallet_reported', null][index % 4],
  segment: index,
  payment_hash: index % 2 ? null : 'b'.repeat(64),
  amount_msat: 1000 * (index + 1)
}))

describe('settlement wording gate', () => {
  it('the settlement copy exists and carries no banned phrase', () => {
    const copy = settlementCopy()
    expect(copy.length).toBeGreaterThan(8)
    expect(copy.flatMap(([label, text]) => offenders(label, text))).toEqual([])
  })

  it('the settlement sources ship no banned phrase outside comments', () => {
    const problems = SOURCES.flatMap((file) =>
      offenders(file, withoutComments(readFileSync(resolve(HERE, file), 'utf8')))
    )
    expect(problems).toEqual([])
  })

  it('what the views render, for every state, carries no banned phrase', () => {
    const rendered: Array<[string, string]> = []
    for (const state of STATES) {
      const view = settlementRowView({
        ...book(state),
        matched_by_segment_only: state === 'settled'
      })
      if (view)
        rendered.push([
          `row:${state}`,
          [view.chip, view.chipLabel, view.label, view.tooltip, view.providerBook].join(' | ')
        ])
    }
    for (const entry of settlementEntryViews(ENTRIES)) {
      rendered.push([
        `entry:${entry.key}`,
        [entry.phase, entry.sourceLabel, entry.sourceTooltip, entry.amount ?? '', entry.referenceNote ?? ''].join(' | ')
      ])
    }
    const counts = {
      paid_exchanges: 3,
      settled_payer_observed: 1,
      no_settlement_seen: 2,
      provider_book: 'not_available'
    }
    rendered.push(['peer', peerSettlementText(counts) ?? ''])
    const close = settlementCloseCounts([], () => false)
    for (const payments of ['on', 'off', 'unknown', undefined] as const) {
      rendered.push([
        `close:${payments}`,
        settlementCloseLine({ ...close, paid: 2, settled: 1, settledWithoutReference: 1 }, payments)
      ])
      rendered.push([`close-empty:${payments}`, settlementCloseLine(close, payments)])
    }
    rendered.push(['unjoined', unjoinedSettlementText(['x'], 2) ?? ''])
    rendered.push(['provider', providerSettlementText(2) ?? ''])
    expect(rendered.flatMap(([label, text]) => offenders(label, text))).toEqual([])
  })

  it('no payment records renders nothing at all: never "unpaid"', () => {
    expect(settlementRowView(null)).toBeNull()
    expect(settlementRowView(undefined)).toBeNull()
    expect(peerSettlementText(undefined)).toBeNull()
  })

  it('each amount is shown as recorded, on its own entry: never added up', () => {
    const amounts = settlementEntryViews(ENTRIES).map((entry) => entry.amount)
    expect(amounts).toEqual(ENTRIES.map((entry) => `recorded ${entry.amount_msat} msat`))
    const sum = ENTRIES.reduce((acc, entry) => acc + (entry.amount_msat ?? 0), 0)
    const everything = JSON.stringify(settlementEntryViews(ENTRIES))
    expect(everything).not.toContain(String(sum))
  })

  it('the gate catches the phrases it bans', () => {
    for (const phrase of [
      'unpaid',
      'your balance',
      'debts',
      'amount owed',
      'price',
      'priced',
      'pricing',
      'relayed',
      'total',
      'fee'
    ]) {
      expect(offenders('probe', `a ${phrase} here`).length, phrase).toBe(1)
    }
    // Lifecycle facts pass.
    expect(
      offenders('probe', 'terms accepted · paid · settled by your wallet · Final amount · recorded 3000 msat')
    ).toEqual([])
  })
})

import { describe, expect, it } from 'vitest'
import type { PaneCRow, PayerBook, PeerSettlementCounts } from '@/features/capsules/api/sidecarTypes'
import {
  peerSettlementText,
  providerSettlementView,
  settlementCloseCounts,
  settlementCloseLine,
  settlementEntryViews,
  settlementRowView,
  unjoinedSettlementText,
  providerSettlementText
} from '@/features/capsules/lib/settlement-view'

function book(state: string): PayerBook {
  return {
    observed_by: 'payer',
    state,
    terms_digests: ['t'.repeat(64)],
    provider_book: 'not_available',
    entries: [
      {
        capsule_id: 'c1',
        timestamp: '2026-09-27T00:00:00Z',
        phase: 'output_invoice_issued',
        source: 'provider_asserted',
        segment: 1,
        payment_hash: 'b'.repeat(64),
        amount_msat: 123457
      },
      {
        capsule_id: 'c2',
        timestamp: '2026-09-27T00:00:01Z',
        phase: 'output_settlement_observed',
        source: 'wallet_reported',
        segment: 1,
        payment_hash: 'b'.repeat(64),
        amount_msat: 123457
      }
    ]
  }
}

function paneRow(key: string, settlement: PayerBook | null): PaneCRow {
  return {
    exchange_key: key,
    role_tag: 'ASKED',
    header_state: 'absent',
    properties: null,
    has_issue: false,
    mine: { state: 'present-unverified', capsule_id: 'd'.repeat(64) },
    theirs: { state: 'absent', capsule_id: null },
    unilateral: true,
    timestamp: '2026-09-27T00:00:00Z',
    settlement
  }
}

describe('settlementRowView', () => {
  it('renders nothing for an exchange with no payment records, never an unpaid state', () => {
    expect(settlementRowView(null)).toBeNull()
    expect(settlementRowView(undefined)).toBeNull()
  })

  it('names the payer-book state and always says the provider’s book is not available', () => {
    const view = settlementRowView(book('settled'))
    expect(view?.label).toBe('settled · your wallet')
    expect(view?.tone).toBe('good')
    expect(view?.providerBook).toBe('provider’s book: not available')
    expect(settlementRowView(book('no_settlement_seen'))?.label).toBe('no payment seen')
    expect(settlementRowView(book('unmatched_settlement'))?.tone).toBe('bad')
  })

  it('shows an unrecognised state as recorded instead of mapping it onto a known one', () => {
    const view = settlementRowView(book('refunded'))
    expect(view?.stateKey).toBe('unrecognised')
    expect(view?.label).toBe('refunded')
  })
})

describe('settlementRowView refinements', () => {
  it('terms accepted with no invoice reads "terms accepted", not paid', () => {
    expect(settlementRowView(book('terms_only'))?.chip).toBe('terms')
    expect(settlementRowView(book('terms_only'))?.chipLabel).toBe('terms accepted')
    expect(settlementRowView(book('settled'))?.chip).toBe('paid')
  })

  it('a settlement matched by segment alone says it had no reference, never the same-reference sentence', () => {
    const view = settlementRowView({ ...book('settled'), matched_by_segment_only: true })
    expect(view?.stateKey).toBe('settled_without_reference')
    expect(view?.label).toBe('settled · your wallet · no reference')
    expect(view?.tooltip).not.toMatch(/same payment reference/)
  })

  it('the page never accepts its own refinement as a state from the wire', () => {
    expect(settlementRowView(book('settled_without_reference'))?.stateKey).toBe('unrecognised')
  })

  it('records naming different terms say so', () => {
    expect(settlementRowView({ ...book('settled'), terms_digests: ['a', 'b'] })?.termsNote).toBe(
      'these payment records name different terms'
    )
    expect(settlementRowView(book('settled'))?.termsNote).toBeNull()
  })
})

describe('settlementEntryViews', () => {
  it('keeps each recorded amount as recorded and names who stated it', () => {
    const [invoice, settled] = settlementEntryViews(book('settled').entries)
    expect(invoice.phase).toBe('Invoice for the answer')
    expect(invoice.sourceLabel).toBe('they stated')
    expect(invoice.amount).toBe('recorded 123457 msat')
    expect(settled.sourceLabel).toBe('your wallet reported')
  })

  it('a payment step with no payment reference says so', () => {
    const [entry] = settlementEntryViews([{ ...book('settled').entries[1], payment_hash: null }])
    expect(entry.referenceNote).toBe('no payment reference recorded')
    expect(settlementEntryViews(book('settled').entries)[1].referenceNote).toBeNull()
  })

  it('an entry with no amount shows none rather than zero', () => {
    const [entry] = settlementEntryViews([{ ...book('settled').entries[0], amount_msat: null }])
    expect(entry.amount).toBeNull()
  })
})

describe('peerSettlementText', () => {
  const counts: PeerSettlementCounts = {
    paid_exchanges: 3,
    settled_payer_observed: 2,
    no_settlement_seen: 1,
    provider_book: 'not_available'
  }

  it('counts paid and settled from your own records, and names the provider’s book as not available', () => {
    const text = peerSettlementText(counts)
    expect(text).toBe('3 paid · 2 settled by your wallet · 1 no payment seen · provider’s book: not available')
    expect(text).not.toMatch(/lapsed|debt/)
  })

  it('never renders an amount or a rate', () => {
    expect(peerSettlementText(counts)).not.toMatch(/msat|%|rate/)
  })

  it('says nothing for a peer with no paid exchange', () => {
    expect(peerSettlementText(undefined)).toBeNull()
    expect(
      peerSettlementText({ ...counts, paid_exchanges: 0, settled_payer_observed: 0, no_settlement_seen: 0 })
    ).toBeNull()
  })
})

describe('Close card counts', () => {
  it('counts inference CLOSED through the gate it is given, and settled from each row', () => {
    const rows = [paneRow('a', book('settled')), paneRow('b', book('no_settlement_seen')), paneRow('c', null)]
    const counts = settlementCloseCounts(rows, (row) => row.exchange_key === 'a')
    expect(counts).toEqual({ exchanges: 3, closed: 1, paid: 2, settled: 1, settledWithoutReference: 0 })
    expect(settlementCloseLine(counts, 'on')).toBe('2 paid · 1 settled by your wallet · provider’s book: not available')
  })

  it('a row that reads "no reference" is counted and worded the same way on the card', () => {
    const noReference = { ...book('settled'), matched_by_segment_only: true }
    expect(settlementRowView(noReference)?.label).toBe('settled · your wallet · no reference')
    const counts = settlementCloseCounts([paneRow('a', noReference), paneRow('b', book('settled'))], () => false)
    expect(counts).toMatchObject({ paid: 2, settled: 1, settledWithoutReference: 1 })
    expect(settlementCloseLine(counts, 'on')).toBe(
      '2 paid · 1 settled by your wallet · 1 settled · no reference · provider’s book: not available'
    )
  })

  it('one exchange book attached to two rows is counted once', () => {
    const shared = { ...book('settled'), exchange_ids: ['ex-1'] }
    const other = { ...book('no_settlement_seen'), exchange_ids: ['ex-2'] }
    const counts = settlementCloseCounts(
      [paneRow('a', shared), paneRow('a#2', shared), paneRow('b', other)],
      () => false
    )
    expect(counts).toMatchObject({ paid: 2, settled: 1, settledWithoutReference: 0 })
  })

  it('payments off reads as off, never as zero settled', () => {
    const counts = settlementCloseCounts([paneRow('a', null)], () => false)
    expect(settlementCloseLine(counts, 'off')).toBe('Payments: off on this node.')
    expect(settlementCloseLine(counts, 'unknown')).toBe('Payments: not known for this node.')
    expect(settlementCloseLine(counts, undefined)).toBe('Payments: not known for this node.')
  })

  it('an exchange with terms accepted and no invoice is not counted as paid', () => {
    const counts = settlementCloseCounts([paneRow('a', book('terms_only'))], () => false)
    expect(counts.paid).toBe(0)
  })

  it('records held on a node now reporting payments off are still counted', () => {
    const counts = settlementCloseCounts([paneRow('a', book('settled'))], () => false)
    expect(settlementCloseLine(counts, 'off')).toBe(
      '1 paid · 1 settled by your wallet · provider’s book: not available'
    )
  })
})

describe('providerSettlementText', () => {
  it('counts the records kept as the provider, and says nothing when there are none', () => {
    expect(providerSettlementText(1)).toBe('1 payment record kept as the provider.')
    expect(providerSettlementText(3)).toBe('3 payment records kept as the provider.')
    expect(providerSettlementText(0)).toBeNull()
    expect(providerSettlementText(undefined)).toBeNull()
  })
})

describe('unjoinedSettlementText', () => {
  it('counts records no row carries instead of dropping them', () => {
    expect(unjoinedSettlementText(['x'])).toBe('1 payment record not matched to an exchange here.')
    expect(unjoinedSettlementText(['x', 'y'])).toBe('2 payment records not matched to an exchange here.')
    expect(unjoinedSettlementText([])).toBeNull()
    expect(unjoinedSettlementText(undefined)).toBeNull()
    expect(unjoinedSettlementText([], 1)).toBe('1 payment record not matched to an exchange here.')
    expect(unjoinedSettlementText(['x'], 2)).toBe('3 payment records not matched to an exchange here.')
  })
})

const settled_entry = () => ({
  capsule_id: 'p',
  timestamp: null,
  phase: 'input_settlement_observed',
  source: 'wallet_reported',
  segment: 0,
  payment_hash: 'aa',
  amount_msat: 120
})

describe('the paid-exchange view: what was charged, who paid, and outcomes the host did not report', () => {
  it('an invoice with no reported outcome says so, as a warning, never as unpaid', () => {
    const view = settlementRowView({ ...book('outcome_not_reported'), final_accounted_msat: null })
    expect(view?.stateKey).toBe('outcome_not_reported')
    expect(view?.label).toBe('outcome not reported')
    expect(view?.tone).toBe('warn')
    expect(view?.tooltip).toMatch(/uncertain/)
    expect(view?.tooltip).toMatch(/can’t say which/)
    expect(view?.finalAmount).toBeNull()
    expect(view?.label).not.toMatch(/unpaid/)
    expect(`${view?.label} ${view?.tooltip}`).not.toMatch(/balance/)
  })

  it('shows the payer’s own final amount as recorded, never a sum made here', () => {
    const view = settlementRowView({ ...book('settled'), final_accounted_msat: 579 })
    expect(view?.finalAmount).toBe('final amount, as you recorded it: 579 msat')
  })

  it('lists the wallet’s credited amount, fee and a delivered watermark on each step that carries them', () => {
    const [plain, settled, delivered] = settlementEntryViews([
      { capsule_id: 'a', timestamp: null, phase: 'input_invoice_issued', source: 'provider_asserted', segment: 0, payment_hash: 'aa', amount_msat: 120 },
      { capsule_id: 'b', timestamp: null, phase: 'input_settlement_observed', source: 'wallet_reported', segment: 0, payment_hash: 'aa', amount_msat: 120, credited_msat: 119, fee_msat: 1 },
      { capsule_id: 'c', timestamp: null, phase: 'delivered', source: 'provider_asserted', segment: null, payment_hash: null, amount_msat: 0, tokens: 42 }
    ], 'provider')
    expect(plain.walletNote).toBeNull()
    expect(settled.walletNote).toBe('credited 119 msat · 1 msat deducted')
    expect(settlementEntryViews([{ ...settled_entry(), fee_msat: 2 }])[0].walletNote).toBe('2 msat to route it')
    expect(delivered.walletNote).toBe('42 tokens')
    expect(delivered.phase).toBe('Delivered')
    // On the provider's own book, the provider is "you".
    expect(plain.sourceLabel).toBe('you stated')
    expect(settled.sourceLabel).toBe('your wallet reported')
  })

  it('the provider’s own book names who paid, the state its wallet reported, and the delivered watermark', () => {
    const provider: PayerBook = {
      observed_by: 'provider',
      who_paid: 'requester',
      state: 'settled',
      terms_digests: ['t'.repeat(64)],
      entries: [],
      delivered_tokens: 42
    }
    const view = providerSettlementView(provider, 'node:aa11bb22cc33')
    expect(view?.label).toBe('received · your wallet')
    expect(view?.whoPaid).toMatch(/^paid by /)
    expect(view?.delivered).toBe('42 tokens delivered, as you recorded')
    // A served row with no recorded counterparty says so; it never guesses.
    expect(providerSettlementView(provider, null)?.whoPaid).toBe('payer not recorded')
    expect(providerSettlementView(provider, undefined)?.whoPaid).toBe('payer not recorded')
    expect(providerSettlementView(provider, '')?.whoPaid).toBe('payer not recorded')
    expect(providerSettlementView({ ...provider, state: 'outcome_not_reported' }, null)?.tone).toBe('warn')
    expect(providerSettlementView({ ...provider, state: 'something_new' }, null)?.stateKey).toBe('unrecognised')
    expect(providerSettlementView(null, null)).toBeNull()
  })

  it('the Peers line counts outcomes not reported, and still no amounts', () => {
    const counts: PeerSettlementCounts = {
      paid_exchanges: 2,
      settled_payer_observed: 1,
      no_settlement_seen: 0,
      outcome_not_reported: 1,
      provider_book: 'not_available'
    }
    const text = peerSettlementText(counts)
    expect(text).toContain('1 outcome not reported')
    expect(text).not.toMatch(/msat|balance/)
  })
})

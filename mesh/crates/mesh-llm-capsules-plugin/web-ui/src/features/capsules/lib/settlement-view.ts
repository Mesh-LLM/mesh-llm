// Settlement on the Evidence tab: this node's own sealed records of the
// payer-side payment lifecycle, as the host joins them to exchanges
// (`capsule_panes_settlement.rs`). Everything here reads those records; it
// never adds amounts up, never shows a balance, and never says "unpaid" for an
// exchange that simply has no payment records.
import type {
  PaneCRow,
  PayerBook,
  PaymentsPresence,
  PeerSettlementCounts,
  SettlementEntry
} from '@/features/capsules/api/sidecarTypes'
import { shortId } from '@/features/capsules/lib/short-id'
import {
  SETTLEMENT_PROVIDER_SOURCE_TOOLTIPS,
  SETTLEMENT_PROVIDER_STATE_TOOLTIPS,
  SETTLEMENT_SOURCE_TOOLTIPS,
  SETTLEMENT_STATE_TOOLTIPS
} from '@/features/capsules/lib/tooltip-copy'

/** Said wherever settlement appears, until the provider side emits its own
 *  observations: only the payer's records exist on this node. */
export const PROVIDER_BOOK_NOT_AVAILABLE_TEXT = 'provider’s book: not available'

export type SettlementStateKey = keyof typeof SETTLEMENT_STATE_TOOLTIPS

const STATE_LABEL: Record<SettlementStateKey, string> = {
  settled: 'settled · your wallet',
  settled_without_reference: 'settled · your wallet · no reference',
  no_settlement_seen: 'no payment seen',
  outcome_not_reported: 'outcome not reported',
  terms_only: 'terms only',
  unmatched_settlement: 'payment names no invoice'
}

const STATE_TONE: Record<SettlementStateKey, 'good' | 'warn' | 'bad' | 'muted'> = {
  settled: 'good',
  settled_without_reference: 'good',
  no_settlement_seen: 'warn',
  outcome_not_reported: 'warn',
  terms_only: 'muted',
  unmatched_settlement: 'bad'
}

function isStateKey(state: string): state is SettlementStateKey {
  return Object.hasOwn(STATE_LABEL, state)
}

export type SettlementRowView = {
  /** `paid` once an invoice exists; `terms` for terms accepted with no
   *  invoice. */
  chip: 'paid' | 'terms'
  /** The chip's words. */
  chipLabel: string
  stateKey: SettlementStateKey | 'unrecognised'
  label: string
  tooltip: string
  tone: 'good' | 'warn' | 'bad' | 'muted'
  providerBook: string
  /** Said when the records disagree on the terms they name. */
  termsNote: string | null
  /** The payer's own final amount, as recorded (`final_accounted`), or
   *  `null` when there is none. Never a sum made here, never a balance. */
  finalAmount: string | null
}

/** The payer's final amount, as it recorded it. */
function finalAmountText(settlement: PayerBook): string | null {
  return typeof settlement.final_accounted_msat === 'number'
    ? `final amount, as you recorded it: ${settlement.final_accounted_msat} msat`
    : null
}

/** The row face for a paid exchange, or `null` when the row has no payment
 *  records -- a free exchange, payments off, or a request that failed before
 *  it was invoiced. `null` renders nothing: no chip, and never "unpaid". */
export function settlementRowView(settlement: PayerBook | null | undefined): SettlementRowView | null {
  if (!settlement) return null
  const providerBook = PROVIDER_BOOK_NOT_AVAILABLE_TEXT
  const chip = settlement.state === 'terms_only' ? 'terms' : 'paid'
  const chipLabel = chip === 'paid' ? 'paid' : 'terms accepted'
  const termsNote = settlement.terms_digests.length > 1 ? 'these payment records name different terms' : null
  const finalAmount = finalAmountText(settlement)
  // Only the host's own states are read; the page's refinement of `settled`
  // is never accepted from the wire.
  if (!isStateKey(settlement.state) || settlement.state === 'settled_without_reference') {
    // A state this page doesn't know is shown as what it is, not mapped onto
    // a known one.
    return {
      chip,
      chipLabel,
      stateKey: 'unrecognised',
      label: settlement.state,
      tooltip: 'A payment state this page does not recognise, shown as recorded.',
      tone: 'muted',
      providerBook,
      termsNote,
      finalAmount
    }
  }
  const state: SettlementStateKey =
    settlement.state === 'settled' && settlement.matched_by_segment_only
      ? 'settled_without_reference'
      : settlement.state
  return {
    chip,
    chipLabel,
    stateKey: state,
    label: STATE_LABEL[state],
    tooltip: SETTLEMENT_STATE_TOOLTIPS[state],
    tone: STATE_TONE[state],
    providerBook,
    termsNote,
    finalAmount
  }
}

export type ProviderStateKey = keyof typeof SETTLEMENT_PROVIDER_STATE_TOOLTIPS

const PROVIDER_STATE_LABEL: Record<ProviderStateKey, string> = {
  settled: 'received · your wallet',
  outcome_not_reported: 'outcome not reported',
  terms_only: 'terms only',
  unmatched_settlement: 'payment names no invoice'
}

const PROVIDER_STATE_TONE: Record<ProviderStateKey, 'good' | 'warn' | 'bad' | 'muted'> = {
  settled: 'good',
  outcome_not_reported: 'warn',
  terms_only: 'muted',
  unmatched_settlement: 'bad'
}

export type ProviderSettlementView = {
  stateKey: ProviderStateKey | 'unrecognised'
  label: string
  tooltip: string
  tone: 'good' | 'warn' | 'bad' | 'muted'
  /** Who paid: the node that requested the exchange, by its id. When the row
   *  records no counterparty, it says so rather than guessing. */
  whoPaid: string
  /** The delivered-token watermark this node recorded, or `null`. */
  delivered: string | null
  termsNote: string | null
}

/** The face of this node's own book as the provider of a paid exchange, or
 *  `null` when it has none. */
export function providerSettlementView(
  book: PayerBook | null | undefined,
  requester: string | null | undefined
): ProviderSettlementView | null {
  if (!book) return null
  const whoPaid = requester ? `paid by ${shortId(requester)}` : 'payer not recorded'
  const delivered =
    typeof book.delivered_tokens === 'number' ? `${book.delivered_tokens} tokens delivered, as you recorded` : null
  const termsNote = book.terms_digests.length > 1 ? 'these payment records name different terms' : null
  if (!Object.hasOwn(PROVIDER_STATE_LABEL, book.state)) {
    return {
      stateKey: 'unrecognised',
      label: book.state,
      tooltip: 'A payment state this page does not recognise, shown as recorded.',
      tone: 'muted',
      whoPaid,
      delivered,
      termsNote
    }
  }
  const state = book.state as ProviderStateKey
  return {
    stateKey: state,
    label: PROVIDER_STATE_LABEL[state],
    tooltip: SETTLEMENT_PROVIDER_STATE_TOOLTIPS[state],
    tone: PROVIDER_STATE_TONE[state],
    whoPaid,
    delivered,
    termsNote
  }
}

const PHASE_LABEL: Record<string, string> = {
  terms_accepted: 'Terms accepted',
  input_invoice_issued: 'Invoice for the request',
  input_settlement_observed: 'Request invoice paid',
  output_invoice_issued: 'Invoice for the answer',
  output_settlement_observed: 'Answer invoice paid',
  final_accounted: 'Final amount',
  delivered: 'Delivered'
}

const SOURCE_LABEL: Record<keyof typeof SETTLEMENT_SOURCE_TOOLTIPS, string> = {
  payer_asserted: 'you recorded',
  provider_asserted: 'they stated',
  wallet_reported: 'your wallet reported'
}

/** On this node's own provider book, "they" and "you" swap for what each
 *  side asserted. */
const PROVIDER_SOURCE_LABEL: Record<keyof typeof SETTLEMENT_SOURCE_TOOLTIPS, string> = {
  payer_asserted: 'they recorded',
  provider_asserted: 'you stated',
  wallet_reported: 'your wallet reported'
}

export type SettlementEntryView = {
  key: string
  phase: string
  sourceKey: keyof typeof SETTLEMENT_SOURCE_TOOLTIPS | null
  sourceLabel: string
  sourceTooltip: string
  /** The recorded amount, as recorded -- `null` when the record carries none. */
  amount: string | null
  paymentHash: string | null
  /** Said when a payment step carries no payment reference to match on. */
  referenceNote: string | null
  /** The wallet's credited amount and fee, and a delivered watermark, as
   *  recorded -- `null` when the record carries none. */
  walletNote: string | null
  timestamp: string | null
}

/** The wallet's own numbers on a step: what it credited, and what routing
 *  the payment took from the payer (paid on top) or the provider (deducted). */
function walletNote(entry: SettlementEntry, perspective: 'payer' | 'provider'): string | null {
  const parts: string[] = []
  if (typeof entry.credited_msat === 'number') parts.push(`credited ${entry.credited_msat} msat`)
  if (typeof entry.fee_msat === 'number')
    parts.push(
      perspective === 'provider' ? `${entry.fee_msat} msat deducted` : `${entry.fee_msat} msat to route it`
    )
  if (typeof entry.tokens === 'number') parts.push(`${entry.tokens} tokens`)
  return parts.length > 0 ? parts.join(' · ') : null
}

function isSourceKey(source: string | null): source is keyof typeof SETTLEMENT_SOURCE_TOOLTIPS {
  return source !== null && Object.hasOwn(SOURCE_LABEL, source)
}

export function settlementEntryViews(
  entries: readonly SettlementEntry[],
  perspective: 'payer' | 'provider' = 'payer'
): SettlementEntryView[] {
  const labels = perspective === 'provider' ? PROVIDER_SOURCE_LABEL : SOURCE_LABEL
  const tooltips = perspective === 'provider' ? SETTLEMENT_PROVIDER_SOURCE_TOOLTIPS : SETTLEMENT_SOURCE_TOOLTIPS
  return entries.map((entry, index) => {
    const source = typeof entry.source === 'string' ? entry.source : null
    const known = isSourceKey(source)
    return {
      key: entry.capsule_id ?? `${entry.phase}-${index}`,
      phase: PHASE_LABEL[entry.phase] ?? entry.phase,
      sourceKey: known ? source : null,
      sourceLabel: known ? labels[source] : (source ?? 'source not recorded'),
      sourceTooltip: known ? tooltips[source] : 'Who stated this value was not recorded.',
      amount: typeof entry.amount_msat === 'number' ? `recorded ${entry.amount_msat} msat` : null,
      paymentHash: entry.payment_hash,
      referenceNote:
        entry.payment_hash === null && entry.phase.endsWith('_settlement_observed')
          ? 'no payment reference recorded'
          : null,
      walletNote: walletNote(entry, perspective),
      timestamp: entry.timestamp
    }
  })
}

/** The Peers line: counts of this peer's paid exchanges, never amounts or a
 *  rate. The provider's own book is named as not available, never counted as
 *  zero. `null` when the peer has no paid exchange on record. */
export function peerSettlementText(counts: PeerSettlementCounts | undefined): string | null {
  if (!counts || counts.paid_exchanges === 0) return null
  const parts = [`${counts.paid_exchanges} paid`, `${counts.settled_payer_observed} settled by your wallet`]
  if (counts.no_settlement_seen > 0) parts.push(`${counts.no_settlement_seen} no payment seen`)
  if ((counts.outcome_not_reported ?? 0) > 0) parts.push(`${counts.outcome_not_reported} outcome not reported`)
  parts.push(PROVIDER_BOOK_NOT_AVAILABLE_TEXT)
  return parts.join(' · ')
}

export type SettlementCloseCounts = {
  exchanges: number
  closed: number
  paid: number
  /** Rows that read "settled · your wallet". */
  settled: number
  /** Rows that read "settled · your wallet · no reference". */
  settledWithoutReference: number
}

/** The Close card's counts over the pane's rows: CLOSED through the predicate
 *  passed in (Integrity passes the one its "Confirmed by the other side" tile
 *  uses), paid and settled from each row's payer-book state, read through the
 *  row's own face so the card and the rows can never disagree. A row whose
 *  terms were accepted with no invoice is not paid. A book the host
 *  attaches to two rows (one exchange id on both) is counted once. */
export function settlementCloseCounts(
  rows: readonly PaneCRow[],
  isClosed: (row: PaneCRow) => boolean
): SettlementCloseCounts {
  let closed = 0
  let paid = 0
  let settled = 0
  let settledWithoutReference = 0
  const counted = new Set<string>()
  for (const row of rows) {
    if (isClosed(row)) closed += 1
    const view = settlementRowView(row.settlement)
    if (view?.chip !== 'paid') continue
    const ids = row.settlement?.exchange_ids
    const book = ids && ids.length > 0 ? `ids:${[...ids].sort().join(' ')}` : `row:${row.exchange_key}`
    if (counted.has(book)) continue
    counted.add(book)
    paid += 1
    if (view.stateKey === 'settled') settled += 1
    if (view.stateKey === 'settled_without_reference') settledWithoutReference += 1
  }
  return { exchanges: rows.length, closed, paid, settled, settledWithoutReference }
}

/** The Close card's payment line, in the rows' own words. Payments off says
 *  so; unknown says so; neither ever reads as "0 settled". */
export function settlementCloseLine(counts: SettlementCloseCounts, payments: PaymentsPresence | undefined): string {
  if (payments === 'off' && counts.paid === 0) return 'Payments: off on this node.'
  if (payments !== 'on' && counts.paid === 0) return 'Payments: not known for this node.'
  const parts = [`${counts.paid} paid`, `${counts.settled} settled by your wallet`]
  if (counts.settledWithoutReference > 0) parts.push(`${counts.settledWithoutReference} settled · no reference`)
  parts.push(PROVIDER_BOOK_NOT_AVAILABLE_TEXT)
  return parts.join(' · ')
}

/** Settlement records the host could not join to any row -- by an exchange
 *  id no row carries, or with no exchange id at all -- are still records this
 *  node holds; they are counted, never dropped. */
/** This node's own records as the provider of a paid exchange. They are its
 *  book, not the payer's, so they are counted and never shown in a payer row. */
export function providerSettlementText(count: number | undefined): string | null {
  if (!count) return null
  return `${count} payment record${count === 1 ? '' : 's'} kept as the provider.`
}

export function unjoinedSettlementText(ids: readonly string[] | undefined, missingExchangeId = 0): string | null {
  const count = (ids?.length ?? 0) + missingExchangeId
  if (count === 0) return null
  return `${count} payment record${count === 1 ? '' : 's'} not matched to an exchange here.`
}

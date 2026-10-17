// component-level enforcement of v3 §2's
// normative rules, on top of the pure-function tests in
// `exchange-row-state.test.ts` / `exchange-stream.test.ts`.
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ExchangeStreamRow } from '@/features/capsules/components/ExchangeStreamRow'
import { exchangeRowDomId } from '@/features/capsules/lib/exchange-pages'
import { formatExchangeTimestamp } from '@/features/capsules/lib/local-time'
import {
  CLOSED_FROM_FETCH_NOT_SAVED,
  OWN_RECORD_FAILS_WARNING,
  SEE_IN_LOGS_NOT_ON_THIS_PAGE,
  SERVED_FOR_PAY_NO_PAYER_TOOLTIP
} from '@/features/capsules/lib/tooltip-copy'
import { buildExchangeLedgerRows, type ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import { ASK_FOR_RECORD_AFTER_MS, type RightCellStateKind } from '@/features/capsules/lib/exchange-row-state'
import { durationText, formatModelIdentity, tokenFlowText } from '@/features/capsules/lib/serving-provenance'
import type { RailSegment } from '@/features/capsules/lib/exchange-stream'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import {
  usePeerLedgerRecompute,
  useRecomputedIdentity,
  type PeerRecomputeState
} from '@/features/capsules/lib/recompute-identity'
import { fixtureHalfBody, fixtureMineCell, fixtureTheirsCell } from '@/features/capsules/lib/pushed-half-fixtures'

const REQUEST_DIGEST = 'a'.repeat(64)
const RESPONSE_DIGEST = 'b'.repeat(64)

/** Our own record, carrying the §6.2/L-G digests a peer's fetched record
 *  must cite for `deriveRightCellState` to land on CLOSED. Passed via
 *  `toggleProps()` on every render -- harmless for every non-`closed` kind,
 *  since `deriveRightCellState` never reads `localRecord` unless
 *  `idMatch`/`signatureOk` both already came back true. */
/** Our requested record: the digests, the node that served it, and the client
 *  nonce both records carry -- what "Ask them for their record" names the
 *  exchange by. */
const LOCAL_RECORD_WITH_DIGESTS: CapsuleRecord = (() => {
  const body = fixtureHalfBody({ capsuleId: 'mine-1' })
  const poc = (body.model_attestation as Record<string, Record<string, Record<string, unknown>>>).compute_attestation[
    'x-mesh-poc-v1'
  ]
  poc.role = 'requested'
  poc.client_nonce = 'nonce-mine-1'
  return body as CapsuleRecord
})()

// finding 1: `ExchangeStreamRow`
// now derives its own right-cell state from `row.raw` + a live
// `usePeerLedgerRecompute` fetch, ignoring any `rightCellState` a caller
// stuffs into the row prop directly (that field is only ever the ledger's
// at-rest default, never something a component trusts for CLOSED/
// CONTRADICTED). Mocked here so component tests can still drive every
// state, including the two ('closed'/'contradicted') that structurally
// require a resolved fetch this render-only harness never performs for
// real.
vi.mock('@/features/capsules/lib/recompute-identity', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/features/capsules/lib/recompute-identity')>()
  return {
    ...actual,
    usePeerLedgerRecompute: vi.fn(),
    useRecomputedIdentity: vi.fn(actual.useRecomputedIdentity)
  }
})

const NOT_FETCHED: PeerRecomputeState = {
  status: 'not_fetched',
  idMatch: null,
  signatureOk: null,
  peerRecord: null,
  fetch: vi.fn()
}

/** The `raw.theirs` shape and mocked fetch outcome that together make
 *  `deriveRightCellState` land on exactly `kind`, mirroring the real
 *  producer/fetch combinations `exchange-row-state.test.ts` exercises at
 *  the pure-function level. */
function fixturesFor(kind: RightCellStateKind): { theirs: PaneCRow['theirs']; recompute: PeerRecomputeState } {
  switch (kind) {
    case 'closed':
      return {
        theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' },
        recompute: {
          ...NOT_FETCHED,
          status: 'found',
          idMatch: true,
          signatureOk: true,
          peerRecord: fixtureHalfBody({ capsuleId: 'a'.repeat(64) })
        }
      }
    case 'contradicted':
      return {
        theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' },
        recompute: { ...NOT_FETCHED, status: 'found', idMatch: false, signatureOk: false, peerRecord: {} }
      }
    case 'open_refused':
      return {
        theirs: {
          state: 'absent',
          capsule_id: null,
          evidence_outcome: 'signed_refusal',
          evidence_outcome_date: '4 Sep'
        },
        recompute: NOT_FETCHED
      }
    case 'open_absent':
      return {
        theirs: {
          state: 'absent',
          capsule_id: null,
          evidence_outcome: 'recorded_absence',
          evidence_outcome_date: '4 Sep'
        },
        recompute: NOT_FETCHED
      }
    case 'open_asked':
      return {
        theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'unanswered', evidence_outcome_date: '4 Sep' },
        recompute: NOT_FETCHED
      }
    case 'open_not_held':
      return {
        theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' },
        recompute: NOT_FETCHED
      }
    case 'open_not_given':
      // Present peer, but only a non-digest self-minted correlation marker --
      // nothing fetchable.
      return {
        theirs: { state: 'NOT_CHECKED', capsule_id: 'capsule-chatcmpl-1', peer_id: 'peer-1' },
        recompute: NOT_FETCHED
      }
    case 'open_not_asked':
      return { theirs: { state: 'absent', capsule_id: null }, recompute: NOT_FETCHED }
    default: {
      const exhaustiveCheck: never = kind
      return exhaustiveCheck
    }
  }
}

function makeRow(kind: RightCellStateKind, overrides: Partial<ExchangeLedgerRow> = {}): ExchangeLedgerRow {
  const { theirs, recompute } = fixturesFor(kind)
  vi.mocked(usePeerLedgerRecompute).mockReturnValue(recompute)
  return {
    exchangeKey: `exch-${kind}`,
    timestamp: '2026-09-08T16:58:05Z',
    roleTag: 'ASKED',
    counterparty: 'node:aa11bb22',
    confirmed: kind === 'closed',
    localOnly: false,
    hasIssue: kind === 'contradicted',
    checksText: '—',
    rightCellState: {
      kind,
      date: kind === 'open_refused' || kind === 'open_absent' || kind === 'open_asked' ? '4 Sep' : null
    },
    sessionId: null,
    twinBracketId: null,
    contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'not_asked', date: null } },
    raw: {
      mine: { state: 'present', capsule_id: 'mine-1' },
      theirs
    } as PaneCRow,
    ...overrides
  }
}

/** Our record when it names no other side: nothing to ask, whatever the time. */
const RECORD_NAMING_NO_PEER: CapsuleRecord = fixtureHalfBody({
  capsuleId: 'mine-1',
  servedBy: 'unknown'
}) as CapsuleRecord

const NO_RAIL: RailSegment = { hasRail: false, isSegmentStart: false }

/** Every test needs these two now that the modal is gone -- named to make
 *  call sites read like "row props", not boilerplate. */
function toggleProps() {
  return {
    localRecord: LOCAL_RECORD_WITH_DIGESTS,
    onToggleChecks: vi.fn(),
    onToggleContent: vi.fn(),
    onAskForRecord: vi.fn()
  }
}

describe('ExchangeStreamRow — a served exchange with no other side', () => {
  const localRow = () =>
    makeRow('open_not_asked', { roleTag: 'SERVED', counterparty: null, localOnly: true })

  it('is badged LOCAL ("this node only"), muted, never OPEN', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={RECORD_NAMING_NO_PEER}
        rail={NO_RAIL}
        row={localRow()}
      />
    )
    const badge = screen.getByText('LOCAL')
    expect(badge.closest('[data-row-tone]')).toHaveAttribute('data-row-tone', 'muted')
    expect(screen.queryByText(/^OPEN/)).not.toBeInTheDocument()
    expect(screen.getByText('Local — no other side')).toBeInTheDocument()
    expect(document.querySelector('[data-bracket-strip="true"]')).toHaveTextContent('Theirs ○ no other side')
    expect(screen.getByRole('button', { name: /the LOCAL state/i })).toBeInTheDocument()
  })

  /** A served row with no counterparty and no ask target, classified by the
   *  real `buildExchangeLedgerRows` (never a hand-set `localOnly`). */
  const classifiedServedRow = (providerSettlement: PaneCRow['provider_settlement']) => {
    const base = makeRow('open_not_asked')
    const raw = {
      ...base.raw,
      exchange_key: base.exchangeKey,
      timestamp: base.timestamp,
      role_tag: 'SERVED',
      counterparty: null,
      has_issue: false,
      provider_settlement: providerSettlement
    } as PaneCRow
    const [row] = buildExchangeLedgerRows([raw], new Map(), new Map([['mine-1', RECORD_NAMING_NO_PEER]]))
    return row
  }
  const renderClassified = (row: ExchangeLedgerRow) =>
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={RECORD_NAMING_NO_PEER}
        rail={NO_RAIL}
        row={row}
      />
    )

  it('a row served for pay with no payer recorded is OPEN, never LOCAL or "served this yourself"', () => {
    const row = classifiedServedRow({
      observed_by: 'provider',
      who_paid: 'requester',
      state: 'settled',
      terms_digests: ['t'.repeat(64)],
      entries: []
    })
    expect(row.rightCellState.kind).toBe('open_not_asked')
    renderClassified(row)
    expect(screen.getByText('OPEN')).toBeInTheDocument()
    expect(screen.queryByText('LOCAL')).not.toBeInTheDocument()
    expect(screen.queryByText('Local — no other side')).not.toBeInTheDocument()
    expect(screen.getByText('Served for pay — payer not recorded')).toBeInTheDocument()
    // The badge's (i) says it may stay open, and why.
    expect(screen.getAllByText(SERVED_FOR_PAY_NO_PAYER_TOOLTIP).length).toBeGreaterThan(0)
    expect(document.body).not.toHaveTextContent(/served this yourself/)
  })

  it('the same row with no provider book is LOCAL, through the same classifier', () => {
    renderClassified(classifiedServedRow(null))
    expect(screen.getByText('LOCAL')).toBeInTheDocument()
    expect(screen.getByText('Local — no other side')).toBeInTheDocument()
    expect(document.body).not.toHaveTextContent(SERVED_FOR_PAY_NO_PAYER_TOOLTIP)
  })

  it('a served row that names its counterparty stays OPEN', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={RECORD_NAMING_NO_PEER}
        rail={NO_RAIL}
        row={makeRow('open_not_asked', { roleTag: 'SERVED' })}
      />
    )
    expect(screen.getByText('OPEN')).toBeInTheDocument()
    expect(screen.queryByText('LOCAL')).not.toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — L-A/L-B alarm styling', () => {
  it('L-B: CONTRADICTED renders the bad tone (alarm)', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('contradicted')} />)
    const badge = screen.getByText('CONTRADICTED')
    expect(badge.style.color).toBe('var(--color-bad-text)')
  })

  it('L-A: every OPEN state renders the neutral/muted tone, never bad', () => {
    const openStatuses: Array<[RightCellStateKind, string]> = [
      ['open_refused', 'OPEN · refused'],
      ['open_absent', 'OPEN · absent'],
      ['open_asked', 'OPEN · asked'],
      ['open_not_given', 'OPEN'],
      ['open_not_asked', 'OPEN']
    ]
    for (const [kind, status] of openStatuses) {
      const { unmount } = render(
        <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow(kind)} />
      )
      const badge = screen.getByText(status)
      expect(badge.style.color).not.toBe('var(--color-bad-text)')
      unmount()
    }
  })

  it('L-A: CLOSED also renders the neutral tone, never bad', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    const badge = screen.getByText('CLOSED')
    expect(badge.style.color).not.toBe('var(--color-bad-text)')
  })
})

describe('ExchangeStreamRow — the states render distinct text/status/action', () => {
  const cases: Array<{ kind: RightCellStateKind; text: string; status: string; action: string | null }> = [
    { kind: 'contradicted', text: '✗ Their record differs', status: 'CONTRADICTED', action: 'Compare' },
    {
      kind: 'open_refused',
      text: 'They declined, and signed the refusal — 4 Sep',
      status: 'OPEN · refused',
      action: 'View refusal'
    },
    {
      kind: 'open_absent',
      text: 'They say they have no record of this — 4 Sep',
      status: 'OPEN · absent',
      action: 'View statement'
    },
    { kind: 'open_asked', text: 'Asked 4 Sep. No reply yet.', status: 'OPEN · asked', action: 'Ask again' },
    {
      kind: 'open_not_given',
      text: 'Their record hasn’t arrived yet.',
      status: 'OPEN',
      action: 'Ask them for their record'
    },
    {
      kind: 'open_not_asked',
      text: 'You haven’t asked for their record.',
      status: 'OPEN',
      action: 'Ask them for their record'
    }
  ]

  for (const { kind, text, status, action } of cases) {
    it(`renders ${kind} correctly`, () => {
      const { unmount } = render(
        <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow(kind)} />
      )
      expect(screen.getByText(text)).toBeInTheDocument()
      expect(screen.getByText(status)).toBeInTheDocument()
      // no whole-row click target and no modal
      // left to open -- every row always carries the two `▸ content`/
      // `▸ checks` toggle buttons + the state (i) and the whole-row (i) info
      // glyphs, plus one more when an action exists.
      if (action) {
        expect(screen.getByRole('button', { name: action })).toBeInTheDocument()
        expect(screen.getAllByRole('button')).toHaveLength(5)
      } else {
        expect(screen.getAllByRole('button')).toHaveLength(4)
      }
      unmount()
    })
  }

  it('UX §3: a CLOSED row says it in a sentence on the face; the per-property cells lead the expansion', () => {
    const { rerender } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />
    )
    expect(screen.getByText('CLOSED')).toBeInTheDocument()
    expect(screen.getByText('✓ They recorded the same request and answer')).toBeInTheDocument()
    // Collapsed: no engineer cells on the face -- the two toggles, the
    // state (i) and the whole-row (i) only.
    expect(screen.queryByText('signature ✓')).not.toBeInTheDocument()
    expect(screen.getAllByRole('button')).toHaveLength(4)

    rerender(
      <ExchangeStreamRow checksExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />
    )
    // The four cells: their id · signature ✓ · request = · response = --
    // each restating a fact the gate's own inputs established -- first in
    // the expansion, ahead of the checks panel.
    const head = document.querySelector('[data-expansion-head="true"]') as HTMLElement
    expect(within(head).getByText(`their id ${'a'.repeat(12)}…`)).toBeInTheDocument()
    expect(within(head).getByText('signature ✓')).toBeInTheDocument()
    expect(within(head).getByText('request =')).toBeInTheDocument()
    expect(within(head).getByText('response =')).toBeInTheDocument()
    const firstCell = head.querySelector('[data-closed-property-cell]') as HTMLElement
    expect(firstCell).toHaveAttribute('data-closed-property-cell', 'their_id')
    // Opening Logs at this exchange needs a host hook, so the
    // expansion says so in plain text, never a dead link.
    expect(within(head).getByText(SEE_IN_LOGS_NOT_ON_THIS_PAGE)).toBeInTheDocument()
    expect(within(head).queryByRole('button', { name: /see in Logs/ })).not.toBeInTheDocument()
    // Pointing Chat at one node needs a host hook: no such button here.
    expect(within(head).queryByRole('button', { name: 'Chat with this node' })).not.toBeInTheDocument()
  })

  it('UX §8: each CLOSED property cell carries its own one-sentence (i)', () => {
    render(
      <ExchangeStreamRow checksExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />
    )
    const expected: Array<[string, string]> = [
      [`their id ${'a'.repeat(12)}…`, 'The id of their record; it’s a fingerprint of the record itself.'],
      ['signature ✓', 'Signed with the key this peer announces.'],
      ['request =', 'The same request as in your record.'],
      ['response =', 'The same answer as in your record.']
    ]
    for (const [label, sentence] of expected) {
      const glyph = screen.getByRole('button', { name: `About ${label}` })
      expect(document.getElementById(glyph.getAttribute('aria-describedby') as string)).toHaveTextContent(sentence)
    }
  })

  it('D4(d) ADVERSARIAL: the per-property cells render ONLY on a gate-closed row — an OPEN row never borrows them', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_held')} />)
    expect(screen.queryByText('signature ✓')).not.toBeInTheDocument()
    expect(screen.queryByText('request =')).not.toBeInTheDocument()
  })

  it('Item 4: the state carries an (i) whose aria-describedby holds the fuller story (terse cell on the face)', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    // Terse status on the face.
    expect(screen.getByText('CLOSED')).toBeInTheDocument()
    // The (i), wired to the fuller CLOSED story.
    const glyph = screen.getByRole('button', { name: 'About the CLOSED state' })
    const description = document.getElementById(glyph.getAttribute('aria-describedby') as string)
    expect(description).toHaveTextContent('They sent their own signed record of this exchange')
    expect(description).toHaveTextContent('same request, answer and model weights as yours')
  })

  it('Item 4: an OPEN · not held state says their record has not arrived yet, behind its (i)', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_held')} />)
    const glyph = screen.getByRole('button', { name: 'About the OPEN · not held state' })
    const description = document.getElementById(glyph.getAttribute('aria-describedby') as string)
    expect(description).toHaveTextContent('Their record hasn’t arrived')
  })

  it('LOAD-BEARING: not-asked and unanswered render visibly distinct text on the row', () => {
    const { unmount: unmountA } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_asked')} />
    )
    const notAskedText = screen.getByText('You haven’t asked for their record.').textContent
    unmountA()
    const { unmount: unmountB } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_asked')} />
    )
    const unansweredText = screen.getByText(/No reply yet\.$/).textContent
    unmountB()
    expect(notAskedText).not.toBe(unansweredText)
  })
})

describe('ExchangeStreamRow — bilateral-retention-decay-property (agent-action-capsule @7f8a78d8, ratified 2026-09-23): one-half-unavailable renders the honest OPEN sub-state, never CONTRADICTED, never CLOSED/"attested by both"', () => {
  it('not_found (peer legitimately holds nothing -- retention decay or never held) renders OPEN · not held', () => {
    const row = makeRow('open_not_held')
    vi.mocked(usePeerLedgerRecompute).mockReturnValue({
      status: 'not_found',
      idMatch: null,
      signatureOk: null,
      peerRecord: null,
      fetch: vi.fn()
    })
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(screen.getByText('OPEN · not held')).toBeInTheDocument()
    expect(screen.queryByText('CONTRADICTED')).not.toBeInTheDocument()
    expect(screen.queryByText('CLOSED')).not.toBeInTheDocument()
  })

  it('error (transport/verification failure -- not a disagreement) renders OPEN · not held, never CONTRADICTED', () => {
    const row = makeRow('open_not_held')
    vi.mocked(usePeerLedgerRecompute).mockReturnValue({
      status: 'error',
      idMatch: null,
      signatureOk: null,
      peerRecord: null,
      errorMessage: 'transport error',
      fetch: vi.fn()
    })
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(screen.getByText('OPEN · not held')).toBeInTheDocument()
    expect(screen.queryByText('CONTRADICTED')).not.toBeInTheDocument()
  })

  it('MUTANT: the only shape this row ever renders CONTRADICTED for is a completed fetch whose recomputed id demonstrably disagrees -- every other unavailable shape must go red if it starts reading CONTRADICTED', () => {
    const row = makeRow('open_not_held')
    const unavailableShapes: PeerRecomputeState[] = [
      { status: 'not_fetched', idMatch: null, signatureOk: null, peerRecord: null, fetch: vi.fn() },
      { status: 'fetching', idMatch: null, signatureOk: null, peerRecord: null, fetch: vi.fn() },
      { status: 'not_found', idMatch: null, signatureOk: null, peerRecord: null, fetch: vi.fn() },
      { status: 'error', idMatch: null, signatureOk: null, peerRecord: null, fetch: vi.fn() },
      { status: 'found', idMatch: null, signatureOk: null, peerRecord: { x: 1 }, fetch: vi.fn() }
    ]
    for (const recompute of unavailableShapes) {
      vi.mocked(usePeerLedgerRecompute).mockReturnValue(recompute)
      const { unmount } = render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
      expect(screen.queryByText('CONTRADICTED')).not.toBeInTheDocument()
      unmount()
    }
  })
})

describe('ExchangeStreamRow — counterparty field', () => {
  it('renders the node id when a counterparty is attributed', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    expect(screen.getByText('node:aa11bb22')).toBeInTheDocument()
    expect(screen.queryByText('counterparty not recorded')).not.toBeInTheDocument()
  })

  it('ADVERSARIAL: reads "counterparty not recorded" — never "unknown peer", never "peer identity not resolved yet" — when unattributed', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        rail={NO_RAIL}
        row={makeRow('closed', { counterparty: null })}
      />
    )
    expect(screen.getByText('counterparty not recorded')).toBeInTheDocument()
    expect(screen.queryByText(/unknown peer/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/peer identity not resolved yet/i)).not.toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — counterparty gating (unaffected by the modal removal)', () => {
  it('ASKED row, no recorded counterparty: "Other side: not known" gated text, no ask button', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={RECORD_NAMING_NO_PEER}
        rail={NO_RAIL}
        row={makeRow('open_not_asked', { counterparty: null })}
      />
    )
    // A remote exchange whose peer is unknown/unrecorded -- there IS an other
    // side, we just don't know it. Never the old "nothing to ask yet".
    expect(screen.getByText('Other side: not known')).toBeInTheDocument()
    expect(screen.queryByText('nothing to ask yet')).not.toBeInTheDocument()
    expect(screen.queryByText('You haven’t asked for their record.')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Ask them for their record' })).not.toBeInTheDocument()
    // The two disclosure toggles still render -- only the ask action is gated.
    expect(screen.getByRole('button', { name: 'What was said ▸' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'How we checked ▸' })).toBeInTheDocument()
  })

  it('SERVED row, no recorded counterparty: "Local — no other side" gated text (a distinct truth from the ASKED case)', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={RECORD_NAMING_NO_PEER}
        rail={NO_RAIL}
        row={makeRow('open_not_asked', { counterparty: null, roleTag: 'SERVED' })}
      />
    )
    // Served locally: no remote counterparty exists on the other side at all.
    expect(screen.getByText('Local — no other side')).toBeInTheDocument()
    expect(screen.queryByText('Other side: not known')).not.toBeInTheDocument()
    expect(screen.queryByText('nothing to ask yet')).not.toBeInTheDocument()
  })

  it('no recorded counterparty on an open_asked row: gated text, no "Ask again" button', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={RECORD_NAMING_NO_PEER}
        rail={NO_RAIL}
        row={makeRow('open_asked', { counterparty: null })}
      />
    )
    expect(screen.getByText('Other side: not known')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Ask again' })).not.toBeInTheDocument()
  })

  it('a recorded counterparty leaves states that are not ask actions untouched (no gating on Compare/View)', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        rail={NO_RAIL}
        row={makeRow('open_refused', { counterparty: null })}
      />
    )
    expect(screen.getByText('They declined, and signed the refusal — 4 Sep')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View refusal' })).toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — the two row toggles replace the modal', () => {
  it('never renders a dialog/alertdialog, in any state, expanded or not', () => {
    for (const checksExpanded of [false, true]) {
      for (const contentExpanded of [false, true]) {
        const { unmount } = render(
          <ExchangeStreamRow
            checksExpanded={checksExpanded}
            contentExpanded={contentExpanded}
            onAction={vi.fn()}
            {...toggleProps()}
            rail={NO_RAIL}
            row={makeRow('closed')}
          />
        )
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
        expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
        unmount()
      }
    }
  })

  it('renders both toggles collapsed by default, independent of each other', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    const contentToggle = screen.getByRole('button', { name: 'What was said ▸' })
    const checksToggle = screen.getByRole('button', { name: 'How we checked ▸' })
    expect(contentToggle).toHaveAttribute('aria-expanded', 'false')
    expect(checksToggle).toHaveAttribute('aria-expanded', 'false')
  })

  it('flips its own label and aria-expanded when its prop is true, independent of the other toggle', () => {
    render(
      <ExchangeStreamRow checksExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />
    )
    expect(screen.getByRole('button', { name: 'How we checked ▾' })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('button', { name: 'What was said ▸' })).toHaveAttribute('aria-expanded', 'false')
  })

  it('a CLOSED row whose own copy fails its checks says so, never a silent CLOSED', () => {
    vi.mocked(useRecomputedIdentity).mockReturnValue({ idMatch: false, signatureOk: true })
    try {
      render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
      expect(screen.getByText('CLOSED')).toBeInTheDocument()
      expect(screen.getByText(/Your own copy fails its checks/)).toBeInTheDocument()
    } finally {
      vi.mocked(useRecomputedIdentity).mockReset()
    }
    // Your copy checks out: no warning.
    vi.mocked(useRecomputedIdentity).mockReturnValue({ idMatch: true, signatureOk: true })
    try {
      cleanup()
      render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
      expect(screen.queryByText(/Your own copy fails its checks/)).not.toBeInTheDocument()
    } finally {
      vi.mocked(useRecomputedIdentity).mockReset()
    }
  })

  it('Compare on a CONTRADICTED row opens its checks (yours beside theirs), never a silent no-op', async () => {
    const user = userEvent.setup()
    const onAction = vi.fn()
    const onToggleChecks = vi.fn()
    const row = makeRow('contradicted')
    render(
      <ExchangeStreamRow
        onAction={onAction}
        onToggleChecks={onToggleChecks}
        onToggleContent={vi.fn()}
        rail={NO_RAIL}
        row={row}
      />
    )
    await user.click(screen.getByRole('button', { name: 'Compare' }))
    expect(onToggleChecks).toHaveBeenCalledWith(row)
    expect(onAction).not.toHaveBeenCalled()
  })

  it('clicking `▸ content` calls onToggleContent with this row only; clicking `▸ checks` calls onToggleChecks only', async () => {
    const user = userEvent.setup()
    const onToggleContent = vi.fn()
    const onToggleChecks = vi.fn()
    const row = makeRow('closed')
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        onToggleChecks={onToggleChecks}
        onToggleContent={onToggleContent}
        rail={NO_RAIL}
        row={row}
      />
    )

    await user.click(screen.getByRole('button', { name: 'What was said ▸' }))
    expect(onToggleContent).toHaveBeenCalledWith(row)
    expect(onToggleChecks).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'How we checked ▸' }))
    expect(onToggleChecks).toHaveBeenCalledWith(row)
    expect(onToggleContent).toHaveBeenCalledTimes(1)
  })

  it('the ask/compare action cell button and the two toggles are independent siblings, never nested', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_asked')} />)
    const actionButton = screen.getByRole('button', { name: 'Ask them for their record' })
    const contentToggle = screen.getByRole('button', { name: 'What was said ▸' })
    const checksToggle = screen.getByRole('button', { name: 'How we checked ▸' })
    expect(actionButton.contains(contentToggle)).toBe(false)
    expect(contentToggle.contains(actionButton)).toBe(false)
    expect(actionButton.contains(checksToggle)).toBe(false)
  })

  it('clicking the ask/compare action never fires either toggle callback', async () => {
    const user = userEvent.setup()
    const onToggleContent = vi.fn()
    const onToggleChecks = vi.fn()
    render(
      <ExchangeStreamRow
        localRecord={LOCAL_RECORD_WITH_DIGESTS}
        onAskForRecord={vi.fn()}
        onAction={vi.fn()}
        onToggleChecks={onToggleChecks}
        onToggleContent={onToggleContent}
        rail={NO_RAIL}
        row={makeRow('open_not_asked')}
      />
    )
    await user.click(screen.getByRole('button', { name: 'Ask them for their record' }))
    expect(onToggleContent).not.toHaveBeenCalled()
    expect(onToggleChecks).not.toHaveBeenCalled()
  })
})

describe('ExchangeStreamRow — focus/highlight/checks toggle', () => {
  it('renders at a stable, addressable DOM id derived from the exchange key', () => {
    const row = makeRow('closed')
    const { container } = render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(container.querySelector(`#${exchangeRowDomId(row.exchangeKey)}`)).toBeInTheDocument()
  })

  it('marks the deep-link target row with aria-current and a data-highlighted flag', () => {
    render(
      <ExchangeStreamRow highlighted onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />
    )
    const rowEl = screen.getByLabelText(`Exchange ${makeRow('closed').exchangeKey}`)
    expect(rowEl).toHaveAttribute('aria-current', 'true')
    expect(rowEl).toHaveAttribute('data-highlighted', 'true')
  })

  it('a non-highlighted, non-focused row carries neither flag', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    const rowEl = screen.getByLabelText(`Exchange ${makeRow('closed').exchangeKey}`)
    expect(rowEl).not.toHaveAttribute('aria-current')
    expect(rowEl).not.toHaveAttribute('data-highlighted')
    expect(rowEl).not.toHaveAttribute('data-focused')
  })

  it('the `c` toggle reveals the security view inline, hidden by default', () => {
    const row = makeRow('closed')
    const { rerender } = render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(screen.queryByRole('region', { name: /Security checks/ })).not.toBeInTheDocument()
    rerender(<ExchangeStreamRow checksExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(screen.getByRole('region', { name: /Security checks/ })).toBeInTheDocument()
    // Never a modal (v3 §4) -- the toggle stays inline under the row.
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — UX §3: the left cell is the event in words; no leading glyph', () => {
  it('states the role in words, and carries no unlabelled leading ◐/● glyph', () => {
    const { unmount } = render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        rail={{ hasRail: true, isSegmentStart: true }}
        row={makeRow('open_not_asked', { roleTag: 'SERVED' })}
      />
    )
    expect(screen.getByText('You served')).toBeInTheDocument()
    expect(screen.queryByText('◐')).not.toBeInTheDocument()
    unmount()

    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    expect(screen.getByText('You asked')).toBeInTheDocument()
    expect(screen.queryByText('●')).not.toBeInTheDocument()
  })

  it('reads peer · model · tokens · duration from the record, and keeps the ids off the face', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={{
          capsule_id: 'mine-1',
          model_attestation: {
            model_id: 'local-gguf/7089c7abcdef0123456789',
            compute_attestation: {
              'x-mesh-poc-v1': {
                latency_ms: 1432,
                serving_provenance: { usage: { prompt_tokens: 212, completion_tokens: 256 } }
              }
            }
          }
        }}
        rail={NO_RAIL}
        row={makeRow('closed', { counterparty: 'key:71eb7777' })}
      />
    )
    const line = document.querySelector('[data-event-line="true"]') as HTMLElement
    // The model's name, never the local-gguf/<hash> path (§3); the full
    // reference stays on hover.
    expect(line).toHaveTextContent('You asked key:71eb7777 · local model · 212 → 256 tokens · 1.4 s')
    expect(line).not.toHaveTextContent('local-gguf')
    expect(screen.queryByText('exch-closed')).not.toBeInTheDocument()
    expect(screen.queryByText('mine-1')).not.toBeInTheDocument()
  })

  it('leaves out a duration the record never measured (latency 0), never "0 ms"', () => {
    expect(durationText(0)).toBeNull()
    expect(durationText('0.000')).toBeNull()
    expect(durationText(undefined)).toBeNull()
    expect(durationText(320)).toBe('320 ms')
    expect(tokenFlowText(null, null)).toBeNull()
    expect(tokenFlowText(46, 2)).toBe('46 → 2 tokens')
  })

  it('offers "Ask them for their record" only after the timeout -- a fresh exchange just waits for the push', () => {
    const fresh = new Date(Date.now() - 10_000).toISOString()
    const { unmount } = render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        rail={NO_RAIL}
        row={makeRow('open_not_asked', { timestamp: fresh })}
      />
    )
    expect(screen.queryByRole('button', { name: 'Ask them for their record' })).not.toBeInTheDocument()
    unmount()

    const stale = new Date(Date.now() - ASK_FOR_RECORD_AFTER_MS - 1000).toISOString()
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        rail={NO_RAIL}
        row={makeRow('open_not_asked', { timestamp: stale })}
      />
    )
    expect(screen.getByRole('button', { name: 'Ask them for their record' })).toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — UX §3: the bracket strip is drawn in words, without notation', () => {
  function stripText(): string {
    return (document.querySelector('[data-bracket-strip="true"]') as HTMLElement).textContent ?? ''
  }

  it('CLOSED: both records held, joined by a line', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    expect(stripText()).toBe('Yours ● sealed —— Theirs ● same')
  })

  it('OPEN · not held: theirs is an open dot, "not received", never joined', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_held')} />)
    expect(stripText()).toBe('Yours ● sealed   Theirs ○ not received')
  })

  it('refused / asked / contradicted / not asked each say their own word; no braces or arrows anywhere', () => {
    const cases: Array<[RightCellStateKind, string]> = [
      ['open_refused', 'Yours ● sealed   Theirs ○ refused'],
      ['open_asked', 'Yours ● sealed   Theirs ○ asked, no reply yet'],
      ['contradicted', 'Yours ● sealed —— Theirs ● differs'],
      ['open_not_asked', 'Yours ● sealed   Theirs ○ not asked for']
    ]
    for (const [kind, strip] of cases) {
      const { unmount } = render(
        <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow(kind)} />
      )
      expect(stripText()).toBe(strip)
      expect(stripText()).not.toMatch(/[{}⟷⇄⇥⇢→◔◌⊘]/)
      unmount()
    }
  })
})

describe('ExchangeStreamRow — toggle ① content', () => {
  it('hidden by default, revealed by contentExpanded', () => {
    const row = makeRow('closed', {
      raw: {
        mine: { state: 'present', capsule_id: 'mine-1', text: 'Summarise this thread…' },
        theirs: { state: 'present', capsule_id: 'theirs-1' }
      } as PaneCRow
    })
    const { rerender } = render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(screen.queryByText(/Summarise this thread/)).not.toBeInTheDocument()
    rerender(<ExchangeStreamRow contentExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(screen.getByText(/Summarise this thread/)).toBeInTheDocument()
  })

  it('Case A (you asked): both halves render on the left; the right renders the their-claim voice', () => {
    const row = makeRow('closed', {
      roleTag: 'ASKED',
      contentToggleState: {
        your: { kind: 'populated', date: null },
        their: { kind: 'recorded_absence', date: '4 Sep' }
      },
      raw: {
        mine: {
          state: 'present',
          capsule_id: 'mine-1',
          text: 'Summarise this thread…',
          reply_text: 'The thread covers three…'
        },
        theirs: {
          state: 'absent',
          capsule_id: null,
          evidence_outcome: 'recorded_absence',
          evidence_outcome_date: '4 Sep'
        }
      } as PaneCRow
    })
    render(<ExchangeStreamRow contentExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    const content = document.querySelector('[data-content-toggle="expanded"]') as HTMLElement
    expect(within(content).getByText(/You asked/)).toBeInTheDocument()
    expect(screen.getByText(/Summarise this thread/)).toBeInTheDocument()
    expect(screen.getByText(/They streamed back/)).toBeInTheDocument()
    expect(screen.getByText(/The thread covers three/)).toBeInTheDocument()
    expect(screen.getByText('They state they hold no payload for this exchange — signed 4 Sep.')).toBeInTheDocument()
  })

  it('Case B (they asked, you served): the left renders the fixed fact-you-know string; the right renders "not visible to you"', () => {
    const row = makeRow('closed', {
      roleTag: 'SERVED',
      contentToggleState: {
        your: { kind: 'streamed_not_retained', date: null },
        their: { kind: 'not_visible_holds', date: null }
      }
    })
    render(<ExchangeStreamRow contentExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(screen.getByText('No content. You streamed this response and did not retain it.')).toBeInTheDocument()
    expect(screen.getByText('Their content — not visible to you. The requester holds it.')).toBeInTheDocument()
  })

  it('deletion renders as a distinct state, not a blank', () => {
    const row = makeRow('closed', {
      roleTag: 'ASKED',
      contentToggleState: {
        your: { kind: 'populated_deleted', date: '5 Sep' },
        their: { kind: 'not_asked', date: null }
      }
    })
    render(<ExchangeStreamRow contentExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    expect(screen.getByText('Content deleted 5 Sep · record still verifies')).toBeInTheDocument()
  })

  it('the three their-content sub-states + not-visible-holds each render distinct text', () => {
    const cases: Array<{
      contentToggleState: ExchangeLedgerRow['contentToggleState']
      text: string
    }> = [
      {
        contentToggleState: {
          your: { kind: 'populated', date: null },
          their: { kind: 'recorded_absence', date: '4 Sep' }
        },
        text: 'They state they hold no payload for this exchange — signed 4 Sep.'
      },
      {
        contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'not_asked', date: null } },
        text: 'Not compared. They would be expected to hold none.'
      },
      {
        contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'unanswered', date: '3 Sep' } },
        text: 'Asked 3 Sep. No reply yet.'
      }
    ]
    for (const { contentToggleState, text } of cases) {
      const { unmount } = render(
        <ExchangeStreamRow
          contentExpanded
          onAction={vi.fn()}
          {...toggleProps()}
          rail={NO_RAIL}
          row={makeRow('closed', { contentToggleState })}
        />
      )
      expect(screen.getByText(text)).toBeInTheDocument()
      unmount()
    }
  })

  it('the never-asked their-content state carries an action button wired to onAction', () => {
    const onAction = vi.fn()
    const row = makeRow('closed', {
      contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'not_asked', date: null } }
    })
    render(<ExchangeStreamRow contentExpanded onAction={onAction} {...toggleProps()} rail={NO_RAIL} row={row} />)
    screen.getByRole('button', { name: 'Ask them to state it' }).click()
    expect(onAction).toHaveBeenCalledWith(row)
  })

  it('LOAD-BEARING (L-H): the your-empty voice and the their-empty voice are visibly different strings', () => {
    const servedRow = makeRow('closed', {
      roleTag: 'SERVED',
      contentToggleState: {
        your: { kind: 'streamed_not_retained', date: null },
        their: { kind: 'not_visible_holds', date: null }
      }
    })
    const { unmount } = render(
      <ExchangeStreamRow contentExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={servedRow} />
    )
    const yourVoice = screen.getByText(/No content\. You streamed/).textContent
    unmount()

    const askedRow = makeRow('closed', {
      roleTag: 'ASKED',
      contentToggleState: {
        your: { kind: 'populated', date: null },
        their: { kind: 'recorded_absence', date: '4 Sep' }
      }
    })
    render(<ExchangeStreamRow contentExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={askedRow} />)
    const theirVoice = screen.getByText(/They state they hold no payload/).textContent

    expect(yourVoice).not.toBe(theirVoice)
  })
})

describe('ExchangeStreamRow — §3A one colour per state', () => {
  it('a row with no held bytes (OPEN) renders no colour and the invitation to act', () => {
    const { container } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_asked')} />
    )
    expect(screen.getByText('You haven’t asked for their record.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Ask them for their record' })).toBeInTheDocument()
    // Exactly one tone marker on the row, and it's muted -- no colour.
    const toneEls = container.querySelectorAll('[data-row-tone]')
    expect(toneEls).toHaveLength(1)
    expect(toneEls[0]).toHaveAttribute('data-row-tone', 'muted')
  })

  it('MUTANT: every OPEN state (not just open_not_asked) renders the muted/no-colour tone', () => {
    const openKinds: RightCellStateKind[] = [
      'open_refused',
      'open_absent',
      'open_asked',
      'open_not_held',
      'open_not_given',
      'open_not_asked'
    ]
    for (const kind of openKinds) {
      const { container, unmount } = render(
        <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow(kind)} />
      )
      const tone = container.querySelector('[data-row-tone]')?.getAttribute('data-row-tone')
      expect(tone === 'good' || tone === 'bad', `${kind} rendered tone "${tone}"`).toBe(false)
      unmount()
    }
  })

  it('a CONTRADICTED row renders the bad/red tone exactly once', () => {
    const { container } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('contradicted')} />
    )
    const toneEls = container.querySelectorAll('[data-row-tone]')
    expect(toneEls).toHaveLength(1)
    expect(container.querySelectorAll('[data-row-tone="bad"]')).toHaveLength(1)
  })

  it('a CLOSED row renders the good/green tone, not muted, not bad', () => {
    const { container } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />
    )
    expect(container.querySelectorAll('[data-row-tone="good"]')).toHaveLength(1)
    expect(container.querySelectorAll('[data-row-tone="bad"], [data-row-tone="muted"]')).toHaveLength(0)
  })
})

describe('ExchangeStreamRow — a Confirmed badge beside a failed check on your own copy', () => {
  it('says your own copy fails its checks, and only then', async () => {
    const closed = makeRow('closed')
    const tampered = { ...LOCAL_RECORD_WITH_DIGESTS, capsule_id: 'f'.repeat(64) }
    const { rerender } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={closed} />
    )
    expect(screen.queryByText(OWN_RECORD_FAILS_WARNING)).not.toBeInTheDocument()
    rerender(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={tampered}
        rail={NO_RAIL}
        row={closed}
      />
    )
    // The fingerprint check runs asynchronously on this machine.
    expect(await screen.findByText(OWN_RECORD_FAILS_WARNING)).toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — confirmed from a record this page asked for', () => {
  it('says the confirmation is on this page only, not saved on the node', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    expect(screen.getByText('CLOSED')).toBeInTheDocument()
    expect(screen.getByText(CLOSED_FROM_FETCH_NOT_SAVED)).toBeInTheDocument()
  })

  it('never on a row confirmed from a record the node holds (pushed and saved)', () => {
    const base = makeRow('open_not_asked')
    const pushed = { ...base, raw: { ...base.raw, mine: fixtureMineCell(), theirs: fixtureTheirsCell('agrees') } }
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={pushed} />)
    expect(screen.getByText('CLOSED')).toBeInTheDocument()
    expect(screen.queryByText(CLOSED_FROM_FETCH_NOT_SAVED)).not.toBeInTheDocument()
  })

  it('says nothing of the kind on a row that is not confirmed', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_held')} />)
    expect(screen.queryByText(CLOSED_FROM_FETCH_NOT_SAVED)).not.toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — the referee', () => {
  it('badges a row that is this node’s answer as a referee, and no other row', () => {
    const referee = makeRow('open_not_asked')
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        rail={NO_RAIL}
        row={{ ...referee, raw: { ...referee.raw, referee_call: true } }}
      />
    )
    expect(screen.getByText('referee answer')).toBeInTheDocument()
  })

  it('shows a verdict delivered about this exchange on its row', () => {
    const judged = makeRow('open_not_asked')
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        rail={NO_RAIL}
        row={{
          ...judged,
          raw: {
            ...judged.raw,
            adjudication: {
              verdict: 'corroborated',
              verdict_capsule_id: 'v'.repeat(64),
              referee_node_id: 'a70d3967bea3b22f'.repeat(4),
              received_at: '2026-09-28T15:00:00Z',
              about_this_node: false
            }
          }
        }}
      />
    )
    expect(screen.getByText('A referee (node a70d3967be…) found this answer corroborated.')).toBeInTheDocument()
  })

  it('shows no referee badge on an ordinary row', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_asked')} />)
    expect(screen.queryByText('referee answer')).not.toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — row time', () => {
  it('shows the exchange time in local time, never an ISO/UTC stamp', () => {
    const { container } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />
    )
    const time = container.querySelector('time')
    expect(time?.getAttribute('datetime')).toBe('2026-09-08T16:58:05Z')
    expect(time?.textContent).toBe(formatExchangeTimestamp('2026-09-08T16:58:05Z'))
    expect(container.textContent).not.toMatch(/\d{4}-\d\d-\d\d[T ]\d\d:\d\d|\bUTC\b|\d\dZ\b/)
  })
})

describe('ExchangeStreamRow — §3A chip strip -> checks jump', () => {
  it('renders the five chips in plain words, never sig / inclusion / registered', () => {
    const { container } = render(
      <ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_asked')} />
    )
    for (const label of ['words match', 'signed', 'in a checkpoint', 'witnessed', 'their record']) {
      expect(screen.getByLabelText(`${label}: jump to that check`)).toBeInTheDocument()
    }
    expect(container.textContent).not.toMatch(/\bsig\b|\binclusion\b|\bregistered\b/)
  })

  it('a chip is a link, never counted among the row action buttons', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('open_not_asked')} />)
    // Two toggles + one ask action + the state's (i) + the whole-row (i) == 5,
    // unchanged by the chip strip ("the two
    // toggles are the entire detail surface" -- chips jump to detail already
    // on the page, they don't add a row action).
    expect(screen.getAllByRole('button')).toHaveLength(5)
    expect(screen.getByLabelText('signed: jump to that check')).toHaveAttribute('role', 'link')
  })

  it('a chip click on a collapsed panel expands checks, then scrolls to and highlights the matching property once it mounts', async () => {
    const user = userEvent.setup()
    const onToggleChecks = vi.fn()
    const row = makeRow('open_not_asked')
    const scrollSpy = vi.spyOn(HTMLElement.prototype, 'scrollIntoView')

    const { rerender } = render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        onToggleChecks={onToggleChecks}
        rail={NO_RAIL}
        row={row}
      />
    )
    await user.click(screen.getByLabelText('signed: jump to that check'))
    expect(onToggleChecks).toHaveBeenCalledWith(row)
    expect(scrollSpy).not.toHaveBeenCalled()

    // The parent (LedgerPage) honors the toggle -- checksExpanded flips true.
    rerender(
      <ExchangeStreamRow
        checksExpanded
        onAction={vi.fn()}
        {...toggleProps()}
        onToggleChecks={onToggleChecks}
        rail={NO_RAIL}
        row={row}
      />
    )
    expect(await screen.findByRole('region', { name: /Security checks/ })).toBeInTheDocument()
    expect(scrollSpy).toHaveBeenCalled()
    scrollSpy.mockRestore()
  })

  it('a chip click while the panel is already expanded scrolls immediately, without re-toggling', async () => {
    const user = userEvent.setup()
    const onToggleChecks = vi.fn()
    const scrollSpy = vi.spyOn(HTMLElement.prototype, 'scrollIntoView')

    render(
      <ExchangeStreamRow
        checksExpanded
        onAction={vi.fn()}
        {...toggleProps()}
        onToggleChecks={onToggleChecks}
        rail={NO_RAIL}
        row={makeRow('open_not_asked')}
      />
    )
    await user.click(screen.getByLabelText('words match: jump to that check'))
    expect(onToggleChecks).not.toHaveBeenCalled()
    expect(scrollSpy).toHaveBeenCalled()
    scrollSpy.mockRestore()
  })
})

describe('ExchangeStreamRow — §3A model identity + short/copyable ids', () => {
  afterEach(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
  })

  it('renders the model by name, with the full ref on hover, when the record carries one', () => {
    const row = makeRow('closed')
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={{
          capsule_id: 'mine-1',
          effect: { request_digest: REQUEST_DIGEST, response_digest: RESPONSE_DIGEST },
          model_attestation: { model_id: 'local-gguf/7089c7abcdef0123456789' }
        }}
        rail={NO_RAIL}
        row={row}
      />
    )
    const modelEl = screen.getByText('local model')
    expect(modelEl).toHaveAttribute('title', 'local-gguf/7089c7abcdef0123456789')
  })

  it('names the model and counts from the other side’s record when ours has none, and says whose they are', () => {
    const row = makeRow('closed')
    row.raw.theirs.record = {
      capsule_id: 'theirs-1',
      model_attestation: {
        model_id: 'local-gguf/abc',
        compute_attestation: {
          'x-mesh-poc-v1': {
            latency_ms: '2300.000',
            serving_provenance: {
              architecture: 'llama',
              parameter_size: '3B',
              usage: { prompt_tokens: 12, completion_tokens: 40 }
            }
          }
        }
      }
    }
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        localRecord={{ capsule_id: 'mine-1', model_attestation: { model_id: 'local-gguf/abc' } }}
        rail={NO_RAIL}
        row={row}
      />
    )
    const line = document.querySelector('[data-event-line="true"]') as HTMLElement
    expect(line).toHaveTextContent('12 → 40 tokens (their count)')
    expect(line).toHaveTextContent('2.3 s (their measure)')
  })

  it('a CLOSED row says where their record sits in their log, once our records cite it', () => {
    const row = makeRow('closed')
    row.raw.theirs.in_their_log = { leaf_index: 6, checkpoint_leaves: 8 }
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={row} />)
    // No number: their checkpoint's leaves include their padding.
    expect(screen.getByText('CLOSED · in their log')).toBeInTheDocument()
    expect(screen.queryByText(/checkpoint 8|8 records/)).not.toBeInTheDocument()
  })

  it('shows the time in the viewer’s own zone, never an ISO/UTC stamp', () => {
    expect(formatExchangeTimestamp('2026-09-28T04:56:00Z')).not.toMatch(/Z$|T\d/)
    expect(formatExchangeTimestamp(null)).toBe('timestamp unavailable')
  })

  it('drops a sha256- algorithm prefix before shortening -- never "local-gguf/sha256…"', () => {
    expect(formatModelIdentity('local-gguf/sha256-6c1a2b41161032677be168d354123594')).toBe('local-gguf/6c1a2b…')
  })

  it('renders nothing for the model when the record carries no model ref -- never a placeholder', () => {
    render(<ExchangeStreamRow onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />)
    expect(screen.queryByText(/local-gguf/)).not.toBeInTheDocument()
  })

  it('a short exchange key/record id renders unchanged in the expansion (never truncated for an already-short id)', () => {
    render(
      <ExchangeStreamRow checksExpanded onAction={vi.fn()} {...toggleProps()} rail={NO_RAIL} row={makeRow('closed')} />
    )
    const head = document.querySelector('[data-expansion-head="true"]') as HTMLElement
    expect(within(head).getByText('exch-closed')).toBeInTheDocument()
    expect(within(head).getByText('mine-1')).toBeInTheDocument()
  })

  it('a long (real digest-shaped) exchange key renders short, with the full string on hover and copy', async () => {
    const user = userEvent.setup()
    const longKey = 'e'.repeat(64)
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    render(
      <ExchangeStreamRow
        checksExpanded
        onAction={vi.fn()}
        {...toggleProps()}
        rail={NO_RAIL}
        row={makeRow('closed', { exchangeKey: longKey })}
      />
    )
    expect(screen.queryByText(longKey)).not.toBeInTheDocument()
    const shortEl = screen.getByText('eeee…eeee')
    expect(shortEl).toHaveAttribute('title', longKey)
    await user.click(shortEl)
    expect(writeText).toHaveBeenCalledWith(longKey)
  })
})

describe('ExchangeStreamRow — a row closed from the record "Ask them for their record" brought', () => {
  it('says the confirmation is for this page only, not saved', () => {
    const theirs = { ...LOCAL_RECORD_WITH_DIGESTS, capsule_id: 'theirs-1' } as Record<string, unknown>
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        askOutcome={{
          kind: 'record',
          at: '2026-09-28T23:00:00Z',
          evidence: { status: 'found', idMatch: true, signatureOk: true, peerRecord: theirs, fetch: vi.fn() }
        }}
        rail={NO_RAIL}
        row={makeRow('open_not_given')}
      />
    )
    expect(screen.getByText('CLOSED')).toBeInTheDocument()
    expect(screen.getByText(CLOSED_FROM_FETCH_NOT_SAVED)).toBeInTheDocument()
  })
})

describe('ExchangeStreamRow — whether their reply to "Ask them for their record" was verified', () => {
  it('says a reply the plugin could not verify is not verified, and why', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        askOutcome={{
          kind: 'no_reply',
          at: '2026-09-28T23:00:00Z',
          detail: 'their reply did not verify: a proof does not verify against the anchor'
        }}
        rail={NO_RAIL}
        row={makeRow('open_not_given')}
      />
    )
    const note = document.querySelector('[data-ask-reply-verified]') as HTMLElement
    expect(note).toHaveAttribute('data-ask-reply-verified', 'false')
    expect(note).toHaveTextContent('Their reply is not verified: their reply did not verify: a proof does not verify')
  })

  it('says a record the plugin proved in their log is verified', () => {
    const theirs = { ...LOCAL_RECORD_WITH_DIGESTS, capsule_id: 'theirs-1' } as Record<string, unknown>
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        askOutcome={{
          kind: 'record',
          at: '2026-09-28T23:00:00Z',
          evidence: { status: 'found', idMatch: true, signatureOk: true, peerRecord: theirs, fetch: vi.fn() }
        }}
        rail={NO_RAIL}
        row={makeRow('open_not_given')}
      />
    )
    const note = document.querySelector('[data-ask-reply-verified]') as HTMLElement
    expect(note).toHaveAttribute('data-ask-reply-verified', 'true')
    expect(note).toHaveTextContent('Their reply is verified')
  })

  it('never says verified for a record whose signature does not hold', () => {
    const theirs = { ...LOCAL_RECORD_WITH_DIGESTS, capsule_id: 'theirs-1' } as Record<string, unknown>
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        askOutcome={{
          kind: 'record',
          at: '2026-09-28T23:00:00Z',
          evidence: { status: 'found', idMatch: true, signatureOk: false, peerRecord: theirs, fetch: vi.fn() }
        }}
        rail={NO_RAIL}
        row={makeRow('open_not_given')}
      />
    )
    const note = document.querySelector('[data-ask-reply-verified]') as HTMLElement
    expect(note).toHaveAttribute('data-ask-reply-verified', 'false')
    expect(note).toHaveTextContent('it is not signed with their announced key')
  })

  it('shows no note while the ask is in flight', () => {
    render(
      <ExchangeStreamRow
        onAction={vi.fn()}
        {...toggleProps()}
        askOutcome={{ kind: 'asking', at: '2026-09-28T23:00:00Z' }}
        rail={NO_RAIL}
        row={makeRow('open_not_given')}
      />
    )
    expect(document.querySelector('[data-ask-reply-verified]')).toBeNull()
  })
})

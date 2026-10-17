// Pure view-model derivation for a Pane B row.
// Kept separate from the rendering components so the honesty-backbone
// invariants (denominators always present, NOT_CHECKED never summarized as
// corroborated, with-you and their-chain never summed) are unit-testable
// without mounting a component.
import type { PaneBConfirmedSibling, PaneBRow, PaneCRow } from '@/features/capsules/api/sidecarTypes'
import { deriveRightCellState } from '@/features/capsules/lib/exchange-row-state'
import type { PeerMeshStatus } from '@/features/capsules/lib/peer-mesh-status'
import { peerTwinTally, peerTwinText } from '@/features/capsules/lib/peer-twins'
import { peerSettlementText } from '@/features/capsules/lib/settlement-view'
import { PEER_ATTENTION, PEER_COLUMN_TOOLTIPS, SELF_REPORTED_TOOLTIP } from '@/features/capsules/lib/tooltip-copy'

export const STATE_NOT_CHECKED = 'NOT_CHECKED'
export const STATE_CONTRADICTED = 'contradicted'
export const STATE_FAILED = 'failed'
export const STATE_REFUSED = 'refused'
export const STATE_VERIFIED = 'verified'

/** `null` means this row carries no counterparty identity at all -- pane-b's
 *  own honest signal that these exchanges are unattributed, not that one
 *  peer's identity failed to resolve (design chooser-v1 §7 S1.1). Never
 *  invent a placeholder string for this case; callers render "counterparty
 *  not recorded" / exclude the row from the Peers list instead. */
export function peerDisplayId(row: PaneBRow): string | null {
  return row.peer_id ?? row.node?.peer_id ?? null
}

/** True when this row has no counterparty identity (see `peerDisplayId`) --
 *  the Peers list never renders a card for these; they're rolled into the
 *  unattributed-exchanges line instead. */
export function isUnattributedPeerRow(row: PaneBRow): boolean {
  return peerDisplayId(row) === null
}

/** The Peers-section headline for exchanges with no counterparty evidence
 *  at all -- a stated fact (R-C), never phrased as pending work. */
export function unattributedExchangesLine(count: number): string {
  const subject = count === 1 ? 'exchange has' : 'exchanges have'
  return `${count} ${subject} no counterparty recorded yet. They appear under Exchanges.`
}

// ---------------------------------------------------------------------------
// Peer identity aliases (D3) -- one
// row per peer, joined on the signing key, with the other id spaces shown as
// ALIASES on that row. Only evidence-backed aliases render; a node-id-only
// row (no signing key linked by any evidence yet) says so instead of
// pretending the spaces are one.
// ---------------------------------------------------------------------------

export type PeerIdentityAliases = { signingKeyId: string | null; endpointId: string | null; nodeId: string | null }

function nonEmpty(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null
}

export function peerIdentityAliases(row: PaneBRow): PeerIdentityAliases {
  const identity = row.identity ?? null
  return {
    signingKeyId: nonEmpty(identity?.signing_key_id),
    endpointId: nonEmpty(identity?.endpoint_id),
    nodeId: nonEmpty(identity?.node_id)
  }
}

/** The one alias line under a peer row's id: "signed by <key16> · node
 *  <id16>… · endpoint <id>" for a key-joined peer (only the aliases the
 *  evidence carries), the honest unlinked note for a node-id-only row, and
 *  `null` when the row carries no identity evidence at all. */
export function peerAliasLine(row: PaneBRow): string | null {
  const { signingKeyId, endpointId, nodeId } = peerIdentityAliases(row)
  if (!signingKeyId) {
    // The unlinked case: the id came from this node's own records (a
    // requester half naming the server it routed to); no pushed,
    // door-verified half links it to a signing key yet.
    if (nodeId && !endpointId) return 'node id from your own records — no signing key linked yet'
    return null
  }
  const parts = [`signed by ${signingKeyId.slice(0, 16)}`]
  if (nodeId) parts.push(`node ${nodeId.slice(0, 16)}…`)
  // Never a 64-hex id on the row face (§7.4): the full id is in the drill.
  if (endpointId) parts.push(`endpoint ${endpointId.length > 16 ? `${endpointId.slice(0, 16)}…` : endpointId}`)
  return parts.join(' · ')
}

/** Used by the modal's Overview tab (`PeerInspector`) for the self-reported
 *  model/quant/context line -- the summary table no longer renders this
 *  (accountability, not operational-announcement),
 *  but the deep-dive inspector still does. */
export function meshMetaLine(meshStatus: PeerMeshStatus | null): string | null {
  if (!meshStatus) return null
  const parts = [
    meshStatus.modelName,
    meshStatus.quant,
    meshStatus.contextLengthK != null ? `${meshStatus.contextLengthK}k ctx` : null
  ].filter((part): part is string => Boolean(part))
  return parts.length > 0 ? parts.join(' · ') : null
}

// ---------------------------------------------------------------------------
// With-you (verifiable, front) -- never summed with their-chain facts.
// Still used by the modal's Overview tab (`PeerInspector`).
// ---------------------------------------------------------------------------

export type WithYouCounts = { requested: number; served: number; confirmed: number }

/** `requested`/`served` from `role_and_count_cell` (direction of exchange,
 *  never a trust signal); `confirmed` from the ONE gate
 *  (`confirmedByOtherSide`, the same figure the Peers table's Confirmed
 *  column renders) -- NEVER `pair_cell.verified`, which no native payload
 *  populates (always 0 live), so the inspector Overview used to contradict
 *  the table on the same peer (fold-in
 *  finding 1). One predicate, one number, two surfaces. */
export function withYouCounts(row: PaneBRow): WithYouCounts {
  return {
    requested: row.role?.you_to_them_count ?? 0,
    served: row.role?.them_to_you_count ?? 0,
    confirmed: confirmedByOtherSide(row).confirmed
  }
}

export function withYouCountsText(counts: WithYouCounts): string {
  return `${counts.requested} requested · ${counts.served} served · ${counts.confirmed} confirmed`
}

// ---------------------------------------------------------------------------
// Adjudication -- always carries a denominator; NOT_CHECKED is never a
// silent pass.
// ---------------------------------------------------------------------------

export type AdjudicationSummary = {
  checked: number
  denominator: number
  corroborated: number
  contradicted: number
  inconclusive: number
  notChecked: boolean
}

/** `denominator` is `exchange_count` (every exchange this row has with the
 *  peer) -- `verdicts_cell` draws its tally from adjudications naming ANY
 *  of this peer's capsule ids, not only the requested-by-you half, so a
 *  narrower denominator would overclaim scope. */
export function adjudicationSummary(row: PaneBRow): AdjudicationSummary {
  // The referee-signed verdicts about this peer, when the plugin sends them:
  // the only verdicts the page counts. A not-comparable ruling isn't a
  // judgment, so it isn't counted as checked.
  const referee = row.referee_verdicts
  if (referee) {
    const corroborated = referee.corroborated?.verdict_capsule_ids.length ?? 0
    const contradicted = referee.contradicted?.verdict_capsule_ids.length ?? 0
    const inconclusive = referee.inconclusive?.verdict_capsule_ids.length ?? 0
    const checked = corroborated + contradicted + inconclusive
    return {
      checked,
      denominator: row.exchange_count ?? 0,
      corroborated,
      contradicted,
      inconclusive,
      notChecked: checked === 0
    }
  }
  const tally = row.verdicts?.tally
  const corroborated = tally?.corroborated ?? 0
  const contradicted = tally?.contradicted ?? 0
  const inconclusive = tally?.inconclusive ?? 0
  const checked = corroborated + contradicted + inconclusive
  const notChecked = row.verdicts?.state === STATE_NOT_CHECKED || checked === 0
  return {
    checked,
    denominator: row.exchange_count ?? 0,
    corroborated,
    contradicted,
    inconclusive,
    notChecked
  }
}

/** Never returns a corroborated-sounding sentence when `notChecked` is
 *  true -- that branch is checked first and returns before the tally is
 *  ever read. */
export function adjudicationSummaryText(summary: AdjudicationSummary): string {
  if (summary.notChecked) {
    return 'Not yet checked — no adjudication sealed for this peer yet.'
  }
  const parts: string[] = []
  if (summary.corroborated) parts.push(`${summary.corroborated} corroborated`)
  if (summary.contradicted) parts.push(`${summary.contradicted} contradicted`)
  if (summary.inconclusive) parts.push(`${summary.inconclusive} inconclusive`)
  return `${summary.checked} of ${summary.denominator} adjudicated · ${parts.join(' · ')}`
}

/** Compact form of `adjudicationSummaryText` for the Peers table's
 *  narrower ADJUDICATION column -- same honesty invariant (never
 *  "corroborated"-sounding when `notChecked`), just terser wording. */
export function adjudicationCompactText(summary: AdjudicationSummary): string {
  if (summary.notChecked) return 'none'
  const parts: string[] = []
  if (summary.corroborated) parts.push(`${summary.corroborated} corroborated`)
  if (summary.contradicted) parts.push(`${summary.contradicted} contradicted`)
  if (summary.inconclusive) parts.push(`${summary.inconclusive} inconclusive`)
  return `${summary.checked} of ${summary.denominator} · ${parts.join(' · ')}`
}

// ---------------------------------------------------------------------------
// Their chain (secondary) -- you verified it's unbroken/unforked, never its
// contents. Never summed with with-you facts.
// ---------------------------------------------------------------------------

export type ChainSummary = {
  text: string
  state: string
  /** True when this cell has no peer-fetch data and is only carrying this
   *  node's OWN chain as `mine_for_reference` -- the UI must label that
   *  clearly as "for reference", never present it as the peer's. */
  ownChainForReferenceOnly: boolean
}

export function theirChainSummary(row: PaneBRow): ChainSummary {
  const cell = row.history
  const state = cell?.state ?? STATE_NOT_CHECKED

  if (state === STATE_VERIFIED && cell?.history_summary) {
    const checkpointCount = cell.history_summary.checkpoint_count
    const bundles = cell.history_summary.verified_bundles
    return {
      text: `Unbroken — you verified their chain (their count: ${checkpointCount ?? '?'} checkpoint(s), ${bundles ?? '?'} bundle(s)).`,
      state,
      ownChainForReferenceOnly: false
    }
  }
  if (state === STATE_FAILED) {
    return {
      text: 'Verification failed — could not confirm their chain is unbroken.',
      state,
      ownChainForReferenceOnly: false
    }
  }
  if (state === STATE_REFUSED) {
    return { text: 'Peer refused to share their chain.', state, ownChainForReferenceOnly: false }
  }
  return {
    text: "Not yet checked — this view has no way to fetch the peer's own chain yet.",
    state: STATE_NOT_CHECKED,
    ownChainForReferenceOnly: Boolean(cell?.mine_for_reference)
  }
}

// ---------------------------------------------------------------------------
// Confirmed by the other side -- routed through the ONE gate
//, the SAME predicate Pane C uses. A peer's
// pushed halves arrive correlated + provenance-carrying in
// `row.confirmed_siblings` (`capsule_panes_native.rs`), each supplying the two
// gate inputs (`theirs.signature_ok` + `digest_match`). Every sibling is run
// through `deriveRightCellState` -- never a second, browser-peer-fetch
// predicate on `history` -- and a `closed` verdict counts as confirmed. A peer
// with no correlated pushed half has an empty list, which the gate reads as
// "not confirmed": the honest "waiting for the other side" state, never a
// fabricated zero and never phrased as pending work the user must trigger.
// ---------------------------------------------------------------------------

export type ConfirmedByOtherSide = { confirmed: number; total: number; note: string | null }

/** Runs one supplied sibling through the ONE gate as a minimal Pane C row --
 *  the gate reads only `mine.record` + `theirs` (body, door verdict, in-browser
 *  id recompute) here, so the rest of a full `PaneCRow` is stubbed honestly. */
function siblingGateState(sibling: PaneBConfirmedSibling): ReturnType<typeof deriveRightCellState> {
  const row: PaneCRow = {
    exchange_key: '',
    role_tag: '',
    header_state: '',
    properties: null,
    has_issue: false,
    mine: sibling.mine ?? { state: 'present-unverified', capsule_id: null },
    theirs: sibling.theirs,
    unilateral: false,
    digest_match: sibling.digest_match,
    timestamp: null
  }
  return deriveRightCellState(row)
}

export function confirmedByOtherSide(row: PaneBRow): ConfirmedByOtherSide {
  const total = row.exchange_count ?? 0
  const siblings = row.confirmed_siblings ?? []
  const confirmed = siblings.filter((sibling) => siblingGateState(sibling).kind === 'closed').length
  const contradicted = siblings.filter((sibling) => siblingGateState(sibling).kind === 'contradicted').length
  // A confirmation the gate reads as CONTRADICTED is a disagreement, not a
  // silent zero -- name it so the count and the alarm can never diverge.
  if (contradicted > 0) {
    return { confirmed, total, note: `${contradicted} contradicted` }
  }
  // Nothing the other side sent has closed through the gate yet -- covers both
  // "no half received" and "a half received but not yet verifiable". Stated as
  // a fact, never as pending work the user has to go and do.
  if (confirmed === 0) {
    return { confirmed: 0, total, note: 'none confirmed yet' }
  }
  return { confirmed, total, note: null }
}

/** UX §2: "3 of 3" -- "of" reads as a count, "/" as a ratio. The one note
 *  kept is a disagreement; "0 of 2" already says none are confirmed. */
export function confirmedByOtherSideText(summary: ConfirmedByOtherSide): string {
  const base = `${summary.confirmed} of ${summary.total}`
  const differ = /^(\d+) contradicted$/.exec(summary.note ?? '')
  return differ ? `${base} · ${differ[1]} differ` : base
}

// ---------------------------------------------------------------------------
// Match -- the clean/mismatch tally of the halves the other side actually
// sent, derived through the SAME ONE gate: a
// `closed` sibling is clean (digests cite our half), a `contradicted` sibling
// is a mismatch. `clean`/`mismatch` always both render (even at zero);
// `contradicted` (the adjudication tally) only appears when nonzero -- a peer
// with no contradiction has never had one, not "0 of them", the same
// zero-is-a-fact-not-silence discipline as `adjudicationSummaryText`.
// ---------------------------------------------------------------------------

export type MatchTally = { clean: number; mismatch: number; contradicted: number }

export function matchTally(row: PaneBRow): MatchTally {
  const siblings = row.confirmed_siblings ?? []
  return {
    clean: siblings.filter((sibling) => siblingGateState(sibling).kind === 'closed').length,
    // A half the door refused because its signed claims contradict
    // our record is a disagreement too, though it is never held as a sibling.
    mismatch:
      siblings.filter((sibling) => siblingGateState(sibling).kind === 'contradicted').length +
      (row.claims_refused ?? 0),
    contradicted: row.verdicts?.tally?.contradicted ?? 0
  }
}

/** UX §2 "Same request & answer": how many of their records hold the same
 *  request and answer as yours, and how many differ. Disputes are their own
 *  column, so the adjudication tally is not repeated here. */
export function matchTallyText(tally: MatchTally): string {
  return `${tally.clean} · ${tally.mismatch} differ`
}

// ---------------------------------------------------------------------------
// Alarm -- the most-recent negative signal, surfaced even collapsed.
// ---------------------------------------------------------------------------

export type AlarmSignal = { present: boolean; text: string; tone: 'bad' | 'warn' }

/** `resolveTimestamp` is an optional local-ledger lookup (the Ledger page
 *  already has `recordsById` from the raw capsule ledger) so a contradiction
 *  date can be shown when it's genuinely recoverable -- never fabricated
 *  when it isn't. */
export function alarmSignal(row: PaneBRow, resolveTimestamp?: (capsuleId: string) => string | null): AlarmSignal {
  const verdicts = row.verdicts
  if (row.referee_verdicts && adjudicationSummary(row).contradicted > 0) {
    return { present: true, text: 'A referee found their answer contradicted', tone: 'bad' }
  }
  if (!row.referee_verdicts && verdicts?.state === STATE_CONTRADICTED) {
    const capsuleId = verdicts.adjudication_capsule_id
    const when = capsuleId ? (resolveTimestamp?.(capsuleId) ?? null) : null
    return { present: true, text: when ? `Contradiction found ${when}` : 'Contradiction found', tone: 'bad' }
  }
  if (row.history?.state === STATE_FAILED) {
    return { present: true, text: 'Chain verification failed', tone: 'bad' }
  }
  if (row.history?.state === STATE_REFUSED || row.served?.state === STATE_REFUSED) {
    return { present: true, text: 'Peer refused a request', tone: 'warn' }
  }
  return { present: false, text: '', tone: 'warn' }
}

/** When the contradicting verdict was sealed, if the local ledger holds it. */
function adjudicationDate(row: PaneBRow, resolveTimestamp?: (capsuleId: string) => string | null): string | null {
  const capsuleId = row.verdicts?.adjudication_capsule_id
  return capsuleId ? (resolveTimestamp?.(capsuleId) ?? null) : null
}

export type PeerAttentionKey = keyof typeof PEER_ATTENTION
export type PeerAttentionItem = { key: PeerAttentionKey; label: string; tooltip: string; tone: 'bad' | 'warn' }

/** The specific things on a peer row that need a look, each counted from the
 *  payload -- never a number the payload doesn't carry. `disagreements` are the
 *  exchanges the ONE gate reads as CONTRADICTED; `differingAnswers` the sealed
 *  comparisons that found a contradiction. A failed log check and a refusal are
 *  states with no count, so they are named, not numbered. */
export function peerAttention(
  row: PaneBRow,
  resolveTimestamp?: (capsuleId: string) => string | null
): PeerAttentionItem[] {
  const items: PeerAttentionItem[] = []
  const disagreements = matchTally(row).mismatch
  if (disagreements > 0) {
    items.push({
      key: 'disagreements',
      label: PEER_ATTENTION.disagreements.label(disagreements),
      tooltip: PEER_ATTENTION.disagreements.tooltip(disagreements),
      tone: 'bad'
    })
  }
  const differing = row.referee_verdicts
    ? adjudicationSummary(row).contradicted
    : row.verdicts?.state === STATE_CONTRADICTED
      ? (row.verdicts.tally?.contradicted ?? 0)
      : 0
  if (differing > 0) {
    items.push({
      key: 'differingAnswers',
      label: PEER_ATTENTION.differingAnswers.label(differing),
      tooltip: PEER_ATTENTION.differingAnswers.tooltip(differing, adjudicationDate(row, resolveTimestamp)),
      tone: 'bad'
    })
  }
  if (row.history?.state === STATE_FAILED) {
    items.push({
      key: 'logFailed',
      label: PEER_ATTENTION.logFailed.label(),
      tooltip: PEER_ATTENTION.logFailed.tooltip(),
      tone: 'bad'
    })
  }
  if (row.history?.state === STATE_REFUSED || row.served?.state === STATE_REFUSED) {
    items.push({
      key: 'refused',
      label: PEER_ATTENTION.refused.label(),
      tooltip: PEER_ATTENTION.refused.tooltip(),
      tone: 'warn'
    })
  }
  return items
}

// ---------------------------------------------------------------------------
// Sort -- closest-first, alarms float to top. Never a trust ordering.
// ---------------------------------------------------------------------------

/** Rows with something to look at -- a disagreement, a dispute that found a
 *  difference, a log that didn't check out, a refusal -- come first (UX §2:
 *  "should I stop dealing with anyone?"). Never a trust ordering. */
export function peerSortKey(row: PaneBRow, latencyMs: number | null): readonly [number, number] {
  const alarmRank = alarmSignal(row).present || peerAttention(row).length > 0 ? 0 : 1
  const latency = latencyMs ?? Number.POSITIVE_INFINITY
  return [alarmRank, latency] as const
}

export function sortPeerRows<T>(rows: readonly T[], keyOf: (row: T) => readonly [number, number]): T[] {
  return [...rows].sort((a, b) => {
    const [aRank, aLatency] = keyOf(a)
    const [bRank, bLatency] = keyOf(b)
    return aRank - bRank || aLatency - bLatency
  })
}

export const SELF_REPORTED_NOTE = 'self-reported — not independently attested'

/** The full sentence behind the `self-reported` chip's (i) -- the terse chip
 *  on the row's face, the fuller honest statement on hover. */
export const SELF_REPORTED_DETAIL = SELF_REPORTED_TOOLTIP

// ---------------------------------------------------------------------------
// Witness -- no pane exposes a peer-scoped witness registration count today
// (`history.mine_for_reference.witnessed` is THIS node's OWN chain, carried
// for reference only, never the peer's -- see `PaneBHistoryCell`), so this
// states the absence honestly instead of borrowing that field.
// ---------------------------------------------------------------------------

export const WITNESS_COVERAGE_COMPACT_TEXT = 'not shown yet'

// ---------------------------------------------------------------------------
// Last dealt with -- the date of a row's latest exchange (UX §2), rendered by
// `periodRangeText`. `—` when there is no exchange history (the "advertised
// but unused" group).
// ---------------------------------------------------------------------------

function periodDateParts(iso: string): { day: number; month: string; year: number } {
  const parsed = new Date(iso)
  return {
    day: parsed.getUTCDate(),
    month: parsed.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' }),
    year: parsed.getUTCFullYear()
  }
}

export function periodRangeText(firstSeen: string | null, lastSeen: string | null): string {
  if (!firstSeen || !lastSeen) return '—'
  const first = periodDateParts(firstSeen)
  const last = periodDateParts(lastSeen)
  if (first.year === last.year && first.month === last.month) {
    return first.day === last.day ? `${first.day} ${first.month}` : `${first.day}–${last.day} ${first.month}`
  }
  if (first.year === last.year) {
    return `${first.day} ${first.month} – ${last.day} ${last.month}`
  }
  return `${first.day} ${first.month} ${first.year} – ${last.day} ${last.month} ${last.year}`
}

// ---------------------------------------------------------------------------
// Unified row view -- one shape for both row groups ("dealt with" rows
// carry the full Pane B row; "advertised but unused" rows have no exchange
// history at all, so every accountability column degrades to the honest
// "—" placeholder rather than a fabricated zero-of-zero).
//
// Accountability-only: no online status,
// latency, or route-to-chat field here (Network's job) -- see
// `PeerTableRow.tsx`.
// ---------------------------------------------------------------------------

export type PeerTableRowView = {
  key: string
  displayId: string
  hasDealings: boolean
  /** The row's alias evidence, one line (`peerAliasLine`): the other id
   *  spaces this ONE peer is known by -- never rendered as extra peers.
   *  `null` when the row carries no identity evidence. */
  aliasLine: string | null
  /** One honesty caveat under the identity, stated once (chooser-v2 §3
   *  discipline carried over): empty for a dealt-with peer (the legend says it once),
   *  "no exchanges yet" for one only advertised. */
  identityNote: string
  exchangeCount: number
  confirmedByOtherSide: string
  match: string
  adjudicationCompact: string
  witnessCompact: string
  period: string
  alarm: AlarmSignal
  /** What needs a look, each named and counted -- replaces the old generic
   *  ⚠ chip on the face (UX §8). Empty when nothing does. */
  attention: PeerAttentionItem[]
  /** Payments with this peer, counted (`peerSettlementText`); `null` when
   *  there is no paid exchange on record. */
  payments: string | null
  /** This peer's twin pairs, as facts (`peerTwinText`); `null` when no twin
   *  pair names it. */
  twins: string | null
  row: PaneBRow | null
}

function twinsText(peerId: string, exchangeRows: readonly PaneCRow[]): string | null {
  const tally = peerTwinTally(peerId, exchangeRows)
  return tally ? peerTwinText(tally) : null
}

export function dealtWithRowView(
  row: PaneBRow,
  resolveTimestamp?: (capsuleId: string) => string | null,
  /** The Exchanges rows (pane C), for this peer's twin pairs. */
  exchangeRows: readonly PaneCRow[] = []
): PeerTableRowView {
  // `peerDisplayId` returns `null` for a row with no counterparty identity
  // at all (see its own doc comment) -- callers filter those out before
  // reaching here, but this fallback keeps the view honest, never `null`,
  // if ever called directly with one.
  const displayId = peerDisplayId(row) ?? 'counterparty not recorded'
  return {
    key: row.peer_id ?? displayId,
    displayId,
    hasDealings: true,
    aliasLine: peerAliasLine(row),
    // UX §2: "self-reported" is true of every row, so it lives once in the
    // table legend, not on each row.
    identityNote: '',
    exchangeCount: row.exchange_count ?? 0,
    confirmedByOtherSide: confirmedByOtherSideText(confirmedByOtherSide(row)),
    match: matchTallyText(matchTally(row)),
    adjudicationCompact: adjudicationCompactText(adjudicationSummary(row)),
    witnessCompact: WITNESS_COVERAGE_COMPACT_TEXT,
    period: periodRangeText(row.last_seen, row.last_seen),
    alarm: alarmSignal(row, resolveTimestamp),
    attention: peerAttention(row, resolveTimestamp),
    payments: peerSettlementText(row.settlement),
    twins: twinsText(displayId, exchangeRows),
    row
  }
}

/** "Nodes advertised but unused" -- a peer the mesh knows about but this
 *  node has never exchanged with. Every accountability column renders `—`
 *  (nothing to report, not "0 of 0") -- there is no `PaneBRow` to derive
 *  one from. */
// ---------------------------------------------------------------------------
// Column keys -- shared between `PeerTableRow` and `LedgerPeersTable`'s
// Columns toggle. Lives here (a non-component module) rather than in either
// component file so exporting it never trips the fast-refresh
// only-export-components lint rule.
// ---------------------------------------------------------------------------

export type PeerTableColumnKey = 'exchanges' | 'confirmed' | 'match' | 'adjudication' | 'witness' | 'period'

export const ALL_PEER_TABLE_COLUMNS: ReadonlySet<PeerTableColumnKey> = new Set<PeerTableColumnKey>([
  'exchanges',
  'confirmed',
  'match',
  'adjudication',
  'witness',
  'period'
])

// The per-column (i) hover copy -- what the column means and what evidence
// backs it, moved off the face and behind the glyph. Each states the evidence,
// never a standing or score reading (the three-sharer-answers framing: what it
// is, what backs it, what it does NOT establish). The `confirmed` line is the
// task's blessed wording verbatim.
export const PEER_COLUMN_INFO: Record<PeerTableColumnKey, string> = PEER_COLUMN_TOOLTIPS

export function advertisedOnlyRowView(displayId: string): PeerTableRowView {
  return {
    key: displayId,
    displayId,
    hasDealings: false,
    aliasLine: null,
    identityNote: 'no exchanges yet',
    exchangeCount: 0,
    confirmedByOtherSide: '—',
    match: '—',
    adjudicationCompact: '—',
    witnessCompact: '—',
    period: '—',
    alarm: { present: false, text: '', tone: 'warn' },
    attention: [],
    payments: null,
    twins: null,
    row: null
  }
}

// Pure view model for the peer drill's
// "Their history" tab (UX review §2 drill, §7.2 names). Four parts, each with
// its own "not asked" state that is never rendered as a zero:
//   (a) Their log, as shown to you -- the peer's chain segment as your node
//       fetched it: checkpoints, per-checkpoint counts by kind, witness entries
//       the checkpoint lists. Never record ids.
//   (b) What others say -- "asked N references, M answered".
//   (c) Verdicts on exchanges they took part in, by provenance -- sealed by you
//       / delivered to them (with their ack or rebuttal) / reported by others.
//       Three columns, never added together (ADJUDICATIONS-ON-HISTORY-CARD.md).
//   (d) Asked of you -- requests this peer made of your node.
// Counts only, never sums across checkpoints or provenances, never a ratio.
// Peer-supplied tokens this view has no words for are never echoed onto the
// screen: a peer controls those strings.
import type {
  AdjudicationVerdictKind,
  AskedOfYouEntry,
  PaneBDeliveredVerdict,
  PaneBRow
} from '@/features/capsules/api/sidecarTypes'
import {
  confirmedByOtherSide,
  confirmedByOtherSideText,
  peerIdentityAliases,
  STATE_FAILED,
  STATE_REFUSED,
  STATE_VERIFIED
} from '@/features/capsules/lib/peer-row-view'

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] as const

/** `26 Sep 05:06 UTC` -- UTC and a fixed month table, so a fixture renders the
 *  same on every machine and locale. */
export function shortUtcTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  const day = date.getUTCDate()
  const month = MONTHS[date.getUTCMonth()]
  const hh = String(date.getUTCHours()).padStart(2, '0')
  const mm = String(date.getUTCMinutes()).padStart(2, '0')
  return `${day} ${month} ${hh}:${mm} UTC`
}

// ---------------------------------------------------------------------------
// Refusal reasons -- the evidence-request registry's tokens, in plain words.
// ---------------------------------------------------------------------------

const REASON_LABELS: Record<string, string> = {
  not_authorized: 'not shared under the sharing setting',
  no_such_record: 'no such record',
  coverage_unsatisfiable: 'that range isn’t available',
  request_malformed: 'the request was malformed',
  policy_decline: 'declined by policy'
}

export function reasonLabel(reason: string): string {
  return REASON_LABELS[reason] ?? 'another reason'
}

// ---------------------------------------------------------------------------
// (a) Their log, as shown to you
// ---------------------------------------------------------------------------

/** Chain-segment kinds (`SHARING-POLICY.md` "History segments") in plain words. */
const KIND_LABELS: Record<string, readonly [string, string]> = {
  exchange: ['exchange', 'exchanges'],
  exchange_twin: ['twin exchange', 'twin exchanges'],
  card: ['history card', 'history cards'],
  adjudication: ['verdict', 'verdicts'],
  adjudication_delivery_receipt: ['verdict delivered to them', 'verdicts delivered to them'],
  ack: ['acknowledgement', 'acknowledgements'],
  rebuttal: ['dispute', 'disputes'],
  close: ['period close', 'period closes'],
  stamp: ['checkpoint stamp', 'checkpoint stamps'],
  capsule: ['other record', 'other records']
}

const OTHER_KIND: readonly [string, string] = ['entry of another kind', 'entries of another kind']

/** `3 exchanges`, `1 twin exchange`. A kind this view has no words for is
 *  counted as "another kind", never shown as the peer's own token. */
export function kindCountText(kind: string, count: number): string {
  const [one, many] = KIND_LABELS[kind] ?? OTHER_KIND
  return plural(count, one, many)
}

export type KindCount = { kind: string; text: string; count: number }
/** `witnessEntries` is what the checkpoint lists; this view doesn't check them. */
export type CheckpointView = { size: number; time: string; witnessEntries: number; counts: KindCount[] }

export type TheirLogView =
  | { kind: 'not_asked'; text: string }
  | { kind: 'no_answer'; text: string }
  | { kind: 'refused'; text: string }
  | { kind: 'failed'; text: string }
  | { kind: 'shown'; text: string; checkpoints: CheckpointView[] }

/** True only when the producer counted the requests your node sent this peer
 *  (`asked` state other than `absent`) and the count is above zero. */
function askedThisPeer(row: PaneBRow): boolean {
  return row.asked?.state !== 'absent' && (row.asked?.count ?? 0) > 0
}

export function theirLogView(row: PaneBRow): TheirLogView {
  const cell = row.history
  const state = cell?.state
  if (state === STATE_REFUSED) {
    const why = cell?.refusal_reason ? ` (${reasonLabel(cell.refusal_reason)})` : ''
    return { kind: 'refused', text: `They refused to show their log${why}.` }
  }
  if (state === STATE_FAILED) {
    return { kind: 'failed', text: 'Their log didn’t check out on this machine.' }
  }
  if (state !== STATE_VERIFIED) {
    // A request sent with no checked answer back must not read as never asked.
    if (askedThisPeer(row)) {
      return { kind: 'no_answer', text: 'Asked — no answer from them has been checked here yet.' }
    }
    return { kind: 'not_asked', text: 'Never asked — your node hasn’t asked for their log.' }
  }
  // Coarsened (sharing policy default): per-checkpoint counts only. The UI
  // never reads `leaf_digests`, even when a peer volunteers them.
  const checkpoints: CheckpointView[] = (cell?.segment?.links ?? []).map((link) => ({
    size: link.checkpoint.mmr_size,
    time: shortUtcTime(link.checkpoint.timestamp),
    witnessEntries: link.checkpoint.witnesses?.length ?? 0,
    counts: Object.entries(link.leaf_counts)
      .filter(([, count]) => count > 0)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([kind, count]) => ({ kind, text: kindCountText(kind, count), count }))
  }))
  if (checkpoints.length === 0) {
    const n = cell?.history_summary?.checkpoint_count
    const seen = n != null ? ` (${plural(Number(n), 'checkpoint')} seen)` : ''
    return {
      kind: 'shown',
      text: `Your node fetched their log and it passed its checks${seen}. Counts per checkpoint weren’t included.`,
      checkpoints
    }
  }
  return {
    kind: 'shown',
    text: `${plural(checkpoints.length, 'checkpoint')} from their log, as your node fetched and checked it.`,
    checkpoints
  }
}

// ---------------------------------------------------------------------------
// Your dealings with them -- two counts, never a ratio. Same gate and the
// same "differ" note as the Peers table's confirmed column.
// ---------------------------------------------------------------------------

export function dealingsText(row: PaneBRow): string {
  const summary = confirmedByOtherSide(row)
  const differ = /· (\d+) differ$/.exec(confirmedByOtherSideText(summary))
  const differText = differ ? ` · ${differ[1]} differ` : ''
  return `${plural(row.exchange_count ?? 0, 'exchange')} · ${summary.confirmed} confirmed by the other side${differText}`
}

// ---------------------------------------------------------------------------
// (b) What others say
// ---------------------------------------------------------------------------

export type OthersSayView =
  { kind: 'never_asked'; text: string } | { kind: 'asked'; text: string; asked: number; answered: number }

/** `references_asked` is present only when the producer carries a references
 *  result (`peer_accountability_tab.verdicts_cell`); absent reads "never asked". */
export function othersSayView(row: PaneBRow): OthersSayView {
  const asked = row.verdicts?.references_asked
  if (asked === undefined) {
    return { kind: 'never_asked', text: 'Never asked — you haven’t asked other nodes about this peer.' }
  }
  const answered = row.verdicts?.references_answered ?? 0
  // `ack_refusals`: records the references hold of THIS peer refusing a
  // verdict delivered to it -- a count of records, not of references.
  const refusals = row.verdicts?.ack_refusals ?? 0
  const refusalsText =
    refusals > 0 ? ` · ${plural(refusals, 'record')} of them refusing a verdict delivered to them` : ''
  return {
    kind: 'asked',
    text: `Asked ${plural(asked, 'reference')} · ${answered} answered${refusalsText}`,
    asked,
    answered
  }
}

// ---------------------------------------------------------------------------
// (c) Verdicts on exchanges they took part in, by provenance. A verdict on a
// twin can find against either side, so these are not all "about" this peer:
// the owner a `contradicted:<owner>` verdict names is dropped by the
// producers before these counts arrive.
// ---------------------------------------------------------------------------

export const VERDICT_KINDS: readonly AdjudicationVerdictKind[] = ['corroborated', 'contradicted', 'inconclusive']

export type ProvenanceKey = 'byYou' | 'deliveredToThem' | 'byOthers'

export const PROVENANCE_LABELS: Record<ProvenanceKey, string> = {
  byYou: 'Sealed by you',
  deliveredToThem: 'Delivered to them, and their reply',
  byOthers: 'Reported by others'
}

export type VerdictCell = { known: false; text: string } | { known: true; text: string; count: number }

export type VerdictRow = { verdict: AdjudicationVerdictKind } & Record<ProvenanceKey, VerdictCell>

function countCell(count: number): VerdictCell {
  return { known: true, text: String(count), count }
}

function deliveredCell(entry: PaneBDeliveredVerdict | undefined): VerdictCell {
  const delivered = entry?.delivered ?? 0
  if (delivered === 0) return countCell(0)
  return {
    known: true,
    text: `${delivered} · ${entry?.acknowledged ?? 0} acknowledged · ${entry?.disputed ?? 0} disputed`,
    count: delivered
  }
}

/** One row per verdict kind, one cell per provenance. A provenance with no
 *  data is "not known" in every row -- never a column of zeros. There is no
 *  total row or column: the three provenances are different claims about who
 *  stands behind a verdict, so adding them would merge them. */
export function verdictRows(row: PaneBRow): VerdictRow[] {
  const ownTally = row.verdicts?.tally
  const theirAdjudications = row.history?.state === STATE_VERIFIED ? row.history.adjudications : undefined
  const deliveredKnown = theirAdjudications?.state === 'enriched'
  const referencesAsked = row.verdicts?.references_asked !== undefined

  return VERDICT_KINDS.map((verdict) => ({
    verdict,
    byYou: ownTally ? countCell(ownTally[verdict]) : { known: false, text: 'not counted' },
    deliveredToThem: deliveredKnown
      ? deliveredCell(theirAdjudications?.delivered?.[verdict])
      : { known: false, text: 'not shown' },
    byOthers: referencesAsked
      ? countCell(row.verdicts?.references_tally?.[verdict] ?? 0)
      : { known: false, text: 'never asked' }
  }))
}

// ---------------------------------------------------------------------------
// (d) Asked of you. The inbound log is node-wide and keyed by the requester's
// SELF-DECLARED id, so a line belongs in this drill only when that id is one
// this peer's row is known by. A line naming no one belongs to no peer, and a
// record push is the peer sending you a record, not a request: both are left
// out here.
// ---------------------------------------------------------------------------

const SUBJECT_LABELS: Record<string, string> = {
  record: 'one record',
  correlation: 'records about an exchange',
  chain_segment: 'a piece of your log',
  range: 'a range of your log'
}

export function subjectLabel(subjectKind: string | null): string {
  if (subjectKind === null) return 'not stated'
  return SUBJECT_LABELS[subjectKind] ?? 'something else'
}

const OUTCOME_LABELS: Record<Exclude<AskedOfYouEntry['status'], 'received'>, string> = {
  answered: 'answered',
  refused: 'refused'
}

export type AskedOfYouLine = { when: string; who: string; subject: string; outcome: string; reason: string | null }

export type AskedOfYouView =
  | { kind: 'not_shown'; text: string }
  | { kind: 'shown'; text: string; asked: number; answered: number; lines: AskedOfYouLine[] }

/** Every id this peer's row is known by: its row key and each alias. */
function idsForPeer(row: PaneBRow): Set<string> {
  const aliases = peerIdentityAliases(row)
  return new Set(
    [row.peer_id, row.node?.peer_id, aliases.signingKeyId, aliases.endpointId, aliases.nodeId].filter(
      (id): id is string => typeof id === 'string' && id.length > 0
    )
  )
}

type AskedEntry = AskedOfYouEntry & { requester_id: string; status: 'answered' | 'refused' }

function isRequestFromPeer(entry: AskedOfYouEntry, ids: ReadonlySet<string>): entry is AskedEntry {
  return (
    entry.path === 'evidence-request' &&
    entry.status !== 'received' &&
    entry.requester_id !== null &&
    ids.has(entry.requester_id)
  )
}

/** Newest first. `answered` counts every request your node did not refuse. */
export function askedOfYouView(row: PaneBRow): AskedOfYouView {
  const log = row.asked_of_you
  if (!log) {
    return { kind: 'not_shown', text: 'Not shown — this node isn’t keeping a log of the requests made of it.' }
  }
  const ids = idsForPeer(row)
  const entries = log.entries
    .filter((entry): entry is AskedEntry => isRequestFromPeer(entry, ids))
    .sort((a, b) => b.ts.localeCompare(a.ts))
  const answered = entries.filter((entry) => entry.status === 'answered').length
  return {
    kind: 'shown',
    // The requester id is self-declared, so the visible line says only what
    // the request named, never who sent it.
    text: `${plural(entries.length, 'request')} that named you · ${answered} answered`,
    asked: entries.length,
    answered,
    lines: entries.map((entry) => ({
      when: shortUtcTime(entry.ts),
      who: entry.requester_id,
      subject: subjectLabel(entry.subject_kind),
      outcome: OUTCOME_LABELS[entry.status],
      reason: entry.reason ? reasonLabel(entry.reason) : null
    }))
  }
}

// The two-sided Ledger row's right-cell state.
// Eight states -- v3 §2's table enumerates six ("closed"/
// "contradicted"/refused/absent/asked/not-asked), plus `open_not_held`
// (finding 1: a peer-asserted
// capsule id with no bytes fetched yet is its own honest state, not a fast
// path into `closed`) and `open_not_given` (a non-digest correlation marker
// is nothing fetchable, not a fetch waiting to happen).
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import { isDigestShaped } from '@/features/capsules/lib/canonical'
import { CLOSED_CELL_TOOLTIPS, ROW_STATE_TOOLTIPS } from '@/features/capsules/lib/tooltip-copy'
import type { PeerRecomputeState } from '@/features/capsules/lib/recompute-identity'

export type RightCellStateKind =
  | 'closed' // artifact, agrees
  | 'contradicted' // artifact, disagrees
  | 'open_refused' // signed_refusal
  | 'open_absent' // recorded_absence
  | 'open_asked' // unanswered
  | 'open_not_held' // peer id known (digest-shaped, fetchable), bytes not fetched
  | 'open_not_given' // peer asserted only a non-digest correlation marker: nothing fetchable
  | 'open_not_asked' // not asked

export type RightCellState = {
  kind: RightCellStateKind
  /** The date carried by the underlying signed statement/ask, when the row
   *  data actually carries one -- never invented when it doesn't. */
  date: string | null
}

// Wire vocabulary from v3 §2's "evidence" column. The evidence-request
// carrier that would populate this is still unwired end-to-end as of this
// batch (capsule-emit-mesh's `evidence_responder.py` docstring: "Carrier
// wiring is out of scope here ... not yet reachable over the wire") -- this
// mapping exists so the UI is forward-compatible the moment it is.
const EVIDENCE_OUTCOME_TO_KIND: Record<string, RightCellStateKind> = {
  // This node's door refused the half the provider pushed: signed with its
  // key, it named another server or other model weights than our record.
  claims_refused: 'contradicted',
  signed_refusal: 'open_refused',
  recorded_absence: 'open_absent',
  unanswered: 'open_asked',
  not_asked: 'open_not_asked'
}

/**
 * Derives the right-cell state from a Pane C row (`deriveRightCellState`,
 * below the helpers).
 *
 * If the row carries `theirs.evidence_outcome` (one of the four
 * evidence-request outcomes above), that value wins outright. A row with no
 * counterparty recorded at all (`theirs.state === 'absent'`) is
 * `open_not_asked`.
 *
 * Otherwise CLOSED/CONTRADICTED come from ONE predicate
 * (`counterpartyHalfState`), whichever way the counterparty's bytes arrived:
 *   - a live `mesh_ledger_fetch` this browser ran (`theirsRecompute`), or
 *   - the half the peer PUSHED, correlated by the pane from our own citing
 *     record (`pushedHalfRecompute`: the held body, the door's signature
 *     verdict, and the `capsule_id` recompute done before the pane query
 *     resolved).
 * CLOSED needs all four: the signature verified against the peer's announced
 * key, the body recomputes to its `capsule_id`, both `effect` digests equal
 * ours (§6.2/L-G), and the provider check (`providerMatches`, provisional).
 * CONTRADICTED: the body does not recompute to its id, both sides carry
 * real digests that differ, or the provider's half names other model weights
 * than it served us under (`modelSwapped`). A PUSHED half from a node that did not serve the
 * exchange is never CONTRADICTED (any peer can push). Anything else -- not recomputed yet, unsigned,
 * a missing digest, a different or unnamed provider -- stays OPEN
 * (`open_not_held`): never a verdict we did not check.
 *
 * With no evidence at all, a digest-shaped peer id is `open_not_held` (known,
 * fetchable) and a non-digest correlation marker is `open_not_given`
 * (**L-C: the right cell never renders a state we inferred**).
 */
/** Reads a dotted-path string field off a loosely-typed record, `null` when
 *  the path doesn't resolve to a non-empty string -- same discipline as
 *  `security-checks-view.ts`'s own `recordString`. */
function recordString(record: Record<string, unknown> | null | undefined, path: readonly string[]): string | null {
  let cursor: unknown = record
  for (const key of path) {
    if (typeof cursor !== 'object' || cursor === null) return null
    cursor = (cursor as Record<string, unknown>)[key]
  }
  return typeof cursor === 'string' && cursor.length > 0 ? cursor : null
}

/** §6.2/L-G: CLOSED's "cites your half by digest" claim, checked for real.
 *  `ours` must actually be a digest (never `unavailable`/`not a digest`
 *  read as a coincidental string match), and the peer's own fetched record
 *  must carry that exact value at the same path -- absent on either side
 *  reads as "does not cite", never a fabricated match. */
function digestFieldCitesOurs(
  localRecord: CapsuleRecord | null | undefined,
  peerRecord: Record<string, unknown> | null,
  path: readonly string[]
): boolean {
  const ours = recordString(localRecord, path)
  if (!ours || !isDigestShaped(ours)) return false
  return recordString(peerRecord, path) === ours
}

/** Both halves of §6.2/L-G's digest citation -- request AND response. */
function digestsCiteOurHalf(
  localRecord: CapsuleRecord | null | undefined,
  peerRecord: Record<string, unknown> | null
): boolean {
  return (
    digestFieldCitesOurs(localRecord, peerRecord, ['effect', 'request_digest']) &&
    digestFieldCitesOurs(localRecord, peerRecord, ['effect', 'response_digest'])
  )
}

const DIGEST_FIELDS = [
  ['effect', 'request_digest'],
  ['effect', 'response_digest']
] as const

/** Both halves carry a real digest for the same field and they differ: the
 *  two sides are on record disagreeing about this exchange. A field missing
 *  on either side is never read as disagreement. */
function digestsDisagree(
  localRecord: CapsuleRecord | null | undefined,
  peerRecord: Record<string, unknown> | null
): boolean {
  return DIGEST_FIELDS.some((path) => {
    const ours = recordString(localRecord, path)
    const theirs = recordString(peerRecord, path)
    return !!ours && !!theirs && isDigestShaped(ours) && isDigestShaped(theirs) && ours !== theirs
  })
}

/** Every weights file digest a record names: the producer's
 *  `compute_attestation.weights_digest.digest` and the host's
 *  `serving_provenance.model.weights_digest`. Both are the loaded file's SHA-256. */
export function weightsClaims(record: Record<string, unknown> | null | undefined): Set<string> {
  const claims = new Set<string>()
  const add = (value: string | null) => {
    const v = value?.trim().toLowerCase()
    if (v && /^[0-9a-f]{64}$/.test(v)) claims.add(v)
  }
  add(recordString(record, ['model_attestation', 'compute_attestation', 'weights_digest', 'digest']))
  add(recordString(record, ['model_attestation', 'compute_attestation', 'x-mesh-poc-v1', 'serving_provenance', 'model', 'weights_digest']))
  return claims
}

/** The `sha256-<hex>` inside `model_attestation.model_id`. mesh-llm's
 *  `local-gguf/sha256-<source identity>` is the file's own SHA-256 only for a
 *  single local file (a Hugging Face GGUF hashes its repository, revision and
 *  files), so it names the model, not the weights file. */
export function modelIdDigest(record: Record<string, unknown> | null | undefined): string | null {
  const modelId = recordString(record, ['model_attestation', 'model_id'])
  return modelId?.toLowerCase().match(/sha256[-:]([0-9a-f]{64})/)?.[1] ?? null
}

/** The provider's half names a different model than the one it served us
 *  under: its own weights claims disagree with each other, or they or its
 *  model id disagree with our record's. Weights are held to weights and ids
 *  to ids; only a half naming no model id is held, by its weights, to the id
 *  we asked for. Names alone never count (aliases). */
function modelSwapped(localRecord: CapsuleRecord | null | undefined, peerRecord: Record<string, unknown> | null): boolean {
  const local = localRecord as Record<string, unknown> | null | undefined
  const theirs = weightsClaims(peerRecord)
  if (theirs.size > 1) return true
  const disagree = (asked: Set<string>, claimed: Set<string>) =>
    asked.size > 0 && claimed.size > 0 && ![...claimed].some((w) => asked.has(w))
  const askedWeights = weightsClaims(local)
  if (disagree(askedWeights, theirs)) return true
  const askedId = modelIdDigest(local)
  const askedIds = new Set(askedId ? [askedId] : [])
  const theirId = modelIdDigest(peerRecord)
  if (theirId) return disagree(askedIds, new Set([theirId]))
  return askedWeights.size === 0 && disagree(askedIds, theirs)
}

const SERVED_BY_PATH = ['model_attestation', 'compute_attestation', 'x-mesh-poc-v1', 'serving_provenance', 'served_by_node_id'] as const

/** "Obtained from the provider": both halves name the same serving node.
 *
 *  PROVISIONAL -- check (i). The rule is `received_from == served_by`, but the
 *  citing record's `received_from` is the plugin's endpoint id, not a mesh
 *  node id, and the host cannot yet supply the sender's node id. Until the
 *  peer ANNOUNCEMENT's key -> node join lands, this compares the node the
 *  provider-signed body names as its server (its own signature, verified at
 *  the door against the announced key for `received_from`) with the node OUR
 *  record says served us. When the announcement join lands, replace this with
 *  check (ii): the announced node id for the signing key must equal our
 *  `served_by_node_id`. */
function providerMatches(
  localRecord: CapsuleRecord | null | undefined,
  peerRecord: Record<string, unknown> | null
): boolean {
  const ours = recordString(localRecord, SERVED_BY_PATH)
  const theirs = recordString(peerRecord, SERVED_BY_PATH)
  return !!ours && ours !== 'unknown' && ours === theirs
}

/** The counterparty half the pane correlated from OUR citing record, as the
 *  same evidence shape a live fetch produces -- the held body, the door's
 *  signature verdict, and the in-browser `capsule_id` recompute
 *  (`pushed-half-recompute.ts`). `null` when the row carries no pushed body. */
export function pushedHalfRecompute(row: PaneCRow): PeerRecomputeState | null {
  const theirs = row.theirs
  if (!theirs.record || theirs.signature_ok === undefined) return null
  return {
    status: 'found',
    idMatch: theirs.id_match ?? null,
    signatureOk: theirs.signature_ok === true,
    peerRecord: theirs.record,
    fetch: () => {}
  }
}

/** THE gate. CLOSED iff all four hold for provider-signed bytes this node
 *  holds (pushed or fetched): the signature verified against the peer's
 *  announced key, the body recomputes to its `capsule_id`, both digests equal
 *  ours, and it came from the provider (`providerMatches`). */
function counterpartyHalfState(
  evidence: PeerRecomputeState,
  localRecord: CapsuleRecord | null | undefined,
  source: 'fetched' | 'pushed'
): RightCellStateKind {
  // Recompute not run / could not run: inconclusive, never a verdict.
  if (evidence.idMatch === null) return 'open_not_held'
  const fromProvider = providerMatches(localRecord, evidence.peerRecord)
  // Any peer can push a half. One from a node that did not serve this
  // exchange says nothing about it: never a contradiction, never a close. A
  // fetched half came from the peer this browser asked.
  if (source === 'pushed' && !fromProvider) return 'open_not_held'
  // The peer's own bytes don't produce the id claimed for them.
  if (!evidence.idMatch) return 'contradicted'
  // Unauthenticated bytes prove nothing either way.
  if (evidence.signatureOk !== true) return 'open_not_held'
  if (!fromProvider) return 'open_not_held'
  if (digestsDisagree(localRecord, evidence.peerRecord)) return 'contradicted'
  // Matching digests prove the two sides saw the same bytes, not which model
  // made them: a provider half naming other weights is never CLOSED.
  if (modelSwapped(localRecord, evidence.peerRecord)) return 'contradicted'
  if (digestsCiteOurHalf(localRecord, evidence.peerRecord)) return 'closed'

  return 'open_not_held'
}

/** A real sealed capsule id is 64 lower-hex (`capsule-emit-mesh`
 *  capsule-emit `DIGEST_LEN`); the self-minted per-response marker
 *  `capsule-<response.id>` (e.g. `capsule-chatcmpl-…`) is not. Only a
 *  digest-shaped id can be fetched and closed against real bytes, so a
 *  non-digest peer assertion must render "not given" with nothing to fetch,
 *  never "known but not fetched" (a fetch that can only fail). Mirrors the
 *  host's `capsule_id_is_digest_shaped` (openai-frontend). */
export function capsuleIdIsDigestShaped(id: string | null | undefined): boolean {
  return typeof id === 'string' && /^[0-9a-f]{64}$/.test(id)
}

export function deriveRightCellState(
  row: PaneCRow,
  theirsRecompute?: PeerRecomputeState,
  localRecord?: CapsuleRecord | null
): RightCellState {
  const outcome = row.theirs.evidence_outcome
  const mappedKind = outcome ? EVIDENCE_OUTCOME_TO_KIND[outcome] : undefined
  if (mappedKind) {
    return { kind: mappedKind, date: row.theirs.evidence_outcome_date ?? null }
  }

  // `theirs.state === 'absent'` means no counterparty is recorded for this
  // row at all -- structurally, `peerFetchJoinKey` (`recompute-identity.
  // ts`) never produces a join key for such a row, so no real fetch could
  // have happened against it. Checked before `theirsRecompute` so this fact
  // always wins, even over a `theirsRecompute` object a caller passed in
  // that no longer matches this row (a stale prop during a re-render, a
  // test double) -- the row's own recorded state is the ground truth.
  if (row.theirs.state === 'absent') {
    return { kind: 'open_not_asked', date: null }
  }

  // One code path for both sources: a live fetch this browser ran wins; else
  // the pushed half the pane correlated from our citing record. Our half is
  // the caller's record when supplied, else the body the pane sent with the
  // pair.
  const fetchedHalf = theirsRecompute?.status === 'found' ? theirsRecompute : null
  const evidence = fetchedHalf ?? pushedHalfRecompute(row)
  if (evidence) {
    const ours = localRecord ?? (row.mine.record as CapsuleRecord | undefined) ?? null
    return { kind: counterpartyHalfState(evidence, ours, fetchedHalf ? 'fetched' : 'pushed'), date: null }
  }

  // No confirmed fetch yet. A peer id is "known but not fetched" (fetchable)
  // ONLY when it is digest-shaped; a self-minted correlation marker
  // (`capsule-chatcmpl-…`) is not fetchable, so the honest state is "not
  // given", not a fetch waiting to happen that can only fail.
  if (!capsuleIdIsDigestShaped(row.theirs.capsule_id)) {
    return { kind: 'open_not_given', date: null }
  }

  return { kind: 'open_not_held', date: null }
}

function dateOrFallback(date: string | null): string {
  return date ?? 'date unavailable'
}

// ---------------------------------------------------------------------------
// Bracket strip (mesh-evidence-tab-double-entry-design-2026-09-23 §7, redrawn
// per the 2026-09-26 UX review §3). The row carries no leading glyph: the
// right cell already states the exchange in a sentence, so a second, unlabelled
// glyph was one more thing to decode. The strip is a pure derivation from the
// SAME row + state the badge renders -- never a new predicate.
// ---------------------------------------------------------------------------

/** One side of the strip: a dot and the word that says what it means. */
export type BracketSide = { glyph: '●' | '○'; word: string }

/** UX §3: the strip is drawn without notation -- two dots under two labels,
 *  `Yours ● sealed` and `Theirs ○ not received`, joined by a line only when
 *  both records are held. The word carries the meaning, so no glyph legend is
 *  needed. */
export type BracketStrip = { yours: BracketSide; joined: boolean; theirs: BracketSide }

function theirSide(kind: RightCellStateKind): BracketSide {
  switch (kind) {
    case 'closed':
      return { glyph: '●', word: 'same' }
    case 'contradicted':
      return { glyph: '●', word: 'differs' }
    case 'open_refused':
      return { glyph: '○', word: 'refused' }
    case 'open_absent':
      return { glyph: '○', word: 'they have no record' }
    case 'open_asked':
      return { glyph: '○', word: 'asked, no reply yet' }
    case 'open_not_held':
    case 'open_not_given':
      return { glyph: '○', word: 'not received' }
    case 'open_not_asked':
      return { glyph: '○', word: 'not asked for' }
    default: {
      const exhaustiveCheck: never = kind
      return exhaustiveCheck
    }
  }
}

export function bracketStrip(row: PaneCRow, state: RightCellState): BracketStrip {
  // Ours is sealed whenever our half is held; `mine.state === 'absent'` is the
  // received-without-a-commitment row shape.
  const yours: BracketSide =
    row.mine.state === 'absent' ? { glyph: '○', word: 'not held' } : { glyph: '●', word: 'sealed' }
  const theirs = theirSide(state.kind)
  return { yours, joined: yours.glyph === '●' && theirs.glyph === '●', theirs }
}

export function bracketStripText(strip: BracketStrip): string {
  const link = strip.joined ? ' —— ' : '   '
  return `Yours ${strip.yours.glyph} ${strip.yours.word}${link}Theirs ${strip.theirs.glyph} ${strip.theirs.word}`
}

/** UX §3 "Put `Ask for their record` on the row only after a timeout": with
 *  push on, their record normally arrives when the exchange finishes, so an
 *  ask is offered only once this long has passed without it. A minute: a
 *  node that runs no record-push door never receives a push, and asking is
 *  then the only way its row closes, so the wait stays short. Committed
 *  times are minute-granular, so the ask appears within a minute or two. */
export const ASK_FOR_RECORD_AFTER_MS = 60 * 1000

/** True once the exchange is old enough that their record should have
 *  arrived. A row with no parseable timestamp cannot be timed, so the ask is
 *  offered rather than hidden. */
export function askForRecordIsDue(timestamp: string | null, nowMs: number): boolean {
  const at = timestamp ? Date.parse(timestamp) : Number.NaN
  if (Number.isNaN(at)) return true
  return nowMs - at >= ASK_FOR_RECORD_AFTER_MS
}

/** D4(d): the CLOSED right column as
 *  compact per-property cells -- their id · signature ✓ · request digest = ·
 *  response digest =. Rendered ONLY when the ONE gate already returned
 *  `closed`, i.e. each cell restates a fact the gate's own inputs
 *  established (`signatureOk === true`, both digests cite our half) -- no
 *  second predicate, no re-derivation. */
export type ClosedPropertyCell = { key: keyof typeof CLOSED_CELL_TOOLTIPS; label: string; tooltip: string }

/** The CLOSED row's per-property cells, each with its own one-sentence (i). */
export function closedPropertyCellItems(row: PaneCRow): ClosedPropertyCell[] {
  const theirId = row.theirs.capsule_id
  return [
    {
      key: 'their_id',
      label: theirId ? `their id ${theirId.slice(0, 12)}…` : 'their id —',
      tooltip: CLOSED_CELL_TOOLTIPS.their_id
    },
    { key: 'signature', label: 'signature ✓', tooltip: CLOSED_CELL_TOOLTIPS.signature },
    { key: 'request', label: 'request =', tooltip: CLOSED_CELL_TOOLTIPS.request },
    { key: 'response', label: 'response =', tooltip: CLOSED_CELL_TOOLTIPS.response }
  ]
}

export function closedPropertyCells(row: PaneCRow): string[] {
  return closedPropertyCellItems(row).map((cell) => cell.label)
}

/** The right cell's rendered sentence (UX §3: the state as a sentence).
 *  Every OPEN variant reads as a presence fact, never a problem (L-A) -- only
 *  `contradicted` names an alarm word (`differs`). */
export function rightCellText(state: RightCellState): string {
  switch (state.kind) {
    case 'closed':
      return '✓ They recorded the same request and answer'
    case 'contradicted':
      return '✗ Their record differs'
    case 'open_refused':
      return `They declined, and signed the refusal — ${dateOrFallback(state.date)}`
    case 'open_absent':
      return `They say they have no record of this — ${dateOrFallback(state.date)}`
    case 'open_asked':
      return `Asked ${dateOrFallback(state.date)}. No reply yet.`
    case 'open_not_held':
      return 'Their record hasn’t arrived yet.'
    case 'open_not_given':
      return 'Their record hasn’t arrived yet.'
    case 'open_not_asked':
      return 'You haven’t asked for their record.'
    default: {
      const exhaustiveCheck: never = state.kind
      return exhaustiveCheck
    }
  }
}

/** The fuller story behind a state's terse cell -- moved off the face and
 *  behind the (i). Each states what the terse cell means and what evidence
 *  backs it (or honestly, what is missing), never a score. The CLOSED and
 *  not-held cases are the ones Item 4 calls out; the rest carry the same
 *  discipline so every state's (i) has an honest sentence. */
export function rightCellDetail(state: RightCellState): string {
  return ROW_STATE_TOOLTIPS[state.kind]
}

/** The row's state label, with "in their log" on a CLOSED row once our
 *  records cite their inclusion proof for it (§7.6b). No number: their
 *  checkpoint's leaf count includes their padding, so it is never a count of
 *  their records. */
export function rowStatusLabel(state: RightCellState, inTheirLog: object | null | undefined): string {
  const label = rightCellStatusLabel(state)
  return state.kind === 'closed' && inTheirLog ? `${label} · in their log` : label
}

export function rightCellStatusLabel(state: RightCellState): string {
  switch (state.kind) {
    case 'closed':
      return 'CLOSED'
    case 'contradicted':
      return 'CONTRADICTED'
    case 'open_refused':
      return 'OPEN · refused'
    case 'open_absent':
      return 'OPEN · absent'
    case 'open_asked':
      return 'OPEN · asked'
    case 'open_not_held':
      return 'OPEN · not held'
    case 'open_not_given':
      return 'OPEN'
    case 'open_not_asked':
      return 'OPEN'
    default: {
      const exhaustiveCheck: never = state.kind
      return exhaustiveCheck
    }
  }
}

/** The action control that lives inside the cell, or `null` for `closed`
 *  (nothing to do once a row agrees and cites your half) and
 *  `open_not_held` (the real action -- `fetch peer capsule &
 *  recompute here` -- lives inside the `▸ checks` panel, next to the
 *  identity it fetches for, not as a second copy of the same button at the
 *  row summary level). */
export function rightCellAction(state: RightCellState): string | null {
  switch (state.kind) {
    case 'closed':
      return null
    case 'contradicted':
      return 'Compare'
    case 'open_refused':
      return 'View refusal'
    case 'open_absent':
      return 'View statement'
    case 'open_asked':
      return 'Ask again'
    case 'open_not_held':
      return null
    case 'open_not_given':
      return null
    case 'open_not_asked':
      return 'Ask them for their record'
    default: {
      const exhaustiveCheck: never = state.kind
      return exhaustiveCheck
    }
  }
}

/** The two states whose action is literally "ask a counterparty"
 * -- every other action (`Compare` / `View
 *  refusal` / `View statement`) inspects evidence this node already holds,
 *  not a live counterparty to contact, so only these two require a
 *  recorded counterparty before the action can render. */
const ASK_ACTION_KINDS: ReadonlySet<RightCellStateKind> = new Set(['open_not_asked', 'open_asked'])

export function isAskAction(kind: RightCellStateKind): boolean {
  return ASK_ACTION_KINDS.has(kind)
}

/** L-B: "Only CONTRADICTED gets alarm styling. It is the one state where
 *  two signed records disagree. Everything else is a presence fact." This
 *  predicate is also what enforces L-A ("an open row is never styled as a
 *  problem") -- every OPEN kind returns `false` here, same as `closed`. */
export function isAlarmState(state: RightCellState): boolean {
  return state.kind === 'contradicted'
}

/** The ledger's state toolbar filter values (v3 §2a: "useful filters are
 *  states, not qualities: open / closed / contradicted / asked-no-reply /
 *  twins only"). `twins` isn't a right-cell state at all -- it's whether a
 *  row belongs to a twin bracket, which callers derive separately (see
 *  `exchange-pages.ts`; no bracket data exists until B6). */
export type LedgerStateFilterValue = 'open' | 'closed' | 'contradicted' | 'asked_no_reply'

export const LEDGER_STATE_FILTER_VALUES: readonly LedgerStateFilterValue[] = [
  'closed',
  'contradicted',
  'asked_no_reply',
  'open'
]

/** Buckets the seven right-cell states into the toolbar's state filter
 *  values: `open_asked` (a real ask, no reply yet) gets its own
 *  `asked_no_reply` bucket, and the four "we hold no reply/no fetch at all"
 *  states collapse into the broader `open` bucket. */
export function ledgerStateFilterValue(state: RightCellState): LedgerStateFilterValue {
  switch (state.kind) {
    case 'closed':
      return 'closed'
    case 'contradicted':
      return 'contradicted'
    case 'open_asked':
      return 'asked_no_reply'
    case 'open_refused':
    case 'open_absent':
    case 'open_not_held':
    case 'open_not_given':
    case 'open_not_asked':
      return 'open'
    default: {
      const exhaustiveCheck: never = state.kind
      return exhaustiveCheck
    }
  }
}

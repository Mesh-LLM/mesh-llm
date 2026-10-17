// Toggle ② the security view (v3 §4) -- pure
// view-model derivation, kept separate from `SecurityChecksView.tsx`'s
// rendering so the honesty rules below are unit-testable without mounting a
// component. Exact block order: IDENTITY -> HEADER -> WHAT IT COMMITS TO ->
// CHECKS -> raw.
//
// The mockup in v3 §4 draws a fully two-sided CHECKS table (a YOURS and a
// THEIRS column, both populated). The real wire contract doesn't back that:
// `PaneCRow.theirs` carries only `{state, capsule_id, ...}` -- no bytes, no
// signature, no per-property checks -- and `useRecomputedIdentity`'s own
// contract is explicit that a counterparty's half in Pane C has no full
// record to recompute against (`recompute-identity.ts`). So THEIRS never
// renders a state this page didn't actually check: every THEIRS cell is
// `NOT_CHECKED` with an honest reason, or the whole column is absent when
// `theirs.state === 'absent'` (L-G, the same rule the row's own "WHAT IT
// COMMITS TO" comparison already lives by).
import type { CapsuleRecord } from '@/features/capsules/api/types'
import type { PaneCRow, TeeFact } from '@/features/capsules/api/sidecarTypes'
import { pushedHalfRecompute, type RightCellStateKind } from '@/features/capsules/lib/exchange-row-state'
import type { PeerRecomputeState, RecomputedIdentity } from '@/features/capsules/lib/recompute-identity'
import { peerFetchJoinKey } from '@/features/capsules/lib/recompute-identity'
import { NINE_PROPERTY_LABELS, PROPERTY_GROUP, RECOMPUTED_PROPERTIES } from '@/features/capsules/lib/nine-properties'
import { labelForState } from '@/features/capsules/lib/assurance-tone'
import { isDigestShaped } from '@/features/capsules/lib/canonical'
import { OWNER_LINKED_PHRASE, OWNER_NOT_LINKED_PHRASE } from '@/features/capsules/lib/tooltip-copy'

function boolToWireState(value: boolean | null): string {
  return value === null ? 'NOT_CHECKED' : value ? 'PASS' : 'FAIL'
}

/** A peer-asserted id is on the record and known -- NOT that any bytes are
 *  held. Governs only whether this row's IDENTITY/CHECKS columns show an
 *  honestly-labelled `NOT_CHECKED`/"as given, not recomputed" cell at all;
 *  it must never gate a `✓ same` claim (that requires `peerRecordFor`
 *  below to return non-null) -- conflating the two was finding 2 of the
 *  2026-09-23 assessment. */
function theirsHeld(row: PaneCRow): boolean {
  return row.theirs.state !== 'absent'
}

/** True only when `theirsRecompute` carries a REAL, arrived result for
 *  content_binding/producer_signature -- never merely because a fetch is
 *  in flight or queued (the same "recomputed means it actually ran"
 *  discipline `buildChecksRows` already applies to `mine`, see
 *  `actuallyRecomputed` below). */
function theirsActuallyRecomputed(theirsRecompute: PeerRecomputeState | undefined): boolean {
  return theirsRecompute?.status === 'found'
}

/** The peer's actual capsule record, present only once a `mesh_ledger_fetch`
 *  this browser ran has actually arrived (`recompute-identity.ts`'s
 *  `peerRecord`) -- `null` at every other status, including a successful
 *  fetch whose `capsule_id` recompute didn't run. Header/commits-to rows
 *  use this, never `theirsHeld`, to decide whether a `theirs` cell can show
 *  a real value at all (finding 2: no field is `✓ same` until this is
 *  non-null AND the fetched field's own value actually equals ours). */
function peerRecordFor(theirsRecompute: PeerRecomputeState | undefined): Record<string, unknown> | null {
  return theirsRecompute?.status === 'found' ? theirsRecompute.peerRecord : null
}

/** piece 4 -- a row is fetchable the moment
 *  `usePeerLedgerRecompute`'s own join-key rule says so (re-exported here so
 *  the component layer has one place to ask "should I show the fetch
 *  action", never a second copy of the join-key rule). */
export const theirsFetchable = peerFetchJoinKey

// ---------------------------------------------------------------------------
// IDENTITY
// ---------------------------------------------------------------------------

export type IdentityCell = { value: string; note: string }
export type IdentityRow = { label: string; yours: IdentityCell; theirs: IdentityCell | null }

/** Witness-level wording only -- "recomputed here", never second-party
 *  judgment wording such as "verified by" (a browser recompute is never
 *  rendered as someone else's judgment; the test lists the banned phrases).
 *  RENDERING NOTE (design §7, 2026-09-23): a peer-asserted capsule id with
 *  no held bytes carries the outline glyph `◔` -- id known, bytes not held
 *  -- never the filled `●`/CLOSED reading finding 1 found the tab
 *  defaulting to. */
function theirsIdentityNote(theirsRecompute: PeerRecomputeState | undefined): string {
  if (theirsActuallyRecomputed(theirsRecompute)) {
    if (theirsRecompute?.idMatch === true) return '✓ recomputed here, matches'
    if (theirsRecompute?.idMatch === false) return '✕ recomputed here, MISMATCH'
  }
  if (theirsRecompute?.status === 'fetching') return 'fetching…'
  if (theirsRecompute?.status === 'not_found') return 'they have no such record'
  if (theirsRecompute?.status === 'error') return `fetch failed: ${theirsRecompute.errorMessage ?? 'unknown error'}`
  return '◔ as given, not recomputed'
}

export function buildIdentityRow(
  row: PaneCRow,
  identity: RecomputedIdentity,
  theirsRecompute?: PeerRecomputeState
): IdentityRow {
  const yoursNote =
    identity.idMatch === true
      ? '✓ recomputed here, matches'
      : identity.idMatch === false
        ? '✕ recomputed here, MISMATCH'
        : 'not yet recomputed'
  // Finding 5: a peer-asserted id that isn't digest-shaped (a chat-
  // completion id forwarded as-is, `capsule-chatcmpl-…`) is not a capsule
  // id -- never rendered as their identity, and never fetchable
  // (`peerFetchJoinKey` already refuses it, so `theirsFetchable` disables
  // the fetch action for exactly this row).
  const theirsCapsuleId = row.theirs.capsule_id && isDigestShaped(row.theirs.capsule_id) ? row.theirs.capsule_id : null
  return {
    label: 'capsule id',
    yours: { value: row.mine.capsule_id ?? '—', note: yoursNote },
    theirs: theirsHeld(row)
      ? { value: theirsCapsuleId ?? 'not given', note: theirsIdentityNote(theirsRecompute) }
      : null
  }
}

// ---------------------------------------------------------------------------
// HEADER -- trimmed to fields this page actually has: a per-record wire
// timestamp, this system's one fixed signing algorithm (EdDSA/Ed25519, the
// only one `cose.ts` ever verifies against -- a true system-wide constant,
// not a per-record invention), the local record's own `key_id` (a real top-
// level field -- see `canonical.ts`'s exclusion list), and the exchange id
// itself. Fields the wire contract doesn't carry at all today (content type,
// capture boundary/rule as distinct fields, a client nonce) are left out
// rather than rendered as a permanently-"unavailable" row for a concept that
// doesn't exist in this data yet -- capture boundary/rule is covered by the
// CHECKS block's own `capture_coverage` line instead of being duplicated
// here.
// ---------------------------------------------------------------------------

export type HeaderCell = { value: string; note?: string }
export type HeaderRow = { label: string; yours: HeaderCell; theirs: HeaderCell | null }

/** Finding 2, reworked (2026-09-23 bounce): the whole `theirs` column is
 *  `null` (no cell at all, not even a placeholder) only when `!held` --
 *  `row.theirs.state === 'absent'`, no counterparty recorded, nothing to
 *  ever fetch (L-G). Once a counterparty IS recorded (`held`), a field this
 *  panel could compare renders a real value if the fetch resolved one, or
 *  the honest `{ value: '—', note: 'not held' }` placeholder while it
 *  hasn't -- an EMPTY DIV there used to read as "nothing to show", visually
 *  indistinguishable from the true not-held-a-counterparty case this same
 *  function already renders correctly for `absent` rows. Never a fabricated
 *  value either way. */
function heldCell(held: boolean, value: string | null): HeaderCell | null {
  if (!held) return null
  return value ? { value } : { value: '—', note: 'not held' }
}

export function buildHeaderRows(
  row: PaneCRow,
  localRecord: CapsuleRecord | null,
  theirsRecompute?: PeerRecomputeState
): HeaderRow[] {
  const keyId = localRecord && typeof localRecord.key_id === 'string' ? (localRecord.key_id as string) : null
  const held = theirsHeld(row)
  const peerRecord = peerRecordFor(theirsRecompute)
  const peerTimestamp = peerRecord && typeof peerRecord.timestamp === 'string' ? peerRecord.timestamp : null
  const peerKeyId = peerRecord && typeof peerRecord.key_id === 'string' ? peerRecord.key_id : null
  return [
    {
      label: 'sealed at',
      yours: { value: row.timestamp ?? 'timestamp unavailable' },
      theirs: heldCell(held, peerTimestamp)
    },
    {
      label: 'algorithm',
      yours: { value: 'EdDSA / Ed25519' },
      // A true system-wide constant, not a per-record claim -- safe to
      // state once real peer bytes are held, without needing to re-read it
      // off the fetched record.
      theirs: heldCell(held, peerRecord ? 'EdDSA / Ed25519' : null)
    },
    {
      label: 'key id',
      yours: { value: keyId ?? 'unavailable' },
      theirs: heldCell(held, peerKeyId)
    },
    {
      // No peer-record field is this node's own `exchange_key` (a per-node
      // grouping derivation, `exchange_key_for` -- `capsule_panes_native.
      // rs`) -- there is nothing to compare even once bytes are held, so
      // `theirs` never renders here, not even a `not held` placeholder: this
      // is a structurally-absent field, not an unfetched value.
      label: 'exchange id',
      yours: { value: row.exchange_key },
      theirs: null
    }
  ]
}

// ---------------------------------------------------------------------------
// WHAT IT COMMITS TO -- the digest-shaped facts a record actually names.
//
// v3 §4: "every ✓ same is a corroboration you
// can see... on an open row, the right column of this block is simply
// absent."
//
// **Finding 2 (2026-09-23 assessment) -- corrected.** This block used to
// treat the row's right-cell state as license to mirror every `yours` value
// into `theirs` labelled `✓ same` the moment the row read CLOSED -- restating
// a fact this page never actually knew (a CLOSED badge, `exchange-row-
// state.ts`'s own fix, now itself requires a fetched-and-verified peer
// artifact, but "the fetched capsule's own id matches what they claimed for
// it" is not the same fact as "this field's value equals mine"). Every
// `theirs` cell below is now `null` (absent) until `theirsRecompute` carries
// the PEER'S OWN fetched record (`peerRecordFor`), and even then it compares
// that record's actual field to ours -- `✓ same` only when they truly match,
// `✕ differs` when they don't, `not held` when the peer's own record simply
// doesn't carry the field. `task binding` and `served by` have no reliable
// peer-record equivalent to compare against (task binding is a locally-
// computed property; "who served" is this row's own role fact, not
// something the peer's record restates the same way) -- their `theirs`
// column stays absent regardless of fetch state, never a fabricated match.
// ---------------------------------------------------------------------------

export type CommitsToCell = { value: string; note: string }
export type CommitsToRow = { label: string; yours: string; theirs: CommitsToCell | null }

function recordString(record: Record<string, unknown> | null, path: readonly string[]): string | null {
  let cursor: unknown = record
  for (const key of path) {
    if (typeof cursor !== 'object' || cursor === null) return null
    cursor = (cursor as Record<string, unknown>)[key]
  }
  return typeof cursor === 'string' && cursor.length > 0 ? cursor : null
}

/** The node a record names as the one that served it
 *  (`serving_provenance.served_by_node_id`), by the pane's own rule
 *  (`evidence_panes.rs`, `peer_fetch_join_key`): a non-empty id other than
 *  the producer's `unknown` placeholder. `null` when the record names none. */
export function recordServedBy(record: object | null): string | null {
  const id = recordString(record as Record<string, unknown> | null, [
    'model_attestation',
    'compute_attestation',
    'x-mesh-poc-v1',
    'serving_provenance',
    'served_by_node_id'
  ])
  return id && id !== 'unknown' ? id : null
}

/** What a record's TDX quote shows (u231): "bound" only when this node
 *  checked all eight checks, including the quote's signatures and its PCK
 *  chain to the pinned root; what stays unchecked is on its own line, and
 *  nothing here says "attested" or "verified". A failed check is shown as
 *  failed, never softened. */
export const TEE_COPY = {
  bound: 'Quote signed by a TDX quoting enclave and bound to this exchange (checked here)',
  absent: 'no quote',
  failed: (check: string) => `Quote failed a check (${check.replace(/_/g, ' ')}): not bound`,
  platform: 'Platform status, revocation and measurements not checked here'
} as const

export function teeQuoteText(fact: TeeFact | undefined): string | null {
  if (!fact) return null
  if (fact.state === 'bound') return TEE_COPY.bound
  if (fact.state === 'failed') return TEE_COPY.failed(fact.check)
  return TEE_COPY.absent
}

function teeRows(row: PaneCRow): CommitsToRow[] {
  const yours = teeQuoteText(row.mine.tee)
  const theirs = teeQuoteText(row.theirs.tee)
  if (yours === null && theirs === null) return []
  const rows: CommitsToRow[] = [
    {
      label: 'TEE quote',
      yours: yours ?? '—',
      theirs: theirs === null ? null : { value: theirs, note: 'checked on this node' }
    }
  ]
  const mineBound = row.mine.tee?.state === 'bound'
  const theirsBound = row.theirs.tee?.state === 'bound'
  if (mineBound || theirsBound) {
    rows.push({
      label: 'TEE platform',
      yours: mineBound ? TEE_COPY.platform : '—',
      theirs: theirsBound ? { value: TEE_COPY.platform, note: 'left to you' } : null
    })
  }
  return rows
}

export function buildCommitsToRows(
  row: PaneCRow,
  localRecord: CapsuleRecord | null,
  theirsRecompute?: PeerRecomputeState
): CommitsToRow[] {
  const effect = (localRecord?.effect ?? null) as { request_digest?: string; response_digest?: string } | null
  const taskBindingCell = row.properties?.task_binding ?? null
  const modelId = localRecord?.model_attestation?.model_id ?? null
  const held = theirsHeld(row)
  const peerRecord = peerRecordFor(theirsRecompute)

  /** `theirs` is `null` (no cell at all) only when `!held` -- no
   *  counterparty is recorded for this row, so there is nothing to ever
   *  fetch (L-G). Once a counterparty IS recorded, this renders the honest
   *  `not held` placeholder until a real peer fetch resolves, and then
   *  compares THAT record's own value at `path` to `yoursValue` -- never a
   *  mirror of `yoursValue` itself, and never a bare empty cell while a
   *  real comparison is merely pending (finding 2, 2026-09-23 bounce). */
  function theirsFor(yoursValue: string, path: readonly string[]): CommitsToCell | null {
    if (!held) return null
    if (!peerRecord) return { value: '—', note: 'not held' }
    const peerValue = recordString(peerRecord, path)
    if (peerValue === null) return { value: '—', note: 'not held' }
    return peerValue === yoursValue ? { value: peerValue, note: '✓ same' } : { value: peerValue, note: '✕ differs' }
  }

  const requestDigestRaw = effect?.request_digest ?? null
  const requestDigest =
    requestDigestRaw === null ? 'unavailable' : isDigestShaped(requestDigestRaw) ? requestDigestRaw : 'not a digest'
  const responseDigestRaw = effect?.response_digest ?? null
  const responseDigest =
    responseDigestRaw === null ? 'unavailable' : isDigestShaped(responseDigestRaw) ? responseDigestRaw : 'not a digest'
  const taskBinding = taskBindingCell?.text ?? 'from the record'
  const modelIdentity = modelId ?? 'unavailable'
  // Who served us, as OUR record names it: its `served_by_node_id`
  // (`recordServedBy`). The pane's `theirs.peer_id` is the same value, but
  // it is sent only while the other half is still to be fetched, so a
  // CLOSED row has none; it is the fallback when this view holds no record.
  // Never the bare word "counterparty"; "not recorded" when neither names one.
  const servedBy =
    row.role_tag === 'SERVED'
      ? 'this node'
      : (recordServedBy(localRecord ?? (row.mine.record as CapsuleRecord | undefined) ?? null) ??
        row.theirs.peer_id ??
        'not recorded')

  return [
    {
      label: 'request digest',
      yours: requestDigest,
      theirs:
        requestDigestRaw !== null && isDigestShaped(requestDigestRaw)
          ? theirsFor(requestDigestRaw, ['effect', 'request_digest'])
          : null
    },
    {
      label: 'response digest',
      yours: responseDigest,
      theirs:
        responseDigestRaw !== null && isDigestShaped(responseDigestRaw)
          ? theirsFor(responseDigestRaw, ['effect', 'response_digest'])
          : null
    },
    { label: 'task binding', yours: taskBinding, theirs: null },
    {
      label: 'model identity',
      yours: modelIdentity,
      theirs: theirsFor(modelIdentity, ['model_attestation', 'model_id'])
    },
    { label: 'served by', yours: servedBy, theirs: null },
    ...teeRows(row)
  ]
}

// ---------------------------------------------------------------------------
// CHECKS -- the ten manifesto properties (capture_coverage renders as one
// full-width line, no YOURS/THEIRS split, matching v3 §4's mockup).
//
// Q1/L-L/L-M/L-G still hold (see below),
// plus this task's own rules:
//  - `local_inclusion`/`checkpoint_signature`/`continuity` and
//    `external_registration` resolve NOT_PRESENT, never the generic
//    NOT_CHECKED default, when the sidecar sends nothing for them --
//    "no checkpoint/receipt covers this record" is a fact, not an
//    unchecked question.
//  - `continuity` is established ONLY if `checkpoint_signature` is
//    established -- a sidecar-asserted continuity PASS is downgraded to
//    NOT_PRESENT if its own dependency isn't established (a checkpoint
//    signature this page cannot see cannot anchor continuity).
//  - `capture_coverage` renders the fixed sentence when the sidecar has a
//    record for it, else the bare `not present` result -- never a PASS
//    chip/variant.
//  - `identity_authority` is two fixed facts, not one state: binding
//    (absent/bound/invalid) and authority (ALWAYS not present -- this
//    system never binds a person). See `buildIdentityAuthorityFacts`.
//  - a property is never rendered `NOT_CHECKED` while flagged `recomputed`
//    -- that combination asserts "recomputed in-browser" and "we didn't
//    check" at once, which is a contradiction (the forbidden mutant).
// ---------------------------------------------------------------------------

export type ChecksSideCell = {
  /** Wire state (`PASS`/`FAIL`/...). */
  state: string
  /** Lowercase manifesto word (Q1) -- `established`/`failed`/... */
  label: string
  /** L-L: the input/policy phrase that must render beside the state,
   *  always -- never a bare state word. */
  detail: string
  /** L-M: true only when this cell's value actually came from an
   *  in-browser recompute that ran (never true merely because the key is
   *  one of `RECOMPUTED_PROPERTIES` -- see the forbidden-mutant note
   *  above). Drives the visual distinction from sidecar-sourced
   *  properties. */
  recomputed: boolean
}

/** One named fact under the `identity_authority` row -- `binding` or
 *  `authority`. Kept as facts under ONE row (not two rows) so the row's
 *  key stays `identity_authority` and the manifesto's ten-key list is
 *  unchanged. */
export type IdentityAuthorityFact = { factLabel: string; cell: ChecksSideCell }

export type ChecksRow = {
  key: string
  label: string
  /** Which of the two labelled groups this row belongs to
   *  (`nine-properties.ts`'s `PROPERTY_GROUP`). */
  group: string
  yours: ChecksSideCell | null
  theirs: ChecksSideCell | null
  /** `capture_coverage` only -- one line, no column split. */
  singleLine?: string
  /** `identity_authority` only -- two facts, no YOURS/THEIRS split. */
  facts?: IdentityAuthorityFact[]
}

// Properties whose absence means "no checkpoint covers this record" --
// distinct wording from `external_registration`'s "no receipt".
const CHECKPOINT_DEPENDENT = new Set(['local_inclusion', 'checkpoint_signature', 'continuity'])

function yoursDetailFor(key: string, state: string, text: string | undefined, actuallyRecomputed: boolean): string {
  if (text) return text
  if (actuallyRecomputed) return 'recomputed here'
  if (RECOMPUTED_PROPERTIES.has(key)) return 'not yet recomputed here'
  if (state === 'NOT_PRESENT') {
    // once-per-node checkpoint/registration
    // facts live on the Integrity section, never repeated per row -- this
    // row links back with one line instead of duplicating them.
    if (key === 'local_inclusion') return 'No checkpoint covers this record — see Integrity.'
    if (CHECKPOINT_DEPENDENT.has(key)) return 'no checkpoint covers this record'
    if (key === 'external_registration') return 'no receipt covers this record'
  }
  return 'from the record'
}

function theirsDetailFor(key: string, theirsRecompute: PeerRecomputeState | undefined): string {
  if (RECOMPUTED_PROPERTIES.has(key)) {
    if (theirsActuallyRecomputed(theirsRecompute)) return 'recomputed here'
    if (theirsRecompute?.status === 'fetching') return 'fetching…'
    if (theirsRecompute?.status === 'not_found') return 'they have no such record'
    if (theirsRecompute?.status === 'error') return `fetch failed: ${theirsRecompute.errorMessage ?? 'unknown error'}`
    return 'their record not received'
  }
  return 'not checked here'
}

/** Look finding 2: whether a checkpoint covers THIS record, from the same
 *  card figure the Integrity strip shades (`checkpointCoverageByRecord`).
 *  `null` = not reported. The native pane sends no per-row checkpoint
 *  properties, and reading their absence as "no checkpoint covers this
 *  record" contradicted Integrity's "All 8 records are sealed into a
 *  checkpoint" on the same screen. */
function checkpointCoverageCell(key: string, covered: boolean | null): { state: string; text: string } | null {
  if (covered === true) {
    return {
      state: 'NOT_CHECKED',
      text:
        key === 'local_inclusion'
          ? 'a checkpoint covers this record — see Integrity; this row hasn’t checked that for itself yet'
          : 'a checkpoint covers this record; not checked here'
    }
  }
  if (covered === false) {
    return {
      state: 'NOT_PRESENT',
      text:
        key === 'local_inclusion'
          ? 'No checkpoint covers this record yet — see Integrity.'
          : 'no checkpoint covers this record yet'
    }
  }
  // Not reported: keep the ruled not-present default (`absentDefaultFor`).
  return null
}

function absentDefaultFor(key: string): string {
  return CHECKPOINT_DEPENDENT.has(key) || key === 'external_registration' ? 'NOT_PRESENT' : 'NOT_CHECKED'
}

function buildIdentityAuthorityFacts(row: PaneCRow, ownerLinked: boolean | null): IdentityAuthorityFact[] {
  const cell = row.properties?.identity_authority ?? null
  const bindingState = cell?.state ?? 'NOT_PRESENT'

  let binding: ChecksSideCell
  if (bindingState === 'PASS') {
    const expiry = typeof cell?.expiry === 'string' ? cell.expiry : 'unknown'
    binding = {
      state: 'PASS',
      label: labelForState('PASS'),
      detail: typeof cell?.text === 'string' ? cell.text : `self-asserted key, valid to ${expiry}`,
      recomputed: false
    }
  } else if (bindingState === 'FAIL') {
    binding = {
      state: 'FAIL',
      label: labelForState('FAIL'),
      detail: typeof cell?.text === 'string' ? cell.text : 'signature invalid',
      recomputed: false
    }
  } else if (!cell && ownerLinked === true) {
    // p2 item 3: the pane carries no binding property, but the node's own
    // status says its key is linked to an owner (verified) -- the same fact
    // Integrity's step 2 states, in the same words.
    binding = {
      state: 'PASS',
      label: labelForState('PASS'),
      detail: OWNER_LINKED_PHRASE,
      recomputed: false
    }
  } else {
    // "no key bound" was wrong: every record IS signed by this node's key
    // (`key_id`); what is missing is the link from that key to an owner.
    binding = {
      state: 'NOT_PRESENT',
      label: labelForState('NOT_PRESENT'),
      detail: OWNER_NOT_LINKED_PHRASE,
      recomputed: false
    }
  }

  // Authority (a person standing behind the key) is always `not present` --
  // this system never binds a person to a key. Not a fact about this
  // record; a fact about the system.
  // p2 item 4: an empty owner never renders as a literal "…".
  const owner = typeof cell?.owner === 'string' && cell.owner.length > 0 ? cell.owner : null
  const authority: ChecksSideCell = {
    state: 'NOT_PRESENT',
    label: labelForState('NOT_PRESENT'),
    detail: owner
      ? `Owner: ${owner} — not bound to a person.`
      : binding.state === 'PASS'
        ? 'Owner linked (self-asserted) — not bound to a person.'
        : 'No owner set — not bound to a person.',
    recomputed: false
  }

  return [
    { factLabel: 'binding', cell: binding },
    { factLabel: 'authority', cell: authority }
  ]
}

/** The ONE gate's verdict on the other side's record (`deriveRightCellState`)
 *  as an outcome-corroboration check -- so the panel's outcome row, the chip
 *  strip, and the row badge all say the same thing. `null` = the gate reached
 *  no verdict (every OPEN state), and the row falls back to the pane. */
function gateOutcomeCell(gateKind: RightCellStateKind | undefined): { state: string; text: string } | null {
  if (gateKind === 'closed') return { state: 'PASS', text: 'their record has the same request and answer' }
  if (gateKind === 'contradicted') return { state: 'FAIL', text: 'their record differs' }
  return null
}

/** Facts from outside the Pane C row that the checks need, so every surface
 *  derives them the same way. */
export type ChecksContext = {
  /** The ONE gate's verdict for this row (`deriveRightCellState`). */
  gateKind?: RightCellStateKind
  /** Whether a checkpoint covers this record (look finding 2); `null` or
   *  absent = not reported. Used only where the pane carries no per-row
   *  checkpoint property. */
  checkpointCovered?: boolean | null
  /** p2 item 3: whether this node's key is linked to an owner, from the SAME
   *  `ownerLinked` derivation Integrity's step 2 reads; `null` = not known.
   *  Used only where the pane carries no `identity_authority` property. */
  ownerLinked?: boolean | null
}

export function buildChecksRows(
  row: PaneCRow,
  identity: RecomputedIdentity,
  theirsRecompute?: PeerRecomputeState,
  { gateKind, checkpointCovered = null, ownerLinked = null }: ChecksContext = {}
): ChecksRow[] {
  const gateOutcome = gateOutcomeCell(gateKind)
  const held = theirsHeld(row)
  const checkpointSignatureState = row.properties?.checkpoint_signature?.state ?? 'NOT_PRESENT'

  return Object.keys(NINE_PROPERTY_LABELS).map((key) => {
    const label = NINE_PROPERTY_LABELS[key]
    const group = PROPERTY_GROUP[key]

    if (key === 'capture_coverage') {
      const cell = row.properties?.[key] ?? null
      const singleLine =
        cell && cell.state !== 'NOT_PRESENT'
          ? (cell.text ?? 'captured by the plugin at this node’s serving boundary (rule: every served exchange)')
          : labelForState('NOT_PRESENT')
      return { key, label, group, yours: null, theirs: null, singleLine }
    }

    if (key === 'identity_authority') {
      return { key, label, group, yours: null, theirs: null, facts: buildIdentityAuthorityFacts(row, ownerLinked) }
    }

    const recomputable = RECOMPUTED_PROPERTIES.has(key)
    let yoursState: string
    if (recomputable) {
      yoursState = boolToWireState(key === 'content_binding' ? identity.idMatch : identity.signatureOk)
    } else {
      const coverageCell =
        CHECKPOINT_DEPENDENT.has(key) && !row.properties?.[key] ? checkpointCoverageCell(key, checkpointCovered) : null
      yoursState =
        (key === 'outcome_corroboration' ? gateOutcome?.state : undefined) ??
        coverageCell?.state ??
        row.properties?.[key]?.state ??
        absentDefaultFor(key)
      // Continuity is established ONLY if checkpoint_signature is: a
      // sidecar-claimed continuity PASS with no established checkpoint
      // signature to anchor to is downgraded. A continuity state that
      // was already something other than PASS (FAIL/INCONCLUSIVE/
      // NOT_CHECKED) is left as-is -- this rule only blocks the upgrade
      // to "established," it doesn't invent a stronger negative result.
      if (key === 'continuity' && yoursState === 'PASS' && checkpointSignatureState !== 'PASS') {
        yoursState = 'NOT_PRESENT'
      }
    }
    const yoursText = recomputable
      ? undefined
      : key === 'outcome_corroboration' && gateOutcome
        ? gateOutcome.text
        : ((CHECKPOINT_DEPENDENT.has(key) && !row.properties?.[key]
            ? checkpointCoverageCell(key, checkpointCovered)?.text
            : undefined) ?? (row.properties?.[key]?.text as string | undefined))
    // The forbidden mutant: a property is only "recomputed" if the
    // recompute actually produced a value -- not merely because its key
    // is recomputable while the recompute hasn't run yet (idMatch/
    // signatureOk still null, so yoursState is NOT_CHECKED).
    const actuallyRecomputed = recomputable && yoursState !== 'NOT_CHECKED'

    const yours: ChecksSideCell = {
      state: yoursState,
      label: labelForState(yoursState),
      detail: yoursDetailFor(key, yoursState, yoursText, actuallyRecomputed),
      recomputed: actuallyRecomputed
    }

    let theirs: ChecksSideCell | null = null
    if (held) {
      // A pushed record the badge already judged (CLOSED / CONTRADICTED)
      // fills THEIRS too, from the same evidence the gate reads
      // (`pushedHalfRecompute`): its id was redone in this browser (checked
      // here); its signature was checked by this node's door when it arrived
      // (node says).
      const pushed = theirsActuallyRecomputed(theirsRecompute) ? null : pushedHalfRecompute(row)
      if (recomputable && pushed && pushed.status === 'found' && pushed.idMatch !== null) {
        const pushedState = boolToWireState(key === 'content_binding' ? pushed.idMatch : pushed.signatureOk)
        theirs = {
          state: pushedState,
          label: labelForState(pushedState),
          detail:
            key === 'content_binding'
              ? 'their record, redone in this browser'
              : 'their record, checked by this node when it arrived',
          recomputed: key === 'content_binding'
        }
      } else if (recomputable && theirsActuallyRecomputed(theirsRecompute)) {
        const theirsRecomputedState = boolToWireState(
          key === 'content_binding' ? (theirsRecompute?.idMatch ?? null) : (theirsRecompute?.signatureOk ?? null)
        )
        theirs = {
          state: theirsRecomputedState,
          label: labelForState(theirsRecomputedState),
          detail: theirsDetailFor(key, theirsRecompute),
          recomputed: true
        }
      } else {
        theirs = {
          state: 'NOT_CHECKED',
          label: labelForState('NOT_CHECKED'),
          detail: theirsDetailFor(key, theirsRecompute),
          recomputed: false
        }
      }
    }

    return { key, label, group, yours, theirs }
  })
}

/** p2 item 2: the THEIRS sentence that repeats down the column ("their log,
 *  no proof given" on seven rows), said once as a column note. `null` when no
 *  sentence repeats. */
export function sharedTheirsDetail(rows: readonly ChecksRow[]): string | null {
  const counts = new Map<string, number>()
  for (const r of rows) {
    if (r.theirs) counts.set(r.theirs.detail, (counts.get(r.theirs.detail) ?? 0) + 1)
  }
  let best: string | null = null
  let bestCount = 1
  for (const [detail, count] of counts) {
    if (count > bestCount) {
      best = detail
      bestCount = count
    }
  }
  return best
}

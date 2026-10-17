// Integrity section view-model --
// completes the section past the chain strip + four stat cards
//: the setup checklist, the
// registration copy, and the once-per-node facts. Pure functions, kept
// separate from `LedgerPage.tsx`'s rendering for the same reason
// `security-checks-view.ts` is -- testable without mounting a component.
//
// `checkpoint_count` / `continuity` / `witnesses` / `owner_added_at` /
// `owner_card_index` are the pre-existing aspirational `card` fields
// (`capsule_panes_native.rs::build_pane_a` returns `card: null` today --
// see that file's module docs on what this native reader does and does not
// compute yet). `registered_no_later_than` and `asked_peer_at` below are
// new fields in the SAME discipline: read if a future backend populates
// them, degrade to the honest "not set up" state otherwise, never
// fabricated. Owner identity is NOT read from `card` -- it comes from the
// live `/api/status` `owner` field (`StatusPayload.owner`, `lib/api/
// types.ts` -- the loose wire shape `PeerInfo['owner']` already carries,
// same field `status-adapter.ts`'s `resolveOwner` reads for the Network
// dashboard), so that one fact is real today, not aspirational.
import type { JsonRecord } from '@/features/capsules/api/types'
import { formatExchangeTimestamp } from '@/features/capsules/lib/local-time'
import {
  CHAIN_STRIP_TOOLTIP,
  INTEGRITY_TILE_TOOLTIPS,
  OWNER_LINKED_PHRASE,
  OWNER_NOT_LINKED_PHRASE,
  WITNESS_OFF
} from '@/features/capsules/lib/tooltip-copy'

/** `StatusPayload.owner` / `PeerInfo['owner']` verbatim (`lib/api/
 *  types.ts`) -- deliberately not re-typed narrower than the wire shape. */
export type StatusOwner = string | { status?: string; verified?: boolean; name?: string; display_name?: string }

/** p2 item 3: linked only when the host VERIFIED the ownership certificate
 *  (`OwnershipStatus::Verified`, api/status.rs). Every other status the host
 *  sends -- `unsigned`, `expired`, `invalid_signature`, `revoked_*`, … -- is a
 *  non-empty string, and the old truthiness test read each of them as bound:
 *  the look showed "linked" on a node whose status said `unsigned`. */
export function ownerLinked(owner: StatusOwner | null | undefined): boolean {
  if (owner == null) return false
  if (typeof owner === 'string') return owner.length > 0
  return owner.verified === true || owner.status === 'verified'
}

const ownerBound = ownerLinked

// ---------------------------------------------------------------------------
// Checkpoint registration -- THE one fact (D1). A local checkpoint is NOT registration: registration means a
// witness this node doesn't run holds the checkpoint. Rung 1, the witness
// line, and the Exchanges headline all derive from this ONE derivation so
// they can never disagree (the live shot had rung 1 say "registered" while
// the same pane said "Registered with 0 witnesses" and Exchanges said "not
// registered" -- the exact §7 overclaim class).
// ---------------------------------------------------------------------------

/** Where this node's checkpoints stand with a witness, the one fact the hero
 *  line, the witness tile, the registration copy and the setup step read:
 *  - `latest`: a witness holds the latest checkpoint;
 *  - `earlier`: a witness holds an earlier checkpoint, not the latest yet
 *    (a checkpoint cut between two offers to the witness);
 *  - `pending`: a witness is set, and none holds a checkpoint yet;
 *  - `off`: no witness is set;
 *  - `unknown`: none holds one, and the node did not say whether one is set.
 *  Only `off` may say the witness is off. */
export type WitnessState = 'latest' | 'earlier' | 'pending' | 'off' | 'unknown'

export type CheckpointRegistration = {
  /** The host reported a checkpoint count at all (`null` card = not
   *  reported -- distinct from a real zero, see `buildSetupSteps`). */
  reported: boolean
  checkpointCount: number | null
  witnessCount: number
  /** At least one checkpoint exists on disk -- a LOCAL fact only. */
  checkpointedLocally: boolean
  /** THE registration fact: a checkpoint held by at least one witness.
   *  Only this may ever render the word "registered". */
  registered: boolean
  /** How many checkpoints a witness holds, the latest or not. */
  witnessedCheckpointCount: number
  witnessState: WitnessState
}

/** The latest checkpoint's receipts that CHECKED: the plugin verified each
 *  against the witness's key and this checkpoint before marking it. A
 *  receipt that is only held, unchecked or refused, never counts. */
export function checkedWitnesses(card: JsonRecord | null | undefined): JsonRecord[] {
  const all = Array.isArray(card?.witnesses) ? (card.witnesses as unknown[]) : []
  return all.filter(
    (witness): witness is JsonRecord =>
      !!witness && typeof witness === 'object' && (witness as JsonRecord).checked === true
  )
}

/** A witness's short name: the host of its URL. */
export function witnessName(url: unknown): string {
  if (typeof url !== 'string' || url === '') return 'a witness'
  const rest = url.includes('://') ? url.split('://')[1] : url
  return rest.split('/')[0].split('@').pop() ?? rest
}

export function checkpointRegistration(card: JsonRecord | null | undefined): CheckpointRegistration {
  const checkpointCount = typeof card?.checkpoint_count === 'number' ? card.checkpoint_count : null
  const witnesses = checkedWitnesses(card)
  const checkpointedLocally = checkpointCount !== null && checkpointCount > 0
  const registered = checkpointedLocally && witnesses.length > 0
  const counted = typeof card?.witnessed_checkpoint_count === 'number' ? card.witnessed_checkpoint_count : 0
  const witnessedCheckpointCount = checkpointedLocally ? Math.max(counted, registered ? 1 : 0) : 0
  const configured = typeof card?.witness_configured === 'boolean' ? card.witness_configured : null
  const witnessState: WitnessState = registered
    ? 'latest'
    : witnessedCheckpointCount > 0
      ? 'earlier'
      : configured === true
        ? 'pending'
        : configured === false
          ? 'off'
          : 'unknown'
  return {
    reported: checkpointCount !== null,
    checkpointCount,
    witnessCount: witnesses.length,
    checkpointedLocally,
    registered,
    witnessedCheckpointCount,
    witnessState
  }
}

/** The witness tile's line under its count, by the same fact. */
export function witnessTileNote(state: WitnessState): string | undefined {
  switch (state) {
    case 'earlier':
      return 'the latest checkpoint is not held yet'
    case 'pending':
      return 'witness on — none holds a checkpoint yet'
    case 'off':
      return WITNESS_OFF
    default:
      return undefined
  }
}

/** Step 2's done state (UX §4): what binding actually established. */
export const OWNER_LINKED_SENTENCE = `Your records are signed by this node’s key, ${OWNER_LINKED_PHRASE}.`

/** The honest rung-1/witness-line copy for a checkpoint no witness holds. */
// Finding 7: "witness: off" is the witness tile's to say ("off — your
// choice"); the step says only what this step's state is.
export const CHECKPOINTED_NOT_REGISTERED_STATUS = 'checkpointed locally · no witness'

// ---------------------------------------------------------------------------
// Setup checklist (ledger-ux-from-the-user-2026-09-09 §6) -- three steps in
// value order, each stating what it buys and what it does not. Never a
// muted "getting started" tip -- this IS the honest state of a node with
// nothing registered, and per the design note's whole point, it should
// still be legible once each step is done.
// ---------------------------------------------------------------------------

export type SetupStep = {
  key: 'checkpoints' | 'identity' | 'ask_peer'
  title: string
  done: boolean
  /** Status word shown next to the title -- "not set up" / "registered" /
   *  "never asked" / "asked <date>". */
  status: string
  /** What it buys and what it does not -- omitted once done, since the
   *  explanatory sentence is written for someone deciding whether to do
   *  the step, not for someone who already has. */
  body: string | null
  /** One thing to do next, when there is one: a link, never a fake button. */
  action?: { label: string; href: string }
}

export function buildSetupSteps(
  card: JsonRecord | null | undefined,
  owner: StatusOwner | null | undefined,
  /** Halves that arrived by push and closed
   *  through the ONE gate (`IntegritySection`'s own `closedByOtherSideCount`,
   *  the same tile figure). Corroboration can arrive by push, not only by an
   *  ask this node sent -- so a nonzero count marks the "ask a peer" step done,
   *  worded to match the tile, never left reading "never asked" while the tile
   *  reads CLOSED. Defaults to 0 (the pre-push-path behaviour). */
  closedByOtherSideCount = 0
): SetupStep[] {
  // Four honest states, never two
  // (D1): `null` = the host did not REPORT a count (not evidence of absence
  // -- distinct from a real zero); `0` = reported, and genuinely none yet;
  // `> 0` with NO witness = checkpointed LOCALLY, which is NOT registration;
  // witnessed = registered. Only the last may say "registered" -- the same
  // ONE fact `buildRegistrationCopy` and the Exchanges headline derive from.
  const registration = checkpointRegistration(card)
  const bound = ownerBound(owner)
  const askedPeerAt = typeof card?.asked_peer_at === 'string' ? card.asked_peer_at : null

  return [
    {
      key: 'checkpoints',
      title: 'Have a witness hold your checkpoints',
      done: registration.witnessedCheckpointCount > 0,
      status: registration.witnessedCheckpointCount > 0
        ? 'witnessed'
        : registration.checkpointedLocally
          ? CHECKPOINTED_NOT_REGISTERED_STATUS
          : registration.reported
            ? 'not set up'
            : 'status not reported',
      body: registration.witnessedCheckpointCount > 0
        ? null
        : registration.reported
          ? 'Right now your records are checkable only against themselves. A witness you don’t run holding a checkpoint is what makes a later rewrite detectable by someone else. It does not make your records true.'
          : 'This node did not report its checkpoint status. That is not the same as having none — the status was not reported, so nothing can be concluded either way.'
    },
    {
      key: 'identity',
      title: 'Bind an owner identity',
      done: bound,
      status: bound ? 'linked (self-asserted)' : OWNER_NOT_LINKED_PHRASE,
      // UX §4: "bound" alone overclaims -- the done state says what it is.
      body: bound
        ? OWNER_LINKED_SENTENCE
        : 'Binding your records to a key you hold (mesh-llm auth init) makes a later denial harder. It is only your own claim about who you are.'
    },
    {
      key: 'ask_peer',
      title: 'Get the other side’s record',
      // A half that arrived by push and closed through the gate corroborates
      // just as an asked-for half does -- the step is done either way.
      done: closedByOtherSideCount > 0 || askedPeerAt !== null,
      // Finding 7: the count is the "Confirmed by the other side" tile's to
      // say; the step says only that it's done.
      status:
        closedByOtherSideCount > 0 ? 'received' : askedPeerAt !== null ? `asked ${askedPeerAt}` : 'none received yet',
      body:
        closedByOtherSideCount > 0 || askedPeerAt !== null
          ? null
          : 'Their record usually arrives on its own when an exchange finishes; you can also ask them for it.'
    }
  ]
}

// ---------------------------------------------------------------------------
// Registration copy -- only renders once a checkpoint exists (v2 §4:
// "registered at N witnesses (M not operated by the producer)"). Derives
// from the SAME `checkpointRegistration` fact as rung 1 and the Exchanges
// headline: an unwitnessed checkpoint reads "checkpointed locally", never
// "Registered with 0 witnesses".
// ---------------------------------------------------------------------------

export type RegistrationCopy = {
  witnessSummary: string
  registeredNoLaterThan: string | null
}

/** `M not operated by this node` -- a witness counts toward M unless it
 *  explicitly says otherwise (`operated_by_producer: true`). Absent
 *  witness data errs toward the more independent-sounding claim being
 *  wrong, not toward silently inflating independence. */
function nonProducerWitnessCount(witnesses: readonly unknown[]): number {
  return witnesses.filter((witness) => {
    const record = witness as { operated_by_producer?: unknown } | null
    return !(record && typeof record === 'object' && record.operated_by_producer === true)
  }).length
}

/** The registration copy for a latest checkpoint no witness holds: why, by
 *  the same fact the hero line reads. */
function unwitnessedSummary(state: WitnessState): string {
  switch (state) {
    case 'earlier':
      return 'Latest checkpoint not held by a witness yet · an earlier one is'
    case 'pending':
      return 'Checkpointed locally · no witness holds it yet (witness: on)'
    case 'off':
      return 'Checkpointed locally · no witness (witness: off)'
    default:
      return 'Checkpointed locally · no witness'
  }
}

export function buildRegistrationCopy(card: JsonRecord | null | undefined): RegistrationCopy | null {
  const registration = checkpointRegistration(card)
  if (!registration.checkpointedLocally) return null

  const witnesses = checkedWitnesses(card)
  const nonProducerCount = nonProducerWitnessCount(witnesses)
  // Local time, never an ISO/UTC stamp on screen.
  const timestamp =
    typeof card?.registered_no_later_than === 'string' ? formatExchangeTimestamp(card.registered_no_later_than) : null

  // An unwitnessed checkpoint is a LOCAL fact -- "registered" (and
  // "registered no later than") would claim a witness holds it. Same ONE
  // fact rung 1 renders (D1).
  if (!registration.registered) {
    return {
      witnessSummary: unwitnessedSummary(registration.witnessState),
      registeredNoLaterThan: timestamp ? `checkpointed no later than ${timestamp}` : null
    }
  }

  return {
    witnessSummary: `Held by ${witnesses.length} witness${witnesses.length === 1 ? '' : 'es'}: ${witnesses
      .map((witness) => witnessName(witness.ts_url))
      .join(', ')} (${nonProducerCount} not operated by this node)`,
    registeredNoLaterThan: timestamp ? `witnessed no later than ${timestamp}` : null
  }
}

// ---------------------------------------------------------------------------
// Chain strip caption -- pulled out of `LedgerPage.tsx`'s ChainStrip so the
// three-state absence handling and the leaf pluralization are unit-testable.
// ---------------------------------------------------------------------------

/** How many RECORDS the latest checkpoint covers: the card's
 *  `covered_record_count`. Its `covered_leaf_count` also counts padding
 *  leaves (covered, but never records), so comparing that with a record
 *  count would read unsealed records as sealed. Null when the host reported
 *  no coverage. */
export function coveredRecordCount(card: JsonRecord | null | undefined): number | null {
  const covered = card?.covered_record_count
  return typeof covered === 'number' ? covered : null
}

export function chainStripCaption(
  sealedCount: number,
  checkpointCount: number | null,
  // The covered-LEAF count from the card (`covered_leaf_count`, inverted from
  // the checkpoint's MMR node count by the host) -- NOT `checkpointCount`, which
  // is the number of checkpoint LINES. The old caption rendered the line count
  // as a leaf count (a live "covered by checkpoint (1 leaves)" for a 1-line,
  // 8-leaf checkpoint). Null when the host did not report a covered-leaf count.
  coveredLeafCount: number | null = null
): string {
  if (checkpointCount !== null && checkpointCount > 0) {
    // A checkpoint exists. Report the covered-LEAF count when the host gave one
    // ("1 leaf" / "N leaves" -- never "1 leaves", D1
    // minor). If it did not, say so honestly rather than
    // reprint the checkpoint-line count as if it were leaves.
    if (coveredLeafCount !== null) {
      // UX §4: "All 8 records are sealed into a checkpoint." -- a tree word
      // ("leaves") never reaches the face.
      if (coveredLeafCount >= sealedCount) {
        return sealedCount === 1
          ? 'The 1 record is sealed into a checkpoint'
          : `All ${sealedCount} records are sealed into a checkpoint`
      }
      const since = sealedCount - coveredLeafCount
      return `${coveredLeafCount} of ${sealedCount} records are sealed into a checkpoint · ${since} since the last checkpoint ${since === 1 ? 'is' : 'are'} unshaded`
    }
    return 'records sealed into a checkpoint (count not reported) · records since the last checkpoint are unshaded'
  }
  const entries = `${sealedCount} entr${sealedCount === 1 ? 'y' : 'ies'}, all sealed`
  // Not reported by the host -- NEVER a false "none exists". A null count
  // (host did not compute/report a checkpoint_count) must not read as "no
  // checkpoint yet"; that conflation is the bug the three states guard
  // against -- Integrity is the highest-cost tab for a false absence.
  return checkpointCount === null
    ? `${entries} · checkpoint status not reported`
    : `${entries} · no checkpoint yet · no witness holds any of it`
}

// ---------------------------------------------------------------------------
// Integrity tile + chain-bar (i) hover copy -- what each tile counts and what
// evidence backs it, moved off the tile face and behind the glyph. Each names
// the evidence and refuses the score reading (three-sharer-answers framing).
// The chain-bar line is the checkpoint-coverage explanation, moved to the
// hover so the caption on the face can stay terse.
// ---------------------------------------------------------------------------

export const INTEGRITY_TILE_INFO = INTEGRITY_TILE_TOOLTIPS

export const CHAIN_BAR_INFO = CHAIN_STRIP_TOOLTIP

// ---------------------------------------------------------------------------
// Once-per-node facts -- retention, capture boundary + rule, identity.
// Never repeated per exchange row (rows link back instead --
// `security-checks-view.ts`'s `local_inclusion` NOT_PRESENT detail).
// ---------------------------------------------------------------------------

/** Retention has no per-node "declared vs. code vs. running image"
 *  comparison wired anywhere in this codebase yet (checked: the config
 *  schema carries a declared `logging.retention_ttl_secs` /
 *  `retention_max_rows`, reachable only through the Settings state hook;
 *  no compiled-default or measured-behaviour source exists to compare it
 *  against). Points at the real setting rather than fabricating a
 *  three-way comparison this build cannot back. */
export const RETENTION_FACT =
  'Retention: set by Settings → Logging (retention length and row cap). This node does not yet compare that declared setting against a running measurement.'

/** Grounded in the real capture point: the capsule plugin seals at the
 *  host's serve boundary (`security-checks-view.ts`'s per-record default
 *  says the same) -- stated once here instead of repeated on every row. */
export const CAPTURE_BOUNDARY_FACT =
  'Capture boundary: the plugin at this node’s serving boundary. Rule: whatever passes through it is what gets sealed; nothing upstream or downstream of it is captured.'

/** Look finding 2: which records the latest checkpoint covers. Chain leaves
 *  are the ledger's records in order, and the checkpoint covers the first
 *  `coveredLeafCount` of them -- the same figure the chain strip shades. An
 *  empty map when the host reported no covered count (unknown, never "none"). */
export function checkpointCoverageByRecord(
  ledgerOrderIds: readonly string[],
  coveredLeafCount: number | null
): Map<string, boolean> {
  const coverage = new Map<string, boolean>()
  if (coveredLeafCount === null) return coverage
  ledgerOrderIds.forEach((id, index) => coverage.set(id, index < coveredLeafCount))
  return coverage
}

/** Finding 3: the Sealed tile's sub-line, so Integrity's record count and
 *  Exchanges' exchange count reconcile on screen. */
/** `routingChoices`: records of your own choice to stop or resume routing to
 *  a peer (§7.5). `paymentRecords`: this node's sealed payment lifecycle
 *  records. Both are sealed like the rest, but not exchanges, so said apart,
 *  and only when there are any. */
export function sealedBreakdownText(
  own: number,
  receivedNotes: number,
  routingChoices = 0,
  paymentRecords = 0
): string {
  let text = `${own} yours · ${receivedNotes} received from the other side`
  if (routingChoices > 0) text += ` · ${routingChoices} ${routingChoices === 1 ? 'routing choice' : 'routing choices'}`
  if (paymentRecords > 0) text += ` · ${paymentRecords} payment records`
  return text
}

export function identityFact(owner: StatusOwner | null | undefined): string {
  if (ownerBound(owner)) {
    return 'Owner: linked (self-asserted) — not bound to a person.'
  }
  return 'Owner: not bound — not bound to a person.'
}

/** Continuity and registration are different facts, stated separately:
 *  continuity is a chain of checkpoints, each binding to the one before it,
 *  so it needs a PRIOR checkpoint; registration is what lets someone else
 *  detect a later rewrite. */
export const CONTINUITY_NOT_ESTABLISHED =
  'Continuity: not established. It needs a prior checkpoint for the next one to bind to.'

/** UX §4: continuity stated from the checkpoint count, separately from
 *  registration (step 1 already explains that, so it is not repeated). With
 *  no checkpoint -- or none reported -- continuity is not established. More
 *  than one checkpoint does not by itself say each binds to the one before:
 *  this node does not report that check, so the sentence doesn't claim it. */
export function continuityFact(checkpointCount: number | null): string {
  if (checkpointCount === null || checkpointCount <= 0) return CONTINUITY_NOT_ESTABLISHED
  if (checkpointCount === 1) {
    return 'Continuity: 1 checkpoint so far. The next one builds on it. A witness is what lets someone else check it too.'
  }
  return `Continuity: ${checkpointCount} checkpoints so far. A witness is what lets someone else check them too.`
}

// ---------------------------------------------------------------------------
// One row per witness (`card.witness_status`): what each holds, by name.
// ---------------------------------------------------------------------------

export type WitnessRow = {
  name: string
  url: string
  /** What this witness holds, in a few words. */
  status: string
  /** Why it does not count yet, or what it holds; null when nothing to add. */
  detail: string | null
  /** Where the key its receipts are checked under came from; null when it
   *  has none yet. */
  key: string | null
  tone: 'good' | 'pending' | 'bad' | 'muted'
}

/** The words for where a witness's key came from. */
export const WITNESS_KEY_CONFIGURED = 'key configured'
export const WITNESS_KEY_PINNED = 'key pinned on first contact, not configured'

export function witnessRows(card: JsonRecord | null | undefined): WitnessRow[] {
  const rows = Array.isArray(card?.witness_status) ? (card.witness_status as unknown[]) : []
  return rows
    .filter((row): row is JsonRecord => !!row && typeof row === 'object')
    .map((row) => {
      const url = typeof row.url === 'string' ? row.url : ''
      const name = typeof row.name === 'string' && row.name !== '' ? row.name : witnessName(url)
      const checked = typeof row.checked_count === 'number' ? row.checked_count : 0
      const held = typeof row.held_count === 'number' ? row.held_count : 0
      const problem = typeof row.problem === 'string' ? row.problem : null
      const latest = row.latest_checked as JsonRecord | null | undefined
      const when =
        latest && typeof latest.timestamp === 'string' ? formatExchangeTimestamp(latest.timestamp) : null
      const key =
        row.key_source === 'configured'
          ? WITNESS_KEY_CONFIGURED
          : row.key_source === 'pinned_on_first_contact'
            ? WITNESS_KEY_PINNED
            : null
      const checkedText = `${checked} checkpoint${checked === 1 ? '' : 's'} held, receipt${checked === 1 ? '' : 's'} checked`
      switch (row.state) {
        case 'latest':
          return { name, url, key, status: 'holds the latest checkpoint', detail: checkedText, tone: 'good' as const }
        case 'earlier':
          return {
            name,
            url,
            key,
            status: 'holds an earlier checkpoint',
            detail: [checkedText, when ? `latest held ${when}` : null, problem].filter(Boolean).join(' · '),
            tone: 'good' as const
          }
        case 'unchecked':
          return {
            name,
            url,
            key,
            status: `${held} receipt${held === 1 ? '' : 's'} not checked`,
            detail: problem,
            tone: 'bad' as const
          }
        case 'removed':
          return {
            name,
            url,
            key,
            status: 'no longer in your settings',
            detail: checked > 0 ? checkedText : null,
            tone: 'muted' as const
          }
        default:
          return { name, url, key, status: 'none held yet', detail: problem, tone: 'pending' as const }
      }
    })
}

/** Said under the witness list when the saved witnesses differ from the ones
 *  checkpoints go to now: the list is read when mesh-llm starts. */
export const WITNESS_RESTART_NOTE = 'Your saved witness list takes effect when mesh-llm restarts; until then checkpoints go to the witnesses above.'

export function witnessRestartNote(card: JsonRecord | null | undefined): string | null {
  return card?.witness_restart_needed === true ? WITNESS_RESTART_NOTE : null
}

/** What is wrong with the configured witness keys (a key not used means its
 *  witness's key would be pinned on first contact instead). */
export function witnessKeyProblems(card: JsonRecord | null | undefined): string[] {
  const problems = card?.witness_key_problems
  return Array.isArray(problems) ? problems.filter((p): p is string => typeof p === 'string') : []
}

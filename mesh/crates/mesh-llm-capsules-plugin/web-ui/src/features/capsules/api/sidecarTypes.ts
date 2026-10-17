// Loose types over the capsule-emit-mesh sidecar's `GET /accountability/
// pane-a|b|c` JSON (L1, `accountability_pane_
// routes.py`) -- byte-for-byte the same payloads `capsule_accountability_
// tab.build_tab_payload` / `peer_accountability_tab.build_peers_payload` /
// `capsule_exchange_tab.build_exchange_list_payload`/`build_exchange_view`
// already produce, so these mirror THOSE shapes, not a new one. Deliberately
// permissive (optional/unknown-tolerant), same discipline as
// `api/types.ts`'s `CapsuleRecord` -- an unrecognised or absent field
// degrades to `undefined`, never a parse failure.
import type { CapsuleRecord, JsonRecord } from '@/features/capsules/api/types'
import type { SplitRowJson } from '@/features/capsules/lib/split-stage'

export type PaneState = { state: string; text?: string | null; [key: string]: unknown }

// ---------------------------------------------------------------------------
// Pane A ("This node") -- capsule_accountability_tab.build_tab_payload
// ---------------------------------------------------------------------------

export type PaneARow = {
  capsule_id: string
  timestamp: string | null
  model_claimed: string | null
  hardware_claimed: string | null
  verify_ok: boolean | null
  rungs: Record<string, PaneState>
  record: CapsuleRecord
  /** `counterparty_half_citation` on a record this node sealed to note a
   *  record received from the other side; absent on its own exchanges. */
  kind?: string | null
}

export type PaneAJson = {
  operator: string | null
  witness_checkpoint_supplied: boolean
  rows: PaneARow[]
  card: JsonRecord | null
}

// ---------------------------------------------------------------------------
// Pane B ("Peers") -- peer_accountability_tab.build_peers_payload. Field
// names below are read verbatim from `peer_accountability_tab.py` (build_
// peer_row / the *_cell functions) on capsule-emit-mesh main -- do not
// invent columns that function doesn't emit. Loose/optional per this file's
// own discipline: an unrecognised field degrades to `undefined`, not a
// parse failure.
// ---------------------------------------------------------------------------

/** `node_cell`. */
export type PaneBNodeCell = PaneState & {
  peer_id?: string | null
  member_kind?: string | null
  exchange_count?: number
}

/** `rung_cell`. `rung`/`distinct_rungs` are the raw ladder values
 *  (`unilateral_fallback` | `acknowledged_receipt` | `full_bilateral`) --
 *  never render the word "rung" itself (ledger grep gate). */
export type PaneBRungCell = PaneState & {
  rung?: string
  distinct_rungs?: string[]
}

/** `role_and_count_cell`. Direction-of-exchange fact, never a trust signal. */
export type PaneBRoleCell = PaneState & {
  role?: 'both' | 'you_to_them' | 'them_to_you' | 'unknown'
  you_to_them_count?: number
  them_to_you_count?: number
  exchange_count?: number
}

/** `peer_history_cell` ("History (theirs)"). Honestly NOT_CHECKED by
 *  default (no peer-fetch carrier wired on most sidecars yet) --
 *  `history_summary` only appears when `peer_fetch_result.status ===
 *  'verified'`. `mine_for_reference` is THIS node's own chain, carried
 *  along for reference only -- never presented as though it were the
 *  peer's. */
export type PaneBHistoryCell = PaneState & {
  fetch_source?: string
  history_summary?: {
    verified_bundles?: number | string
    checkpoint_count?: number | string
    [key: string]: unknown
  }
  mine_for_reference?: {
    history?: PaneState & { history_depth?: number; checkpoint_count?: number; cadence?: Record<string, unknown> }
    continuity?: PaneState & { unforked?: boolean }
    witnessed?: PaneState & { witnesses?: string[] }
    [key: string]: unknown
  } | null
  /** `refusal_reason`, `segment` and
   *  `adjudications` below: no producer emits them yet (the Python sidecar's
   *  `peer_history_cell` and the native reader carry none); each degrades to
   *  omission. `refusal_reason` is the signed refusal's registry token when
   *  `state === 'refused'`. */
  refusal_reason?: string
  /** The peer's `chain_segment` answer (`capsule_emit.chain_segment.ChainSegment`),
   *  present only on a verified fetch. Coarsened: per-checkpoint leaf counts by
   *  kind; any `leaf_digests` a peer volunteers are never read by the UI. */
  segment?: { links: PaneBSegmentLink[] }
  /** The peer's own history-card `adjudications` block (provenance 2 of
   *  `ADJUDICATIONS-ON-HISTORY-CARD.md`): verdicts on twins they were party to,
   *  delivered to them, with their ack/rebuttal. Keyed by bare verdict kind: the
   *  owner a `contradicted:<owner>` verdict names is dropped, so a contradicted
   *  count can be against either side. `state` separates "never enriched" from
   *  "enriched, none". */
  adjudications?: {
    state: 'never_enriched' | 'enriched'
    delivered?: Partial<Record<AdjudicationVerdictKind, PaneBDeliveredVerdict>>
  }
}

export type AdjudicationVerdictKind = 'corroborated' | 'contradicted' | 'inconclusive'

/** One checkpoint of a peer's `chain_segment` (`CheckpointLink.to_dict`). */
export type PaneBSegmentLink = {
  checkpoint: { mmr_size: number; timestamp: string; witnesses?: unknown[] }
  leaf_counts: Record<string, number>
  leaf_digests?: string[]
}

/** `history_card` `adjudications.delivered[<verdict>]`. */
export type PaneBDeliveredVerdict = { delivered: number; acknowledged: number; disputed: number }

/** One `received_log.jsonl` line (`evidence_server.ReceivedLogEntry`).
 *  The log is node-wide and `requester_id` is the
 *  requester's self-declared id (`null` on a record push); the drill keeps only
 *  the `evidence-request` lines whose id matches this peer's row. */
export type AskedOfYouEntry = {
  ts: string
  path: string
  requester_id: string | null
  subject_kind: string | null
  status: 'answered' | 'refused' | 'received'
  reason: string | null
}

/** `served_cell` ("Served (theirs)"). Same peer-fetch-gap discipline as
 *  `history`. */
export type PaneBServedCell = PaneState & {
  source?: string
  served_summary?: {
    n_served?: number | string
    n_completed?: number | string
    n_failed?: number | string
    [key: string]: unknown
  }
  mine_for_reference?: unknown
}

/** `pair_cell` ("Pair (me<->them)") -- digest reconciliation, real. The
 *  only cell that can say "missing" (a lone half with nothing to reconcile
 *  against). */
export type PaneBPairCell = PaneState & {
  verified?: number
  failed?: number
  missing?: number
  details?: Array<{ exchange_id: string; state: string }>
}

/** `verdicts_cell`. `tally` is real only for adjudications THIS node
 *  itself sealed -- "held by others" half is a separate pending reason.
 *  Never render `tally` without also stating the denominator it came
 *  from (`exchange_count`). */
export type PaneBVerdictsCell = PaneState & {
  tally?: { corroborated: number; contradicted: number; inconclusive: number }
  adjudication_capsule_id?: string
  references_tally?: { corroborated: number; contradicted: number; inconclusive: number }
  references_asked?: number
  references_answered?: number
  /** Records the references hold of THIS peer refusing a verdict delivered
   *  to it (`verdicts_cell`'s `ack_refusals`) -- a count of records, not of
   *  references. Present with `references_asked`. */
  ack_refusals?: number
}

/** `asked_cell` -- evidence requests THIS node sent to this peer. Absent
 *  by default (no send-log carrier yet on most sidecars). */
export type PaneBAskedCell = PaneState & {
  count?: number
  send_log?: unknown[]
}

/** One correlated push-primary counterparty half
 *  supplied to the ONE gate, per this peer's asked half that a
 *  provenance-carrying foreign sibling reconciles with
 *  (`capsule_panes_native.rs::confirmed_siblings_for`). Carries both bodies
 *  plus the door's recorded `signature_ok`, the inputs
 *  `exchange-row-state.ts::deriveRightCellState` judges -- so Pane B's
 *  "confirmed by the other side" / MATCH derive from the SAME gate Pane C uses,
 *  never a second browser-peer-fetch predicate. Empty on a peer with no
 *  correlated pushed half; the gate reads that as "not confirmed", honest. */
export type PaneBConfirmedSibling = {
  /** Our asked half, with its body (`mine_pair_cell`). */
  mine?: PaneCRow['mine']
  theirs: PaneCRow['theirs']
  digest_match?: PaneCRow['digest_match']
}

/** D3 -- ONE peer's alias evidence,
 *  joined on the signing key (`capsule_panes_native.rs::PeerIdentity`): the
 *  pushed body's own `key_id` (door-verified against the announced peer key),
 *  the door's `received_from` endpoint id, and a mesh node id only when a
 *  record actually names one. Each field is evidence-backed or null -- the UI
 *  renders these as aliases on one row, never as extra peers. Absent on an
 *  older payload; degrades to no alias line, never a fabricated identity. */
export type PaneBPeerIdentity = {
  signing_key_id?: string | null
  endpoint_id?: string | null
  node_id?: string | null
  /** Where `node_id` came from: this node's own records, or the peer's own
   *  record naming itself. Absent on an older host. */
  node_id_source?: 'your_records' | 'their_record' | null
}

export type PaneBRow = {
  peer_id: string | null
  /** See `PaneBPeerIdentity`. */
  identity?: PaneBPeerIdentity | null
  node: PaneBNodeCell
  rung: PaneBRungCell
  role: PaneBRoleCell
  history: PaneBHistoryCell
  served: PaneBServedCell
  pair: PaneBPairCell
  /** See `PaneBConfirmedSibling`. Absent on an older sidecar that predates the
   *  push-primary path -- `confirmedByOtherSide` reads that as no correlated
   *  half, never a fabricated confirmation. */
  confirmed_siblings?: PaneBConfirmedSibling[]
  verdicts: PaneBVerdictsCell
  /** Referee-signed verdicts about this peer, from this node's own chain
   *  (`verdict_counts.rs`). Absent from a plugin that predates it. */
  referee_verdicts?: RefereeVerdictCounts
  asked: PaneBAskedCell
  /** Your node's `received_log.jsonl` lines,
   *  as the producer carries them to this row. No producer emits this yet; an
   *  absent field reads as "not shown in this view", never as zero requests. */
  /** `requester_id_source` is always `self_declared`: each entry's requester
   *  id is what the request said about itself. */
  /** Exchanges with this peer the door refused because their signed claims
   *  contradict our record (another server named, other weights). Absent
   *  when none. They count as disagreements. */
  claims_refused?: number
  asked_of_you?: { entries: AskedOfYouEntry[]; requester_id_source?: 'self_declared' }
  exchange_count: number
  first_seen: string | null
  last_seen: string | null
  expand?: {
    pair_ledger?: Array<{ exchange_id: string; state: string }>
    their_card?: PaneState
  }
  /** Present only when this node holds settlement records at all. */
  settlement?: PeerSettlementCounts
  [key: string]: unknown
}

/** One bucket of `referee_verdicts`: how many, and which verdict records. */
export type RefereeVerdictBucket = { count: number; verdict_capsule_ids: string[] }

export type RefereeVerdictBucketKey = 'corroborated' | 'contradicted' | 'inconclusive' | 'not_comparable'

/** Only verdicts whose referee signature this node's door verified; all four
 *  buckets present, zero included. */
export type RefereeVerdictCounts = Record<RefereeVerdictBucketKey, RefereeVerdictBucket>

export type PaneBJson = {
  peer_count: number
  rows: PaneBRow[]
  /** See `PaymentsPresence`. */
  payments?: PaymentsPresence
  [key: string]: unknown
}

// ---------------------------------------------------------------------------
// Settlement (`capsule_panes_settlement.rs`): this node's own sealed records
// of the payer-side payment lifecycle, joined to exchanges by `exchange_id`.
// ---------------------------------------------------------------------------

/** Whether this node has a payments provider, as the host answers it. `off`
 *  is the free-only configuration; `unknown` means the lookup failed. */
export type PaymentsPresence = 'on' | 'off' | 'unknown'

/** Who asserted a recorded value: this node, the provider (as relayed to this
 *  node), or this node's wallet. */
export type SettlementSource = 'payer_asserted' | 'provider_asserted' | 'wallet_reported'

export type SettlementPhase =
  | 'terms_accepted'
  | 'input_invoice_issued'
  | 'output_invoice_issued'
  | 'input_settlement_observed'
  | 'output_settlement_observed'
  | 'final_accounted'
  | 'delivered'

/** One sealed observation, values as recorded (amounts are never summed). */
export type SettlementEntry = {
  capsule_id: string | null
  timestamp: string | null
  phase: SettlementPhase | string
  source: SettlementSource | string | null
  segment: number | null
  payment_hash: string | null
  amount_msat: number | null
  /** On a settlement, what the receiving wallet credited, when the host
   *  passed it (provider side). */
  credited_msat?: number
  /** On a settlement, the wallet's fee, when the host passed it (payer: paid
   *  on top; provider: deducted). */
  fee_msat?: number
  /** On a provider's `delivered` step, the delivered-token watermark. */
  tokens?: number
}

/** The payer's book for one exchange: every invoice settled by this node's
 *  wallet (`settled`), an invoice with no settlement seen, terms with no
 *  invoice, or a settlement that names no invoice of this exchange. */
export type PayerBookState =
  | 'settled'
  | 'no_settlement_seen'
  | 'outcome_not_reported'
  | 'terms_only'
  | 'unmatched_settlement'

/** A book of one exchange's payment records: the payer's (`settlement`) or,
 *  on an exchange this node served for pay, its own as the provider
 *  (`provider_settlement`). */
export type PayerBook = {
  observed_by: 'payer' | 'provider'
  /** `this_node` on a payer book; `requester` on a provider book (the row's
   *  counterparty). */
  who_paid?: 'this_node' | 'requester' | string
  /** The payer's own recorded total for the exchange, wallet amounts plus
   *  fees, copied from its `final_accounted` record; `null` when there is
   *  none (and always on a provider book, or a row of several exchanges). */
  final_accounted_msat?: number | null
  /** The provider's delivered-token watermark, on a provider book. */
  delivered_tokens?: number | null
  state: PayerBookState | string
  terms_digests: string[]
  entries: SettlementEntry[]
  /** True when a settlement carried no payment hash and could only be matched
   *  to its segment's invoice. */
  matched_by_segment_only?: boolean
  /** On a payer book: `not_available`, the provider's book is on its node. */
  provider_book?: string
  /** The exchange ids whose books this summary covers; the row's state is the
   *  worst of them. */
  exchange_ids?: string[]
}

/** Per-peer counts, from this node's own records only. The provider's side
 *  is not seen here at all (`provider_book: "not_available"`), so no
 *  provider-side count is sent. */
export type PeerSettlementCounts = {
  /** Exchanges with at least one invoice recorded. */
  paid_exchanges: number
  /** Exchanges whose terms were accepted with no invoice: not paid. */
  terms_only?: number
  settled_payer_observed: number
  no_settlement_seen: number
  /** Exchanges invoiced with no settlement and no final amount: the host
   *  reported no outcome. */
  outcome_not_reported?: number
  provider_book: string
}

// ---------------------------------------------------------------------------
// Pane C ("This exchange") -- capsule_exchange_tab.build_exchange_list_
// payload / build_exchange_view. The nine-property assurance map
// (`properties`) is the ONLY pane surface that carries the five-state chip
// strip today -- Pane A/B's `rungs`/cell states above are still the older
// per-rung ladder vocabulary (tone-compatible, not chip-shaped).
// ---------------------------------------------------------------------------

export type AssuranceProperties = Record<string, PaneState>

/** The four evidence-request outcomes named in the two-sided-ledger design
 *  note (v3 §2's "evidence" column) -- see `exchange-row-state.ts`. Optional
 *  because the carrier that would populate it is still unwired end-to-end
 *  (`evidence_responder.py`: "not yet reachable over the wire") -- an
 *  absent value degrades to `not_asked`, never a guess at one of the other
 *  three. */
/** `claims_refused`: this node's door refused the provider's pushed half for
 *  its signed claims (another server named, other model weights). */
export type EvidenceRequestOutcome = 'signed_refusal' | 'recorded_absence' | 'unanswered' | 'not_asked' | 'claims_refused'

/** item 3 — the mechanical facts of an ambient
 *  twin comparison, when this row is one half of one. Deliberately carries
 *  NO computed verdict field (item 4: observe-only, no verdict published) --
 *  only what the two dispatches themselves recorded. Every field
 *  optional/forward-looking: no sidecar emits this yet (the host-side
 *  dispatch that would populate it is an unwired seam, see
 *  `runtime::twin_sample` on the Rust side), so it degrades to omission,
 *  never a fabricated comparison. */
export type TwinComparison = {
  temperature?: number | null
  seed?: number | null
  model_identity_hash?: string | null
  settings_label?: string | null
  response_digest?: string | null
}

/** The TDX quote check on one record (`producer::tee_attest::page_fact`): run
 *  on the node, never in the browser. `check` names the check that failed. */
export type TeeFact =
  | { state: 'absent'; context?: string | null }
  | { state: 'bound' }
  | {
      state: 'failed'
      check:
        | 'shape'
        | 'digest'
        | 'parse'
        | 'report_data'
        | 'producer_key'
        | 'quote_signature'
        | 'qe_report_signature'
        | 'attestation_key_binding'
        | 'pck_chain'
    }

export type PaneCRow = {
  exchange_key: string
  role_tag: string
  /** A split request's records (docs/DESIGN-split-stage-records.md §7): the
   *  coordinator's main record and the stage records it carried, or -- on a
   *  stage node's own row -- that stage's block. Absent on every other row.
   *  Not served yet; see DATA-ROUTES.md. */
  split?: SplitRowJson
  /** D4(a) -- the peer this row's own
   *  evidence names, using the SAME row key Pane B's peer rows use
   *  (`capsule_panes_native.rs::PeerAttribution`): the pushed sibling's
   *  identity, or -- on a requester-side row -- the server this node's own
   *  record routed to (`served_by_node_id`). Naming whom we asked is not a
   *  claim to hold their half; the right cell still derives independently.
   *  `null`/absent when no record names a peer -- never invented. */
  counterparty?: string | null
  header_state: string
  properties: AssuranceProperties | null
  has_issue: boolean
  mine: {
    state: string
    capsule_id: string | null
    role?: string
    text?: string
    /** What the counterparty streamed back to you -- held only on the
     *  requester's side (v3 §3: "both
     *  halves of the conversation are on the left, because both passed
     *  through you"). Optional/forward-looking: no sidecar emits it yet,
     *  degrades to omitting the second half rather than inventing one. */
    reply_text?: string
    reply_capsule_id?: string | null
    /** L-D: the requester's own copy of populated content, deleted locally.
     *  Only meaningful when `role_tag === 'ASKED'` -- a SERVED row's `mine`
     *  side was never populated to begin with (L-F). */
    deleted?: boolean
    deleted_date?: string | null
    /** OUR half's own body -- sent only on a correlated pushed pair
     *  (`capsule_panes_native.rs::mine_pair_cell`), so the gate compares the
     *  counterparty half against our real digests + `served_by_node_id`. */
    record?: Record<string, unknown>
    /** The TDX quote in our own record, as the node checked it. */
    tee?: TeeFact
  }
  theirs: {
    state: string
    capsule_id: string | null
    role?: string
    text?: string
    evidence_outcome?: EvidenceRequestOutcome
    evidence_outcome_date?: string | null
    /** Why the door refused, with `claims_refused`: `served_by_mismatch` or
     *  `model_mismatch`. */
    evidence_outcome_reason?: string | null
    /** Piece 3 -- the mesh peer id to fetch FROM
     *  (`capsule_panes_native.rs::theirs_cell`'s `served_by_node_id`),
     *  present only alongside `theirs.state === 'NOT_CHECKED'` and a real
     *  `capsule_id`. Never guessed: `null`/absent means this row carries no
     *  fetchable join key, same "never fabricate" discipline as every other
     *  optional field in this file. */
    peer_id?: string | null
    /** The record-push door's provenance triple
     *  for a locally-held, identity-verified counterparty half
     *  (`capsule_panes_native.rs::theirs_sibling_cell`, read off OUR citing
     *  record's `received_half`). `signature_ok` is the door's recorded
     *  verdict of the peer-key signature check -- one of the ONE gate's four
     *  CLOSED conditions, never sufficient alone. Absent on every row with no
     *  citing record; a self-sealed / provenance-less sibling never carries
     *  it, so it never closes. */
    signature_ok?: boolean
    /** The TDX quote in their held record, as this node checked it. */
    tee?: TeeFact
    received_from?: string
    via?: string
    received_at?: string
    /** The held foreign body itself -- the provider-signed bytes the door
     *  verified (`capsule_panes_native.rs::theirs_sibling_cell`). Present only
     *  on a correlated pushed half; the gate recomputes its `capsule_id` and
     *  compares its digests + provider against OUR half. */
    record?: Record<string, unknown>
    /** Whether `record` recomputes to `capsule_id`, computed IN THIS BROWSER
     *  by `attachPushedHalfRecompute` before the pane query resolves (never
     *  sent by the host). `null` when the recompute could not run; absent
     *  until it has run. */
    id_match?: boolean | null
    /** Where their record sits in their own log, from our record citing
     *  their inclusion proof and checkpoint: `checkpoint_leaves` is the leaves
     *  that checkpoint covers, their padding included, so never a record
     *  count. Null/absent until that arrives. */
    in_their_log?: { leaf_index: number; checkpoint_leaves: number } | null
  }
  unilateral: boolean
  /** The STRUCTURAL digest reconciliation of a
   *  correlated pair (`capsule_panes_native.rs::digest_match_state`) -- both
   *  halves' `effect.request_digest`/`effect.response_digest` compared
   *  field-by-field. Display-only: the ONE gate (`exchange-row-state.ts`)
   *  compares the two bodies itself and does not read this. Absent on a
   *  unilateral row (nothing to reconcile). */
  digest_match?: { state: 'verified' | 'failed' | 'absent' | 'present-unverified' }
  timestamp: string | null
  /** The conversation this exchange belongs to, when this node was the
   *  requester (v3 §2 L-O: a served row structurally has none -- this node
   *  was never party to the requester's session). Optional/forward-looking:
   *  no sidecar emits it yet, so it degrades to `null` (no rail), never an
   *  invented grouping. */
  session_id?: string | null
  /** item 2 — the id shared by BOTH halves of an
   *  ambient twin comparison, minted host-side (Rust) per the ticket's
   *  Rust-native ruling. `null`/absent on every row that wasn't ambiently
   *  twinned (the overwhelming majority) -- optional/forward-looking, same
   *  discipline as `session_id`: no sidecar emits it yet, degrades to no
   *  bracket rather than a half-bracket. */
  twin_bracket_id?: string | null
  /** The comparison facts for this row's half of the bracket -- see
   *  `TwinComparison`. Present only alongside a real `twin_bracket_id`. */
  twin_comparison?: TwinComparison | null
  /** Pane C's twin pairing (the plugin pane builder's `attach_twins`): which
   *  row is the other half, and whether the two providers gave the same
   *  answer (their sealed answer-text digests; `null` while either is
   *  missing). `verdict` / `verdict_capsule_id` appear only once a REFEREE
   *  has signed a verdict for this pair. */
  twin?: TwinRowFacts | null
  /** This node's record of a call it answered AS A REFEREE (its client nonce
   *  carries the referee prefix). */
  referee_call?: boolean
  /** On a JUDGED node: a referee's verdict delivered to this node that cites
   *  this row's own half. Absent when there is none. */
  adjudication?: DeliveredAdjudication
  /** On the REFEREE node: the verdict it issued, on the row of the call it
   *  answered as referee. Absent when there is none. */
  adjudication_issued?: IssuedAdjudication
  /** The payer-book summary of the settlement records this exchange's ids
   *  join. `null`/absent means no payment lifecycle was recorded for it --
   *  free, payments off, or failed before authorization -- never unpaid. */
  settlement?: PayerBook | null
  /** On an exchange this node served for pay: its own book as the provider,
   *  joined by the exchange id the host named its lifecycle events with. */
  provider_settlement?: PayerBook | null
}

export type TwinRowFacts = {
  bracket_id: string
  same_answer: boolean | null
  other_row: string | null
  verdict?: string | null
  verdict_capsule_id?: string | null
  referee_node_id?: string | null
  /** The pair's outcome as this node recorded it (`referee::request`):
   *  adjudicated, or not adjudicated with the reason. `null` until the pair
   *  is complete here. */
  referee_row?: import('@/features/capsules/lib/referee-row').RefereeRow | null
}

/** `"corroborated"` or `"contradicted:<node_id>"`, as the referee sealed it. */
export type VerdictWire = string

export type DeliveredAdjudication = {
  verdict: VerdictWire
  verdict_capsule_id: string
  referee_node_id: string
  received_at: string
  /** The verdict names this node as the one contradicted. */
  about_this_node: boolean
}

export type IssuedAdjudication = {
  verdict: VerdictWire
  verdict_capsule_id: string
  bracket_id?: string
  halves: string[]
}

/** `http/ledger/verdict?capsule_id=…`: one signed verdict record, with the
 *  plugin's own check of its signature. */
export type VerdictRecordJson = {
  capsule: Record<string, unknown>
  signed_by_key_id: string | null
  /** An earlier run of this node recorded this verdict. This node has no
   *  referee now and doesn't check the signer against the referee's
   *  announced key, so a legacy verdict is never `verify_ok`. */
  legacy?: boolean
  /** The referee's signature verifies AND this node's chain records the
   *  verdict. */
  verify_ok: boolean
  /** How this node's chain holds it: a verdict it issued, one delivered to
   *  it, or neither. */
  recorded_as?: 'issued' | 'received' | null
  referee_node_id?: string | null
  /** Why the signature didn't verify, when it didn't. */
  signature_error?: string | null
}

export type PaneCListJson = {
  row_count: number
  default_sort: string
  filters: string[]
  rows: PaneCRow[]
  next_after_seq: number | null
  archived_segments: unknown[]
  /** The independent check's state for this list: whether it is on, and
   *  whether any client marked a pair (no twin bracket on any row: "Not
   *  adjudicated: no client marked these as a pair"). */
  referee?: {
    adjudicate_differing_twins: boolean
    twins_marked: boolean
    unmarked_reason: string | null
  }
  /** See `PaymentsPresence`. */
  payments?: PaymentsPresence
  /** Exchange ids with settlement records that no row carries. */
  settlement_unjoined?: string[]
  /** Settlement records with no exchange id at all. */
  settlement_missing_exchange_id?: number
  /** Settlement records this node sealed as the provider of a paid exchange. */
  settlement_provider_records?: number
  /** Provider exchange ids no row carries. */
  settlement_provider_unjoined?: string[]
}

export type PaneCDrilldownJson =
  | { exchange_key: string; found: false }
  | { exchange_key: string; found: true; view: JsonRecord & { properties?: AssuranceProperties; capsule_id?: string } }

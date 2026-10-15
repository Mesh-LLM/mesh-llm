import { describe, expect, it } from 'vitest'
import {
  deriveRightCellState,
  isAlarmState,
  isAskAction,
  ledgerStateFilterValue,
  rightCellAction,
  rightCellDetail,
  rightCellStatusLabel,
  rightCellText,
  type RightCellState,
  type RightCellStateKind
} from '@/features/capsules/lib/exchange-row-state'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import type { PeerRecomputeState } from '@/features/capsules/lib/recompute-identity'
import { fixtureHalfBody, fixtureMineCell, fixtureTheirsCell } from '@/features/capsules/lib/pushed-half-fixtures'
import { STANDING_WORD } from '@/features/capsules/lib/banned-terms'

function paneCRow(overrides: Partial<PaneCRow> = {}): PaneCRow {
  return {
    exchange_key: 'exch-1',
    role_tag: 'ASKED',
    header_state: 'ok',
    properties: null,
    has_issue: false,
    mine: { state: 'present', capsule_id: 'mine-1' },
    theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' },
    unilateral: false,
    timestamp: '2026-09-08T00:00:00Z',
    ...overrides
  }
}

/** Our own record, carrying the two §6.2/L-G digests CLOSED requires a
 *  peer's fetched record to cite. */
function localRecordWithDigests(): CapsuleRecord {
  return fixtureHalfBody({ capsuleId: 'mine-1' }) as CapsuleRecord
}

/** A peer record that actually cites both of `localRecordWithDigests()`'s
 *  digests -- the ONLY peerRecord shape `digestsCiteOurHalf` reads as
 *  CLOSED-eligible. */
function citingPeerRecord(): Record<string, unknown> {
  return fixtureHalfBody({ capsuleId: 'a'.repeat(64) })
}

function fetched(overrides: Partial<PeerRecomputeState> = {}): PeerRecomputeState {
  return {
    status: 'found',
    idMatch: true,
    signatureOk: true,
    peerRecord: { capsule_id: 'a'.repeat(64) },
    fetch: () => {},
    ...overrides
  }
}

const ALL_KINDS: RightCellStateKind[] = [
  'closed',
  'contradicted',
  'open_refused',
  'open_absent',
  'open_asked',
  'open_not_held',
  'open_not_given',
  'open_not_asked'
]

describe('deriveRightCellState — finding 1 (2026-09-23 assessment): CLOSED requires a held artifact', () => {
  it('MUTANT: a peer-asserted id alone (theirs.state !== absent, no fetch) is never CLOSED', () => {
    const row = paneCRow({ theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' } })
    expect(deriveRightCellState(row).kind).not.toBe('closed')
    expect(deriveRightCellState(row).kind).toBe('open_not_held')
  })

  it('a real fetch that matches, is signed, AND cites our half by digest -> CLOSED', () => {
    const row = paneCRow()
    const state = deriveRightCellState(
      row,
      fetched({ idMatch: true, signatureOk: true, peerRecord: citingPeerRecord() }),
      localRecordWithDigests()
    )
    expect(state.kind).toBe('closed')
  })

  it('a real fetch that does NOT match -> CONTRADICTED, never CLOSED', () => {
    const row = paneCRow()
    expect(deriveRightCellState(row, fetched({ idMatch: false })).kind).toBe('contradicted')
  })

  it('BOUNCE REPRO (finding 1): idMatch true but signatureOk false is NOT CLOSED -- an id match proves nothing without a verified signature over the fetched bytes', () => {
    const row = paneCRow()
    const state = deriveRightCellState(
      row,
      fetched({ idMatch: true, signatureOk: false, peerRecord: citingPeerRecord() }),
      localRecordWithDigests()
    )
    expect(state.kind).not.toBe('closed')
    expect(state.kind).toBe('open_not_held')
  })

  it('idMatch true, signature verified, but the peer record does not cite our request/response digests (§6.2/L-G) -- NOT CLOSED', () => {
    const row = paneCRow()
    const state = deriveRightCellState(
      row,
      fetched({ idMatch: true, signatureOk: true, peerRecord: { capsule_id: 'a'.repeat(64) } }),
      localRecordWithDigests()
    )
    expect(state.kind).not.toBe('closed')
    expect(state.kind).toBe('open_not_held')
  })

  it('idMatch true, signature verified, digests cite, but with no localRecord passed at all -- NOT CLOSED (never a fabricated match against nothing)', () => {
    const row = paneCRow()
    const state = deriveRightCellState(
      row,
      fetched({ idMatch: true, signatureOk: true, peerRecord: citingPeerRecord() })
    )
    expect(state.kind).not.toBe('closed')
    expect(state.kind).toBe('open_not_held')
  })

  it('a fetch still in flight is not evidence of anything -> stays open_not_held', () => {
    const row = paneCRow()
    const state = deriveRightCellState(row, fetched({ status: 'fetching', idMatch: null, peerRecord: null }))
    expect(state.kind).toBe('open_not_held')
  })

  it('a non-digest self-minted peer marker (capsule-chatcmpl-…) is not fetchable -> open_not_given, never pending fetch', () => {
    const row = paneCRow({
      theirs: { state: 'NOT_CHECKED', capsule_id: 'capsule-chatcmpl-1790354430635', peer_id: 'peer-1' }
    })
    expect(deriveRightCellState(row).kind).toBe('open_not_given')
  })

  it('a digest-shaped peer id with no confirmed fetch -> open_not_held (fetchable)', () => {
    const row = paneCRow({
      theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' }
    })
    expect(deriveRightCellState(row).kind).toBe('open_not_held')
  })

  it('a fetch that came back not_found is not evidence of a match -> stays open_not_held, never CLOSED', () => {
    const row = paneCRow()
    const state = deriveRightCellState(row, fetched({ status: 'not_found', idMatch: null, peerRecord: null }))
    expect(state.kind).toBe('open_not_held')
  })

  it('a fetch whose id-recompute could not run (idMatch null) is inconclusive, never CLOSED', () => {
    const row = paneCRow()
    const state = deriveRightCellState(row, fetched({ status: 'found', idMatch: null, peerRecord: { x: 1 } }))
    expect(state.kind).toBe('open_not_held')
  })

  it('theirs.state === absent is open_not_asked regardless of any fetch state (no join key exists to fetch from)', () => {
    const row = paneCRow({ theirs: { state: 'absent', capsule_id: null } })
    expect(deriveRightCellState(row, fetched()).kind).toBe('open_not_asked')
  })
})

describe('deriveRightCellState — the ONE gate: pushed and fetched halves take the same predicate', () => {
  // CLOSED = provider-signed bytes this node holds (pushed or fetched), checked
  // locally: signature verified against the peer's announced key, the body
  // recomputes to its capsule_id, both digests equal ours, and it came from the
  // provider (provisional check (i): both halves name the same server). The
  // real in-browser recompute of `id_match` is exercised in
  // `pushed-half-recompute.test.ts`; here it is supplied as the pane query
  // delivers it.
  function pushedRow(theirs: PaneCRow['theirs'], overrides: Partial<PaneCRow> = {}): PaneCRow {
    return paneCRow({
      unilateral: false,
      mine: fixtureMineCell(),
      theirs,
      digest_match: { state: 'verified' },
      ...overrides
    })
  }

  it('a pushed half that agrees closes on the LIST path (no recompute, no local record passed)', () => {
    expect(deriveRightCellState(pushedRow(fixtureTheirsCell('agrees'))).kind).toBe('closed')
  })

  it('MUTANT no provenance -> OPEN: a held body with no door verdict from a citing record never closes', () => {
    const { signature_ok: _dropped, ...noVerdict } = fixtureTheirsCell('agrees')
    const kind = deriveRightCellState(pushedRow(noVerdict)).kind
    expect(kind).not.toBe('closed')
    expect(kind).not.toBe('contradicted')
  })

  it('MUTANT self-sealed sibling -> OPEN: the pane files our own second half under mine, so theirs is absent', () => {
    const row = pushedRow({ state: 'absent', capsule_id: null }, { unilateral: true })
    expect(deriveRightCellState(row).kind).toBe('open_not_asked')
  })

  it('MUTANT provenance ok, digests differ -> CONTRADICTED', () => {
    expect(deriveRightCellState(pushedRow(fixtureTheirsCell('disagrees'))).kind).toBe('contradicted')
  })

  it('reads the bodies, not the structural summary: a stale "verified" digest_match over disagreeing bodies still CONTRADICTS', () => {
    const row = pushedRow(fixtureTheirsCell('disagrees'), { digest_match: { state: 'verified' } })
    expect(deriveRightCellState(row).kind).toBe('contradicted')
  })

  it('before the capsule_id recompute has run (id_match absent) a pushed half is never CLOSED', () => {
    const { id_match: _dropped, ...unrecomputed } = fixtureTheirsCell('agrees')
    expect(deriveRightCellState(pushedRow(unrecomputed)).kind).toBe('open_not_held')
  })

  it('a pushed body that does not recompute to its capsule_id -> CONTRADICTED', () => {
    expect(deriveRightCellState(pushedRow({ ...fixtureTheirsCell('agrees'), id_match: false })).kind).toBe(
      'contradicted'
    )
  })

  it('a door verdict other than true -> not CLOSED', () => {
    expect(deriveRightCellState(pushedRow({ ...fixtureTheirsCell('agrees'), signature_ok: false })).kind).toBe(
      'open_not_held'
    )
  })

  it('a pushed half from a node that did not serve us never CONTRADICTS, whatever its bytes', () => {
    const theirs = fixtureTheirsCell('disagrees')
    const record = fixtureHalfBody({ capsuleId: theirs.capsule_id ?? undefined, servedBy: 'd'.repeat(64) })
    expect(deriveRightCellState(pushedRow({ ...theirs, record })).kind).toBe('open_not_held')
    expect(deriveRightCellState(pushedRow({ ...theirs, record, id_match: false })).kind).toBe('open_not_held')
  })

  it('a FETCHED half keeps its id-recompute contradiction (it came from the peer we asked)', () => {
    const peerRecord = fixtureHalfBody({ capsuleId: 'a'.repeat(64), servedBy: 'd'.repeat(64) })
    const state = deriveRightCellState(pushedRow(fixtureTheirsCell('agrees')), fetched({ idMatch: false, peerRecord }))
    expect(state.kind).toBe('contradicted')
  })

  describe('attack D: a swapped model never reads CLOSED', () => {
    const asked = 'a'.repeat(64)
    const swapped = 'b'.repeat(64)
    function withWeights(
      body: Record<string, unknown>,
      modelId: string,
      attested?: string,
      served?: string
    ): Record<string, unknown> {
      const attestation = body.model_attestation as Record<string, Record<string, unknown>>
      const compute = attestation.compute_attestation
      const poc = compute['x-mesh-poc-v1'] as Record<string, Record<string, unknown>>
      return {
        ...body,
        model_attestation: {
          ...attestation,
          model_id: modelId,
          compute_attestation: {
            ...compute,
            ...(attested ? { weights_digest: { digest: attested } } : {}),
            'x-mesh-poc-v1': {
              ...poc,
              serving_provenance: {
                ...poc.serving_provenance,
                ...(served ? { model: { weights_digest: served } } : {})
              }
            }
          }
        }
      }
    }
    const kindFor = (ourModelId: string, theirModelId: string, attested: string, served: string) => {
      const theirs = fixtureTheirsCell('agrees')
      const record = withWeights(theirs.record as Record<string, unknown>, theirModelId, attested, served)
      const ours = withWeights(fixtureHalfBody({ capsuleId: 'mine-1' }), ourModelId) as CapsuleRecord
      return deriveRightCellState(pushedRow({ ...theirs, record }), undefined, ours).kind
    }
    const askedId = `local-gguf/sha256-${asked}`
    const swappedId = `local-gguf/sha256-${swapped}`

    it('control: every weights claim names the asked weights -> CLOSED', () => {
      expect(kindFor(askedId, askedId, asked, asked)).toBe('closed')
    })
    it('case 1: only the serving provenance weights swapped -> CONTRADICTED', () => {
      expect(kindFor(askedId, askedId, asked, swapped)).toBe('contradicted')
    })
    it('case 2: model_id and both weights fields swapped together -> CONTRADICTED', () => {
      expect(kindFor(askedId, swappedId, swapped, swapped)).toBe('contradicted')
    })
    it('alias: our model_id is a name with no weights -> CLOSED', () => {
      expect(kindFor('qwen', askedId, asked, asked)).toBe('closed')
    })
  })

  it('attack B: the door refused our provider’s half on its claims -> CONTRADICTED, dated', () => {
    const row = paneCRow({
      theirs: {
        state: 'absent',
        capsule_id: null,
        evidence_outcome: 'claims_refused',
        evidence_outcome_date: '2026-09-28T08:00:00Z',
        evidence_outcome_reason: 'model_mismatch'
      }
    })
    expect(deriveRightCellState(row)).toEqual({ kind: 'contradicted', date: '2026-09-28T08:00:00Z' })
  })

  it('PROVISIONAL provider check (i): a pushed body naming a different server -> not CLOSED', () => {
    const theirs = fixtureTheirsCell('agrees')
    const record = fixtureHalfBody({ capsuleId: theirs.capsule_id ?? undefined, servedBy: 'd'.repeat(64) })
    expect(deriveRightCellState(pushedRow({ ...theirs, record })).kind).toBe('open_not_held')
  })

  it('the fetch path takes the SAME predicate: agreeing, signed, same provider -> CLOSED', () => {
    expect(
      deriveRightCellState(paneCRow(), fetched({ peerRecord: citingPeerRecord() }), localRecordWithDigests()).kind
    ).toBe('closed')
  })

  it('the fetch path takes the SAME predicate: digests differ -> CONTRADICTED', () => {
    const peerRecord = fixtureHalfBody({ capsuleId: 'a'.repeat(64), responseDigest: 'e'.repeat(64) })
    expect(deriveRightCellState(paneCRow(), fetched({ peerRecord }), localRecordWithDigests()).kind).toBe(
      'contradicted'
    )
  })

  it('a PUSHED half from a node that did not serve us never contradicts: junk digests or a bad id stay OPEN', () => {
    const theirs = fixtureTheirsCell('disagrees')
    const record = fixtureHalfBody({ capsuleId: theirs.capsule_id ?? undefined, responseDigest: 'e'.repeat(64), servedBy: 'd'.repeat(64) })
    expect(deriveRightCellState(pushedRow({ ...theirs, record })).kind).toBe('open_not_held')
    expect(deriveRightCellState(pushedRow({ ...theirs, record, id_match: false })).kind).toBe('open_not_held')
  })

  it('the fetch path takes the SAME predicate: a different provider -> not CLOSED', () => {
    const peerRecord = fixtureHalfBody({ capsuleId: 'a'.repeat(64), servedBy: 'd'.repeat(64) })
    expect(deriveRightCellState(paneCRow(), fetched({ peerRecord }), localRecordWithDigests()).kind).toBe(
      'open_not_held'
    )
  })

  it('a live fetch this browser ran wins over the pushed evidence on the same row', () => {
    const row = pushedRow(fixtureTheirsCell('agrees'))
    expect(deriveRightCellState(row, fetched({ idMatch: false }), localRecordWithDigests()).kind).toBe('contradicted')
  })
})

describe('deriveRightCellState — bilateral-retention-decay-property (agent-action-capsule @7f8a78d8, ratified 2026-09-23): one-half-unavailable MUST NOT collapse into both-present-disagreeing', () => {
  it('a peer fetch that legitimately comes back not_found (their retention decayed the record away, or they never held it) renders OPEN — never CONTRADICTED, never CLOSED/"attested by both"', () => {
    const row = paneCRow()
    const state = deriveRightCellState(
      row,
      fetched({ status: 'not_found', idMatch: null, signatureOk: null, peerRecord: null })
    )
    expect(state.kind).not.toBe('contradicted')
    expect(state.kind).not.toBe('closed')
    expect(state.kind).toBe('open_not_held')
  })

  it('an errored peer fetch (transport/verification failure, not a disagreement) renders OPEN — never CONTRADICTED', () => {
    const row = paneCRow()
    const state = deriveRightCellState(
      row,
      fetched({ status: 'error', idMatch: null, signatureOk: null, peerRecord: null })
    )
    expect(state.kind).not.toBe('contradicted')
    expect(state.kind).toBe('open_not_held')
  })

  it('MUTANT: only an ACTUAL fetched-and-compared idMatch===false is CONTRADICTED — every other "half unavailable" shape (not_fetched, fetching, not_found, error, or a fetch whose own id-recompute could not run) must go red if it starts reading as CONTRADICTED', () => {
    const row = paneCRow()
    const unavailableShapes: PeerRecomputeState[] = [
      fetched({ status: 'not_fetched', idMatch: null, signatureOk: null, peerRecord: null }),
      fetched({ status: 'fetching', idMatch: null, signatureOk: null, peerRecord: null }),
      fetched({ status: 'not_found', idMatch: null, signatureOk: null, peerRecord: null }),
      fetched({ status: 'error', idMatch: null, signatureOk: null, peerRecord: null }),
      fetched({ status: 'found', idMatch: null, signatureOk: null, peerRecord: { x: 1 } })
    ]
    for (const recompute of unavailableShapes) {
      expect(deriveRightCellState(row, recompute).kind).not.toBe('contradicted')
    }
    // The one and only shape that IS a real disagreement: a completed
    // fetch whose recomputed id demonstrably does not match the peer's
    // own claimed id -- the "both-present-disagreeing" case the property
    // says MUST stay a contradiction, never downgraded to missing-half.
    expect(deriveRightCellState(row, fetched({ status: 'found', idMatch: false })).kind).toBe('contradicted')
  })
})

describe('deriveRightCellState — all eight states reachable', () => {
  it('signed_refusal evidence_outcome -> open_refused, carrying its date', () => {
    const row = paneCRow({
      theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'signed_refusal', evidence_outcome_date: '4 Sep' }
    })
    const state = deriveRightCellState(row)
    expect(state.kind).toBe('open_refused')
    expect(state.date).toBe('4 Sep')
  })

  it('recorded_absence evidence_outcome -> open_absent', () => {
    const row = paneCRow({
      theirs: {
        state: 'absent',
        capsule_id: null,
        evidence_outcome: 'recorded_absence',
        evidence_outcome_date: '4 Sep'
      }
    })
    expect(deriveRightCellState(row).kind).toBe('open_absent')
  })

  it('unanswered evidence_outcome -> open_asked', () => {
    const row = paneCRow({
      theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'unanswered', evidence_outcome_date: '3 Sep' }
    })
    expect(deriveRightCellState(row).kind).toBe('open_asked')
  })

  it('not_asked evidence_outcome -> open_not_asked', () => {
    const row = paneCRow({
      theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'not_asked' }
    })
    expect(deriveRightCellState(row).kind).toBe('open_not_asked')
  })

  it('ADVERSARIAL — L-C: an absent theirs record with NO evidence_outcome carried never becomes open_asked (we hold no ask-log, so we cannot claim we asked)', () => {
    const row = paneCRow({ theirs: { state: 'absent', capsule_id: null } })
    const state = deriveRightCellState(row)
    expect(state.kind).toBe('open_not_asked')
    expect(state.kind).not.toBe('open_asked')
  })

  it('every one of the seven kinds is reachable', () => {
    const reached = new Set<RightCellStateKind>()
    reached.add(
      deriveRightCellState(
        paneCRow(),
        fetched({ idMatch: true, signatureOk: true, peerRecord: citingPeerRecord() }),
        localRecordWithDigests()
      ).kind
    )
    reached.add(deriveRightCellState(paneCRow(), fetched({ idMatch: false })).kind)
    reached.add(
      deriveRightCellState(
        paneCRow({ theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'signed_refusal' } })
      ).kind
    )
    reached.add(
      deriveRightCellState(
        paneCRow({ theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'recorded_absence' } })
      ).kind
    )
    reached.add(
      deriveRightCellState(paneCRow({ theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'unanswered' } }))
        .kind
    )
    reached.add(
      deriveRightCellState(paneCRow({ theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'p' } }))
        .kind
    )
    reached.add(
      deriveRightCellState(
        paneCRow({ theirs: { state: 'NOT_CHECKED', capsule_id: 'capsule-chatcmpl-1', peer_id: 'p' } })
      ).kind
    )
    reached.add(deriveRightCellState(paneCRow({ theirs: { state: 'absent', capsule_id: null } })).kind)
    for (const kind of ALL_KINDS) expect(reached.has(kind)).toBe(true)
    expect(reached.size).toBe(8)
  })
})

function stateOf(kind: RightCellStateKind, date: string | null = null): RightCellState {
  return { kind, date }
}

describe('rightCellText — the load-bearing distinction', () => {
  it('LOAD-BEARING: not-asked and unanswered never render the same string', () => {
    const notAsked = rightCellText(stateOf('open_not_asked'))
    const unanswered = rightCellText(stateOf('open_asked', '3 Sep'))
    expect(notAsked).not.toBe(unanswered)
  })

  it('LOAD-BEARING: pending-fetch never renders the same string as not-asked or CLOSED', () => {
    const pendingFetch = rightCellText(stateOf('open_not_held'))
    expect(pendingFetch).not.toBe(rightCellText(stateOf('open_not_asked')))
    expect(pendingFetch).not.toBe(rightCellText(stateOf('closed')))
  })

  it('renders the exact copy from v3 §2 for each state', () => {
    expect(rightCellText(stateOf('closed'))).toBe('✓ They recorded the same request and answer')
    expect(rightCellText(stateOf('contradicted'))).toBe('✗ Their record differs')
    expect(rightCellText(stateOf('open_refused', '4 Sep'))).toBe('They declined, and signed the refusal — 4 Sep')
    expect(rightCellText(stateOf('open_absent', '4 Sep'))).toBe('They say they have no record of this — 4 Sep')
    expect(rightCellText(stateOf('open_asked', '3 Sep'))).toBe('Asked 3 Sep. No reply yet.')
    expect(rightCellText(stateOf('open_not_asked'))).toBe('You haven’t asked for their record.')
    // UX §7 ruling: "not held" and "id not given" share one face; the (i)
    // tells them apart.
    expect(rightCellText(stateOf('open_not_held'))).toBe('Their record hasn’t arrived yet.')
    expect(rightCellText(stateOf('open_not_given'))).toBe('Their record hasn’t arrived yet.')
  })

  it('never invents a date when none is carried', () => {
    expect(rightCellText(stateOf('open_refused'))).toContain('date unavailable')
  })
})

describe('rightCellDetail — Item 4: the fuller story behind each state, moved behind the (i)', () => {
  it('gives every state a distinct honest sentence', () => {
    const details = ALL_KINDS.map((kind) => rightCellDetail(stateOf(kind)))
    expect(new Set(details).size).toBe(ALL_KINDS.length)
    for (const detail of details) expect(detail.length).toBeGreaterThan(0)
  })

  it('CLOSED names every condition the gate checks -- signed, from the node that served you, same request, answer and weights', () => {
    const text = rightCellDetail(stateOf('closed'))
    expect(text).toContain('their own signed record')
    expect(text).toContain('from the node that served you')
    expect(text).toContain('checks out on this machine')
    expect(text).toContain('same request, answer and model weights')
  })

  it('CONTRADICTED names every cause the gate contradicts on', () => {
    const text = rightCellDetail(stateOf('contradicted'))
    for (const cause of ['the request', 'the answer', 'the model', 'who served it', 'doesn’t match its own id']) {
      expect(text).toContain(cause)
    }
  })

  it('the not-held states say their record has not arrived; "id not given" adds why it cannot be asked for', () => {
    // Both cases the gate returns open_not_held for: nothing yet, or
    // something arrived that couldn't be confirmed as theirs.
    expect(rightCellDetail(stateOf('open_not_held'))).toContain('Their record hasn’t arrived')
    expect(rightCellDetail(stateOf('open_not_held'))).toContain('couldn’t be confirmed as theirs')
    expect(rightCellDetail(stateOf('open_not_given'))).toContain('they didn’t send an id to ask for it by')
  })

  it('never names a banned word, even to deny it, and none of the engineer’s words', () => {
    for (const kind of ALL_KINDS) {
      expect(rightCellDetail(stateOf(kind))).not.toMatch(new RegExp(`\\b(${STANDING_WORD}|judgement|capsule id|half|halves|recomputed?)\\b`, 'i'))
    }
  })

  it('never uses a score/rating/proven word', () => {
    for (const kind of ALL_KINDS) {
      expect(rightCellDetail(stateOf(kind))).not.toMatch(/\b(score|rating|proven)\b/i)
    }
  })
})

describe('rightCellStatusLabel', () => {
  it('names seven distinct status badges (open_not_given and open_not_asked share the plain OPEN badge)', () => {
    const labels = ALL_KINDS.map((kind) => rightCellStatusLabel(stateOf(kind)))
    expect(new Set(labels).size).toBe(7)
    expect(labels).toEqual([
      'CLOSED',
      'CONTRADICTED',
      'OPEN · refused',
      'OPEN · absent',
      'OPEN · asked',
      'OPEN · not held',
      'OPEN',
      'OPEN'
    ])
  })
})

describe('rightCellAction', () => {
  it('closed, open_not_held, and open_not_given have no row-level action; every other state has one', () => {
    expect(rightCellAction(stateOf('closed'))).toBeNull()
    expect(rightCellAction(stateOf('open_not_held'))).toBeNull()
    expect(rightCellAction(stateOf('open_not_given'))).toBeNull()
    for (const kind of ALL_KINDS.filter((k) => k !== 'closed' && k !== 'open_not_held' && k !== 'open_not_given')) {
      expect(rightCellAction(stateOf(kind))).not.toBeNull()
    }
  })

  it('names the exact action from v3 §2', () => {
    expect(rightCellAction(stateOf('contradicted'))).toBe('Compare')
    expect(rightCellAction(stateOf('open_refused'))).toBe('View refusal')
    expect(rightCellAction(stateOf('open_absent'))).toBe('View statement')
    expect(rightCellAction(stateOf('open_asked'))).toBe('Ask again')
    expect(rightCellAction(stateOf('open_not_asked'))).toBe('Ask them for their record')
  })
})

describe('isAlarmState — L-A/L-B enforcement', () => {
  it('L-B: only CONTRADICTED is an alarm state', () => {
    expect(isAlarmState(stateOf('contradicted'))).toBe(true)
  })

  it('L-A: every OPEN state, and CLOSED, is never styled as a problem', () => {
    for (const kind of ALL_KINDS.filter((k) => k !== 'contradicted')) {
      expect(isAlarmState(stateOf(kind))).toBe(false)
    }
  })
})

describe('isAskAction — counterparty-gating predicate', () => {
  it('only open_not_asked and open_asked are ask actions -- open_not_held is a FETCH action, not an ask', () => {
    expect(isAskAction('open_not_asked')).toBe(true)
    expect(isAskAction('open_asked')).toBe(true)
    expect(isAskAction('open_not_held')).toBe(false)
    for (const kind of ALL_KINDS.filter((k) => k !== 'open_not_asked' && k !== 'open_asked')) {
      expect(isAskAction(kind)).toBe(false)
    }
  })
})

describe('ledgerStateFilterValue — v3 §2a toolbar buckets', () => {
  it('closed and contradicted map to themselves', () => {
    expect(ledgerStateFilterValue(stateOf('closed'))).toBe('closed')
    expect(ledgerStateFilterValue(stateOf('contradicted'))).toBe('contradicted')
  })

  it('open_asked (a real ask, no reply yet) gets its own asked_no_reply bucket', () => {
    expect(ledgerStateFilterValue(stateOf('open_asked', '3 Sep'))).toBe('asked_no_reply')
  })

  it('the four no-reply/no-fetch states collapse into the broader open bucket', () => {
    expect(ledgerStateFilterValue(stateOf('open_refused'))).toBe('open')
    expect(ledgerStateFilterValue(stateOf('open_absent'))).toBe('open')
    expect(ledgerStateFilterValue(stateOf('open_not_held'))).toBe('open')
    expect(ledgerStateFilterValue(stateOf('open_not_asked'))).toBe('open')
  })
})

describe('deriveRightCellState — a swapped model is never CLOSED (attack D)', () => {
  const W_ASKED = '1'.repeat(64)
  const W_OTHER = '2'.repeat(64)
  function withModel(
    body: Record<string, unknown>,
    m: { modelId?: string; caWeights?: string; spWeights?: string }
  ): Record<string, unknown> {
    const ma = structuredClone(body.model_attestation) as Record<string, any>
    if (m.modelId) ma.model_id = m.modelId
    if (m.caWeights) ma.compute_attestation.weights_digest = { digest_alg: 'SHA-256', digest: m.caWeights, scope: 'file' }
    if (m.spWeights) ma.compute_attestation['x-mesh-poc-v1'].serving_provenance.model = { weights_digest: m.spWeights }
    return { ...body, model_attestation: ma }
  }
  const ours = () => withModel(localRecordWithDigests(), { modelId: `local-gguf/sha256-${W_ASKED}` }) as CapsuleRecord
  const gate = (peer: Record<string, unknown>) =>
    deriveRightCellState(paneCRow(), fetched({ peerRecord: peer }), ours()).kind

  it('control: the provider names the weights we asked for, digests agree -> CLOSED', () => {
    const peer = withModel(citingPeerRecord(), { modelId: `local-gguf/sha256-${W_ASKED}`, caWeights: W_ASKED, spWeights: W_ASKED })
    expect(gate(peer)).toBe('closed')
  })

  it('case 1: only serving_provenance.model.weights_digest swapped -> CONTRADICTED, never CLOSED', () => {
    const peer = withModel(citingPeerRecord(), { modelId: `local-gguf/sha256-${W_ASKED}`, caWeights: W_ASKED, spWeights: W_OTHER })
    expect(gate(peer)).toBe('contradicted')
  })

  it('case 2: a consistent liar (every weights field and model_id swapped) -> CONTRADICTED, never CLOSED', () => {
    const peer = withModel(citingPeerRecord(), { modelId: `local-gguf/sha256-${W_OTHER}`, caWeights: W_OTHER, spWeights: W_OTHER })
    expect(gate(peer)).toBe('contradicted')
  })

  it('the same holds for a pushed half', () => {
    const theirs = fixtureTheirsCell('agrees')
    const record = withModel(fixtureHalfBody({ capsuleId: theirs.capsule_id ?? undefined }), { caWeights: W_OTHER })
    const row = paneCRow({ unilateral: false, mine: fixtureMineCell(), theirs: { ...theirs, record }, digest_match: { state: 'verified' } })
    const mine = withModel(row.mine.record as Record<string, unknown>, { modelId: `local-gguf/sha256-${W_ASKED}` })
    expect(deriveRightCellState({ ...row, mine: { ...row.mine, record: mine } }).kind).toBe('contradicted')
  })

  it('a Hugging Face id (its digest is not the file digest) naming the model we asked for -> CLOSED', () => {
    const hfId = `local-gguf/sha256-${'3'.repeat(64)}`
    const local = withModel(localRecordWithDigests(), { modelId: hfId }) as CapsuleRecord
    const peer = withModel(citingPeerRecord(), { modelId: hfId, caWeights: W_ASKED, spWeights: W_ASKED })
    expect(deriveRightCellState(paneCRow(), fetched({ peerRecord: peer }), local).kind).toBe('closed')
  })

  it('a model name alone (an alias, no weights on our side) never contradicts', () => {
    const local = withModel(localRecordWithDigests(), { modelId: 'qwen' }) as CapsuleRecord
    const peer = withModel(citingPeerRecord(), { modelId: `local-gguf/sha256-${W_OTHER}`, caWeights: W_OTHER })
    expect(deriveRightCellState(paneCRow(), fetched({ peerRecord: peer }), local).kind).toBe('closed')
  })
})

describe('deriveRightCellState — a provider half the door refused for its claims (attack B/D)', () => {
  it('reads CONTRADICTED, never open and never closed', () => {
    const row = paneCRow({
      theirs: { ...paneCRow().theirs, evidence_outcome: 'claims_refused', evidence_outcome_date: '2026-09-28T08:00:00Z' }
    })
    expect(deriveRightCellState(row).kind).toBe('contradicted')
  })
})

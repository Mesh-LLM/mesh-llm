import { describe, expect, it } from 'vitest'
import { SECOND_SIGNATURE_WORD } from '@/features/capsules/lib/banned-terms'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import { checkpointCoverageByRecord } from '@/features/capsules/lib/integrity-view'
import {
  NINE_PROPERTY_LABELS,
  WHAT_ACTUALLY_HAPPENED_GROUP,
  WHAT_NODE_SAID_GROUP
} from '@/features/capsules/lib/nine-properties'
import type { PeerRecomputeState, RecomputedIdentity } from '@/features/capsules/lib/recompute-identity'
import { peerFetchJoinKey } from '@/features/capsules/lib/recompute-identity'
import { fixtureMineCell, fixtureTheirsCell } from '@/features/capsules/lib/pushed-half-fixtures'
import {
  buildChecksRows,
  buildCommitsToRows,
  recordServedBy,
  TEE_COPY,
  buildHeaderRows,
  buildIdentityRow,
  theirsFetchable
} from '@/features/capsules/lib/security-checks-view'

// The manifesto's ten names, verbatim from `manifesto-the-tenth-check-v6`'s
// table -- the accept test's ground truth (this repo's field names, per
// `NINE_PROPERTY_LABELS`'s own header comment).
const MANIFESTO_TEN_PROPERTY_KEYS = [
  'content_binding',
  'producer_signature',
  'task_binding',
  'local_inclusion',
  'checkpoint_signature',
  'external_registration',
  'continuity',
  'identity_authority',
  'capture_coverage',
  'outcome_corroboration'
]

function paneCRow(overrides: Partial<PaneCRow> = {}): PaneCRow {
  return {
    exchange_key: 'exch-cle',
    role_tag: 'ASKED',
    header_state: 'ok',
    properties: null,
    has_issue: false,
    mine: { state: 'present', capsule_id: 'mine-capsule-1' },
    theirs: { state: 'present', capsule_id: 'theirs-capsule-1' },
    unilateral: false,
    timestamp: '2026-09-08T16:58:05.412Z',
    ...overrides
  }
}

const NOT_RECOMPUTED: RecomputedIdentity = { idMatch: null, signatureOk: null }
const RECOMPUTED_MATCH: RecomputedIdentity = { idMatch: true, signatureOk: true }

describe('buildChecksRows — rendered property set (accept: == manifesto ten names exactly)', () => {
  it('emits exactly the manifesto ten property keys, in order', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED)
    expect(rows.map((r) => r.key)).toEqual(MANIFESTO_TEN_PROPERTY_KEYS)
  })

  it('NINE_PROPERTY_LABELS itself carries all ten (9+1, not 8+1)', () => {
    expect(Object.keys(NINE_PROPERTY_LABELS)).toEqual(MANIFESTO_TEN_PROPERTY_KEYS)
    expect(NINE_PROPERTY_LABELS.task_binding).toBe('task binding')
  })

  it('every row but outcome_corroboration is grouped "what this node said it did"; outcome_corroboration alone is "what actually happened"', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    for (const key of MANIFESTO_TEN_PROPERTY_KEYS) {
      if (key === 'outcome_corroboration') {
        expect(byKey[key].group).toBe(WHAT_ACTUALLY_HAPPENED_GROUP)
      } else {
        expect(byKey[key].group).toBe(WHAT_NODE_SAID_GROUP)
      }
    }
  })
})

describe('buildChecksRows — Q1: five results render lowercase, never shouty caps', () => {
  it('established/failed/not present/not checked/inconclusive, not PASS/FAIL/...', () => {
    const row = paneCRow({
      properties: {
        task_binding: { state: 'PASS' },
        local_inclusion: { state: 'FAIL' },
        checkpoint_signature: { state: 'PASS' },
        external_registration: { state: 'NOT_CHECKED' },
        continuity: { state: 'INCONCLUSIVE' }
      }
    })
    const rows = buildChecksRows(row, NOT_RECOMPUTED)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.task_binding.yours?.label).toBe('established')
    expect(byKey.local_inclusion.yours?.label).toBe('failed')
    expect(byKey.checkpoint_signature.yours?.label).toBe('established')
    expect(byKey.external_registration.yours?.label).toBe('not checked')
    expect(byKey.continuity.yours?.label).toBe('inconclusive')
    for (const r of rows) {
      if (r.yours) expect(r.yours.label).not.toMatch(/[A-Z]/)
    }
  })
})

describe('buildChecksRows — checkpoint-dependent properties resolve NOT_PRESENT, never NOT_CHECKED, when absent', () => {
  it('local_inclusion/checkpoint_signature/continuity default to not present with no properties at all', () => {
    const rows = buildChecksRows(paneCRow({ properties: null }), NOT_RECOMPUTED)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.local_inclusion.yours?.label).toBe('not present')
    expect(byKey.local_inclusion.yours?.detail).toBe('No checkpoint covers this record — see Integrity.')
    expect(byKey.checkpoint_signature.yours?.label).toBe('not present')
    expect(byKey.continuity.yours?.label).toBe('not present')
  })

  it('external_registration defaults to not present, with a receipt-specific detail, not the checkpoint one', () => {
    const rows = buildChecksRows(paneCRow({ properties: null }), NOT_RECOMPUTED)
    const externalRegistration = rows.find((r) => r.key === 'external_registration')
    expect(externalRegistration?.yours?.label).toBe('not present')
    expect(externalRegistration?.yours?.detail).toBe('no receipt covers this record')
  })

  it('a property with no absence rule (task_binding) still defaults to not checked, unchanged', () => {
    const rows = buildChecksRows(paneCRow({ properties: null }), NOT_RECOMPUTED)
    const taskBinding = rows.find((r) => r.key === 'task_binding')
    expect(taskBinding?.yours?.label).toBe('not checked')
  })
})

describe('buildChecksRows — look finding 2: per-record checkpoint coverage agrees with Integrity', () => {
  it('REGRESSION: a record the checkpoint covers never reads "no checkpoint covers this record"', () => {
    // The look: Integrity said "All 8 records are sealed into a checkpoint"
    // while every row's panel said "no checkpoint covers this record" --
    // the native pane sends no per-row properties, and their absence was
    // read as not covered.
    const coveredId = 'mine-1'
    const coverage = checkpointCoverageByRecord([coveredId, 'mine-2'], 1)
    const rows = buildChecksRows(paneCRow({ properties: null }), NOT_RECOMPUTED, undefined, {
      checkpointCovered: coverage.get(coveredId) ?? null
    })
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    for (const key of ['local_inclusion', 'checkpoint_signature', 'continuity']) {
      expect(byKey[key].yours?.detail).not.toMatch(/no checkpoint covers/)
      // Covered, but the inclusion proof isn't checked in the browser --
      // never an "established" this page did not check.
      expect(byKey[key].yours?.state).toBe('NOT_CHECKED')
    }
    expect(byKey.local_inclusion.yours?.detail).toBe(
      'a checkpoint covers this record — see Integrity; this row hasn’t checked that for itself yet'
    )
  })

  it('a record after the last checkpoint says "not ... yet"', () => {
    const coverage = checkpointCoverageByRecord(['mine-1', 'mine-2'], 1)
    const rows = buildChecksRows(paneCRow({ properties: null }), NOT_RECOMPUTED, undefined, {
      checkpointCovered: coverage.get('mine-2') ?? null
    })
    const localInclusion = rows.find((r) => r.key === 'local_inclusion')
    expect(localInclusion?.yours?.state).toBe('NOT_PRESENT')
    expect(localInclusion?.yours?.detail).toBe('No checkpoint covers this record yet — see Integrity.')
  })

  it('an unreported covered count maps nothing -- coverage unknown, never "none"', () => {
    expect(checkpointCoverageByRecord(['mine-1'], null).size).toBe(0)
  })
})

describe('buildChecksRows — continuity established only if checkpoint_signature established', () => {
  it('downgrades a claimed continuity PASS to not present when checkpoint_signature is not established', () => {
    const row = paneCRow({
      properties: {
        checkpoint_signature: { state: 'NOT_PRESENT' },
        continuity: { state: 'PASS' }
      }
    })
    const rows = buildChecksRows(row, NOT_RECOMPUTED)
    const continuity = rows.find((r) => r.key === 'continuity')
    expect(continuity?.yours?.state).toBe('NOT_PRESENT')
    expect(continuity?.yours?.label).toBe('not present')
  })

  it('leaves continuity PASS alone when checkpoint_signature is also established', () => {
    const row = paneCRow({
      properties: {
        checkpoint_signature: { state: 'PASS' },
        continuity: { state: 'PASS' }
      }
    })
    const rows = buildChecksRows(row, NOT_RECOMPUTED)
    const continuity = rows.find((r) => r.key === 'continuity')
    expect(continuity?.yours?.state).toBe('PASS')
    expect(continuity?.yours?.label).toBe('established')
  })

  it('does not invent a stronger negative -- a continuity FAIL/INCONCLUSIVE from the sidecar passes through unchanged', () => {
    const row = paneCRow({
      properties: {
        checkpoint_signature: { state: 'NOT_PRESENT' },
        continuity: { state: 'FAIL', text: 'checkpoint mismatch at leaf 12' }
      }
    })
    const rows = buildChecksRows(row, NOT_RECOMPUTED)
    const continuity = rows.find((r) => r.key === 'continuity')
    expect(continuity?.yours?.state).toBe('FAIL')
    expect(continuity?.yours?.detail).toBe('checkpoint mismatch at leaf 12')
  })
})

describe('buildChecksRows — capture_coverage: fixed sentence or not present, never a PASS variant', () => {
  it('renders the fixed sentence when the sidecar has a record for it', () => {
    const rows = buildChecksRows(paneCRow({ properties: { capture_coverage: { state: 'PASS' } } }), NOT_RECOMPUTED)
    const captureCoverage = rows.find((r) => r.key === 'capture_coverage')
    expect(captureCoverage?.singleLine).toBe(
      'captured by the plugin at this node’s serving boundary (rule: every served exchange)'
    )
    expect(captureCoverage?.yours).toBeNull()
    expect(captureCoverage?.theirs).toBeNull()
  })

  it('renders "not present" when the sidecar sends nothing for it', () => {
    const rows = buildChecksRows(paneCRow({ properties: null }), NOT_RECOMPUTED)
    const captureCoverage = rows.find((r) => r.key === 'capture_coverage')
    expect(captureCoverage?.singleLine).toBe('not present')
  })

  it('an explicit NOT_PRESENT state also renders "not present", not the sentence', () => {
    const rows = buildChecksRows(
      paneCRow({ properties: { capture_coverage: { state: 'NOT_PRESENT' } } }),
      NOT_RECOMPUTED
    )
    const captureCoverage = rows.find((r) => r.key === 'capture_coverage')
    expect(captureCoverage?.singleLine).toBe('not present')
  })
})

describe('buildChecksRows — identity/authority: two facts, not one state', () => {
  it('binding is "not present" and authority is always "not present" when nothing is bound', () => {
    const rows = buildChecksRows(paneCRow({ properties: null }), NOT_RECOMPUTED)
    const identityAuthority = rows.find((r) => r.key === 'identity_authority')
    expect(identityAuthority?.yours).toBeNull()
    expect(identityAuthority?.facts?.map((f) => f.factLabel)).toEqual(['binding', 'authority'])
    const binding = identityAuthority?.facts?.find((f) => f.factLabel === 'binding')
    const authority = identityAuthority?.facts?.find((f) => f.factLabel === 'authority')
    expect(binding?.cell.label).toBe('not present')
    expect(authority?.cell.label).toBe('not present')
    expect(authority?.cell.detail).toMatch(/not bound to a person/)
  })

  it('binding renders "self-asserted key, valid to <expiry>" when a key is bound', () => {
    const rows = buildChecksRows(
      paneCRow({ properties: { identity_authority: { state: 'PASS', expiry: '2026-12-01' } } }),
      NOT_RECOMPUTED
    )
    const identityAuthority = rows.find((r) => r.key === 'identity_authority')
    const binding = identityAuthority?.facts?.find((f) => f.factLabel === 'binding')
    expect(binding?.cell.label).toBe('established')
    expect(binding?.cell.detail).toBe('self-asserted key, valid to 2026-12-01')
  })

  it('binding renders "failed" when the bound key is invalid', () => {
    const rows = buildChecksRows(paneCRow({ properties: { identity_authority: { state: 'FAIL' } } }), NOT_RECOMPUTED)
    const identityAuthority = rows.find((r) => r.key === 'identity_authority')
    const binding = identityAuthority?.facts?.find((f) => f.factLabel === 'binding')
    expect(binding?.cell.label).toBe('failed')
  })

  it('authority is always not present, even when binding is established', () => {
    const rows = buildChecksRows(
      paneCRow({ properties: { identity_authority: { state: 'PASS', expiry: '2026-12-01' } } }),
      NOT_RECOMPUTED
    )
    const identityAuthority = rows.find((r) => r.key === 'identity_authority')
    const authority = identityAuthority?.facts?.find((f) => f.factLabel === 'authority')
    expect(authority?.cell.label).toBe('not present')
  })
})

describe('buildChecksRows — L-L: every check row names its inputs and policy inline', () => {
  it('no property row carries an empty detail (a bare state word is forbidden)', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED)
    for (const r of rows) {
      if (r.key === 'capture_coverage') {
        expect(r.singleLine).toBeTruthy()
        continue
      }
      if (r.key === 'identity_authority') {
        for (const fact of r.facts ?? []) expect(fact.cell.detail).toBeTruthy()
        continue
      }
      expect(r.yours?.detail).toBeTruthy()
    }
  })

  it('a sidecar-supplied text wins over the generic default', () => {
    const row = paneCRow({
      properties: { continuity: { state: 'FAIL', text: 'checkpoint mismatch at leaf 12' } }
    })
    const rows = buildChecksRows(row, NOT_RECOMPUTED)
    const continuityRow = rows.find((r) => r.key === 'continuity')
    expect(continuityRow?.yours?.detail).toBe('checkpoint mismatch at leaf 12')
  })
})

describe('buildChecksRows — L-M: recomputed-here and from-sidecar are never the same', () => {
  it('content_binding/producer_signature are flagged recomputed; the rest are not', () => {
    const rows = buildChecksRows(paneCRow(), RECOMPUTED_MATCH)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.content_binding.yours?.recomputed).toBe(true)
    expect(byKey.producer_signature.yours?.recomputed).toBe(true)
    expect(byKey.task_binding.yours?.recomputed).toBe(false)
    expect(byKey.local_inclusion.yours?.recomputed).toBe(false)
    expect(byKey.outcome_corroboration.yours?.recomputed).toBe(false)
  })

  it('recomputed properties always say "recomputed here", never a sidecar phrase', () => {
    const rows = buildChecksRows(paneCRow(), RECOMPUTED_MATCH)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.content_binding.yours?.detail).toBe('recomputed here')
    expect(byKey.producer_signature.yours?.detail).toBe('recomputed here')
  })

  it('content_binding reflects idMatch, never upgraded to established when unrecomputed', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED)
    const contentBinding = rows.find((r) => r.key === 'content_binding')
    expect(contentBinding?.yours?.state).toBe('NOT_CHECKED')
    expect(contentBinding?.yours?.label).toBe('not checked')
  })
})

describe('buildChecksRows — forbidden mutant: never NOT_CHECKED together with recomputed:true', () => {
  it('content_binding/producer_signature are NOT flagged recomputed while the recompute has not run yet', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.content_binding.yours?.state).toBe('NOT_CHECKED')
    expect(byKey.content_binding.yours?.recomputed).toBe(false)
    expect(byKey.producer_signature.yours?.state).toBe('NOT_CHECKED')
    expect(byKey.producer_signature.yours?.recomputed).toBe(false)
  })

  it('says "not yet recomputed here", not "recomputed here", while unrecomputed', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.content_binding.yours?.detail).toBe('not yet recomputed here')
  })

  it('no row, in any identity state, ever pairs NOT_CHECKED with recomputed:true', () => {
    for (const identity of [
      NOT_RECOMPUTED,
      RECOMPUTED_MATCH,
      { idMatch: false, signatureOk: null } as RecomputedIdentity,
      { idMatch: null, signatureOk: false } as RecomputedIdentity
    ]) {
      const rows = buildChecksRows(paneCRow(), identity)
      for (const r of rows) {
        if (r.yours) expect(r.yours.state === 'NOT_CHECKED' && r.yours.recomputed).toBe(false)
        if (r.theirs) expect(r.theirs.state === 'NOT_CHECKED' && r.theirs.recomputed).toBe(false)
      }
    }
  })
})

describe('buildChecksRows — THEIRS column (v1 §P5 / recompute-identity.ts: no full record for a counterparty half)', () => {
  it('theirs column is entirely absent when theirs.state is absent (L-G)', () => {
    const row = paneCRow({ theirs: { state: 'absent', capsule_id: null } })
    const rows = buildChecksRows(row, NOT_RECOMPUTED)
    for (const r of rows) {
      if (r.key === 'capture_coverage' || r.key === 'identity_authority') continue
      expect(r.theirs).toBeNull()
    }
  })

  it('theirs is always NOT_CHECKED when held -- never a claimed PASS this page cannot back', () => {
    const rows = buildChecksRows(paneCRow(), RECOMPUTED_MATCH)
    for (const r of rows) {
      if (r.key === 'capture_coverage' || r.key === 'identity_authority') continue
      expect(r.theirs?.state).toBe('NOT_CHECKED')
      expect(r.theirs?.recomputed).toBe(false)
      expect(r.theirs?.detail).toBeTruthy()
    }
  })
})

// piece 4: once a real peer fetch actually
// resolves (status 'found'), content_binding/producer_signature's THEIRS
// cells render the REAL recomputed result -- but every other property still
// has no bytes to check, so it stays NOT_CHECKED even with a `found` peer
// fetch (R4 other half: a found fetch must not upgrade properties it never
// touched).
describe('buildChecksRows — THEIRS column reflects a real peer recompute once one has run', () => {
  const FOUND_MATCH: PeerRecomputeState = {
    status: 'found',
    idMatch: true,
    signatureOk: true,
    peerRecord: null,
    fetch: () => {}
  }
  const FOUND_MISMATCH: PeerRecomputeState = {
    status: 'found',
    idMatch: false,
    signatureOk: false,
    peerRecord: null,
    fetch: () => {}
  }
  const NOT_FETCHED: PeerRecomputeState = {
    status: 'not_fetched',
    idMatch: null,
    signatureOk: null,
    peerRecord: null,
    fetch: () => {}
  }
  const FETCHING: PeerRecomputeState = {
    status: 'fetching',
    idMatch: null,
    signatureOk: null,
    peerRecord: null,
    fetch: () => {}
  }
  const NOT_FOUND: PeerRecomputeState = {
    status: 'not_found',
    idMatch: null,
    signatureOk: null,
    peerRecord: null,
    fetch: () => {}
  }
  const ERRORED: PeerRecomputeState = {
    status: 'error',
    idMatch: null,
    signatureOk: null,
    peerRecord: null,
    errorMessage: 'peer unroutable',
    fetch: () => {}
  }

  it('a found+matching peer fetch renders content_binding/producer_signature as PASS, recomputed:true, "recomputed here"', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED, FOUND_MATCH)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.content_binding.theirs?.state).toBe('PASS')
    expect(byKey.content_binding.theirs?.recomputed).toBe(true)
    expect(byKey.content_binding.theirs?.detail).toBe('recomputed here')
    expect(byKey.producer_signature.theirs?.state).toBe('PASS')
    expect(byKey.producer_signature.theirs?.recomputed).toBe(true)
  })

  it('a found+mismatching peer fetch renders FAIL, not a silently-passed match', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED, FOUND_MISMATCH)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.content_binding.theirs?.state).toBe('FAIL')
    expect(byKey.producer_signature.theirs?.state).toBe('FAIL')
  })

  it('every non-recomputable property stays NOT_CHECKED even once theirs is found (R4: found never leaks onto untouched properties)', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED, FOUND_MATCH)
    for (const r of rows) {
      if (['content_binding', 'producer_signature', 'capture_coverage', 'identity_authority'].includes(r.key)) continue
      expect(r.theirs?.state).toBe('NOT_CHECKED')
      expect(r.theirs?.recomputed).toBe(false)
    }
  })

  // (negative, R4 other half) not_fetched/fetching/not_found/error must
  // never be rendered as though a real recompute ran -- recomputed stays
  // false and state stays NOT_CHECKED for every one of them.
  it.each([
    ['not_fetched', NOT_FETCHED],
    ['fetching', FETCHING],
    ['not_found', NOT_FOUND],
    ['error', ERRORED]
  ])('a %s peer-fetch status never renders as recomputed', (_label, state) => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED, state)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.content_binding.theirs?.state).toBe('NOT_CHECKED')
    expect(byKey.content_binding.theirs?.recomputed).toBe(false)
  })

  it('surfaces the real not_found/error reasons in the detail text, not a generic placeholder', () => {
    const notFoundRows = buildChecksRows(paneCRow(), NOT_RECOMPUTED, NOT_FOUND)
    expect(notFoundRows.find((r) => r.key === 'content_binding')?.theirs?.detail).toMatch(/no such record/)

    const erroredRows = buildChecksRows(paneCRow(), NOT_RECOMPUTED, ERRORED)
    expect(erroredRows.find((r) => r.key === 'content_binding')?.theirs?.detail).toMatch(/peer unroutable/)
  })

  // Vocabulary check (state the check ran and the
  // verdict, not the grepped terms): confirmed the recomputed-theirs wording
  // above never uses second-party-judgment language -- clean.
  it('never renders theirs as a second-party judgment ("verified by", "confirmed by", a second signature)', () => {
    const rows = buildChecksRows(paneCRow(), NOT_RECOMPUTED, FOUND_MATCH)
    for (const r of rows) {
      if (r.theirs?.detail) {
        expect(r.theirs.detail.toLowerCase()).not.toMatch(new RegExp(`verified by|confirmed by|${SECOND_SIGNATURE_WORD}`))
      }
    }
  })
})

describe('theirsFetchable / peerFetchJoinKey', () => {
  it('is the SAME function re-exported, not a second copy of the join-key rule', () => {
    expect(theirsFetchable).toBe(peerFetchJoinKey)
  })

  it('a row with theirs.state NOT_CHECKED and a real digest-shaped capsule_id+peer_id is fetchable', () => {
    const row = paneCRow({ theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-node-3' } })
    expect(theirsFetchable(row)).toEqual({ capsuleId: 'a'.repeat(64), peerId: 'peer-node-3' })
  })

  // (negative, R4 other half) NOT_CHECKED with a hole in the join key, or
  // any other state, must never report fetchable -- there is nowhere to
  // fetch FROM.
  it.each([
    ['absent state', { state: 'absent', capsule_id: null }],
    ['NOT_CHECKED with no capsule_id', { state: 'NOT_CHECKED', capsule_id: null, peer_id: 'peer-node-3' }],
    ['NOT_CHECKED with no peer_id', { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: null }],
    // a non-NOT_CHECKED state carrying a real join key anyway must still
    // never report fetchable -- `theirs_cell` (capsule_panes_native.rs)
    // only ever emits a join key alongside literal NOT_CHECKED, so any
    // other state means something upstream is inconsistent, not fetchable.
    [
      'a real join key under an unexpected state',
      { state: 'present', capsule_id: 'a'.repeat(64), peer_id: 'peer-node-3' }
    ]
  ])('%s is never reported fetchable', (_label, theirs) => {
    const row = paneCRow({ theirs })
    expect(theirsFetchable(row)).toBeNull()
  })

  // Finding 5 (2026-09-23 assessment): a peer-forwarded chat-completion id
  // is not a capsule id -- MUTANT: drop the digest-shape check and this
  // goes green when it must stay red.
  it('MUTANT-GUARD: a peer-asserted id that is not digest-shaped (a chat-completion id) is never reported fetchable', () => {
    const row = paneCRow({
      theirs: { state: 'NOT_CHECKED', capsule_id: 'capsule-chatcmpl-1790147257740', peer_id: 'peer-node-3' }
    })
    expect(theirsFetchable(row)).toBeNull()
  })
})

describe('buildIdentityRow', () => {
  it('yours reflects the actual recompute result, never a hardcoded match', () => {
    const mismatch: RecomputedIdentity = { idMatch: false, signatureOk: null }
    const identityRow = buildIdentityRow(paneCRow(), mismatch)
    expect(identityRow.yours.note).toMatch(/MISMATCH/)
  })

  it('theirs is absent (null) when theirs.state is absent', () => {
    const row = paneCRow({ theirs: { state: 'absent', capsule_id: null } })
    const identityRow = buildIdentityRow(row, NOT_RECOMPUTED)
    expect(identityRow.theirs).toBeNull()
  })

  it('theirs never claims "recomputed" when no peer recompute was supplied -- it was never recomputed', () => {
    const identityRow = buildIdentityRow(paneCRow(), RECOMPUTED_MATCH)
    expect(identityRow.theirs?.note).not.toMatch(/recomputed here, matches/)
  })

  it('theirs renders the REAL recomputed match once a peer fetch actually found and verified', () => {
    const found: PeerRecomputeState = {
      status: 'found',
      idMatch: true,
      signatureOk: true,
      peerRecord: null,
      fetch: () => {}
    }
    const identityRow = buildIdentityRow(paneCRow(), NOT_RECOMPUTED, found)
    expect(identityRow.theirs?.note).toBe('✓ recomputed here, matches')
  })

  it('theirs renders a real MISMATCH, never silently upgraded to a match', () => {
    const found: PeerRecomputeState = {
      status: 'found',
      idMatch: false,
      signatureOk: false,
      peerRecord: null,
      fetch: () => {}
    }
    const identityRow = buildIdentityRow(paneCRow(), NOT_RECOMPUTED, found)
    expect(identityRow.theirs?.note).toBe('✕ recomputed here, MISMATCH')
  })

  // (negative, R4 other half) a fetch that found nothing or errored must
  // never render the "matches" note -- honesty about failure, not a
  // silently-dropped-back-to-unrecomputed placeholder that could be
  // confused with success.
  it('a not_found peer fetch never renders as a match', () => {
    const notFound: PeerRecomputeState = {
      status: 'not_found',
      idMatch: null,
      signatureOk: null,
      peerRecord: null,
      fetch: () => {}
    }
    const identityRow = buildIdentityRow(paneCRow(), NOT_RECOMPUTED, notFound)
    expect(identityRow.theirs?.note).not.toMatch(/matches/)
    expect(identityRow.theirs?.note).toBe('they have no such record')
  })

  // RENDERING NOTE (design §7, 2026-09-23): id known, bytes not held -- the
  // outline glyph, never the filled CLOSED reading.
  it('a peer-asserted, unfetched id carries the ◔ outline glyph, not a bare "as given" claim', () => {
    const row = paneCRow({ theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' } })
    const identityRow = buildIdentityRow(row, NOT_RECOMPUTED)
    expect(identityRow.theirs?.note).toBe('◔ as given, not recomputed')
  })

  // Finding 5: a peer-forwarded chat-completion id is not a capsule id.
  it('MUTANT-GUARD: a non-digest-shaped peer-asserted id renders "not given", never the raw non-id value', () => {
    const row = paneCRow({
      theirs: { state: 'NOT_CHECKED', capsule_id: 'capsule-chatcmpl-1790147257740', peer_id: 'peer-1' }
    })
    const identityRow = buildIdentityRow(row, NOT_RECOMPUTED)
    expect(identityRow.theirs?.value).toBe('not given')
    expect(identityRow.theirs?.value).not.toBe('capsule-chatcmpl-1790147257740')
  })
})

describe('buildHeaderRows / buildCommitsToRows — no fabricated fields', () => {
  it('degrades to an honest placeholder rather than inventing a value', () => {
    const headerRows = buildHeaderRows(paneCRow(), null)
    const keyId = headerRows.find((r) => r.label === 'key id')
    expect(keyId?.yours.value).toBe('unavailable')
  })

  // Finding 2, reworked (2026-09-23 bounce): every comparable header field
  // ('sealed at' / 'algorithm' / 'key id') on a HELD-but-unfetched row must
  // render the 'not held' placeholder, not a bare empty cell -- checked
  // per property so a regression in any one field's wiring is caught, not
  // just averaged over the whole row.
  it.each(['sealed at', 'algorithm', 'key id'])(
    "held/not-held per property: '%s' renders 'not held' when a counterparty is recorded but nothing has been fetched",
    (label) => {
      const row = paneCRow({ theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' } })
      const headerRows = buildHeaderRows(row, null)
      expect(headerRows.find((r) => r.label === label)?.theirs).toEqual({ value: '—', note: 'not held' })
    }
  )

  it.each(['sealed at', 'algorithm', 'key id'])(
    "held/not-held per property: '%s' has no theirs cell at all when theirs.state is absent (no counterparty recorded)",
    (label) => {
      const row = paneCRow({ theirs: { state: 'absent', capsule_id: null } })
      const headerRows = buildHeaderRows(row, null)
      expect(headerRows.find((r) => r.label === label)?.theirs).toBeNull()
    }
  )

  it.each(['sealed at', 'algorithm', 'key id'])(
    "held/not-held per property: '%s' renders the real fetched value once a peer fetch actually resolves",
    (label) => {
      const row = paneCRow({ theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' } })
      const theirsRecompute: PeerRecomputeState = {
        status: 'found',
        idMatch: true,
        signatureOk: true,
        peerRecord: { timestamp: '2026-09-09T00:00:00Z', key_id: 'peer-key-1' },
        fetch: () => {}
      }
      const headerRows = buildHeaderRows(row, null, theirsRecompute)
      expect(headerRows.find((r) => r.label === label)?.theirs).not.toEqual({ value: '—', note: 'not held' })
      expect(headerRows.find((r) => r.label === label)?.theirs).not.toBeNull()
    }
  )

  it('commits-to reads real digests off the local record when present', () => {
    const rows = buildCommitsToRows(paneCRow(), {
      effect: { request_digest: 'a'.repeat(64), response_digest: 'b'.repeat(64) }
    } as never)
    const requestDigest = rows.find((r) => r.label === 'request digest')
    expect(requestDigest?.yours).toBe('a'.repeat(64))
  })

  // Finding 4 (2026-09-23 assessment): the producer's own honest-absence
  // sentinel (`unknown-request:<model>`, `capsule_emit.rs`) is not a
  // digest -- the viewer must flag it, never tick it.
  it('MUTANT-GUARD: a non-digest-shaped request_digest (the producer sentinel) renders "not a digest", never ticked as real', () => {
    const rows = buildCommitsToRows(paneCRow(), {
      effect: { request_digest: 'unknown-request:local-gguf/sha256-7089c7' }
    } as never)
    const requestDigest = rows.find((r) => r.label === 'request digest')
    expect(requestDigest?.yours).toBe('not a digest')
  })
})

describe('buildCommitsToRows — theirs column, finding 2 corrected', () => {
  const LOCAL_RECORD = { effect: { request_digest: 'a'.repeat(64), response_digest: 'b'.repeat(64) } } as never

  // MUTANT (finding 2): the old behaviour mirrored `yours` into `theirs`
  // labelled `✓ same` the instant the row read CLOSED -- restating our own
  // value as though it were an independent fact. `theirs` must never render
  // that mirrored `✓ same` until this browser actually holds the peer's OWN
  // record.
  it('MUTANT-GUARD: no theirsRecompute supplied -- no theirs cell ever reads ✓ same (never a mirror of our own value)', () => {
    const rows = buildCommitsToRows(paneCRow(), LOCAL_RECORD)
    for (const row of rows) {
      expect(row.theirs?.note).not.toBe('✓ same')
    }
  })

  // Finding 2, reworked (2026-09-23 bounce): a comparable field on a HELD
  // row (a counterparty is recorded) with no fetch yet is the "unfetched"
  // case an empty div used to hide -- it must render the SAME honest
  // `not held` placeholder a found-but-missing-field fetch already gets,
  // never a bare `null`/empty cell that looks indistinguishable from the
  // true absent-counterparty case below.
  it.each(['request digest', 'response digest', 'model identity'])(
    "held/not-held per property: '%s' renders the 'not held' placeholder on an unfetched but HELD row, never an empty cell",
    (label) => {
      const rows = buildCommitsToRows(paneCRow(), LOCAL_RECORD)
      expect(rows.find((r) => r.label === label)?.theirs).toEqual({ value: '—', note: 'not held' })
    }
  )

  it.each(['request digest', 'response digest', 'model identity'])(
    "held/not-held per property: '%s' has no theirs cell at all when theirs.state is absent (no counterparty recorded)",
    (label) => {
      const rows = buildCommitsToRows(paneCRow({ theirs: { state: 'absent', capsule_id: null } }), LOCAL_RECORD)
      expect(rows.find((r) => r.label === label)?.theirs).toBeNull()
    }
  )

  it('a found fetch whose peer record genuinely matches renders check same off the peers own field, not a mirror', () => {
    const theirsRecompute: PeerRecomputeState = {
      status: 'found',
      idMatch: true,
      signatureOk: true,
      peerRecord: { effect: { request_digest: 'a'.repeat(64), response_digest: 'c'.repeat(64) } },
      fetch: () => {}
    }
    const rows = buildCommitsToRows(paneCRow(), LOCAL_RECORD, theirsRecompute)
    const requestDigest = rows.find((r) => r.label === 'request digest')
    const responseDigest = rows.find((r) => r.label === 'response digest')
    expect(requestDigest?.theirs).toEqual({ value: 'a'.repeat(64), note: '✓ same' })
    // Different peer value -- MUST read differs, never silently pass.
    expect(responseDigest?.theirs).toEqual({ value: 'c'.repeat(64), note: '✕ differs' })
  })

  it('a found fetch whose peer record omits the field renders "not held", never a value we never received', () => {
    const theirsRecompute: PeerRecomputeState = {
      status: 'found',
      idMatch: true,
      signatureOk: true,
      peerRecord: { effect: {} },
      fetch: () => {}
    }
    const rows = buildCommitsToRows(paneCRow(), LOCAL_RECORD, theirsRecompute)
    const requestDigest = rows.find((r) => r.label === 'request digest')
    expect(requestDigest?.theirs?.note).toBe('not held')
    expect(requestDigest?.theirs?.value).not.toBe('a'.repeat(64))
  })

  it('an OPEN row (theirs absent) has no theirs column at all -- MUTANT: never render a ✓/✕ marker with nothing to compare', () => {
    const rows = buildCommitsToRows(paneCRow({ theirs: { state: 'absent', capsule_id: null } }), null)
    for (const row of rows) {
      expect(row.theirs).toBeNull()
    }
  })

  // Finding 3: `served by: counterparty` was a placeholder rendered as a
  // fact, and compared. It reads the real forwarded peer id, or an honest
  // absence -- and is never claimed as compared either way.
  it('finding 3: served by reads the real forwarded peer id on an ASKED row, "not recorded" when absent, and is never compared', () => {
    const withPeerId = buildCommitsToRows(
      paneCRow({
        role_tag: 'ASKED',
        theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-node-3' }
      }),
      LOCAL_RECORD
    )
    expect(withPeerId.find((r) => r.label === 'served by')?.yours).toBe('peer-node-3')
    expect(withPeerId.find((r) => r.label === 'served by')?.theirs).toBeNull()

    const withoutPeerId = buildCommitsToRows(
      paneCRow({ role_tag: 'ASKED', theirs: { state: 'absent', capsule_id: null } }),
      LOCAL_RECORD
    )
    expect(withoutPeerId.find((r) => r.label === 'served by')?.yours).toBe('not recorded')
    expect(withoutPeerId.find((r) => r.label === 'served by')?.yours).not.toBe('counterparty')
  })

  it('a SERVED row honestly names this node -- a real fact, not a placeholder', () => {
    const rows = buildCommitsToRows(paneCRow({ role_tag: 'SERVED' }), LOCAL_RECORD)
    expect(rows.find((r) => r.label === 'served by')?.yours).toBe('this node')
  })

  it('task binding never claims a theirs comparison -- no peer-record equivalent exists', () => {
    const theirsRecompute: PeerRecomputeState = {
      status: 'found',
      idMatch: true,
      signatureOk: true,
      peerRecord: { effect: { request_digest: 'a'.repeat(64) } },
      fetch: () => {}
    }
    const rows = buildCommitsToRows(paneCRow(), LOCAL_RECORD, theirsRecompute)
    expect(rows.find((r) => r.label === 'task binding')?.theirs).toBeNull()
  })
})

describe('buildChecksRows — a pushed record the badge judged fills THEIRS', () => {
  it('their id is redone in this browser (checked here); their signature was checked by this node (node says)', () => {
    const row = {
      exchange_key: 'exch-pushed',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: null,
      has_issue: false,
      mine: fixtureMineCell(),
      theirs: fixtureTheirsCell('agrees'),
      unilateral: false,
      timestamp: '2026-09-28T00:00:00Z'
    } as PaneCRow
    const rows = buildChecksRows(row, { idMatch: true, signatureOk: true }, undefined, { gateKind: 'closed' })
    const content = rows.find((r) => r.key === 'content_binding')?.theirs
    const signature = rows.find((r) => r.key === 'producer_signature')?.theirs
    expect(content).toMatchObject({ state: 'PASS', recomputed: true })
    expect(signature).toMatchObject({ state: 'PASS', recomputed: false })
    expect(signature?.detail).toMatch(/checked by this node when it arrived/)
  })
})

describe('buildCommitsToRows — served by, from our own record (item 5)', () => {
  const PROVIDER = 'a1'.repeat(32)
  const recordServedByProvider = (id: string | null) =>
    ({
      model_attestation: {
        compute_attestation: { 'x-mesh-poc-v1': { serving_provenance: id === null ? {} : { served_by_node_id: id } } }
      }
    }) as never
  // A payer's CLOSED row: their half is held, so the pane sends no `theirs.peer_id`.
  const closedPayerRow = (mineRecord?: Record<string, unknown>) =>
    paneCRow({
      role_tag: 'ASKED',
      mine: { state: 'present', capsule_id: 'm', ...(mineRecord ? { record: mineRecord } : {}) },
      theirs: { state: 'present', capsule_id: 't', signature_ok: true, received_from: 'node-b' }
    })
  const servedBy = (rows: ReturnType<typeof buildCommitsToRows>) => rows.find((r) => r.label === 'served by')?.yours

  it("MUTANT-GUARD: a payer's CLOSED row shows the provider its record names", () => {
    expect(servedBy(buildCommitsToRows(closedPayerRow(), recordServedByProvider(PROVIDER)))).toBe(PROVIDER)
  })

  it('reads the row\'s own record when the view holds no local record', () => {
    const mine = recordServedByProvider(PROVIDER) as unknown as Record<string, unknown>
    expect(servedBy(buildCommitsToRows(closedPayerRow(mine), null))).toBe(PROVIDER)
  })

  it('a record that names no server, or the producer\'s "unknown", is still "not recorded"', () => {
    expect(servedBy(buildCommitsToRows(closedPayerRow(), recordServedByProvider(null)))).toBe('not recorded')
    expect(servedBy(buildCommitsToRows(closedPayerRow(), recordServedByProvider('unknown')))).toBe('not recorded')
    expect(servedBy(buildCommitsToRows(closedPayerRow(), recordServedByProvider('')))).toBe('not recorded')
  })

  it("falls back to the pane's join key only when no record is held, and a served row is this node", () => {
    const fetchable = paneCRow({ role_tag: 'ASKED', theirs: { state: 'NOT_CHECKED', capsule_id: 't', peer_id: 'node-x' } })
    expect(servedBy(buildCommitsToRows(fetchable, null))).toBe('node-x')
    expect(servedBy(buildCommitsToRows(paneCRow({ role_tag: 'SERVED' }), recordServedByProvider(PROVIDER)))).toBe(
      'this node'
    )
  })

  it('our record wins over the pane\'s join key when they name different nodes', () => {
    const row = paneCRow({
      role_tag: 'ASKED',
      mine: { state: 'present', capsule_id: 'm' },
      theirs: { state: 'NOT_CHECKED', capsule_id: 't', peer_id: 'node-other' }
    })
    expect(servedBy(buildCommitsToRows(row, recordServedByProvider(PROVIDER)))).toBe(PROVIDER)
  })

  it('a row with no record of ours never reads theirs: "not recorded"', () => {
    const receivedOnly = paneCRow({
      role_tag: 'ASKED',
      mine: { state: 'absent', capsule_id: null },
      theirs: {
        state: 'present',
        capsule_id: 't',
        record: recordServedByProvider('b2'.repeat(32)) as unknown as Record<string, unknown>
      } as never
    })
    expect(servedBy(buildCommitsToRows(receivedOnly, null))).toBe('not recorded')
  })

  it('recordServedBy follows the pane\'s rule', () => {
    expect(recordServedBy(recordServedByProvider(PROVIDER))).toBe(PROVIDER)
    expect(recordServedBy(recordServedByProvider('unknown'))).toBeNull()
    expect(recordServedBy(null)).toBeNull()
  })
})

describe('buildCommitsToRows — the TDX quote (u229)', () => {
  const rowsFor = (overrides: Partial<PaneCRow>) =>
    buildCommitsToRows(paneCRow(overrides), null).filter((r) => r.label.startsWith('TEE'))

  it('shows no TEE rows when neither record says anything about a quote', () => {
    expect(rowsFor({})).toEqual([])
  })

  it('a quote bound to our record: the binding on one line, the unchecked platform on another', () => {
    const rows = rowsFor({ mine: { state: 'present', capsule_id: 'm', tee: { state: 'bound' } } })
    expect(rows.map((r) => r.label)).toEqual(['TEE quote', 'TEE platform'])
    expect(rows[0].yours).toBe(TEE_COPY.bound)
    expect(rows[1].yours).toBe(TEE_COPY.platform)
  })

  it("their held record's quote shows in their column, checked on this node", () => {
    const rows = rowsFor({
      mine: { state: 'present', capsule_id: 'm', tee: { state: 'absent' } },
      theirs: { state: 'present', capsule_id: 't', tee: { state: 'bound' } }
    })
    expect(rows[0].yours).toBe(TEE_COPY.absent)
    expect(rows[0].theirs).toEqual({ value: TEE_COPY.bound, note: 'checked on this node' })
    const platform = rows.find((r) => r.label === 'TEE platform')
    expect(platform?.yours).toBe('—')
    expect(platform?.theirs).toEqual({ value: TEE_COPY.platform, note: 'left to you' })
  })

  it('MUTANT-GUARD: a quote that fails a check is said to fail, and no platform line implies a measurement', () => {
    const rows = rowsFor({ mine: { state: 'present', capsule_id: 'm', tee: { state: 'failed', check: 'report_data' } } })
    expect(rows.map((r) => r.label)).toEqual(['TEE quote'])
    expect(rows[0].yours).toBe(TEE_COPY.failed('report_data'))
    expect(rows[0].yours).toContain('report data')
    expect(rows[0].yours).toContain('not bound')
    expect(rows[0].yours).not.toBe(TEE_COPY.bound)
  })

  it('never says attested or verified about a TEE, and says the signatures are not checked', () => {
    const rows = rowsFor({
      mine: { state: 'present', capsule_id: 'm', tee: { state: 'bound' } },
      theirs: { state: 'present', capsule_id: 't', tee: { state: 'failed', check: 'digest' } }
    })
    const shown = rows.flatMap((r) => [r.label, r.yours, r.theirs?.value ?? '', r.theirs?.note ?? ''])
    const words = [...shown, TEE_COPY.absent].join(' ').toLowerCase()
    expect(words).not.toMatch(/attest|verif/)
    expect(TEE_COPY.bound).toContain('signed by a TDX quoting enclave')
    expect(TEE_COPY.platform).toBe('Platform status, revocation and measurements not checked here')
  })
})

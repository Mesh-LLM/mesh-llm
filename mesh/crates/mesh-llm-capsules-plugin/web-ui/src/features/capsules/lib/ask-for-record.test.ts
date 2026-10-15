import { ed25519 } from '@noble/curves/ed25519'
import { describe, expect, it } from 'vitest'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import {
  FIXTURE_PROVIDER_NODE,
  FIXTURE_REQUEST_DIGEST,
  FIXTURE_RESPONSE_DIGEST,
  fixtureHalfBody
} from '@/features/capsules/lib/pushed-half-fixtures'
import { ASK_FOR_RECORD_AFTER_MS } from '@/features/capsules/lib/exchange-row-state'
import type { EvidenceVerification } from '@/features/capsules/api/evidenceRequestClient'
import {
  askIsOffered,
  askReplyNote,
  askTarget,
  HOST_SERVED_NO_NONCE,
  judgeAskReply,
  producerSignatureVerifies,
  refusalSigningBody,
  stateAfterAsk,
  type AskJudges
} from './ask-for-record'

const ASKED_AT = '2026-09-28T21:00:00Z'
const NONCE = 'nonce-of-the-exchange'

function ourRequestedRecord(): Record<string, unknown> {
  const body = fixtureHalfBody({ capsuleId: 'ours-1' })
  const poc = (body.model_attestation as Record<string, Record<string, Record<string, unknown>>>).compute_attestation[
    'x-mesh-poc-v1'
  ]
  poc.role = 'requested'
  poc.client_nonce = NONCE
  return body
}

/** A row still waiting for their record: known peer, nothing received. */
function waitingRow(): PaneCRow {
  return {
    exchange_key: `digest:${FIXTURE_REQUEST_DIGEST}`,
    mine: { state: 'present', record: ourRequestedRecord() },
    theirs: { state: 'NOT_CHECKED', capsule_id: 'capsule-chatcmpl-1', peer_id: FIXTURE_PROVIDER_NODE }
  } as unknown as PaneCRow
}

const hex = (bytes: Uint8Array) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')

/** The peer's own key, as this node was told it (`announced_key_id`). */
const PEER_SECRET = ed25519.utils.randomSecretKey()
const ANNOUNCED = hex(ed25519.getPublicKey(PEER_SECRET))
/** The digest of the request bytes the plugin sent. */
const SENT = 'f'.repeat(64)

function signedRefusal(reason: string, { secret = PEER_SECRET, requestDigest = SENT } = {}) {
  const refusal = { request_digest: requestDigest, reason, issued_at: '2026-09-28T21:00:05Z' }
  const sig = ed25519.sign(refusalSigningBody(refusal), secret)
  return { ...refusal, key_id: hex(ed25519.getPublicKey(secret)), sig: hex(sig) }
}

/** The tool's reply: the peer's answer, and what the plugin's verification
 *  found (by default, what a verified answer of that shape reads as). */
function answer(
  body: Record<string, unknown>,
  announcedKeyId: string | null = ANNOUNCED,
  verification?: EvidenceVerification
) {
  const found: EvidenceVerification =
    verification ??
    (typeof body.reason === 'string'
      ? { state: 'refusal', reason: body.reason }
      : { state: 'artifact', records: artifactLeafIndices(body) })
  return { kind: 'answer' as const, answer: body, verification: found, requestDigest: SENT, announcedKeyId }
}

/** An artifact answer carrying `records` (the -00 wire the plugin serves:
 *  the artifact as exact JSON text, each record's body hex-encoded). */
function artifact(records: Record<string, unknown>[]): Record<string, unknown> {
  const served = records.map((record, i) => ({
    leaf_index: i,
    digest: 'd'.repeat(64),
    body: hex(new TextEncoder().encode(JSON.stringify(record)))
  }))
  return {
    artifact: JSON.stringify({ anchor: 'a'.repeat(64), evidence_stream: 'log', subject: { correlation: NONCE }, records: served }),
    material: '{}',
    envelope: {}
  }
}

function artifactLeafIndices(body: Record<string, unknown>): number[] {
  const parsed = JSON.parse(String(body.artifact)) as { records: { leaf_index: number }[] }
  return parsed.records.map((r) => r.leaf_index)
}

/** A COSE_Sign1 (EdDSA, attached payload) over `payload`, as the producer
 *  signs a record's `capsule_id`. */
function coseSign1(payload: Uint8Array, secret: Uint8Array): string {
  const protectedHeader = Uint8Array.of(0xa1, 0x01, 0x27)
  const sigStructure = Uint8Array.of(
    0x84,
    0x6a,
    ...new TextEncoder().encode('Signature1'),
    0x43,
    ...protectedHeader,
    0x40,
    0x58,
    payload.length,
    ...payload
  )
  const sig = ed25519.sign(sigStructure, secret)
  return hex(
    Uint8Array.of(0xd2, 0x84, 0x43, ...protectedHeader, 0xa0, 0x58, payload.length, ...payload, 0x58, 0x40, ...sig)
  )
}

function signedRecord(secret: Uint8Array): Record<string, unknown> {
  const id = new Uint8Array(32).fill(7)
  return { capsule_id: hex(id), key_id: hex(ed25519.getPublicKey(secret)), signature: coseSign1(id, secret) }
}

const trustingJudges: AskJudges = {
  recomputeIdMatch: async () => true,
  producerSignatureVerifies: () => true
}

describe('askTarget', () => {
  const DIGESTS = { request: FIXTURE_REQUEST_DIGEST, response: FIXTURE_RESPONSE_DIGEST }
  const poc = (record: Record<string, unknown>) =>
    (record.model_attestation as Record<string, Record<string, Record<string, unknown>>>).compute_attestation[
      'x-mesh-poc-v1'
    ]

  it('names the node that served our request, our nonce and both digests', () => {
    expect(askTarget(ourRequestedRecord())).toEqual({ peerId: FIXTURE_PROVIDER_NODE, nonce: NONCE, digests: DIGESTS })
  })

  it('never names the exchange by the placeholder nonce every host-served half shares', () => {
    const placeholder = ourRequestedRecord()
    poc(placeholder).client_nonce = HOST_SERVED_NO_NONCE
    expect(askTarget(placeholder)).toEqual({ peerId: FIXTURE_PROVIDER_NODE, nonce: null, digests: DIGESTS })
  })

  it('offers nothing without a full peer id, or without a nonce or both digests', () => {
    const noNonce = ourRequestedRecord()
    delete poc(noNonce).client_nonce
    expect(askTarget(noNonce)).toEqual({ peerId: FIXTURE_PROVIDER_NODE, nonce: null, digests: DIGESTS })
    delete (noNonce.effect as Record<string, unknown>).response_digest
    expect(askTarget(noNonce)).toBeNull()
    expect(askTarget(fixtureHalfBody({ servedBy: 'unknown' }))).toBeNull()
  })
})

describe('askIsOffered', () => {
  const target = { peerId: FIXTURE_PROVIDER_NODE, nonce: NONCE, digests: null }
  const at = '2026-09-28T20:00:00Z'
  const now = Date.parse(at)

  it('waits for the timeout, then offers the ask on a row whose record has not arrived', () => {
    expect(askIsOffered('open_not_given', target, at, now + ASK_FOR_RECORD_AFTER_MS - 1)).toBe(false)
    expect(askIsOffered('open_not_given', target, at, now + ASK_FOR_RECORD_AFTER_MS)).toBe(true)
    expect(askIsOffered('open_not_held', target, at, now + ASK_FOR_RECORD_AFTER_MS)).toBe(true)
  })

  it('never offers it on a closed or contradicted row, or with no one to ask', () => {
    const late = now + ASK_FOR_RECORD_AFTER_MS
    expect(askIsOffered('closed', target, at, late)).toBe(false)
    expect(askIsOffered('contradicted', target, at, late)).toBe(false)
    expect(askIsOffered('open_not_given', null, at, late)).toBe(false)
  })
})

describe('judgeAskReply', () => {
  it('reads a signed no_such_subject as their signed statement that they have no record', async () => {
    const outcome = await judgeAskReply(answer(signedRefusal('no_such_subject')), null, ASKED_AT)
    expect(outcome).toEqual({ kind: 'no_record', at: '2026-09-28T21:00:05Z' })
    expect(stateAfterAsk(waitingRow(), outcome, null)).toEqual({ kind: 'open_absent', date: '2026-09-28T21:00:05Z' })
  })

  it('reads any other signed reason as a signed decline', async () => {
    const outcome = await judgeAskReply(answer(signedRefusal('policy_declined')), null, ASKED_AT)
    expect(outcome).toEqual({ kind: 'refused', at: '2026-09-28T21:00:05Z', reason: 'policy_declined' })
    expect(stateAfterAsk(waitingRow(), outcome, null).kind).toBe('open_refused')
  })

  it('reads a signed coverage lag as not yet provable: the row stays asked and can ask again', async () => {
    const outcome = await judgeAskReply(answer(signedRefusal('coverage_unsatisfiable')), null, ASKED_AT)
    expect(outcome.kind).toBe('no_reply')
    expect(stateAfterAsk(waitingRow(), outcome, null)).toEqual({ kind: 'open_asked', date: ASKED_AT })
  })

  it('never takes a refusal whose signature does not verify', async () => {
    const forged = { ...signedRefusal('no_such_subject'), reason: 'policy_declined' }
    const outcome = await judgeAskReply(answer(forged), null, ASKED_AT)
    expect(outcome.kind).toBe('no_reply')
    expect(stateAfterAsk(waitingRow(), outcome, null)).toEqual({ kind: 'open_asked', date: ASKED_AT })
  })

  // Adversarial read: a refusal must be theirs, and about this ask.
  it('never takes a refusal signed with a throwaway key instead of their announced one', async () => {
    const throwaway = signedRefusal('no_such_subject', { secret: ed25519.utils.randomSecretKey() })
    const outcome = await judgeAskReply(answer(throwaway), null, ASKED_AT)
    expect(outcome).toMatchObject({ kind: 'no_reply', detail: 'their reply is not signed with the key they announced' })
    expect(stateAfterAsk(waitingRow(), outcome, null).kind).toBe('open_asked')
  })

  it('never takes a refusal replayed from a different request', async () => {
    const replayed = signedRefusal('no_such_subject', { requestDigest: 'e'.repeat(64) })
    const outcome = await judgeAskReply(answer(replayed), null, ASKED_AT)
    expect(outcome).toMatchObject({ kind: 'no_reply', detail: 'their reply answers a different request' })
  })

  it('takes nothing as their refusal when this node has no announced key for them', async () => {
    const outcome = await judgeAskReply(answer(signedRefusal('no_such_subject'), null), null, ASKED_AT)
    expect(outcome.kind).toBe('no_reply')
  })

  it('takes nothing the plugin could not verify, and says why', async () => {
    const theirs = { ...fixtureHalfBody({ capsuleId: 'theirs-1' }), signature: 'aa', key_id: 'bb' }
    const outcome = await judgeAskReply(
      answer(artifact([theirs]), ANNOUNCED, { state: 'not_evidence', why: 'a proof does not verify against the anchor' }),
      FIXTURE_REQUEST_DIGEST,
      ASKED_AT,
      trustingJudges
    )
    expect(outcome).toEqual({
      kind: 'no_reply',
      at: ASKED_AT,
      detail: 'their reply did not verify: a proof does not verify against the anchor'
    })
    expect(stateAfterAsk(waitingRow(), outcome, ourRequestedRecord() as never).kind).toBe('open_asked')
    expect(askReplyNote(outcome)).toEqual({
      verified: false,
      text: 'Their reply is not verified: their reply did not verify: a proof does not verify against the anchor'
    })
  })

  it('takes nothing when the plugin had no announced key to check the reply under', async () => {
    const outcome = await judgeAskReply(
      answer(signedRefusal('no_such_subject'), null, { state: 'no_announced_key' }),
      null,
      ASKED_AT
    )
    expect(outcome).toMatchObject({ kind: 'no_reply' })
    expect(askReplyNote(outcome)?.verified).toBe(false)
  })

  it('never takes a refusal whose reason differs from the one the plugin verified', async () => {
    const outcome = await judgeAskReply(
      answer(signedRefusal('no_such_subject'), ANNOUNCED, { state: 'refusal', reason: 'policy_declined' }),
      null,
      ASKED_AT
    )
    expect(outcome.kind).toBe('no_reply')
  })

  it('keeps the row asked, no reply yet, when the other side could not be reached', async () => {
    const outcome = await judgeAskReply({ kind: 'no_answer', message: 'peer unreachable' }, null, ASKED_AT)
    expect(stateAfterAsk(waitingRow(), outcome, null)).toEqual({ kind: 'open_asked', date: ASKED_AT })
  })

  it('says a verified reply is verified', async () => {
    const refusal = await judgeAskReply(answer(signedRefusal('no_such_subject')), null, ASKED_AT)
    expect(askReplyNote(refusal)?.verified).toBe(true)
    const theirs = { ...fixtureHalfBody({ capsuleId: 'theirs-1' }), signature: 'aa', key_id: 'bb' }
    const record = await judgeAskReply(answer(artifact([theirs])), FIXTURE_REQUEST_DIGEST, ASKED_AT, trustingJudges)
    expect(askReplyNote(record)).toMatchObject({ verified: true })
    expect(askReplyNote({ kind: 'asking', at: ASKED_AT })).toBeNull()
  })

  it('says a record is verified only when its id recomputes and its signature holds', async () => {
    const theirs = { ...fixtureHalfBody({ capsuleId: 'theirs-1' }), signature: 'aa', key_id: 'bb' }
    const judged = (judges: AskJudges) =>
      judgeAskReply(answer(artifact([theirs])), FIXTURE_REQUEST_DIGEST, ASKED_AT, judges)
    expect(askReplyNote(await judged(trustingJudges))?.verified).toBe(true)
    expect(askReplyNote(await judged({ ...trustingJudges, producerSignatureVerifies: () => false }))).toEqual({
      verified: false,
      text: 'Their record is in their log, but it is not verified: it is not signed with their announced key'
    })
    expect(askReplyNote(await judged({ ...trustingJudges, recomputeIdMatch: async () => false }))).toEqual({
      verified: false,
      text: 'Their record is in their log, but it is not verified: its id does not recompute'
    })
    expect(askReplyNote(await judged({ ...trustingJudges, recomputeIdMatch: async () => null }))?.verified).toBe(false)
  })

  it('reads only the records the plugin proved', async () => {
    const theirs = { ...fixtureHalfBody({ capsuleId: 'theirs-1' }), signature: 'aa', key_id: 'bb' }
    const outcome = await judgeAskReply(
      answer(artifact([theirs]), ANNOUNCED, { state: 'artifact', records: [] }),
      FIXTURE_REQUEST_DIGEST,
      ASKED_AT,
      trustingJudges
    )
    expect(outcome).toMatchObject({ kind: 'no_reply', detail: 'their reply carried no record' })
  })

  // Asked by the exchange's digests, a node can answer with more than one
  // record carrying them: the half it served and, say, a half it requested.
  function theirHalf(capsuleId: string, role: 'served' | 'requested'): Record<string, unknown> {
    const body = fixtureHalfBody({ capsuleId })
    ;(body.model_attestation as Record<string, Record<string, Record<string, unknown>>>).compute_attestation[
      'x-mesh-poc-v1'
    ].role = role
    return { ...body, signature: 'aa', key_id: 'bb' }
  }

  it('prefers the half they served among records with our digests', async () => {
    const served = theirHalf('theirs-served', 'served')
    const requested = theirHalf('theirs-requested', 'requested')
    const outcome = await judgeAskReply(
      answer(artifact([requested, served])),
      FIXTURE_REQUEST_DIGEST,
      ASKED_AT,
      trustingJudges
    )
    expect(outcome).toMatchObject({ kind: 'record', evidence: { peerRecord: { capsule_id: 'theirs-served' } } })
  })

  it('never picks one of several records that could be theirs by order: the reply is ambiguous', async () => {
    const one = theirHalf('theirs-1', 'served')
    const two = theirHalf('theirs-2', 'served')
    const outcome = await judgeAskReply(answer(artifact([one, two])), FIXTURE_REQUEST_DIGEST, ASKED_AT, trustingJudges)
    expect(outcome).toEqual({ kind: 'ambiguous', at: ASKED_AT, count: 2 })
    expect(askReplyNote(outcome)).toEqual({
      verified: false,
      text: 'Their reply carried 2 records that could be theirs for this exchange; none is shown as theirs'
    })
    expect(stateAfterAsk(waitingRow(), outcome, null)).toEqual({ kind: 'open_asked', date: ASKED_AT })
  })

  it('closes the row through the gate when their record verifies and cites our half', async () => {
    const theirs = { ...fixtureHalfBody({ capsuleId: 'theirs-1' }), signature: 'aa', key_id: 'bb' }
    const outcome = await judgeAskReply(
      answer(artifact([theirs])),
      FIXTURE_REQUEST_DIGEST,
      ASKED_AT,
      trustingJudges
    )
    expect(outcome.kind).toBe('record')
    const ours = ourRequestedRecord()
    expect(stateAfterAsk(waitingRow(), outcome, ours as never).kind).toBe('closed')
  })

  it('never closes the row on a record whose signature does not verify', async () => {
    const theirs = { ...fixtureHalfBody({ capsuleId: 'theirs-1' }), signature: 'aa', key_id: 'bb' }
    const outcome = await judgeAskReply(
      answer(artifact([theirs])),
      FIXTURE_REQUEST_DIGEST,
      ASKED_AT,
      { ...trustingJudges, producerSignatureVerifies: () => false }
    )
    expect(stateAfterAsk(waitingRow(), outcome, ourRequestedRecord() as never).kind).not.toBe('closed')
  })

  it('reads their record with a different answer as a disagreement', async () => {
    const theirs = {
      ...fixtureHalfBody({ capsuleId: 'theirs-1', responseDigest: 'e'.repeat(64) }),
      signature: 'aa',
      key_id: 'bb'
    }
    const outcome = await judgeAskReply(
      answer(artifact([theirs])),
      FIXTURE_REQUEST_DIGEST,
      ASKED_AT,
      trustingJudges
    )
    expect(stateAfterAsk(waitingRow(), outcome, ourRequestedRecord() as never).kind).toBe('contradicted')
  })
})

describe('judgeAskReply: their record under the announced key', () => {
  const realSignature: AskJudges = { recomputeIdMatch: async () => true, producerSignatureVerifies }

  it('counts a fetched record as signed only under the key they announced', async () => {
    for (const [secret, expected] of [
      [PEER_SECRET, true],
      [ed25519.utils.randomSecretKey(), false]
    ] as const) {
      const outcome = await judgeAskReply(
        answer(artifact([signedRecord(secret)])),
        null,
        ASKED_AT,
        realSignature
      )
      expect(outcome.kind === 'record' && outcome.evidence.status === 'found' && outcome.evidence.signatureOk).toBe(
        expected
      )
    }
  })
})

describe('producerSignatureVerifies', () => {
  it('holds for a record signed with their announced key', () => {
    expect(producerSignatureVerifies(signedRecord(PEER_SECRET), ANNOUNCED)).toBe(true)
  })

  it('fails for a record validly signed with any other key, or with no announced key', () => {
    const throwaway = signedRecord(ed25519.utils.randomSecretKey())
    expect(producerSignatureVerifies(throwaway, ANNOUNCED)).toBe(false)
    expect(producerSignatureVerifies(signedRecord(PEER_SECRET), null)).toBe(false)
  })

  it('fails when the envelope does not sign the record’s own id', () => {
    const record = signedRecord(PEER_SECRET)
    expect(producerSignatureVerifies({ ...record, capsule_id: '08'.repeat(32) }, ANNOUNCED)).toBe(false)
  })
})

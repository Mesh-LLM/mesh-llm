// "Ask them for their record" (UX §3): after the timeout, a row whose other
// side's record hasn't arrived can ask that node for it. The plugin sends one
// evidence request over the mesh and verifies the reply before handing it
// over (`api/evidenceRequestClient.ts`); this module judges what came back,
// the same way the row judges a record that was pushed: nothing here is
// trusted because it arrived.
//
//   - only a reply the plugin verified counts (`verification`): a refusal
//     signed with the key this node was told the peer signs with, naming OUR
//     request, or an artifact whose records are proven in the peer's log
//     under a checkpoint that key signed. Anything else -- no announced key,
//     a reply that proves nothing -- leaves the row "asked, no reply yet",
//     and the row says why (`askReplyNote`);
//   - a refusal is also checked here, in the browser, under the announced
//     key and the request digest the plugin sent. `no_such_subject` is their
//     signed "I have no record of this"; `coverage_unsatisfiable` means they
//     hold it but no checkpoint covers it yet, so the row stays asked and can
//     ask again; any other reason is a signed decline;
//   - a record goes through THE row gate (`deriveRightCellState`) as a fetched
//     half: it closes the row only if its id recomputes, its signature
//     verifies under the announced key, it came from the node that served us
//     and both digests equal ours.
import { ed25519 } from '@noble/curves/ed25519'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import type { AskSubject, EvidenceAskReply } from '@/features/capsules/api/evidenceRequestClient'
import { verifyCoseSign1 } from '@/features/capsules/lib/cose'
import {
  askForRecordIsDue,
  deriveRightCellState,
  type RightCellState,
  type RightCellStateKind
} from '@/features/capsules/lib/exchange-row-state'
import { recomputeIdMatch, type PeerRecomputeState } from '@/features/capsules/lib/recompute-identity'

/** What asking produced for one row. `at` is when this node asked or when the
 *  other side signed its reply. */
export type AskOutcome =
  | { kind: 'asking'; at: string }
  | { kind: 'no_reply'; at: string; detail: string }
  | { kind: 'refused'; at: string; reason: string }
  | { kind: 'no_record'; at: string }
  | { kind: 'record'; at: string; evidence: PeerRecomputeState }
  /** Their verified reply carried more than one record that could be their
   *  half of this exchange, and nothing picks one: none is shown as theirs. */
  | { kind: 'ambiguous'; at: string; count: number }

/** Whom to ask and how to name the exchange, read from OUR record of it: the
 *  node that served our request (or that asked us, on a row we served), the
 *  client nonce our record carries, and the exchange's two `effect` digests.
 *  `null` when the node is unknown, or when neither the nonce nor both
 *  digests are. */
export type AskTarget = AskSubject & { peerId: string }

/** The nonce a host-served record carries when the host forwarded none
 *  (`evidence_log::HOST_SERVED_NO_NONCE`). Every such record carries it, so
 *  it names no exchange. */
export const HOST_SERVED_NO_NONCE = 'host-served-no-nonce'

const FULL_ID = /^[0-9a-f]{64}$/
/** Their signed "I hold no such record" (the -00 registry token). */
const NO_SUCH_SUBJECT = 'no_such_subject'
/** They hold the record but no checkpoint covers it yet: a lag, never a
 *  decline. Asking again later can succeed. */
const COVERAGE_LAG = 'coverage_unsatisfiable'

function obj(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function str(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null
}

function pocBlock(record: unknown): Record<string, unknown> | null {
  return obj(obj(obj(obj(record)?.model_attestation)?.compute_attestation)?.['x-mesh-poc-v1'])
}

export function askTarget(record: CapsuleRecord | Record<string, unknown> | null | undefined): AskTarget | null {
  const poc = pocBlock(record)
  const provenance = obj(poc?.serving_provenance)
  const recordedNonce = str(poc?.client_nonce)
  const nonce = recordedNonce === HOST_SERVED_NO_NONCE ? null : recordedNonce
  const effect = obj(obj(record)?.effect)
  const request = str(effect?.request_digest)?.toLowerCase()
  const response = str(effect?.response_digest)?.toLowerCase()
  const digests = request && response && FULL_ID.test(request) && FULL_ID.test(response) ? { request, response } : null
  const other = poc?.role === 'served' ? str(provenance?.requested_by_node_id) : str(provenance?.served_by_node_id)
  const peerId = other?.toLowerCase() ?? null
  return peerId && FULL_ID.test(peerId) && (nonce || digests) ? { peerId, nonce, digests } : null
}

/** The row states whose record hasn't arrived, where asking can help. */
const ASKABLE_KINDS: ReadonlySet<RightCellStateKind> = new Set([
  'open_not_held',
  'open_not_given',
  'open_not_asked',
  'open_asked'
])

/** The ask is offered once the exchange is past the timeout, on a row still
 *  waiting for their record, when we know whom to ask and how. */
export function askIsOffered(
  kind: RightCellStateKind,
  target: AskTarget | null,
  timestamp: string | null,
  nowMs: number
): boolean {
  return ASKABLE_KINDS.has(kind) && target !== null && askForRecordIsDue(timestamp, nowMs)
}

function hexBytes(hex: string): Uint8Array | null {
  if (!/^([0-9a-f]{2})+$/i.test(hex)) return null
  const bytes = new Uint8Array(hex.length / 2)
  for (let i = 0; i < bytes.length; i++) bytes[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16)
  return bytes
}

/** The bytes a refusal's signature covers: its three signed fields as JSON
 *  with sorted keys and no spaces (`capsule_emit.evidence_request.Refusal`). */
export function refusalSigningBody(refusal: { request_digest: string; reason: string; issued_at: string }): Uint8Array {
  const body = JSON.stringify({
    issued_at: refusal.issued_at,
    reason: refusal.reason,
    request_digest: refusal.request_digest
  })
  return new TextEncoder().encode(body)
}

/** The announced key's bytes, when the item names that same key. */
function announcedKeyBytes(namedKeyId: unknown, announcedKeyId: string | null): Uint8Array | null {
  const named = str(namedKeyId)?.toLowerCase()
  if (!announcedKeyId || named !== announcedKeyId.toLowerCase()) return null
  const key = hexBytes(announcedKeyId)
  return key && key.length === 32 ? key : null
}

/** Why a refusal proves nothing, or `null` when it holds: signed under the
 *  announced key, over OUR request. */
function refusalProblem(
  refusal: Record<string, unknown>,
  announcedKeyId: string | null,
  sentRequestDigest: string
): string | null {
  if (!announcedKeyId) return NO_KEY_DETAIL
  const key = announcedKeyBytes(refusal.key_id, announcedKeyId)
  if (!key) return 'their reply is not signed with the key they announced'
  const requestDigest = str(refusal.request_digest)
  if (requestDigest?.toLowerCase() !== sentRequestDigest.toLowerCase()) {
    return 'their reply answers a different request'
  }
  const sig = hexBytes(str(refusal.sig) ?? '')
  const reason = str(refusal.reason)
  const issuedAt = str(refusal.issued_at)
  if (!sig || !reason || !issuedAt) return 'their reply is not signed with the key they announced'
  try {
    const body = refusalSigningBody({ request_digest: requestDigest, reason, issued_at: issuedAt })
    return ed25519.verify(sig, body, key) ? null : 'their reply is not signed with the key they announced'
  } catch {
    return 'their reply is not signed with the key they announced'
  }
}

/** A record's producer signature: a COSE_Sign1 over its own 32-byte
 *  `capsule_id`, under the key this node was told the peer signs with. A
 *  record naming any other key, or a peer with no announced key, fails. */
export function producerSignatureVerifies(record: Record<string, unknown>, announcedKeyId: string | null): boolean {
  const envelope = hexBytes(str(record.signature) ?? '')
  const key = announcedKeyBytes(record.key_id, announcedKeyId)
  const id = hexBytes(str(record.capsule_id) ?? '')
  if (!envelope || !key || !id) return false
  const result = verifyCoseSign1(envelope, key)
  return result.verified && result.payload.length === id.length && result.payload.every((b, i) => b === id[i])
}

export type AskJudges = {
  recomputeIdMatch: (record: Record<string, unknown>, capsuleId: string | null) => Promise<boolean | null>
  producerSignatureVerifies: (record: Record<string, unknown>, announcedKeyId: string | null) => boolean
}

const REAL_JUDGES: AskJudges = { recomputeIdMatch, producerSignatureVerifies }

/** The records an artifact answer carries, decoded: each served record's
 *  body is its line in the peer's ledger, hex-encoded, inside the artifact's
 *  exact JSON text. Only the leaf indices the verification proved are kept. */
function artifactRecords(answer: Record<string, unknown>, proven: readonly number[]): Record<string, unknown>[] {
  let artifact: unknown
  try {
    artifact = typeof answer.artifact === 'string' ? JSON.parse(answer.artifact) : null
  } catch {
    return []
  }
  const served = obj(artifact)?.records
  if (!Array.isArray(served)) return []
  const out: Record<string, unknown>[] = []
  for (const item of served) {
    const entry = obj(item)
    if (!entry || typeof entry.leaf_index !== 'number' || !proven.includes(entry.leaf_index)) continue
    const bytes = hexBytes(str(entry.body) ?? '')
    if (!bytes) continue
    try {
      const parsed = obj(JSON.parse(new TextDecoder().decode(bytes)))
      if (parsed) out.push(parsed)
    } catch {
      // not a record: skipped
    }
  }
  return out
}

/** Judge the other side's reply for the exchange whose request digest is
 *  `requestDigest`. */
export async function judgeAskReply(
  reply: EvidenceAskReply,
  requestDigest: string | null,
  askedAt: string,
  judges: AskJudges = REAL_JUDGES
): Promise<AskOutcome> {
  if (reply.kind === 'no_answer') return { kind: 'no_reply', at: askedAt, detail: reply.message }
  const answer = obj(reply.answer)
  const verification = reply.verification
  switch (verification.state) {
    case 'no_announced_key':
      return { kind: 'no_reply', at: askedAt, detail: NO_KEY_DETAIL }
    case 'not_evidence':
      return { kind: 'no_reply', at: askedAt, detail: `their reply did not verify: ${verification.why}` }
    case 'unknown':
      return { kind: 'no_reply', at: askedAt, detail: 'their reply was not verified' }
    case 'refusal': {
      const problem = answer ? refusalProblem(answer, reply.announcedKeyId, reply.requestDigest) : 'no refusal'
      if (problem || answer?.reason !== verification.reason) {
        return { kind: 'no_reply', at: askedAt, detail: problem ?? 'their reply was not verified' }
      }
      const at = str(answer?.issued_at) ?? askedAt
      if (verification.reason === COVERAGE_LAG) {
        return { kind: 'no_reply', at: askedAt, detail: 'they hold the record but cannot prove it yet; ask again later' }
      }
      return verification.reason === NO_SUCH_SUBJECT
        ? { kind: 'no_record', at }
        : { kind: 'refused', at, reason: verification.reason }
    }
    case 'artifact': {
      const receipts = answer ? artifactRecords(answer, verification.records) : []
      const candidates = theirHalfCandidates(receipts, requestDigest)
      if (candidates.length === 0) return { kind: 'no_reply', at: askedAt, detail: 'their reply carried no record' }
      if (candidates.length > 1) return { kind: 'ambiguous', at: askedAt, count: candidates.length }
      const receipt = candidates[0]
      const idMatch = await judges.recomputeIdMatch(receipt, str(receipt.capsule_id))
      return {
        kind: 'record',
        at: askedAt,
        evidence: {
          status: 'found',
          idMatch,
          signatureOk: judges.producerSignatureVerifies(receipt, reply.announcedKeyId),
          peerRecord: receipt,
          fetch: () => {}
        }
      }
    }
  }
}

/** The records in their reply that could be their half of this exchange:
 *  those whose request digest is ours (all of them when none is), and among
 *  those the half they SERVED when there is one. One candidate is judged by
 *  the row gate; more than one is ambiguous, never resolved by order. */
export function theirHalfCandidates(
  receipts: readonly Record<string, unknown>[],
  requestDigest: string | null
): Record<string, unknown>[] {
  const matching = receipts.filter((candidate) => obj(candidate.effect)?.request_digest === requestDigest)
  const pool = matching.length > 0 ? matching : [...receipts]
  const served = pool.filter((candidate) => pocBlock(candidate)?.role === 'served')
  return served.length > 0 ? served : pool
}

const NO_KEY_DETAIL = 'this node has no announced key for them, so their reply cannot be checked'

/** The line a row shows once its ask is answered: whether the reply was
 *  verified, and if not, why. `null` while the ask is in flight. */
export function askReplyNote(outcome: AskOutcome): { verified: boolean; text: string } | null {
  switch (outcome.kind) {
    case 'asking':
      return null
    case 'no_reply':
      return { verified: false, text: `Their reply is not verified: ${outcome.detail}` }
    case 'refused':
    case 'no_record':
      return { verified: true, text: 'Their reply is verified: signed with their announced key, for this request' }
    case 'record': {
      // The plugin proved the record is in their log; the record itself is
      // verified only if its id recomputes AND it is signed with their
      // announced key. Anything less is said, never rounded up.
      const { idMatch, signatureOk } = outcome.evidence
      if (idMatch === true && signatureOk === true) {
        return {
          verified: true,
          text: 'Their reply is verified: the record is in their log, under a checkpoint signed with their announced key'
        }
      }
      const problems = [
        idMatch === true ? null : idMatch === false ? 'its id does not recompute' : 'its id could not be recomputed',
        signatureOk === true ? null : 'it is not signed with their announced key'
      ].filter((p): p is string => p !== null)
      return {
        verified: false,
        text: `Their record is in their log, but it is not verified: ${problems.join('; ')}`
      }
    }
    case 'ambiguous':
      return {
        verified: false,
        text: `Their reply carried ${outcome.count} records that could be theirs for this exchange; none is shown as theirs`
      }
  }
}

/** The row's right cell once it has asked. */
export function stateAfterAsk(
  row: PaneCRow,
  outcome: AskOutcome,
  localRecord: CapsuleRecord | null | undefined
): RightCellState {
  switch (outcome.kind) {
    case 'asking':
    case 'no_reply':
    case 'ambiguous':
      return { kind: 'open_asked', date: outcome.at }
    case 'refused':
      return { kind: 'open_refused', date: outcome.at }
    case 'no_record':
      return { kind: 'open_absent', date: outcome.at }
    case 'record':
      return deriveRightCellState(row, outcome.evidence, localRecord)
  }
}

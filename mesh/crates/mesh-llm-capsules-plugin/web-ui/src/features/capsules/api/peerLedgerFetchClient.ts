// piece 4: the browser-side call for piece 2's
// `ledger-fetch/1` bridge, reached through the plugin's own tool route
// (`tools/mesh_ledger_fetch`, which the console host serves at
// `/api/plugins/capsules/tools/mesh_ledger_fetch`) via the host's
// plugin-scoped fetch. The plugin's `mesh_ledger_fetch` tool answers with its
// own tagged `LedgerFetchResponse` JSON on success
// (`status: "found" | "not_found" | "not_authorized" | "archived" | "error"`) unchanged; a transport-level
// failure (peer unroutable, channel undeclared, tool not served by this plugin
// build, no reply) surfaces as a non-2xx from the host route instead, never a
// hang.
//
// **Witness-level recompute only.** This client carries bytes; it never
// signs, endorses, or judges them -- `recompute-identity.ts`'s in-browser
// recompute is what turns these bytes into a checked fact, same as it
// already does for `mine`.
import { pluginHost } from '@/plugin-host/host'

export const MESH_LEDGER_FETCH_ROUTE = 'tools/mesh_ledger_fetch'

export type PeerLedgerFetchOutcome =
  | { kind: 'found'; capsule: Record<string, unknown>; signedStatementB64: string; nodePubKeyPem: string }
  | { kind: 'not_found' }
  | { kind: 'error'; message: string }
  | { kind: 'transport_error'; message: string }

/** Base64-decodes the responder's `signed_statement_b64` into raw COSE_Sign1
 * bytes -- same shape `fetchSignedStatement` already returns for `mine`, so
 * `recompute-identity.ts` never has to know which side it came from. */
export function decodeSignedStatementB64(b64: string): Uint8Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/** Fetches ONE peer capsule over the mesh, honestly reporting every outcome
 * the wire actually returned -- a fetch that failed or found nothing must
 * never render as though it succeeded (this codebase's own "never fabricate"
 * discipline, e.g. `theirs_cell` in capsule_panes_native.rs). */
export async function fetchPeerLedgerCapsule(peerId: string, capsuleId: string): Promise<PeerLedgerFetchOutcome> {
  let response: Response
  try {
    response = await pluginHost().network.fetchPlugin(MESH_LEDGER_FETCH_ROUTE, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ peer_id: peerId, capsule_id: capsuleId })
    })
  } catch (error) {
    return { kind: 'transport_error', message: error instanceof Error ? error.message : 'network error' }
  }

  if (!response.ok) {
    // The host route reports a plugin-tool error (malformed args, an
    // unroutable peer, a timed-out mesh stream) as a non-2xx whose body is
    // the error text, not the tagged `LedgerFetchResponse` shape.
    const text = await response.text().catch(() => '')
    return { kind: 'transport_error', message: text || `HTTP ${response.status}` }
  }

  let body: unknown
  try {
    body = await response.json()
  } catch {
    return { kind: 'error', message: 'peer response was not valid JSON' }
  }

  if (body && typeof body === 'object' && 'status' in body) {
    const tagged = body as { status: string; [key: string]: unknown }
    if (
      tagged.status === 'found' &&
      typeof tagged.capsule === 'object' &&
      tagged.capsule !== null &&
      typeof tagged.signed_statement_b64 === 'string' &&
      typeof tagged.node_pub_key_pem === 'string'
    ) {
      return {
        kind: 'found',
        capsule: tagged.capsule as Record<string, unknown>,
        signedStatementB64: tagged.signed_statement_b64,
        nodePubKeyPem: tagged.node_pub_key_pem
      }
    }
    if (tagged.status === 'not_found') return { kind: 'not_found' }
    if (tagged.status === 'archived') {
      return { kind: 'error', message: 'the peer set this record aside when it started a new history' }
    }
    if (tagged.status === 'not_authorized') {
      return { kind: 'error', message: 'the peer declined to share this record' }
    }
    if (tagged.status === 'error') {
      return { kind: 'error', message: typeof tagged.message === 'string' ? tagged.message : 'peer reported an error' }
    }
  }
  return { kind: 'error', message: 'unrecognised response shape from peer' }
}

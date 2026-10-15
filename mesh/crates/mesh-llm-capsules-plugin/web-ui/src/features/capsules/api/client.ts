// Fetches the capsule ledger from this plugin's own routes, through the
// console host's plugin-scoped fetch (`host.network.fetchPlugin`). The page
// reads the ledger the plugin writes; it never joins into mesh-llm's own log
// store, and it never calls a host-private route.
//
// Ported from the fork tab, which read the same files through a fork-only host
// route (`/api/capsules/ledger/*`). Plugin HTTP routes answer JSON only, so
// the jsonl ledger, the PEM key, and the COSE bytes arrive wrapped in JSON
// (`PLUGIN_ROUTES` below). Exported names and return shapes are unchanged, so
// every caller and test from the fork reads the same.
import type { CapsuleLedger, CapsuleRecord, DisclosurePreimage } from '@/features/capsules/api/types'
import { decodeSignedStatementB64 } from '@/features/capsules/api/peerLedgerFetchClient'
import { PluginRouteError, getPluginJson } from '@/plugin-host/host'

/** The plugin-relative routes this client reads. The plugin serves them; see
 *  `web-ui/DATA-ROUTES.md`. */
export const PLUGIN_ROUTES = {
  ledger: 'http/ledger',
  signedStatement: (capsuleId: string) => `http/ledger/signed-statement?capsule_id=${encodeURIComponent(capsuleId)}`,
  disclosure: (capsuleId: string) => `http/ledger/disclosure?capsule_id=${encodeURIComponent(capsuleId)}`
} as const

type LedgerBody = { records?: unknown; node_pub_key_pem?: unknown }

function isRecord(value: unknown): value is CapsuleRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export async function fetchCapsuleLedger(): Promise<CapsuleLedger> {
  let body: LedgerBody
  try {
    body = await getPluginJson<LedgerBody>(PLUGIN_ROUTES.ledger)
  } catch (error) {
    // No ledger yet (nothing sealed) reads as an empty ledger, as before.
    if (error instanceof PluginRouteError && error.status === 404) return { records: [], nodePubKeyPem: null }
    throw error
  }
  // Skip a malformed entry rather than failing the whole ledger view.
  const records = Array.isArray(body.records) ? body.records.filter(isRecord) : []
  const nodePubKeyPem = typeof body.node_pub_key_pem === 'string' ? body.node_pub_key_pem : null
  return { records, nodePubKeyPem }
}

/** Fetches capsule_id's detached COSE_Sign1 signed statement, or null if none exists. */
export async function fetchSignedStatement(capsuleId: string): Promise<Uint8Array | null> {
  try {
    const body = await getPluginJson<{ signed_statement_b64?: unknown }>(PLUGIN_ROUTES.signedStatement(capsuleId))
    return typeof body.signed_statement_b64 === 'string' ? decodeSignedStatementB64(body.signed_statement_b64) : null
  } catch {
    return null
  }
}

/**
 * Fetches capsule_id's OPTIONAL local disclosure preimage (the request+
 * response TEXT the plugin keeps next to the ledger), or null when none
 * exists. Most capsules have none: the signed capsule commits to
 * request/response by digest only, and this file is a separate, out-of-band
 * attachment.
 */
export async function fetchDisclosurePreimage(capsuleId: string): Promise<DisclosurePreimage | null> {
  try {
    const body = await getPluginJson<{ disclosure?: unknown }>(PLUGIN_ROUTES.disclosure(capsuleId))
    return isRecord(body.disclosure) ? (body.disclosure as DisclosurePreimage) : null
  } catch {
    return null
  }
}

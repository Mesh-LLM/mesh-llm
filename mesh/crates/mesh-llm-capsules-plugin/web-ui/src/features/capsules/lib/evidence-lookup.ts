// The console's chat and Logs views each give this plugin's contribution the
// host's ids for one exchange; this asks the plugin whether this node sealed
// it (`http/lookup`) and says so in one line. Nothing is guessed: no ids, no
// lookup; no match, nothing shown (see EvidenceChip).
import type { MeshPluginUiContributionSubject } from '@/plugin-host/host-contract'
import { PLUGIN_NAME } from '@/plugin-host/standalone-host'

export type LookupResult =
  | { found: true; exchange_key: string | null; capsule_id: string | null; role?: string }
  | { found: false }

/** The plugin route for this subject, or null when the host gave no id to
 *  look it up by. */
export function lookupPath(subject: MeshPluginUiContributionSubject): string | null {
  if (subject.slot === 'logs_request') {
    return subject.exchangeId ? `http/lookup?exchange_id=${encodeURIComponent(subject.exchangeId)}` : null
  }
  return subject.clientNonce ? `http/lookup?client_nonce=${encodeURIComponent(subject.clientNonce)}` : null
}

/** Where "see the record" goes: the Evidence page, focused on that row. */
export function recordPath(result: LookupResult): string | null {
  if (!result.found || !result.exchange_key) return null
  return `/plugins/${PLUGIN_NAME}/evidence?focusExchangeKey=${encodeURIComponent(result.exchange_key)}`
}

export const SEALED_LINE = 'Sealed on this node · see the record'

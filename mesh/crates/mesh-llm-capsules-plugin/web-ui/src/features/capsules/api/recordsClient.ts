// The owner's own records: the `Your records` status and the three `Clean up
// records` actions. Reached through the host's generic plugin tool-call route
// (plugin-relative `tools/<tool_name>`, same as `peerLedgerFetchClient.ts`)
// -- this plugin's `owner_maintenance.rs` answers. No new host route. Ported
// from the mesh-llm console fork's `api/recordsClient.ts`; only the transport
// differs (the host's `fetchPlugin`, not the console's own API URL).
//
// Every cleanup seals a record of itself on the plugin side; this client only
// carries the request and reports what came back, never a success it didn't
// get.
import { pluginHost } from '@/plugin-host/host'

export type SharingSwitchKey = 'record_at_completion' | 'history_segments' | 'adjudications' | 'witness'

export type SharingSwitchState = { value: string | null; source: 'set' | 'default' }

export type RecordsStatus = {
  records_path: string
  record_count: number
  head: string | null
  log_id: string
  stored_text_count: number
  new_history_pending: { requested_at: string; closing_record_id: string } | null
  sharing: Record<SharingSwitchKey, SharingSwitchState>
  /** Set when the ledger's last line was torn (a write cut short) and was
   *  cut back to the last whole record when the plugin opened it. */
  torn_write_cut_at?: number | null
  /** Every cut of a torn last line this log has had, kept across restarts. */
  ledger_repairs?: Array<{ at: string; torn_write_cut_at: number }>
  /** Why mesh's config file is not being read as written: the sharing
   *  switches are then its last values that parsed, or all off. */
  config_problem?: string | null
  /** Sharing switches set to a value the plugin does not know; each is
   *  read as off. */
  setting_problems?: string[]
  /** Whether this plugin keeps the exchange text it is handed, and for how
   *  long. Absent from a plugin that predates it. */
  exchange_text?: { kept: boolean; retention_days: number }
}

export type SealedCleanup = { capsule_id: string; record_number: number; kind: string }

export type CleanupAction = 'delete_stored_text' | 'rebuild_index' | 'start_new_log'

export type CleanupResult =
  | { action: 'delete_stored_text'; deleted_count: number; sealed: SealedCleanup | null }
  | { action: 'rebuild_index'; records_checked: number; sealed: SealedCleanup }
  | { action: 'start_new_log'; sealed: SealedCleanup }

const OPERATION: Record<CleanupAction, string> = {
  delete_stored_text: 'evidence_delete_stored_text',
  rebuild_index: 'evidence_rebuild_index',
  start_new_log: 'evidence_start_new_history'
}

/** Thrown for any non-2xx or unreadable reply. The host route reports a
 *  plugin-tool error as a non-2xx whose body is the error text. */
export class RecordsToolError extends Error {}

async function callTool<T>(operation: string, args: Record<string, unknown>): Promise<T> {
  let response: Response
  try {
    response = await pluginHost().network.fetchPlugin(`tools/${operation}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(args)
    })
  } catch (error) {
    throw new RecordsToolError(error instanceof Error ? error.message : 'network error')
  }
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new RecordsToolError(text || `HTTP ${response.status}`)
  }
  try {
    return (await response.json()) as T
  } catch {
    throw new RecordsToolError('the node’s reply was not valid JSON')
  }
}

export function fetchRecordsStatus(): Promise<RecordsStatus> {
  return callTool<RecordsStatus>('evidence_records_status', {})
}

export async function runCleanup(action: CleanupAction): Promise<CleanupResult> {
  const args = action === 'start_new_log' ? { confirm: true } : {}
  const body = await callTool<Record<string, unknown>>(OPERATION[action], args)
  return { action, ...body } as CleanupResult
}

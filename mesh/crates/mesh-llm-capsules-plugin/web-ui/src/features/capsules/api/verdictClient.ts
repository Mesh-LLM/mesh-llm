// One signed verdict record, from the plugin's own route
// (`http/ledger/verdict?capsule_id=…`), with the plugin's check of its
// signature. The page shows it; it never re-judges it.
import type { VerdictRecordJson } from '@/features/capsules/api/sidecarTypes'
import { getPluginJson } from '@/plugin-host/host'

export function fetchVerdictRecord(capsuleId: string): Promise<VerdictRecordJson> {
  return getPluginJson<VerdictRecordJson>(`http/ledger/verdict?capsule_id=${encodeURIComponent(capsuleId)}`)
}

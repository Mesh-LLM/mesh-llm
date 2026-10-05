import type { ChatActionMetric } from '@/features/app-tabs/types'
import type { StatusPayload } from '@/lib/api/types'
import { meshAdvertisedVramGB, meshCapacityInputFromStatus } from '@/lib/vram'

/**
 * Header badges for live mode: node count and advertised mesh capacity, derived
 * from the same status payload and helper the Network tab uses so both tabs agree.
 * Client-role nodes are counted as nodes but contribute no capacity, matching
 * the scheduler's aggregate.
 */
export function liveChatActionMetrics(status: StatusPayload | undefined): ChatActionMetric[] {
  if (!status) return []

  const nodeCount = 1 + (status.peers?.length ?? 0)
  const vramGb = meshAdvertisedVramGB(meshCapacityInputFromStatus(status))

  return [
    { id: 'nodes', icon: 'cpu', label: `${nodeCount} node${nodeCount === 1 ? '' : 's'}` },
    { id: 'vram', icon: 'hard-drive', label: `${vramGb.toFixed(1)} GB` }
  ]
}

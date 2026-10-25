import { useQuery } from '@tanstack/react-query'
import { env } from '@/lib/env'

/** True only when this node's wallet policy can pay for inference (`automatic` with a budget).
 *  Any failure — payments not built in, no wallet route, remote console — reads as free-only. */
async function fetchPaidRoutingAllowed(): Promise<boolean> {
  const response = await fetch(`${env.managementApiUrl}/api/wallet`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command: 'policy' })
  })
  if (!response.ok) return false
  const policy: unknown = await response.json().catch(() => null)
  return (policy as { mode?: unknown } | null)?.mode === 'automatic'
}

export function usePaidRoutingQuery(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ['chat', 'paid-routing-allowed'],
    queryFn: fetchPaidRoutingAllowed,
    staleTime: 30_000,
    refetchInterval: 60_000,
    retry: false,
    enabled: options?.enabled ?? true
  })
}

import { useQuery } from '@tanstack/react-query'
import { env } from '@/lib/env'
import { parseModelPayments, type ModelPayment } from '@/features/chat/lib/model-prices'

/** Bound the page walk: meshes far past this size have larger problems than the picker. */
const PAGE_LIMIT = 500
const PAGE_CAP = 5

async function fetchModelPayments() {
  const payments = new Map<string, ModelPayment>()
  let cursor: string | null = null
  for (let page = 0; page < PAGE_CAP; page += 1) {
    const url = cursor
      ? `${env.managementApiUrl}/mesh/v1/models?limit=${PAGE_LIMIT}&cursor=${encodeURIComponent(cursor)}`
      : `${env.managementApiUrl}/mesh/v1/models?limit=${PAGE_LIMIT}`
    const response = await fetch(url)
    if (!response.ok) throw new Error(`/mesh/v1/models returned ${response.status}`)
    const body = (await response.json()) as { next_cursor?: unknown }
    for (const [id, payment] of parseModelPayments(body)) payments.set(id, payment)
    cursor = typeof body?.next_cursor === 'string' && body.next_cursor ? body.next_cursor : null
    if (!cursor) break
  }
  return payments
}

export function useModelPaymentsQuery(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ['chat', 'model-payments'],
    queryFn: fetchModelPayments,
    staleTime: 30_000,
    refetchInterval: 60_000,
    retry: false,
    enabled: options?.enabled ?? true
  })
}

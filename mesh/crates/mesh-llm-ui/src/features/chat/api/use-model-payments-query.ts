import { useQuery } from '@tanstack/react-query'
import { env } from '@/lib/env'
import { parseModelPayments } from '@/features/chat/lib/model-prices'

async function fetchModelPayments() {
  const response = await fetch(`${env.managementApiUrl}/v1/models`)
  if (!response.ok) throw new Error(`/v1/models returned ${response.status}`)
  return parseModelPayments(await response.json())
}

export function useModelPaymentsQuery(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ['chat', 'model-payments'],
    queryFn: fetchModelPayments,
    staleTime: 30_000,
    retry: false,
    enabled: options?.enabled ?? true
  })
}

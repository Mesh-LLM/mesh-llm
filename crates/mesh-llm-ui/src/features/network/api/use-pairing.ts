import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  connectWithPairingOffer,
  createPairingOffer,
  decidePairingSession,
  discoverNearbyPairingDevices,
  fetchPairingSessions
} from '@/features/network/api/pairing'
import { pairingKeys } from '@/lib/query/query-keys'

export function usePairingSessions(enabled: boolean) {
  return useQuery({
    queryKey: pairingKeys.sessions(),
    queryFn: fetchPairingSessions,
    enabled,
    refetchInterval: enabled ? 1_000 : false
  })
}

export function useNearbyPairingDevices(enabled: boolean) {
  return useQuery({
    queryKey: pairingKeys.nearby(),
    queryFn: discoverNearbyPairingDevices,
    enabled,
    staleTime: 5_000,
    refetchInterval: enabled ? 15_000 : false,
    refetchOnWindowFocus: false
  })
}

export function useCreatePairingOffer() {
  return useMutation({ mutationFn: createPairingOffer })
}

export function useConnectPairingOffer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: connectWithPairingOffer,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: pairingKeys.sessions() })
  })
}

export function usePairingDecision() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, decision }: { id: string; decision: 'approve' | 'reject' | 'cancel' }) =>
      decidePairingSession(id, decision),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: pairingKeys.sessions() })
  })
}

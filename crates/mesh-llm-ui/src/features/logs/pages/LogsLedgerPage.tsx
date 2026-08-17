import { useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { logsKeys } from '@/features/logs/api/use-logs-ledger-query'
import { LogsLedger } from '@/features/logs/components/LogsLedger'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { useDataMode } from '@/lib/data-mode'
import type { LogsLedgerSearch } from '@/features/logs/lib/log-search'

export function LogsLedgerPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const dataMode = useDataMode()
  const statusQuery = useStatusQuery({ enabled: dataMode.mode === 'live' })
  const search = useSearch({ from: '/logs' })
  const invalidateLedger = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: logsKeys.all, refetchType: 'active' })
  }, [queryClient])
  const updateSearch = useCallback(
    (nextSearch: LogsLedgerSearch) => {
      void navigate({ to: '/logs', search: nextSearch })
    },
    [navigate]
  )
  return (
    <LogsLedger
      loggingStatus={statusQuery.data?.logging}
      onMaintenanceMutationSucceeded={invalidateLedger}
      onSearchChange={updateSearch}
      search={search}
    />
  )
}

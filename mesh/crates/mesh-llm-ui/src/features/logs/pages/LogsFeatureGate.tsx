import { Navigate } from '@tanstack/react-router'
import type { ReactNode } from 'react'
import { isClientOnlyNode } from '@/features/app-shell/lib/status-helpers'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { useDataMode } from '@/lib/data-mode'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'

type LogsFeatureGateProps = {
  readonly children: ReactNode
}

/** Prevent direct URLs from rendering logging pages while the surface is disabled or the node is client-only. */
export function LogsFeatureGate({ children }: LogsFeatureGateProps) {
  const logsPageEnabled = useBooleanFeatureFlag('global/logsPage')
  const { mode } = useDataMode()
  const statusQuery = useStatusQuery({ enabled: mode === 'live' })

  if (!logsPageEnabled || isClientOnlyNode(statusQuery.data)) return <Navigate replace to="/" />
  return children
}

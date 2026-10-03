import { Navigate } from '@tanstack/react-router'
import type { ReactNode } from 'react'
import { isClientOnlyNode } from '@/features/app-shell/lib/status-helpers'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { useDataMode } from '@/lib/data-mode'

type ConfigurationFeatureGateProps = {
  readonly children: ReactNode
}

/** Prevent direct URLs from rendering configuration pages on a client-only node, whose management API is unreachable. */
export function ConfigurationFeatureGate({ children }: ConfigurationFeatureGateProps) {
  const { mode } = useDataMode()
  const statusQuery = useStatusQuery({ enabled: mode === 'live' })

  if (isClientOnlyNode(statusQuery.data)) return <Navigate replace to="/" />
  return children
}

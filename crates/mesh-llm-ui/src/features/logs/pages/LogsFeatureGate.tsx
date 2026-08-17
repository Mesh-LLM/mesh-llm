import { Navigate } from '@tanstack/react-router'
import type { ReactNode } from 'react'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'

type LogsFeatureGateProps = {
  readonly children: ReactNode
}

/** Prevent direct URLs from rendering logging pages while the surface is disabled. */
export function LogsFeatureGate({ children }: LogsFeatureGateProps) {
  const logsPageEnabled = useBooleanFeatureFlag('global/logsPage')

  if (!logsPageEnabled) return <Navigate replace to="/" />
  return children
}

import type { LogCallerPathType, LogPeerPathType, LogRequest } from '@/features/logs/api/schemas'

const pathTypeLabels: Record<LogCallerPathType | LogPeerPathType, string> = {
  direct: 'Direct',
  local_http: 'Local HTTP',
  relay: 'Relay',
  remote_quic_http: 'Remote QUIC HTTP'
}

export function formatEndpointId(endpointId: string): string {
  return endpointId.length <= 12 ? endpointId : `${endpointId.slice(0, 4)}…${endpointId.slice(-4)}`
}

export function formatNetworkPathType(pathType: LogCallerPathType | LogPeerPathType): string {
  return pathTypeLabels[pathType]
}

export function formatRequestCaller(request: LogRequest): string | undefined {
  const identity = request.callerEndpointId
    ? formatEndpointId(request.callerEndpointId)
    : request.callerAddr
      ? request.callerAddr
      : undefined
  const path = request.callerPathType ? formatNetworkPathType(request.callerPathType) : undefined
  if (!identity && !path) return undefined
  return [identity ?? 'Caller', path].filter(Boolean).join(' · ')
}

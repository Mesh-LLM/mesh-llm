import type { LogEventCategory } from '@/features/logs/lib/log-event-ledger'

export const LOG_EVENT_CATEGORY_LABELS: Record<LogEventCategory, string> = {
  requests: 'Requests',
  system: 'System',
  quic: 'QUIC',
  gossip: 'Gossip',
  iroh: 'Iroh'
}

export const LOG_EVENT_CATEGORY_COLORS: Record<LogEventCategory, string> = {
  requests: 'var(--color-log-requests)',
  system: 'var(--color-log-system)',
  quic: 'var(--color-log-quic)',
  gossip: 'var(--color-log-gossip)',
  iroh: 'var(--color-log-iroh)'
}

export const LOG_EVENT_CATEGORY_MARKER_CLASS: Record<LogEventCategory, string> = {
  requests: 'rounded-[2px]',
  system: 'rounded-full',
  quic: 'rounded-[1px] rotate-45',
  gossip: 'h-1.5 w-2.5 rounded-[1px]',
  iroh: 'h-1.5 w-2.5 rounded-full'
}

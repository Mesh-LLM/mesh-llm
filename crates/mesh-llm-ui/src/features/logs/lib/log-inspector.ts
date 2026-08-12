export const LOG_REQUEST_DETAIL_TABS = ['overview', 'payloads', 'timeline', 'diagnostics'] as const

export type LogRequestDetailTab = (typeof LOG_REQUEST_DETAIL_TABS)[number]

export const DEFAULT_LOG_REQUEST_DETAIL_TAB = 'overview' as const satisfies LogRequestDetailTab

export const LEGACY_LOG_REQUEST_DETAIL_TABS = ['summary', 'request', 'response', 'routing', 'stream', 'errors'] as const

export type LegacyLogRequestDetailTab = (typeof LEGACY_LOG_REQUEST_DETAIL_TABS)[number]

export type LogRequestDetailTabInput = LogRequestDetailTab | LegacyLogRequestDetailTab

export type LogInspector =
  { readonly type: 'request'; readonly id: string } | { readonly type: 'audit'; readonly id: string }

export function isLogRequestDetailTab(value: string): value is LogRequestDetailTab {
  return LOG_REQUEST_DETAIL_TABS.some((tab) => tab === value)
}

export function normalizeLogRequestDetailTab(value: unknown): LogRequestDetailTab | undefined {
  switch (value) {
    case 'overview':
    case 'summary':
      return 'overview'
    case 'payloads':
    case 'request':
    case 'response':
      return 'payloads'
    case 'timeline':
    case 'routing':
    case 'stream':
      return 'timeline'
    case 'diagnostics':
    case 'errors':
      return 'diagnostics'
    default:
      return undefined
  }
}

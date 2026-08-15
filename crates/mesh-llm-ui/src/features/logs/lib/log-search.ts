import { LogPageCursor, LogReplayCursor, LogRequestId } from '@/features/logs/api/ids'
import type { LogsRequestQuery } from '@/features/logs/api/client'
import { LOG_EVENT_CATEGORIES, type LogEventCategory } from '@/features/logs/lib/log-event-ledger'
import {
  DEFAULT_LOG_REQUEST_DETAIL_TAB,
  normalizeLogRequestDetailTab,
  type LogInspector,
  type LogRequestDetailTab,
  type LogRequestDetailTabInput
} from '@/features/logs/lib/log-inspector'
// Time presets replace raw RFC 3339 inputs per shape plan.
export type RelativeTimePreset = '1h' | '6h' | '12h' | '24h' | '7d' | ''

export const RELATIVE_TIME_PRESETS: readonly { value: RelativeTimePreset; label: string }[] = [
  { value: '', label: 'Lifetime' },
  { value: '1h', label: 'Last hour' },
  { value: '6h', label: 'Last 6 hours' },
  { value: '12h', label: 'Last 12 hours' },
  { value: '24h', label: 'Last 24 hours' },
  { value: '7d', label: 'Last week' }
]

export function resolveRelativeTime(
  preset: RelativeTimePreset,
  nowMs = Date.now()
): { from?: string; to?: string } | undefined {
  if (!preset) return undefined
  const hourMs = 3_600_000
  const durationMs: Record<Exclude<RelativeTimePreset, ''>, number> = {
    '1h': hourMs,
    '6h': 6 * hourMs,
    '12h': 12 * hourMs,
    '24h': 24 * hourMs,
    '7d': 7 * 24 * hourMs
  }

  return {
    from: new Date(nowMs - durationMs[preset]).toISOString(),
    to: new Date(nowMs).toISOString()
  }
}

function hoursAgo(isoString: string): number | undefined {
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return undefined
  const diffMs = Date.now() - date.getTime()
  return Math.round(diffMs / 60_000) // minutes for sub-hour, hours otherwise
}

export function formatRelativeTime(isoString: string): string {
  const minsAgo = hoursAgo(isoString) ?? Infinity
  if (minsAgo < 2) return 'just now'
  if (minsAgo < 60) return `${minsAgo}m ago`

  const hours = Math.floor(minsAgo / 60)
  if (hours < 24) {
    const remainMins = minsAgo % 60
    if (remainMins === 0) return `${hours}h ago`
    return `${hours}h ${remainMins}m ago`
  }

  const days = Math.floor(hours / 24)
  if (days < 7) {
    const remainHours = hours % 24
    if (remainHours === 0) return `${days}d ago`
    return `${days}d ${remainHours}h ago`
  }

  // Fallback to date for older entries — still more readable than raw ISO.
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return isoString
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export type LogsFilterKey = 'model' | 'provider' | 'engine' | 'route' | 'source' | 'outcome'

const FILTER_KEYS: readonly LogsFilterKey[] = ['model', 'provider', 'engine', 'route', 'source', 'outcome']

export type LogsLedgerSearch = {
  readonly focusRequestId?: string
  readonly replayCursor?: string
  readonly cursor?: string
  readonly trail?: readonly string[]
  readonly model?: string
  readonly provider?: string
  readonly engine?: string
  readonly route?: string
  readonly source?: string
  readonly outcome?: string
  /** Explicit legacy API bounds retained for deep links and SSE reconnects. */
  readonly from?: string
  readonly to?: string
  readonly timeRange?: RelativeTimePreset | ''
  readonly categories?: readonly LogEventCategory[] | 'none'
  readonly inspectType?: LogInspector['type']
  readonly inspectId?: string
  readonly tab?: LogRequestDetailTab
}

type LogsLedgerSearchWithTabInput = Omit<LogsLedgerSearch, 'tab'> & {
  readonly tab?: LogRequestDetailTabInput
}

function optionalString(value: unknown) {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function timestamp(value: unknown) {
  const candidate = optionalString(value)
  return candidate && !Number.isNaN(Date.parse(candidate)) ? candidate : undefined
}

function cursor(value: unknown) {
  const candidate = optionalString(value)
  if (!candidate) return undefined
  try {
    return LogPageCursor.parse(candidate).toString()
  } catch {
    return undefined
  }
}

function cursorTrail(value: unknown) {
  const entries = Array.isArray(value) ? value : [value]
  return entries.flatMap((entry) => {
    const parsed = cursor(entry)
    return parsed ? [parsed] : []
  })
}

function eventCategories(value: unknown): readonly LogEventCategory[] | 'none' | undefined {
  if (value === 'none') return 'none'
  if (value === undefined) return undefined
  const entries = (Array.isArray(value) ? value : [value]).flatMap((entry) =>
    typeof entry === 'string' ? entry.split(',') : []
  )
  const categories = LOG_EVENT_CATEGORIES.filter((category) => entries.includes(category))
  return categories.length > 0 ? categories : undefined
}

function requestInspectorId(value: string): string | undefined {
  return LogRequestId.tryParse(value)?.toString()
}

export function parseLogsLedgerSearch(search: Record<string, unknown>): LogsLedgerSearch {
  const parsed: Partial<Record<LogsFilterKey, string>> = {}
  for (const key of FILTER_KEYS) parsed[key] = optionalString(search[key])

  // Support timeRange presets and retain explicit legacy API bounds in deep links.
  const rawTimeRange = optionalString(search['timeRange'])
  const timeRange = RELATIVE_TIME_PRESETS.find((preset) => preset.value === rawTimeRange)?.value ?? ''
  const from = timestamp(search['from'])
  const to = timestamp(search['to'])

  const pageCursor = cursor(search['cursor'])

  const focusRequestId = optionalString(search['focusRequestId'])
  const replayCursor = optionalString(search['replayCursor'])
  const trail = cursorTrail(search['trail'])
  const categories = eventCategories(search['categories'])
  const inspectType =
    search['inspectType'] === 'request' || search['inspectType'] === 'audit' ? search['inspectType'] : undefined
  const candidateInspectId = optionalString(search['inspectId'])
  const inspectId =
    inspectType === 'request' && candidateInspectId
      ? requestInspectorId(candidateInspectId)
      : inspectType === 'audit'
        ? candidateInspectId
        : undefined
  const tab =
    inspectType === 'request' && inspectId
      ? (normalizeLogRequestDetailTab(search['tab']) ?? DEFAULT_LOG_REQUEST_DETAIL_TAB)
      : undefined
  return {
    ...parsed,
    ...(from ? { from } : {}),
    ...(to ? { to } : {}),
    ...(timeRange ? { timeRange } : {}),
    ...(focusRequestId ? { focusRequestId } : {}),
    ...(replayCursor && isReplayCursor(replayCursor) ? { replayCursor } : {}),
    ...(pageCursor ? { cursor: pageCursor } : {}),
    ...(trail.length > 0 ? { trail } : {}),
    ...(categories ? { categories } : {}),
    ...(inspectType && inspectId ? { inspectType, inspectId } : {}),
    ...(tab ? { tab } : {})
  }
}

function isReplayCursor(value: string) {
  try {
    LogReplayCursor.parse(value)
    return true
  } catch {
    return false
  }
}

/**
 * Resolves the ledger's server-side scope once for a render.
 *
 * Callers with a rolling preset should pass their shared clock value so the
 * request, audit, chart, and maintenance controls agree on the same bounds.
 */
export function toLogsRequestQuery(search: LogsLedgerSearch, nowMs = Date.now()): LogsRequestQuery {
  const parsedCursor = search.cursor ? LogPageCursor.parse(search.cursor) : undefined
  const timeBounds =
    resolveRelativeTime(search.timeRange ?? '', nowMs) ??
    (search.from || search.to ? { from: search.from, to: search.to } : undefined)
  return {
    cursor: parsedCursor,
    from: timeBounds?.from,
    to: timeBounds?.to,
    model: search.model,

    provider: search.provider,
    engine: search.engine,
    route: search.route,
    source: search.source,
    outcome: search.outcome
  }
}

export function advanceLogsPage(search: LogsLedgerSearch, nextCursor: string | undefined): LogsLedgerSearch {
  if (!nextCursor) {
    const trail = search.trail ?? []
    const previous = trail.at(-1)
    return {
      ...search,
      ...(previous ? { cursor: previous } : {}),
      ...(previous ? { trail: trail.slice(0, -1) } : { cursor: undefined, trail: undefined })
    }
  }
  return {
    ...search,
    cursor: nextCursor,
    trail: search.cursor ? [...(search.trail ?? []), search.cursor] : []
  }
}

export function resetLogsSearch(_search: LogsLedgerSearch): LogsLedgerSearch {
  return {}
}

export function updateLogsFilter(
  search: LogsLedgerSearch,
  key: LogsFilterKey,
  value: string | undefined
): LogsLedgerSearch {
  return { ...search, [key]: value, cursor: undefined, trail: undefined }
}

export function updateLogsTimeRange(search: LogsLedgerSearch, timeRange: RelativeTimePreset | ''): LogsLedgerSearch {
  return { ...search, from: undefined, to: undefined, timeRange, cursor: undefined, trail: undefined }
}

export function updateLogCategories(
  search: LogsLedgerSearch,
  categories: readonly LogEventCategory[] | undefined
): LogsLedgerSearch {
  return {
    ...search,
    categories: categories === undefined ? undefined : categories.length === 0 ? 'none' : [...new Set(categories)]
  }
}

export function openLogInspector(search: LogsLedgerSearch, inspector: LogInspector): LogsLedgerSearch {
  const inspectId = inspector.type === 'request' ? LogRequestId.parse(inspector.id).toString() : inspector.id
  const { tab: _tab, ...ledgerSearch } = search
  return {
    ...ledgerSearch,
    inspectType: inspector.type,
    inspectId,
    ...(inspector.type === 'request' ? { tab: DEFAULT_LOG_REQUEST_DETAIL_TAB } : {})
  }
}

export function closeLogInspector(search: LogsLedgerSearchWithTabInput): LogsLedgerSearch {
  const { inspectType: _inspectType, inspectId: _inspectId, tab: _tab, ...ledgerSearch } = search
  return ledgerSearch
}

export function logInspectorFromSearch(search: LogsLedgerSearch): LogInspector | undefined {
  if (!search.inspectType || !search.inspectId) return undefined
  return { type: search.inspectType, id: search.inspectId }
}

export function legacyRequestInspectorSearch(
  requestId: string,
  search: LogsLedgerSearchWithTabInput
): LogsLedgerSearch {
  const tab = normalizeLogRequestDetailTab(search.tab) ?? DEFAULT_LOG_REQUEST_DETAIL_TAB
  return {
    ...search,
    inspectType: 'request',
    inspectId: LogRequestId.parse(requestId).toString(),
    tab
  }
}

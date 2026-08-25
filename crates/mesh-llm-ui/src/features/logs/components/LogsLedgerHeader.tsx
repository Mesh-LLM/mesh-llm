import type { ReactNode } from 'react'
import { LoaderCircle, Logs, RadioTower, RotateCcw, WifiSync } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { InfoBanner, InfoBannerAnnotation } from '@/components/ui/InfoBanner'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { Tooltip } from '@/components/ui/tooltip'
import type { LogsLiveConnectionState } from '@/features/logs/api/use-logs-live-recovery'
import { liveStateLabel, liveStateTone } from '@/features/logs/components/LogsLedgerSections'
import type { LoggingStatus } from '@/lib/api/types'

type LogsHeaderSourceId = 'requests' | 'operations'

export type LogsHeaderSource = {
  readonly id: LogsHeaderSourceId
  readonly label: string
  readonly error: boolean
  readonly fetching: boolean
  readonly hasLoadedData: boolean
  readonly refetch: () => Promise<unknown>
}

type LogsHeaderLiveState = {
  readonly state: LogsLiveConnectionState
  readonly fallbackPollingActive: boolean
  readonly pollingEnabled: boolean
  readonly togglePolling: () => void
}

type LogsLedgerHeaderProps = {
  readonly cleanup: ReactNode
  readonly hasSupportedWindow: boolean
  readonly live: LogsHeaderLiveState
  readonly loggingStatus: LoggingStatus | undefined
  readonly operationsBounded: boolean
  readonly requestBounded: boolean
  readonly sources: readonly LogsHeaderSource[]
}

type RecoveryStatus = 'Unavailable' | 'Showing last window'

const NORMAL_DESCRIPTION = 'Monitor request activity and operational events from this MeshLLM host.'

const sourceCopy: Record<
  LogsHeaderSourceId,
  { readonly data: string; readonly unavailableVerb: 'is' | 'are'; readonly window: string }
> = {
  requests: { data: 'request history', unavailableVerb: 'is', window: 'request window' },
  operations: { data: 'operational events', unavailableVerb: 'are', window: 'operational window' }
}

function recoveryStatus(source: LogsHeaderSource): RecoveryStatus {
  return source.hasLoadedData ? 'Showing last window' : 'Unavailable'
}

function availableSourceDescription(sources: readonly LogsHeaderSource[]): string {
  const availableSource = sources.find((source) => !source.error && source.hasLoadedData)
  if (!availableSource) return ''
  return ` ${availableSource.label} ${availableSource.id === 'requests' ? 'remains' : 'remain'} available.`
}

function recoveryDescription(sources: readonly LogsHeaderSource[], failedSources: readonly LogsHeaderSource[]): string {
  const firstFailure = failedSources.at(0)
  if (!firstFailure) return NORMAL_DESCRIPTION

  if (failedSources.length === 1) {
    const copy = sourceCopy[firstFailure.id]
    const availableDescription = availableSourceDescription(sources)
    return firstFailure.hasLoadedData
      ? `${firstFailure.label} could not be refreshed. The last loaded ${copy.window} remains visible.${availableDescription}`
      : `${firstFailure.label} could not be loaded. No previously loaded ${copy.data} ${copy.unavailableVerb} available.${availableDescription}`
  }

  const labels = failedSources
    .map((source, index) => (index === 0 ? source.label : source.label.toLowerCase()))
    .join(' and ')
  const loadedFailures = failedSources.filter((source) => source.hasLoadedData)
  if (loadedFailures.length === 0) {
    return `${labels} could not be loaded. No previously loaded log data is available.`
  }
  if (loadedFailures.length === failedSources.length) {
    return `${labels} could not be refreshed. The last loaded request and operational windows remain visible.`
  }

  const loadedSource = loadedFailures.at(0)
  const unavailableSource = failedSources.find((source) => !source.hasLoadedData)
  if (!loadedSource || !unavailableSource) return `${labels} could not be refreshed.`
  const loadedCopy = sourceCopy[loadedSource.id]
  const unavailableCopy = sourceCopy[unavailableSource.id]
  return `${labels} could not be refreshed. The last loaded ${loadedCopy.window} remains visible. No previously loaded ${unavailableCopy.data} ${unavailableCopy.unavailableVerb} available.`
}

function captureStatusLabel(status: LoggingStatus): string {
  switch (status.capture_mode) {
    case 'metadata_only':
      return 'Payloads · Metadata only'
    case 'redacted_artifacts':
      return status.artifact_capture_ready ? 'Payloads · Redacted · Ready' : 'Payloads · Redacted · Unavailable'
    case 'unavailable':
      return 'Payloads · Unavailable'
  }
}

export function LogsLedgerHeader({
  cleanup,
  hasSupportedWindow,
  live,
  loggingStatus,
  operationsBounded,
  requestBounded,
  sources
}: LogsLedgerHeaderProps) {
  const failedSources = sources.filter((source) => source.error)
  const hasError = failedSources.length > 0
  const retrying = failedSources.some((source) => source.fetching)
  const refreshing = !hasError && sources.some((source) => source.fetching)
  const requestSource = sources.find((source) => source.id === 'requests')
  const showStatus = hasSupportedWindow || hasError
  const liveStatusLabel =
    live.fallbackPollingActive && !live.pollingEnabled ? 'Polling paused' : liveStateLabel(live.state)
  const liveStatusTone = live.fallbackPollingActive && !live.pollingEnabled ? 'muted' : liveStateTone(live.state)
  return (
    <InfoBanner
      action={
        hasError ? (
          <Button
            className="ui-control shrink-0 gap-1.5"
            disabled={retrying}
            onClick={() => void Promise.all(failedSources.map((source) => source.refetch()))}
            size="sm"
            variant="outline"
          >
            <RotateCcw aria-hidden="true" className="size-3.5" />
            {retrying ? 'Retrying…' : 'Retry'}
          </Button>
        ) : (
          cleanup
        )
      }
      actionClassName="basis-full justify-start pl-[58px] pt-1 sm:basis-auto sm:justify-end sm:pl-0 sm:pt-0"
      annotation={
        requestBounded || operationsBounded ? (
          <InfoBannerAnnotation ariaLabel="Log window notices" tone="warn">
            <div className="space-y-2">
              {requestBounded ? (
                <div>
                  <div className="font-medium text-foreground">Ledger window is bounded</div>
                  <div>
                    The server returned more than 1,000 matching records. The table, chart, and KPIs show the first
                    1,000 only; narrow the filters for complete totals.
                  </div>
                </div>
              ) : null}
              {operationsBounded ? (
                <div>
                  <div className="font-medium text-foreground">Operational window is bounded</div>
                  <div>
                    The server returned more than 1,000 matching operational records. The unified table shows the first
                    1,000 only; narrow the time range for a complete operational window.
                  </div>
                </div>
              ) : null}
            </div>
          </InfoBannerAnnotation>
        ) : undefined
      }
      className="flex-wrap items-start sm:flex-nowrap"
      description={recoveryDescription(sources, failedSources)}
      leadingIcon={<Logs aria-hidden="true" className="size-4" />}
      leadingIconClassName="size-[38px]"
      status={
        showStatus ? (
          <div aria-live={hasError ? undefined : 'polite'} className="flex flex-wrap items-center gap-2">
            {hasError ? (
              <Tooltip
                content={
                  <div className="space-y-1">
                    {failedSources.map((source) => (
                      <div className="flex items-center justify-between gap-4" key={source.id}>
                        <span className="text-foreground">{source.label}</span>
                        <span className="whitespace-nowrap text-fg-dim">{recoveryStatus(source)}</span>
                      </div>
                    ))}
                  </div>
                }
                side="bottom"
              >
                <button
                  aria-label="Refresh failed. View failed log sources"
                  className="inline-flex cursor-help appearance-none rounded-full border-0 bg-transparent p-0 text-inherit focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                  type="button"
                >
                  <StatusBadge size="caption" tone="bad">
                    Refresh failed
                  </StatusBadge>
                </button>
              </Tooltip>
            ) : null}
            {refreshing ? (
              <StatusBadge size="caption" tone="warn">
                <LoaderCircle aria-hidden="true" className="size-3.5 animate-spin motion-reduce:animate-none" />
                Updating
              </StatusBadge>
            ) : requestSource?.hasLoadedData ? (
              live.fallbackPollingActive ? (
                <button
                  aria-label="Fallback log polling"
                  aria-pressed={live.pollingEnabled}
                  className="inline-flex cursor-pointer appearance-none rounded-full border-0 bg-transparent p-0 text-inherit focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                  onClick={live.togglePolling}
                  type="button"
                >
                  <StatusBadge dot size="caption" tone={liveStatusTone}>
                    {liveStatusLabel}
                  </StatusBadge>
                </button>
              ) : live.state === 'connected' ? (
                <StatusBadge size="caption" tone={liveStatusTone}>
                  <RadioTower aria-hidden="true" className="size-3.5" />
                  {liveStatusLabel}
                </StatusBadge>
              ) : live.state === 'reconnecting' ? (
                <StatusBadge size="caption" tone={liveStatusTone}>
                  <WifiSync aria-hidden="true" className="size-3.5 animate-pulse motion-reduce:animate-none" />
                  {liveStatusLabel}
                </StatusBadge>
              ) : (
                <StatusBadge dot size="caption" tone={liveStatusTone}>
                  {liveStatusLabel}
                </StatusBadge>
              )
            ) : null}
            <StatusBadge size="caption" tone="muted">
              Local only
            </StatusBadge>
            {loggingStatus ? (
              <StatusBadge
                size="caption"
                tone={
                  loggingStatus.capture_mode === 'redacted_artifacts' && !loggingStatus.artifact_capture_ready
                    ? 'warn'
                    : 'muted'
                }
              >
                {captureStatusLabel(loggingStatus)}
              </StatusBadge>
            ) : null}
          </div>
        ) : undefined
      }
      title="System logs"
      titleId="logs-ledger-title"
      titleLevel="h1"
      variant={hasError ? 'error' : 'default'}
    />
  )
}

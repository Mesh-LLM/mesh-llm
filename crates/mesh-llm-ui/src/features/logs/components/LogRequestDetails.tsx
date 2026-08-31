import { ArrowLeft } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { CopyInstructionRow } from '@/components/ui/CopyInstructionRow'
import { TabPanel, type TabPanelItem } from '@/components/ui/TabPanel'
import type { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import {
  useLogRequestArtifactsQuery,
  useLogRequestAttemptsQuery,
  useLogRequestEventsQuery,
  useLogRequestSummaryQuery
} from '@/features/logs/api/use-log-request-details-query'
import { LogRequestDiagnostics } from '@/features/logs/components/LogRequestDiagnostics'
import { LogRequestInspectorFooter } from '@/features/logs/components/LogRequestInspectorFooter'
import { LogRequestOverview } from '@/features/logs/components/LogRequestOverview'
import { LogRequestPayloads } from '@/features/logs/components/LogRequestPayloads'
import { LogRequestTabBody } from '@/features/logs/components/LogRequestTabBody'
import { LogRequestTimeline } from '@/features/logs/components/LogRequestTimeline'
import type { LogRequestDetailTab } from '@/features/logs/lib/log-request-details'

type LogRequestDetailsProps = {
  readonly requestId: LogRequestId
  readonly knownRequest?: LogRequest
  readonly tab: LogRequestDetailTab
  readonly onBack: () => void
  readonly onTabChange: (tab: LogRequestDetailTab) => void
  readonly onMaintenanceMutationSucceeded?: () => void
  readonly embedded?: boolean
}

const TAB_PANEL_CONTENT_CLASS = 'mt-0 flex min-h-0 flex-1 flex-col overflow-hidden p-0'

function RequestSummaryState({ loading, error }: { readonly loading: boolean; readonly error: boolean }) {
  if (loading) return <p className="type-body text-fg-dim">Loading request summary.</p>
  if (error) {
    return (
      <p className="type-body text-fg-dim" role="alert">
        request summary could not be loaded. The local log service did not return a usable response.
      </p>
    )
  }
  return (
    <p className="type-body text-fg-dim" role="status">
      Request summary is unavailable.
    </p>
  )
}

function DetailLimitNotice({ visible }: { readonly visible: boolean }) {
  if (!visible) return null
  return (
    <p
      className="mb-3 rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-warn)_35%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-warn)_8%,var(--color-panel))] px-3 py-2 type-caption text-fg-dim"
      role="status"
    >
      This request exceeds the bounded diagnostic limit. The records shown below are incomplete.
    </p>
  )
}

export function LogRequestDetails({
  requestId,
  knownRequest,
  tab,
  onBack,
  onTabChange,
  onMaintenanceMutationSucceeded,
  embedded = false
}: LogRequestDetailsProps) {
  const headingRef = useRef<HTMLHeadingElement>(null)
  const summaryQuery = useLogRequestSummaryQuery(requestId, knownRequest)
  const artifactsQuery = useLogRequestArtifactsQuery(
    requestId,
    tab === 'overview' || tab === 'payloads' || tab === 'diagnostics'
  )
  const eventsQuery = useLogRequestEventsQuery(
    requestId,
    tab === 'overview' || tab === 'timeline' || tab === 'diagnostics'
  )
  const attemptsQuery = useLogRequestAttemptsQuery(
    requestId,
    tab === 'overview' || tab === 'timeline' || tab === 'diagnostics'
  )
  const tabs = [
    {
      value: 'overview',
      label: 'Overview',
      content: (
        <LogRequestTabBody active={tab === 'overview'}>
          <DetailLimitNotice
            visible={Boolean(
              artifactsQuery.data?.nextCursor || eventsQuery.data?.nextCursor || attemptsQuery.data?.nextCursor
            )}
          />
          {summaryQuery.data ? (
            <div className="space-y-[var(--shell-normal)]">
              {embedded ? null : (
                <CopyInstructionRow label="Request ID" value={summaryQuery.data.requestId.toString()} />
              )}
              <LogRequestOverview
                artifacts={{
                  items: artifactsQuery.data?.items,
                  loading: artifactsQuery.isLoading,
                  error: artifactsQuery.isError
                }}
                attempts={{
                  items: attemptsQuery.data?.items,
                  loading: attemptsQuery.isLoading,
                  error: attemptsQuery.isError
                }}
                events={{
                  items: eventsQuery.data?.items,
                  loading: eventsQuery.isLoading,
                  error: eventsQuery.isError
                }}
                request={summaryQuery.data}
              />
            </div>
          ) : (
            <RequestSummaryState error={summaryQuery.isError} loading={summaryQuery.isLoading} />
          )}
        </LogRequestTabBody>
      )
    },
    {
      value: 'payloads',
      label: 'Payloads',
      content: (
        <LogRequestTabBody active={tab === 'payloads'}>
          <DetailLimitNotice visible={Boolean(artifactsQuery.data?.nextCursor)} />
          <LogRequestPayloads
            artifacts={artifactsQuery.data?.items}
            error={artifactsQuery.isError}
            loading={artifactsQuery.isLoading}
          />
        </LogRequestTabBody>
      )
    },
    {
      value: 'timeline',
      label: 'Timeline',
      content: (
        <LogRequestTabBody active={tab === 'timeline'}>
          <DetailLimitNotice visible={Boolean(eventsQuery.data?.nextCursor || attemptsQuery.data?.nextCursor)} />
          <LogRequestTimeline
            attempts={attemptsQuery.data?.items}
            attemptsError={attemptsQuery.isError}
            attemptsLoading={attemptsQuery.isLoading}
            events={eventsQuery.data?.items}
            eventsError={eventsQuery.isError}
            eventsLoading={eventsQuery.isLoading}
          />
        </LogRequestTabBody>
      )
    },
    {
      value: 'diagnostics',
      label: 'Diagnostics',
      content: (
        <LogRequestTabBody active={tab === 'diagnostics'}>
          <DetailLimitNotice
            visible={Boolean(
              artifactsQuery.data?.nextCursor || eventsQuery.data?.nextCursor || attemptsQuery.data?.nextCursor
            )}
          />
          <LogRequestDiagnostics
            artifacts={artifactsQuery.data?.items}
            artifactsError={artifactsQuery.isError}
            artifactsLoading={artifactsQuery.isLoading}
            attempts={attemptsQuery.data?.items}
            attemptsError={attemptsQuery.isError}
            attemptsLoading={attemptsQuery.isLoading}
            events={eventsQuery.data?.items}
            eventsError={eventsQuery.isError}
            eventsLoading={eventsQuery.isLoading}
            request={summaryQuery.data}
            requestError={summaryQuery.isError}
            requestLoading={summaryQuery.isLoading}
          />
        </LogRequestTabBody>
      )
    }
  ] satisfies readonly TabPanelItem<LogRequestDetailTab>[]

  useEffect(() => {
    if (!embedded) headingRef.current?.focus()
  }, [embedded])

  return (
    <section
      aria-label={embedded ? `Request details for ${requestId.toString()}` : undefined}
      aria-labelledby={embedded ? undefined : 'log-request-details-title'}
      className={
        embedded
          ? 'flex min-h-0 w-full flex-1 flex-col overflow-hidden'
          : 'mx-auto flex w-full max-w-[1440px] flex-col overflow-hidden'
      }
    >
      {embedded ? null : (
        <header className="shrink-0 border-b border-border-soft pb-[var(--panel-y)]">
          <Button
            className="ui-control h-8 gap-1.5 px-2.5 text-[length:var(--density-type-caption)]"
            onClick={onBack}
            size="sm"
            variant="outline"
          >
            <ArrowLeft aria-hidden="true" className="size-3.5" />
            Back to logs
          </Button>
          <div className="mt-4 min-w-0">
            <div className="type-label text-fg-faint">Request inspector</div>
            <h1
              className="type-display mt-1 break-all text-foreground outline-none"
              id="log-request-details-title"
              ref={headingRef}
              tabIndex={-1}
            >
              {requestId.toString()}
            </h1>
          </div>
        </header>
      )}

      <TabPanel
        ariaLabel="Request detail tabs"
        className="flex min-h-0 flex-1 flex-col overflow-hidden"
        contentClassName={TAB_PANEL_CONTENT_CLASS}
        onValueChange={onTabChange}
        tabBarClassName="shrink-0"
        tabs={tabs}
        triggerClassName="min-h-11 lg:h-8 lg:min-h-0"
        value={tab}
      />
      <LogRequestInspectorFooter
        onClose={onBack}
        onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded}
        request={summaryQuery.data}
      />
    </section>
  )
}

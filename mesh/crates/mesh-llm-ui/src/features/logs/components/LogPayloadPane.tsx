import { FileDown, FileInput, FileOutput, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { useId, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { LogArtifact } from '@/features/logs/api/schemas'
import { useLogArtifactContentQuery } from '@/features/logs/api/use-log-artifact-content-query'
import type { JsonFormat } from '@/features/logs/components/JsonPayloadView'
import { LogPayloadContentView, LogPayloadMessage } from '@/features/logs/components/LogPayloadContent'
import {
  decodeLogArtifactContent,
  type AvailableLogArtifact,
  type LogArtifactGroup
} from '@/features/logs/lib/log-payload-content'

export type LogPayloadKind = 'request' | 'response'

type LogPayloadPaneProps = {
  readonly kind: LogPayloadKind
  readonly group: LogArtifactGroup
  readonly loading: boolean
  readonly error: boolean
  readonly format: JsonFormat
  readonly headerAction?: ReactNode
  readonly displayToolbar?: ReactNode
}

const payloadPaneConfig: Record<
  LogPayloadKind,
  { readonly Icon: LucideIcon; readonly label: string; readonly emptyMessage: string }
> = {
  request: {
    Icon: FileInput,
    label: 'Request',
    emptyMessage:
      'No request-body artifact is in this ledger entry. This can be metadata-only capture, a route with no retained body, or retention cleanup; it does not by itself prove capture is disabled.'
  },
  response: {
    Icon: FileOutput,
    label: 'Response',
    emptyMessage:
      'No response-body artifact is in this ledger entry. This can be metadata-only capture, a route with no retained body, or retention cleanup; it does not by itself prove capture is disabled.'
  }
}

function DecodedPayload({
  artifact,
  label,
  format
}: {
  readonly artifact: LogArtifact
  readonly label: string
  readonly format: JsonFormat
}) {
  const content = useMemo(() => decodeLogArtifactContent(artifact), [artifact])
  return <LogPayloadContentView content={content} format={format} label={label} />
}

function RemotePayloadAction({
  artifact,
  label,
  format
}: {
  readonly artifact: AvailableLogArtifact
  readonly label: string
  readonly format: JsonFormat
}) {
  const query = useLogArtifactContentQuery(artifact)
  const loadedArtifact = query.data

  if (query.isFetching) {
    return <LogPayloadMessage detail="Reading retained content from the local log service." title="Loading payload" />
  }
  if (query.isError) {
    return (
      <LogPayloadMessage alert detail="The audited retained-content read failed." title="Payload load failed">
        <Button
          className="ui-control gap-1.5"
          onClick={() => void query.refetch()}
          size="sm"
          type="button"
          variant="outline"
        >
          <FileDown aria-hidden="true" className="size-3.5" />
          Retry load
        </Button>
      </LogPayloadMessage>
    )
  }
  return <DecodedPayload artifact={loadedArtifact ?? artifact} format={format} label={label} />
}

function ArtifactPayload({
  artifact,
  label,
  format
}: {
  readonly artifact: LogArtifact
  readonly label: string
  readonly format: JsonFormat
}) {
  if (artifact.contentState !== 'available') {
    return <LogPayloadContentView content={decodeLogArtifactContent(artifact)} format={format} label={label} />
  }
  return artifact.contentBase64 === undefined ? (
    <RemotePayloadAction artifact={artifact} format={format} label={label} />
  ) : (
    <DecodedPayload artifact={artifact} format={format} label={label} />
  )
}

export function LogPayloadPane({
  kind,
  group,
  loading,
  error,
  format,
  headerAction,
  displayToolbar
}: LogPayloadPaneProps) {
  const titleId = useId()
  const config = payloadPaneConfig[kind]
  const Icon = config.Icon
  const primary = group.primary
  return (
    <section
      aria-labelledby={titleId}
      className="min-w-0 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel"
    >
      <header className="flex min-h-14 flex-wrap items-center justify-between gap-3 px-[var(--panel-x)] py-[var(--panel-y)]">
        <div className="flex w-full min-w-0 items-center gap-2 sm:w-auto sm:flex-1">
          <Icon aria-hidden="true" className="size-4 shrink-0 text-accent" />
          <div className="min-w-0">
            <h3 className="type-panel-title text-foreground" id={titleId}>
              {config.label}
            </h3>
            <p className="truncate font-mono text-[length:var(--density-type-caption)] text-fg-faint">
              {primary?.kind ?? 'No primary artifact'}
            </p>
          </div>
        </div>
        {headerAction}
      </header>
      {displayToolbar ? (
        <div className="border-t border-border-soft bg-panel-strong/55 px-[var(--panel-x)] py-2">{displayToolbar}</div>
      ) : null}
      <ScrollArea
        className="border-t border-border-soft bg-background [&>[data-orientation=horizontal]]:bg-border-soft [&>[data-orientation=horizontal]>div]:bg-fg-dim"
        horizontal
        type="always"
        vertical={false}
        viewportClassName="overscroll-contain pb-2.5 [container-type:inline-size] [scrollbar-gutter:stable] [&>div]:!block [&>div]:min-w-full"
        viewportLabel={`${config.label} payload content`}
      >
        {loading ? (
          <LogPayloadMessage detail="Reading retained artifact metadata." title={`Loading ${kind} payload`} />
        ) : null}
        {!loading && error ? (
          <LogPayloadMessage
            alert
            detail="The local log service did not return usable artifact metadata."
            title={`${config.label} payload unavailable`}
          />
        ) : null}
        {!loading && !error && primary === undefined ? (
          <LogPayloadMessage detail={config.emptyMessage} title={`No retained ${kind} artifact`} />
        ) : null}
        {!loading && !error && primary ? (
          <ArtifactPayload
            artifact={primary}
            format={format}
            key={primary.artifactId.toString()}
            label={config.label}
          />
        ) : null}
      </ScrollArea>
    </section>
  )
}

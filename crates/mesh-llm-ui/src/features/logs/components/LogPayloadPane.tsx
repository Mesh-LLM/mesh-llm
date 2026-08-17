import { Eye, FileDown, FileInput, FileOutput, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { useId, useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { LogArtifact } from '@/features/logs/api/schemas'
import { useLogArtifactContentQuery } from '@/features/logs/api/use-log-artifact-content-query'
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
  readonly headerAction?: ReactNode
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

function RevealedPayload({ artifact, label }: { readonly artifact: LogArtifact; readonly label: string }) {
  const content = useMemo(() => decodeLogArtifactContent(artifact), [artifact])
  return <LogPayloadContentView content={content} label={label} />
}

type PayloadRevealPromptProps = {
  readonly canViewWithoutRead: boolean
  readonly onReveal: () => void
}

function PayloadRevealPrompt({ canViewWithoutRead, onReveal }: PayloadRevealPromptProps) {
  const actionLabel = canViewWithoutRead ? 'View payload' : 'Load payload'
  const ActionIcon = canViewWithoutRead ? Eye : FileDown
  return (
    <LogPayloadMessage
      detail={
        canViewWithoutRead
          ? 'Retained content stays hidden until you choose to view it.'
          : 'This audited read downloads only the selected retained body.'
      }
      title="Payload content is hidden"
    >
      <Button className="ui-control gap-1.5" onClick={onReveal} size="sm" type="button" variant="outline">
        <ActionIcon aria-hidden="true" className="size-3.5" />
        {actionLabel}
      </Button>
    </LogPayloadMessage>
  )
}

function InlinePayloadAction({ artifact, label }: { readonly artifact: AvailableLogArtifact; readonly label: string }) {
  const [revealed, setRevealed] = useState(false)
  return revealed ? (
    <RevealedPayload artifact={artifact} label={label} />
  ) : (
    <PayloadRevealPrompt canViewWithoutRead onReveal={() => setRevealed(true)} />
  )
}

function RemotePayloadAction({ artifact, label }: { readonly artifact: AvailableLogArtifact; readonly label: string }) {
  const [revealed, setRevealed] = useState(false)
  const query = useLogArtifactContentQuery(artifact)
  const loadedArtifact = query.data

  if (!revealed) {
    return (
      <PayloadRevealPrompt
        canViewWithoutRead={loadedArtifact !== undefined}
        onReveal={() => {
          setRevealed(true)
          if (loadedArtifact === undefined) void query.refetch()
        }}
      />
    )
  }

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
  return <RevealedPayload artifact={loadedArtifact ?? artifact} label={label} />
}

function ArtifactPayload({ artifact, label }: { readonly artifact: LogArtifact; readonly label: string }) {
  if (artifact.contentState !== 'available') {
    return <LogPayloadContentView content={decodeLogArtifactContent(artifact)} label={label} />
  }
  return artifact.contentBase64 === undefined ? (
    <RemotePayloadAction artifact={artifact} label={label} />
  ) : (
    <InlinePayloadAction artifact={artifact} label={label} />
  )
}

export function LogPayloadPane({ kind, group, loading, error, headerAction }: LogPayloadPaneProps) {
  const titleId = useId()
  const config = payloadPaneConfig[kind]
  const Icon = config.Icon
  const primary = group.primary
  return (
    <section
      aria-labelledby={titleId}
      className="min-w-0 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel"
    >
      <header className="flex min-h-14 items-center justify-between gap-3 px-[var(--panel-x)] py-[var(--panel-y)]">
        <div className="flex min-w-0 items-center gap-2">
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
      <ScrollArea
        className="h-64 border-t border-border-soft bg-background"
        horizontal
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
          <ArtifactPayload artifact={primary} key={primary.artifactId.toString()} label={config.label} />
        ) : null}
      </ScrollArea>
    </section>
  )
}

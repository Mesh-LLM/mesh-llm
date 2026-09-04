import type { CSSProperties } from 'react'
import {
  BrainCircuit,
  Eye,
  FileIcon,
  FileImage,
  FileText,
  Loader2,
  MessageSquareX,
  Music,
  Send,
  User,
  X
} from 'lucide-react'
import type { MessageRole } from '@/features/app-tabs/types'
import { cn } from '@/lib/cn'
import { ResponseStatsBar } from '@/features/chat/components/ResponseStatsBar'
import { ThinkingDisclosure } from '@/features/chat/components/ThinkingDisclosure'
import { MarkdownMessage } from '@/features/chat/components/messages/MarkdownMessage'
import { isMeshVirtualModel } from '@/features/chat/lib/moa-progress'
import { splitAssistantThinking } from '@/features/chat/components/thinking-segments'

export type MessageAttachmentAction = {
  id: string
  label: string
  kind: 'image' | 'pdf' | 'audio' | 'file'
  fileName: string
  onOpen: () => void
}

function MeshIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width={10}
      height={10}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="5" cy="6" r="2.2" />
      <circle cx="19" cy="6" r="2.2" />
      <circle cx="12" cy="18" r="2.2" />
      <path d="M6.8 7.3L10.7 16.3M17.2 7.3L13.3 16.3M7 6h10" />
    </svg>
  )
}

type MessageRowProps = {
  messageRole: MessageRole
  body: string
  timestamp: string
  model?: string
  state?: 'default' | 'queued' | 'streaming' | 'stopped' | 'error'
  route?: string
  tokens?: string
  inspect?: () => void
  inspectLabel?: string
  inspected?: boolean
  routeNode?: string
  showRouteMetadata?: boolean
  tokPerSec?: string
  ttft?: string
  onStopStreaming?: () => void
  onRemoveQueued?: () => void
  attachments?: MessageAttachmentAction[]
}

function AttachmentIcon({ kind }: { kind: MessageAttachmentAction['kind'] }) {
  if (kind === 'image') return <FileImage className="size-3.5" aria-hidden={true} />
  if (kind === 'pdf') return <FileText className="size-3.5" aria-hidden={true} />
  if (kind === 'audio') return <Music className="size-3.5" aria-hidden={true} />
  return <FileIcon className="size-3.5" aria-hidden={true} />
}

function AssistantMarkdown({
  text,
  linksEnabled,
  variant = 'default',
  streaming = false
}: {
  text: string
  linksEnabled: boolean
  variant?: 'default' | 'thinking'
  streaming?: boolean
}) {
  return (
    <MarkdownMessage
      className="text-[length:var(--density-type-body-lg)] leading-[1.55]"
      content={text}
      linksEnabled={linksEnabled}
      streaming={streaming}
      variant={variant}
    />
  )
}

function AssistantMessageContent({
  body,
  linksEnabled,
  model,
  streaming
}: {
  body: string
  linksEnabled: boolean
  model?: string
  streaming: boolean
}) {
  const meshVirtualModel = isMeshVirtualModel(model)
  const segments = splitAssistantThinking(body, { streaming: streaming && !meshVirtualModel })

  if (segments.length === 0) return null

  return (
    <div className="block space-y-3">
      {segments.map((segment, index) => {
        const key = `${segment.kind}-${index}`

        if (segment.kind === 'thinking') {
          const hasResponseAfter = segments
            .slice(index + 1)
            .some((candidate) => candidate.kind === 'response' && candidate.text.length > 0)
          const active = streaming && (segment.open || (meshVirtualModel && !hasResponseAfter))

          if (meshVirtualModel) {
            return (
              <ThinkingDisclosure active={active} key={key}>
                <AssistantMarkdown
                  text={segment.text}
                  linksEnabled={linksEnabled}
                  streaming={streaming}
                  variant="thinking"
                />
              </ThinkingDisclosure>
            )
          }

          return (
            <div
              className="block rounded-[var(--radius)] border px-3 py-2.5 text-[length:var(--density-type-body)] leading-[1.5] text-fg-dim"
              data-thinking-state={active ? 'active' : 'complete'}
              key={key}
              style={{
                background: 'color-mix(in oklab, var(--color-accent) 5%, var(--color-panel))',
                borderColor: 'color-mix(in oklab, var(--color-accent) 22%, var(--color-border-soft))'
              }}
            >
              <span className="mb-1.5 flex select-none items-center gap-1.5 font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.07em] text-fg-faint">
                {active ? (
                  <Loader2 className="size-3 animate-spin" aria-hidden={true} />
                ) : (
                  <BrainCircuit className="size-3" aria-hidden={true} strokeWidth={1.7} />
                )}
                <span>Thinking</span>
              </span>
              <AssistantMarkdown
                text={segment.text}
                linksEnabled={linksEnabled}
                streaming={streaming}
                variant="thinking"
              />
            </div>
          )
        }

        return (
          <div className="block select-text break-words" key={key} style={{ border: '1px solid transparent' }}>
            <AssistantMarkdown text={segment.text} linksEnabled={linksEnabled} streaming={streaming} />
          </div>
        )
      })}
    </div>
  )
}

export function MessageRow({
  messageRole,
  body,
  timestamp,
  model,
  state = 'default',
  route,
  tokens,
  inspect,
  inspectLabel,
  inspected,
  routeNode,
  showRouteMetadata = true,
  tokPerSec,
  ttft,
  onStopStreaming,
  onRemoveQueued,
  attachments = []
}: MessageRowProps) {
  const isUser = messageRole === 'user'
  const isResponse = !isUser
  const isQueued = state === 'queued'
  const isStreamingPlaceholder = state === 'streaming'
  const isStopped = state === 'stopped'
  const isError = state === 'error'
  const hasAttachmentActions = isUser && attachments.length > 0
  const canInspect = inspect != null
  const canRemoveQueued = isQueued && onRemoveQueued != null
  const routeMetadata = showRouteMetadata && ((isUser && routeNode) || (isResponse && route))
  const displayModel = isQueued ? 'Queued' : model
  const rowClassName =
    'relative -mx-2 mb-5 block w-[calc(100%+16px)] select-none rounded-[var(--radius-lg)] border-0 bg-transparent px-2 py-1 text-left transition-[background,box-shadow] duration-150'
  const rowStyle: CSSProperties = {
    ...(inspected ? { background: 'color-mix(in oklab, var(--color-accent) 4%, transparent)' } : {})
  }
  const userContentStyle: CSSProperties = {
    background: 'var(--chat-user-message-background)',
    borderLeft: isError
      ? '1px solid var(--color-bad)'
      : isQueued
        ? '1px solid color-mix(in oklab, var(--color-fg-faint) 42%, var(--color-border))'
        : '1px solid var(--color-accent)',
    padding: '8px 12px 8px 14px'
  }
  const errorContentStyle: CSSProperties = {
    ...userContentStyle,
    background: 'color-mix(in oklab, var(--color-bad) 8%, var(--chat-user-message-background))'
  }
  const accessibleInspectLabel = inspectLabel ?? `Inspect ${isUser ? 'user' : 'assistant'} message from ${timestamp}`
  const headerMetadata = displayModel ? [displayModel] : []
  const messageContent = (
    <>
      <div className="mb-1.5 flex select-none items-center gap-2 text-[length:var(--density-type-caption)] text-fg-faint">
        <span
          className="inline-flex size-4 items-center justify-center rounded-[var(--radius)]"
          style={{
            background: isUser
              ? 'var(--color-panel-strong)'
              : 'color-mix(in oklab, var(--color-accent) 25%, transparent)',
            color: isUser ? 'var(--color-fg-dim)' : 'var(--color-accent)'
          }}
        >
          {isUser ? <User className="size-[10px]" /> : <MeshIcon />}
        </span>
        <span className="font-medium text-foreground">{isUser ? 'You' : 'Assistant'}</span>
        {headerMetadata.map((metadata) => (
          <span className="contents" key={metadata}>
            <span>·</span>
            <span className="font-mono">{metadata}</span>
          </span>
        ))}
        {routeMetadata ? (
          <>
            <span>·</span>
            {isUser && routeNode ? (
              canInspect ? (
                <button
                  type="button"
                  className="inline-flex items-center gap-[5px] outline-none transition-colors hover:text-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
                  style={{ color: inspected ? 'var(--color-accent)' : 'var(--color-fg-dim)' }}
                  aria-label={accessibleInspectLabel}
                  onClick={(event) => {
                    event.stopPropagation()
                    inspect?.()
                  }}
                >
                  <Send className="size-[10px]" aria-hidden={true} /> sent to{' '}
                  <span className="font-mono">{routeNode}</span>
                </button>
              ) : (
                <span
                  className="inline-flex items-center gap-[5px]"
                  style={{ color: inspected ? 'var(--color-accent)' : 'var(--color-fg-dim)' }}
                >
                  <Send className="size-[10px]" /> sent to <span className="font-mono">{routeNode}</span>
                </span>
              )
            ) : isResponse && route ? (
              <span
                className="inline-flex items-center gap-[5px]"
                style={{ color: inspected ? 'var(--color-accent)' : 'var(--color-fg-dim)' }}
              >
                <span className="size-[5px] rounded-full bg-accent" />
                routed via <span className="font-mono">{routeNode ?? route}</span>
              </span>
            ) : null}
          </>
        ) : null}
        {canInspect && isUser && !routeNode ? (
          <button
            type="button"
            className="inline-flex items-center gap-1 text-fg-faint outline-none transition-colors hover:text-fg-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
            aria-label={accessibleInspectLabel}
            onClick={(event) => {
              event.stopPropagation()
              inspect?.()
            }}
          >
            <Eye className="size-3" aria-hidden={true} />
            <span>Inspect</span>
          </button>
        ) : null}
      </div>
      <div
        className="block select-text text-[length:var(--density-type-body-lg)] leading-[1.55]"
        style={{
          padding: isUser || isError ? '6px 0 6px 14px' : '12px 16px',
          background: 'transparent',
          border: isUser || isError ? '0 solid transparent' : '1px solid transparent',
          maxWidth: isUser || isError ? 720 : 'none'
        }}
      >
        {isError ? (
          <div className="block space-y-1">
            <span className="block font-medium text-bad">Message failed to send</span>
            <span className="block break-words text-fg-muted">{body}</span>
            <span className="block text-[length:var(--density-type-caption)] text-fg-faint">
              Try selecting another model, then send the prompt again.
            </span>
          </div>
        ) : isResponse ? (
          <AssistantMessageContent body={body} linksEnabled={true} model={model} streaming={isStreamingPlaceholder} />
        ) : (
          body
        )}
        {isStreamingPlaceholder ? (
          <span className="mt-3 block">
            <button
              aria-label="Stop streaming"
              className="ui-control ui-control-destructive-hover group/stop inline-flex items-center gap-2 rounded-[var(--radius)] border px-2.5 py-1.5 font-mono text-[length:var(--density-type-caption)] outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-destructive"
              onClick={(event) => {
                event.stopPropagation()
                onStopStreaming?.()
              }}
              type="button"
            >
              <Loader2 className="size-3.5 animate-spin" />
              <span className="group-hover/stop:hidden">Streaming response...</span>
              <span className="hidden group-hover/stop:inline">Stop Streaming</span>
            </button>
          </span>
        ) : null}
      </div>
      {isResponse && !isError ? (
        <ResponseStatsBar
          tokens={tokens}
          tokPerSec={tokPerSec}
          ttft={ttft}
          stopped={isStopped}
          inspect={inspect}
          inspectLabel={accessibleInspectLabel}
        />
      ) : null}
    </>
  )
  const content = isError ? (
    <div className="flex items-center justify-between gap-5" role="alert" style={errorContentStyle}>
      <div className="min-w-0">{messageContent}</div>
      <MessageSquareX aria-hidden={true} className="size-8 shrink-0 text-bad" strokeWidth={1.6} />
    </div>
  ) : isUser ? (
    <div
      className={cn('relative block', hasAttachmentActions && 'flex items-center justify-between gap-4')}
      style={userContentStyle}
    >
      <div className="min-w-0">{messageContent}</div>
      {canRemoveQueued ? (
        <button
          type="button"
          className="ui-control absolute right-3 top-1/2 z-10 inline-flex size-8 -translate-y-1/2 items-center justify-center rounded-[var(--radius)] border text-fg-muted outline-none transition-[background,color,box-shadow,transform] hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          aria-label="Remove queued message"
          title="Remove queued message"
          onClick={(event) => {
            event.stopPropagation()
            onRemoveQueued()
          }}
        >
          <X className="size-4" aria-hidden={true} strokeWidth={1.8} />
        </button>
      ) : null}
      {hasAttachmentActions ? (
        <span className="flex shrink-0 flex-col items-end justify-center gap-1.5">
          {attachments.map((attachment) => (
            <button
              key={attachment.id}
              type="button"
              className="ui-control inline-flex h-8 max-w-[12rem] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-caption)] font-medium text-fg-muted outline-none transition-[background,color,box-shadow,transform] hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              aria-label={`Open ${attachment.fileName}`}
              title={attachment.fileName}
              onClick={(event) => {
                event.stopPropagation()
                attachment.onOpen()
              }}
            >
              <AttachmentIcon kind={attachment.kind} />
              <span className="truncate">{attachment.label}</span>
            </button>
          ))}
        </span>
      ) : null}
    </div>
  ) : (
    messageContent
  )

  return (
    <article className={rowClassName} style={rowStyle}>
      {content}
    </article>
  )
}

import { useState } from 'react'
import { Pager } from '@/components/ui/Pager'
import { JsonPayloadView, type JsonFormat } from '@/features/logs/components/JsonPayloadView'
import type { LogEventStreamData, LogEventStreamFrame } from '@/features/logs/lib/log-payload-content'

export type SsePayloadViewProps = {
  readonly ariaLabel: string
  readonly format: JsonFormat
  readonly frames: readonly LogEventStreamFrame[]
}

function SseFrameData({
  ariaLabel,
  data,
  format
}: {
  readonly ariaLabel: string
  readonly data: LogEventStreamData
  readonly format: JsonFormat
}) {
  switch (data.state) {
    case 'json':
      return (
        <JsonPayloadView
          ariaLabel={`${ariaLabel} JSON data`}
          format={format}
          prettyText={data.prettyText}
          text={data.text}
        />
      )
    case 'text':
      return (
        <section aria-label={`${ariaLabel} plaintext data`}>
          <pre className="whitespace-pre-wrap break-words p-3 font-mono text-[length:var(--density-type-caption)] leading-relaxed text-foreground">
            <code>{data.text}</code>
          </pre>
        </section>
      )
    default:
      return assertNever(data)
  }
}

function SseFrameMetadata({ frame }: { readonly frame: LogEventStreamFrame }) {
  if (frame.event === undefined && frame.id === undefined) return null
  return (
    <dl className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
      {frame.event === undefined ? null : (
        <div className="flex min-w-0 items-baseline gap-1.5">
          <dt className="type-label text-fg-faint">Event</dt>
          <dd className="min-w-0 break-all font-mono type-caption text-foreground">{frame.event}</dd>
        </div>
      )}
      {frame.id === undefined ? null : (
        <div className="flex min-w-0 items-baseline gap-1.5">
          <dt className="type-label text-fg-faint">ID</dt>
          <dd className="min-w-0 break-all font-mono type-caption text-foreground">{frame.id}</dd>
        </div>
      )}
    </dl>
  )
}

export function SsePayloadView({ ariaLabel, format, frames }: SsePayloadViewProps) {
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0)
  const activeFrameIndex = Math.min(selectedFrameIndex, Math.max(frames.length - 1, 0))
  const activeFrame = frames[activeFrameIndex]
  if (activeFrame === undefined) return <section aria-label={ariaLabel} />

  const position = activeFrameIndex + 1
  const frameLabel = `${ariaLabel} frame ${position}`
  return (
    <section aria-label={frameLabel} className="min-w-0">
      <header className="grid min-w-0 gap-2 border-b border-border-soft bg-panel-strong/55 px-3 py-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
        <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
          <span className="type-label text-fg-faint">
            Frame {position} of {frames.length}
          </span>
          <SseFrameMetadata frame={activeFrame} />
        </div>
        <Pager
          ariaLabel="Response frames"
          className="w-full border-t border-border-soft pt-2 sm:w-auto sm:border-t-0 sm:pt-0"
          count={frames.length}
          nextLabel="Next response frame"
          onValueChange={setSelectedFrameIndex}
          pageLabel={(index) => `Response frame ${index + 1} of ${frames.length}`}
          previousLabel="Previous response frame"
          statusLabel={(index, count) => `Frame ${index + 1} of ${count}`}
          value={activeFrameIndex}
          variant="numbered"
        />
      </header>
      <SseFrameData ariaLabel={frameLabel} data={activeFrame.data} format={format} />
    </section>
  )
}

function assertNever(value: never): never {
  throw new Error(`Unhandled event stream data: ${String(value)}`)
}

import { FileText, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { JsonPayloadView, type JsonFormat } from '@/features/logs/components/JsonPayloadView'
import { SsePayloadView } from '@/features/logs/components/SsePayloadView'
import type { LogPayloadContent } from '@/features/logs/lib/log-payload-content'

const UNAVAILABLE_REASON_DETAIL = {
  streaming_response_not_assembled: 'The streaming response was not assembled into a retained body.',
  response_body_not_bounded: 'The response body could not be bounded for safe artifact retention.',
  capture_content_limit_exceeded: 'The body exceeded the configured artifact capture limit.',
  capture_memory_budget_exceeded: 'The body exceeded the aggregate in-memory artifact capture budget.',
  artifact_capture_disabled: 'Artifact body capture was disabled when this request was recorded.',
  artifact_capture_failed: 'Artifact body capture failed before a retained body could be written.'
} as const

type LogPayloadMessageProps = {
  readonly title: string
  readonly detail: string
  readonly alert?: boolean
  readonly children?: ReactNode
}

export function LogPayloadMessage({ title, detail, alert = false, children }: LogPayloadMessageProps) {
  const Icon = alert ? TriangleAlert : FileText
  return (
    <div className="flex h-64 items-center justify-center p-4" role={alert ? 'alert' : 'status'}>
      <div className="max-w-sm text-center">
        <Icon aria-hidden="true" className={`mx-auto size-5 ${alert ? 'text-bad' : 'text-fg-faint'}`} />
        <div className={`type-panel-title mt-2 ${alert ? 'text-bad' : 'text-foreground'}`}>{title}</div>
        <p className="type-caption mt-1 text-fg-dim">{detail}</p>
        {children ? <div className="mt-3 flex justify-center">{children}</div> : null}
      </div>
    </div>
  )
}

function PlaintextPayload({ text, label }: { readonly text: string; readonly label: string }) {
  return (
    <section aria-label={label}>
      <pre className="whitespace-pre-wrap break-words p-3 font-mono text-[length:var(--density-type-caption)] leading-relaxed text-foreground">
        <code>{text}</code>
      </pre>
    </section>
  )
}

export function LogPayloadContentView({
  content,
  format,
  label
}: {
  readonly content: LogPayloadContent
  readonly format: JsonFormat
  readonly label: string
}) {
  switch (content.state) {
    case 'json':
      return (
        <JsonPayloadView
          ariaLabel={`${label} JSON payload`}
          format={format}
          prettyText={content.prettyText}
          text={content.text}
        />
      )
    case 'text':
      return <PlaintextPayload label={`${label} plaintext payload`} text={content.text} />
    case 'event-stream':
      return <SsePayloadView ariaLabel={`${label} event stream`} format={format} frames={content.frames} />
    case 'malformed-json':
      return (
        <div>
          <p className="border-b border-border-soft p-3 type-caption text-warn" role="status">
            Malformed JSON. Showing inert plaintext; no markup is interpreted.
          </p>
          <PlaintextPayload label={`${label} malformed JSON plaintext`} text={content.text} />
        </div>
      )
    case 'binary':
      return (
        <LogPayloadMessage
          detail={`${content.bytes.byteLength} decoded bytes. Use the explicit download control in retained metadata.`}
          title="Binary or unknown content is not rendered"
        />
      )
    case 'not-loaded':
      return (
        <LogPayloadMessage
          detail="The audited read returned metadata without a retained body."
          title="Content not loaded"
        />
      )
    case 'unavailable':
      return (
        <LogPayloadMessage
          detail={
            content.reason === undefined
              ? 'Artifact metadata was retained, but its body is unavailable to this audited read.'
              : UNAVAILABLE_REASON_DETAIL[content.reason]
          }
          title="Capture unavailable"
        />
      )
    case 'missing':
      return (
        <LogPayloadMessage
          detail="Artifact metadata says its body is intentionally no longer retained (for example after retention or deletion)."
          title="Body not retained"
        />
      )
    case 'corrupt':
      return (
        <LogPayloadMessage alert detail="Integrity checks failed; the retained body was not opened." title="Corrupt" />
      )
    case 'too-large':
      return (
        <LogPayloadMessage
          alert
          detail="Retained content exceeds the 16 MiB rendering ceiling and was not decoded."
          title="Payload is too large to render"
        />
      )
    case 'decode-error':
      return (
        <LogPayloadMessage
          alert
          detail={
            content.reason === 'base64'
              ? 'The retained base64 body is malformed and was not rendered.'
              : 'The retained body is not valid UTF-8 and was not rendered as text.'
          }
          title="Content could not be decoded safely"
        />
      )
    default:
      return assertNever(content)
  }
}

function assertNever(value: never): never {
  throw new Error(`Unhandled payload state: ${String(value)}`)
}

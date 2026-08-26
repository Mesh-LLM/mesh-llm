import { useId, useMemo, useState } from 'react'
import { SegmentedControl, type SegmentedControlOption } from '@/components/ui/SegmentedControl'
import type { LogArtifact } from '@/features/logs/api/schemas'
import type { JsonFormat } from '@/features/logs/components/JsonPayloadView'
import { LogArtifactInventory } from '@/features/logs/components/LogArtifactMetadata'
import { type LogPayloadKind, LogPayloadPane } from '@/features/logs/components/LogPayloadPane'
import { classifyLogArtifacts } from '@/features/logs/lib/log-payload-content'

const PAYLOAD_KIND_OPTIONS = [
  { value: 'request', label: 'Request', selectedTone: 'accent' },
  { value: 'response', label: 'Response', selectedTone: 'accent' }
] as const satisfies readonly SegmentedControlOption[]

const JSON_FORMAT_OPTIONS = [
  { value: 'pretty', label: 'Pretty', selectedTone: 'accent' },
  { value: 'raw', label: 'Raw', selectedTone: 'accent' }
] as const satisfies readonly SegmentedControlOption[]

const isPayloadKind = (v: string): v is LogPayloadKind => v === 'request' || v === 'response'
const isJsonFormat = (value: string): value is JsonFormat => value === 'pretty' || value === 'raw'

export type LogRequestPayloadsProps = {
  readonly artifacts: readonly LogArtifact[] | undefined
  readonly loading: boolean
  readonly error: boolean
}

export function LogRequestPayloads({ artifacts, loading, error }: LogRequestPayloadsProps) {
  const classified = useMemo(() => classifyLogArtifacts(artifacts ?? []), [artifacts])
  const [kind, setKind] = useState<LogPayloadKind>('request')
  const [format, setFormat] = useState<JsonFormat>('pretty')
  const displayLabelId = useId()

  const headerAction = (
    <div className="min-w-0 w-full sm:w-auto">
      <SegmentedControl
        ariaLabel="Payload"
        onValueChange={(value) => {
          if (isPayloadKind(value)) setKind(value)
        }}
        options={PAYLOAD_KIND_OPTIONS}
        value={kind}
        variant="pill"
      />
    </div>
  )

  const displayToolbar = (
    <div
      aria-labelledby={displayLabelId}
      className="flex min-w-0 flex-col items-stretch gap-2 sm:flex-row sm:items-center sm:justify-between"
      role="toolbar"
    >
      <span className="type-label shrink-0 text-fg-faint" id={displayLabelId}>
        Display
      </span>
      <SegmentedControl
        ariaLabelledBy={displayLabelId}
        onValueChange={(value) => {
          if (isJsonFormat(value)) setFormat(value)
        }}
        options={JSON_FORMAT_OPTIONS}
        value={format}
        variant="buttons"
      />
    </div>
  )

  return (
    <div className="min-w-0 space-y-[var(--shell-normal)]">
      {kind === 'request' ? (
        <LogPayloadPane
          error={error}
          displayToolbar={displayToolbar}
          format={format}
          group={classified.request}
          headerAction={headerAction}
          kind="request"
          loading={loading}
        />
      ) : (
        <LogPayloadPane
          error={error}
          displayToolbar={displayToolbar}
          format={format}
          group={classified.response}
          headerAction={headerAction}
          kind="response"
          loading={loading}
        />
      )}
      {!loading && !error ? <LogArtifactInventory classified={classified} /> : null}
    </div>
  )
}

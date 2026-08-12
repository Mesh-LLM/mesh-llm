import { useMemo, useState } from 'react'
import type { LogArtifact } from '@/features/logs/api/schemas'
import { SegmentedControl, type SegmentedControlOption } from '@/components/ui/SegmentedControl'
import { LogArtifactInventory } from '@/features/logs/components/LogArtifactMetadata'
import { LogPayloadPane, type LogPayloadKind } from '@/features/logs/components/LogPayloadPane'
import { classifyLogArtifacts } from '@/features/logs/lib/log-payload-content'

const PAYLOAD_KIND_OPTIONS = [
  { value: 'request', label: 'Request', selectedTone: 'accent' },
  { value: 'response', label: 'Response', selectedTone: 'accent' }
] as const satisfies readonly SegmentedControlOption[]

const isPayloadKind = (v: string): v is LogPayloadKind => v === 'request' || v === 'response'

export type LogRequestPayloadsProps = {
  readonly artifacts: readonly LogArtifact[] | undefined
  readonly loading: boolean
  readonly error: boolean
}

export function LogRequestPayloads({ artifacts, loading, error }: LogRequestPayloadsProps) {
  const classified = useMemo(() => classifyLogArtifacts(artifacts ?? []), [artifacts])
  const [kind, setKind] = useState<LogPayloadKind>('request')

  const payloadToggle = (
    <SegmentedControl
      ariaLabel="Payload"
      onValueChange={(v) => {
        if (isPayloadKind(v)) {
          setKind(v)
        }
      }}
      options={PAYLOAD_KIND_OPTIONS}
      value={kind}
      variant="pill"
    />
  )

  return (
    <div className="space-y-[var(--shell-normal)]">
      {kind === 'request' ? (
        <LogPayloadPane
          error={error}
          group={classified.request}
          headerAction={payloadToggle}
          kind="request"
          loading={loading}
        />
      ) : (
        <LogPayloadPane
          error={error}
          group={classified.response}
          headerAction={payloadToggle}
          kind="response"
          loading={loading}
        />
      )}
      {!loading && !error ? <LogArtifactInventory classified={classified} /> : null}
    </div>
  )
}

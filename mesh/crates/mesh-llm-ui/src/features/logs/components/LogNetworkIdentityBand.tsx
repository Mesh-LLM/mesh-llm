import { Copy } from 'lucide-react'
import type { LogCallerPathType, LogPeerPathType } from '@/features/logs/api/schemas'
import { Button } from '@/components/ui/button'
import { formatEndpointId, formatNetworkPathType } from '@/features/logs/lib/log-client-info'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { useClipboardCopy } from '@/lib/useClipboardCopy'

type LogNetworkIdentityBandProps = {
  readonly address?: string
  readonly endpointId?: string
  readonly occurrenceCount?: number
  readonly pathType?: LogCallerPathType | LogPeerPathType
  readonly title: 'Caller' | 'Peer'
}

function addressLabel(address: string | undefined, pathType: LogNetworkIdentityBandProps['pathType']) {
  if (pathType === 'relay') return 'Connected via relay — no direct address observed'
  return address ?? 'Address unknown'
}

function occurrenceLabel(count: number): string {
  return `${count.toLocaleString()} ${count === 1 ? 'occurrence' : 'occurrences'} in the loaded window`
}

export function LogNetworkIdentityBand({
  address,
  endpointId,
  occurrenceCount,
  pathType,
  title
}: LogNetworkIdentityBandProps) {
  const { copyState, copyText } = useClipboardCopy()
  const copyLabel = `${title.toLowerCase()} endpoint ID`

  return (
    <section aria-label={title} className="border-b border-border-soft bg-panel-strong px-4 py-4 sm:px-5">
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="type-panel-title text-foreground">{title}</h2>
          <p className="mt-1.5 break-all font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
            {endpointId ? formatEndpointId(endpointId) : 'Endpoint ID not available'}
          </p>
        </div>
        {endpointId ? (
          <Button
            aria-label={`Copy ${copyLabel}`}
            className="ui-control h-13 min-h-13 shrink-0 gap-1.5 lg:h-8 lg:min-h-8"
            onClick={() => void copyText(endpointId)}
            size="sm"
            type="button"
            variant="outline"
          >
            <Copy aria-hidden="true" className="size-3" />
            {copyStateLabel(copyState)}
          </Button>
        ) : null}
      </div>
      <dl className="mt-3 flex min-w-0 flex-wrap items-baseline gap-x-6 gap-y-1.5 border-t border-border-soft pt-2.5">
        <div className="flex min-w-0 items-baseline gap-2">
          <dt className="type-label shrink-0 text-fg-faint">Address</dt>
          <dd className="min-w-0 break-words font-mono type-caption text-foreground">
            {addressLabel(address, pathType)}
          </dd>
        </div>
        <div className="flex min-w-0 items-baseline gap-2">
          <dt className="type-label shrink-0 text-fg-faint">Path</dt>
          <dd className="font-mono type-caption text-foreground">
            {pathType ? formatNetworkPathType(pathType) : 'Not recorded'}
          </dd>
        </div>
      </dl>
      {occurrenceCount === undefined ? null : (
        <p className="type-caption mt-2 text-fg-dim">{occurrenceLabel(occurrenceCount)}</p>
      )}
    </section>
  )
}

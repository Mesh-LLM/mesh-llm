import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { LogArtifact } from '@/features/logs/api/schemas'

export type LogArtifactStateBadgesProps = {
  readonly artifact: LogArtifact
}

function artifactTone(state: LogArtifact['contentState']): StatusBadgeTone {
  switch (state) {
    case 'available':
      return 'good'
    case 'unavailable':
      return 'warn'
    case 'missing':
    case 'corrupt':
      return 'bad'
    default:
      return assertNever(state)
  }
}

export function LogArtifactStateBadges({ artifact }: LogArtifactStateBadgesProps) {
  const truncationLabel = artifact.truncated
    ? 'Truncated'
    : artifact.contentState === 'available'
      ? 'Not truncated'
      : undefined

  return (
    <div className="flex flex-wrap items-center justify-end gap-1.5">
      <StatusBadge size="caption" tone={artifactTone(artifact.contentState)}>
        {artifact.contentState}
      </StatusBadge>
      <StatusBadge size="caption" tone={artifact.redacted ? 'warn' : 'muted'}>
        {artifact.redacted ? 'Redacted' : 'Not redacted'}
      </StatusBadge>
      {truncationLabel === undefined ? null : (
        <StatusBadge size="caption" tone={artifact.truncated ? 'warn' : 'muted'}>
          {truncationLabel}
        </StatusBadge>
      )}
    </div>
  )
}

function assertNever(value: never): never {
  throw new Error(`Unhandled artifact state: ${String(value)}`)
}

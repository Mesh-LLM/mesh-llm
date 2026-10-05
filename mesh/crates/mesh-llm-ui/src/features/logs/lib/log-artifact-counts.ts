import type { LogArtifact } from '@/features/logs/api/schemas'

export type ArtifactCounts = {
  readonly total: number
  readonly contentStates: string
  readonly redacted: number
  readonly truncated: number
  readonly bytes: number
  readonly versions: readonly number[]
}

const CONTENT_STATES = [
  'available',
  'unavailable',
  'missing',
  'corrupt'
] as const satisfies readonly LogArtifact['contentState'][]

export function deriveArtifactCounts(items: readonly LogArtifact[]): ArtifactCounts {
  const contentStates = CONTENT_STATES.map((state) => ({
    state,
    count: items.filter((item) => item.contentState === state).length
  }))
    .filter(({ count }) => count > 0)
    .map(({ state, count }) => `${count.toLocaleString()} ${state}`)
    .join(' · ')
  const redacted = items.filter((item) => item.redacted).length
  const truncated = items.filter((item) => item.truncated).length
  const bytes = items.reduce((total, item) => total + item.bytes, 0)
  const versions = [...new Set(items.map((item) => item.version))].sort((left, right) => left - right)
  return {
    total: items.length,
    contentStates,
    redacted,
    truncated,
    bytes,
    versions
  }
}

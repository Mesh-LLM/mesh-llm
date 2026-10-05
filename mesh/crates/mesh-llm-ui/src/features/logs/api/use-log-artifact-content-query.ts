import { useQuery } from '@tanstack/react-query'
import { LogsApiClient } from '@/features/logs/api/client'
import type { AvailableLogArtifact } from '@/features/logs/lib/log-payload-content'
import { useDataMode, type DataMode } from '@/lib/data-mode'

export const logArtifactContentKeys = {
  all: ['logs', 'artifact-content'] as const,
  detail: (artifact: AvailableLogArtifact, mode: DataMode) =>
    [...logArtifactContentKeys.all, artifact.artifactId.toString(), mode] as const
}

export function useLogArtifactContentQuery(artifact: AvailableLogArtifact) {
  const dataMode = useDataMode()
  return useQuery({
    queryKey: logArtifactContentKeys.detail(artifact, dataMode.mode),
    queryFn: () => new LogsApiClient().getArtifact(artifact.artifactId, dataMode.mode),
    enabled: true,
    retry: false,
    staleTime: 10_000
  })
}

// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogArtifactId, LogRequestId } from '@/features/logs/api/ids'
import type { LogArtifact } from '@/features/logs/api/schemas'
import { DataModeProvider } from '@/lib/data-mode'

const api = vi.hoisted(() => ({ getArtifact: vi.fn() }))

vi.mock('@/features/logs/api/client', () => ({
  LogsApiClient: class {
    getArtifact = api.getArtifact
  }
}))

import { useLogArtifactContentQuery } from '@/features/logs/api/use-log-artifact-content-query'

const ARTIFACT: Extract<LogArtifact, { contentState: 'available' }> = {
  artifactId: LogArtifactId.parse('00000000-0000-4000-8000-000000000011'),
  requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000001'),
  occurredAt: '2026-08-04T12:00:00Z',
  kind: 'request_body',
  mediaKind: 'application/json',
  checksum: 'sha256:0123456789abcdef',
  bytes: 2,
  version: 1,
  redacted: true,
  truncated: false,
  contentState: 'available',
  contentBase64: undefined
}

afterEach(() => {
  api.getArtifact.mockReset()
})

describe('useLogArtifactContentQuery', () => {
  it('uses the audited artifact endpoint when the selected payload mounts', async () => {
    // Given
    api.getArtifact.mockResolvedValue({ ...ARTIFACT, contentBase64: btoa('{}') })

    // When
    renderHook(() => useLogArtifactContentQuery(ARTIFACT), { wrapper: createWrapper() })

    // Then
    await waitFor(() => expect(api.getArtifact).toHaveBeenCalledOnce())
    expect(api.getArtifact).toHaveBeenCalledWith(ARTIFACT.artifactId, 'live')
  })

  it('keeps automatic artifact reads in harness mode', async () => {
    api.getArtifact.mockResolvedValue({ ...ARTIFACT, contentBase64: btoa('{}') })
    renderHook(() => useLogArtifactContentQuery(ARTIFACT), {
      wrapper: createWrapper('harness')
    })

    await waitFor(() => expect(api.getArtifact).toHaveBeenCalledOnce())
    expect(api.getArtifact).toHaveBeenCalledWith(ARTIFACT.artifactId, 'harness')
  })
})

function createWrapper(initialMode: 'harness' | 'live' = 'live') {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return function Wrapper({ children }: { readonly children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <DataModeProvider initialMode={initialMode} persist={false}>
          {children}
        </DataModeProvider>
      </QueryClientProvider>
    )
  }
}

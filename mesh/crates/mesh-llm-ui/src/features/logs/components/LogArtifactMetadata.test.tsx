// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogArtifactId, LogRequestId } from '@/features/logs/api/ids'
import type { LogArtifact } from '@/features/logs/api/schemas'

const api = vi.hoisted(() => ({ downloadArtifact: vi.fn() }))

vi.mock('@/features/logs/api/client', () => ({
  LogsApiClient: class {
    downloadArtifact = api.downloadArtifact
  }
}))

import { LogArtifactMetadata } from '@/features/logs/components/LogArtifactMetadata'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')
const ARTIFACT_ID = LogArtifactId.parse('00000000-0000-4000-8000-000000000011')

function artifact(
  contentState: LogArtifact['contentState'],
  redacted = true,
  truncated = false,
  unavailableReason?: Extract<LogArtifact, { contentState: 'unavailable' }>['unavailableReason']
): LogArtifact {
  const base = {
    artifactId: ARTIFACT_ID,
    requestId: REQUEST_ID,
    occurredAt: '2026-08-04T12:00:00Z',
    kind: 'request_body',
    mediaKind: 'application/json',
    checksum: 'sha256:0123456789abcdef',
    bytes: 32,
    version: 1,
    redacted,
    truncated
  }
  return contentState === 'available'
    ? { ...base, contentState, contentBase64: btoa('{}') }
    : {
        ...base,
        contentState,
        ...(contentState === 'unavailable' && unavailableReason !== undefined ? { unavailableReason } : {}),
        contentBase64: undefined
      }
}

afterEach(() => {
  api.downloadArtifact.mockReset()
  vi.restoreAllMocks()
})

describe('LogArtifactMetadata', () => {
  it('downloads an available redacted artifact only after its explicit action', async () => {
    // Given
    const user = userEvent.setup()
    const available = artifact('available')
    const createObjectUrl = vi.fn(() => 'blob:artifact')
    const revokeObjectUrl = vi.fn()
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    Object.assign(URL, {
      createObjectURL: createObjectUrl,
      revokeObjectURL: revokeObjectUrl
    })
    api.downloadArtifact.mockResolvedValue({
      state: 'download',
      download: {
        artifact: available,
        bytes: new Uint8Array([1, 2, 3]),
        fileName: 'mesh-llm-redacted-artifact.bin',
        mediaType: 'application/octet-stream'
      }
    })
    render(<LogArtifactMetadata artifact={available} />)

    // When
    const download = screen.getByRole('button', {
      name: 'Download redacted artifact'
    })

    // Then
    expect(api.downloadArtifact).not.toHaveBeenCalled()

    // When
    await user.click(download)

    // Then
    await waitFor(() => expect(api.downloadArtifact).toHaveBeenCalledWith(ARTIFACT_ID))
    expect(createObjectUrl).toHaveBeenCalledOnce()
    await waitFor(() => expect(revokeObjectUrl).toHaveBeenCalledWith('blob:artifact'))
    expect(screen.getByText('Artifact download started.')).toHaveAttribute('role', 'status')
    anchorClick.mockRestore()
  })

  it('does not start a browser download when retained content becomes unavailable', async () => {
    // Given
    const user = userEvent.setup()
    const available = artifact('available')
    const createObjectUrl = vi.fn(() => 'blob:artifact')
    const revokeObjectUrl = vi.fn()
    Object.assign(URL, {
      createObjectURL: createObjectUrl,
      revokeObjectURL: revokeObjectUrl
    })
    api.downloadArtifact.mockResolvedValue({
      state: 'unavailable',
      artifact: artifact('missing')
    })
    render(<LogArtifactMetadata artifact={available} />)

    // When
    await user.click(screen.getByRole('button', { name: 'Download redacted artifact' }))

    // Then
    await waitFor(() => expect(api.downloadArtifact).toHaveBeenCalledWith(ARTIFACT_ID))
    expect(createObjectUrl).not.toHaveBeenCalled()
    expect(revokeObjectUrl).not.toHaveBeenCalled()
    expect(screen.getByText('This artifact is no longer available for download.')).toHaveAttribute('role', 'status')
  })

  it.each(['unavailable', 'missing', 'corrupt'] as const)(
    'keeps %s metadata explicit without a download action',
    (contentState) => {
      render(<LogArtifactMetadata artifact={artifact(contentState)} />)

      expect(screen.getByText(contentState)).toBeInTheDocument()
      expect(screen.queryByText('Complete')).not.toBeInTheDocument()
      expect(screen.queryByText('Not truncated')).not.toBeInTheDocument()
      expect(screen.queryByRole('button', { name: 'Download redacted artifact' })).not.toBeInTheDocument()
    }
  )

  it('labels available untruncated content without claiming completeness', () => {
    // Given / When
    render(<LogArtifactMetadata artifact={artifact('available')} />)

    // Then
    expect(screen.getByText('Not truncated')).toBeInTheDocument()
    expect(screen.queryByText('Complete')).not.toBeInTheDocument()
  })

  it('retains a true truncation marker when artifact content is unavailable', () => {
    // Given / When
    render(<LogArtifactMetadata artifact={artifact('unavailable', true, true)} />)

    // Then
    expect(screen.getByText('Truncated')).toBeInTheDocument()
    expect(screen.queryByText('Complete')).not.toBeInTheDocument()
  })

  it.each([
    ['streaming_response_not_assembled', 'Streaming response was not assembled for retention.'],
    ['response_body_not_bounded', 'Response body exceeded the bounded capture policy.'],
    ['capture_content_limit_exceeded', 'Artifact content exceeded the configured capture limit.'],
    ['capture_memory_budget_exceeded', 'Artifact capture exceeded the configured memory budget.'],
    ['artifact_capture_disabled', 'Artifact capture was disabled when this record was created.'],
    ['artifact_capture_failed', 'Artifact capture failed before content could be retained.'],
    [undefined, 'No specific capture reason was recorded.']
  ] as const)('explains unavailable content with the %s operator-facing reason', (reason, expectedText) => {
    render(<LogArtifactMetadata artifact={artifact('unavailable', false, false, reason)} />)

    expect(screen.getByRole('status')).toHaveTextContent(expectedText)
  })
})

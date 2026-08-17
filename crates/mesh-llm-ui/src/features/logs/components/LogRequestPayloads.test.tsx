// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogArtifactId, LogRequestId } from '@/features/logs/api/ids'
import type { LogArtifact } from '@/features/logs/api/schemas'
import { LOG_PAYLOAD_RENDER_LIMIT_BYTES } from '@/features/logs/lib/log-payload-content'
import { DataModeProvider } from '@/lib/data-mode'

const api = vi.hoisted(() => ({ getArtifact: vi.fn() }))

vi.mock('@/features/logs/api/client', () => ({
  LogsApiClient: class {
    getArtifact = api.getArtifact
  }
}))

import { LogRequestPayloads } from '@/features/logs/components/LogRequestPayloads'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')
const REQUEST_ARTIFACT_ID = LogArtifactId.parse('00000000-0000-4000-8000-000000000011')
const RESPONSE_ARTIFACT_ID = LogArtifactId.parse('00000000-0000-4000-8000-000000000012')

function availableArtifact(
  artifactId: LogArtifactId,
  kind: string,
  contentBase64: string | undefined
): Extract<LogArtifact, { contentState: 'available' }> {
  return {
    artifactId,
    requestId: REQUEST_ID,
    occurredAt: '2026-08-04T12:00:00Z',
    kind,
    mediaKind: 'application/json',
    checksum: 'sha256:0123456789abcdef',
    bytes: 32,
    version: 1,
    redacted: false,
    truncated: false,
    contentState: 'available',
    contentBase64
  }
}

function stateArtifact(
  contentState: Exclude<LogArtifact['contentState'], 'available'>,
  unavailableReason?: Extract<LogArtifact, { contentState: 'unavailable' }>['unavailableReason']
): LogArtifact {
  const base = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', undefined)
  if (contentState === 'unavailable') {
    return { ...base, contentState, unavailableReason, contentBase64: undefined }
  }
  return { ...base, contentState, contentBase64: undefined }
}

function renderPayloads(artifacts: readonly LogArtifact[], queryClient = createQueryClient()) {
  const view = render(
    <QueryClientProvider client={queryClient}>
      <DataModeProvider initialMode="live" persist={false}>
        <LogRequestPayloads artifacts={artifacts} error={false} loading={false} />
      </DataModeProvider>
    </QueryClientProvider>
  )
  return { ...view, queryClient }
}

function createQueryClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } })
}

afterEach(() => {
  api.getArtifact.mockReset()
})

describe('LogRequestPayloads', () => {
  it('shows explicit actions without reading inline or remote body content on entry', async () => {
    // Given
    const user = userEvent.setup()
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa('{"request":true}'))
    const response = availableArtifact(RESPONSE_ARTIFACT_ID, 'response_body', undefined)
    const atobSpy = vi.spyOn(globalThis, 'atob')

    // When
    renderPayloads([request, response])

    // Then — Request pane visible by default
    expect(
      within(screen.getByRole('region', { name: 'Request' })).getByRole('button', { name: 'View payload' })
    ).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Response' })).not.toBeInTheDocument()

    // When — toggle to Response
    await user.click(screen.getByRole('radio', { name: 'Response' }))

    // Then — Response pane visible, Request unmounted
    expect(
      within(screen.getByRole('region', { name: 'Response' })).getByRole('button', { name: 'Load payload' })
    ).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Request' })).not.toBeInTheDocument()

    // No content decoded or fetched on entry to either pane
    expect(api.getArtifact).not.toHaveBeenCalled()
    expect(atobSpy).not.toHaveBeenCalled()
  })

  it('reveals inline content locally and fetches only the selected remote artifact once', async () => {
    // Given
    const user = userEvent.setup()
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa('{"request":true}'))
    const response = availableArtifact(RESPONSE_ARTIFACT_ID, 'response_body', undefined)
    api.getArtifact.mockResolvedValue({ ...response, contentBase64: btoa('{"response":true}') })
    renderPayloads([request, response])

    // When — Request pane visible by default; click its "View payload"
    await user.click(
      within(screen.getByRole('region', { name: 'Request' })).getByRole('button', { name: 'View payload' })
    )

    // Then — inline content revealed, no fetch
    expect(api.getArtifact).not.toHaveBeenCalled()
    expect(
      within(screen.getByRole('region', { name: 'Request' })).getByRole('radio', { name: 'Pretty' })
    ).toBeInTheDocument()

    // When — toggle to Response; click "Load payload"
    await user.click(screen.getByRole('radio', { name: 'Response' }))
    await user.click(
      within(screen.getByRole('region', { name: 'Response' })).getByRole('button', { name: 'Load payload' })
    )

    // Then — single fetch for response artifact
    await waitFor(() => expect(api.getArtifact).toHaveBeenCalledOnce())
    expect(api.getArtifact).toHaveBeenCalledWith(RESPONSE_ARTIFACT_ID, 'live')
    expect(
      within(screen.getByRole('region', { name: 'Response' })).getByRole('radio', { name: 'Pretty' })
    ).toBeInTheDocument()
  })

  it('requires a fresh local reveal after remount without refetching cached content', async () => {
    // Given
    const user = userEvent.setup()
    const queryClient = createQueryClient()
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', undefined)
    api.getArtifact.mockResolvedValue({ ...request, contentBase64: btoa('{"request":true}') })
    const firstVisit = renderPayloads([request], queryClient)

    // When
    await user.click(screen.getByRole('button', { name: 'Load payload' }))

    // Then
    await waitFor(() => expect(api.getArtifact).toHaveBeenCalledOnce())
    expect(screen.getByRole('radio', { name: 'Pretty' })).toBeInTheDocument()

    // When
    firstVisit.unmount()
    renderPayloads([request], queryClient)

    // Then
    expect(screen.queryByRole('radio', { name: 'Pretty' })).not.toBeInTheDocument()

    // When
    await user.click(screen.getByRole('button', { name: 'View payload' }))

    // Then
    expect(screen.getByRole('radio', { name: 'Pretty' })).toBeInTheDocument()
    expect(api.getArtifact).toHaveBeenCalledOnce()
  })

  it('renders malformed hostile JSON as inert text only after explicit viewing', async () => {
    // Given
    const user = userEvent.setup()
    const malformed = '{"markup":<img src=x onerror=alert(1)>}'
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa(malformed))
    const { container } = renderPayloads([request])

    // When
    await user.click(screen.getByRole('button', { name: 'View payload' }))

    // Then
    expect(screen.getByText('Malformed JSON. Showing inert plaintext; no markup is interpreted.')).toBeInTheDocument()
    expect(screen.getByText(malformed)).toBeInTheDocument()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
  })

  it.each([
    {
      name: 'binary content',
      artifact: {
        ...availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa('binary')),
        mediaKind: 'application/octet-stream'
      },
      title: 'Binary or unknown content is not rendered'
    },
    {
      name: 'oversized content',
      artifact: {
        ...availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa('{}')),
        bytes: LOG_PAYLOAD_RENDER_LIMIT_BYTES + 1
      },
      title: 'Payload is too large to render'
    },
    {
      name: 'invalid base64',
      artifact: availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', 'not base64!'),
      title: 'Content could not be decoded safely'
    }
  ])('renders $name safely only after explicit viewing', async ({ artifact, title }) => {
    // Given
    const user = userEvent.setup()
    renderPayloads([artifact])

    // When
    await user.click(screen.getByRole('button', { name: 'View payload' }))

    // Then
    expect(screen.getByText(title)).toBeInTheDocument()
  })

  it('shows a remote read failure and recovers through the explicit retry action', async () => {
    // Given
    const user = userEvent.setup()
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', undefined)
    api.getArtifact.mockRejectedValueOnce(new Error('read failed'))
    api.getArtifact.mockResolvedValueOnce({ ...request, contentBase64: btoa('{"request":true}') })
    renderPayloads([request])

    // When
    await user.click(screen.getByRole('button', { name: 'Load payload' }))

    // Then
    expect(await screen.findByRole('alert')).toHaveTextContent('Payload load failed')
    expect(screen.queryByText('read failed')).not.toBeInTheDocument()

    // When
    await user.click(screen.getByRole('button', { name: 'Retry load' }))

    // Then
    await waitFor(() => expect(api.getArtifact).toHaveBeenCalledTimes(2))
    expect(screen.getByRole('radio', { name: 'Pretty' })).toBeInTheDocument()
  })

  it('renders an explicit state when an audited read returns metadata without a body', async () => {
    // Given
    const user = userEvent.setup()
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', undefined)
    api.getArtifact.mockResolvedValue(request)
    renderPayloads([request])

    // When
    await user.click(screen.getByRole('button', { name: 'Load payload' }))

    // Then
    expect(await screen.findByText('Content not loaded')).toBeInTheDocument()
  })

  it('announces when payload capture retained no request or response body', async () => {
    // Given
    const user = userEvent.setup()
    renderPayloads([])

    // Then — Request empty state visible by default
    expect(
      screen.getByText('No request-body artifact is in this ledger entry.', { exact: false }).closest('[role="status"]')
    ).toBeInTheDocument()

    // When — toggle to Response
    await user.click(screen.getByRole('radio', { name: 'Response' }))

    // Then — Response empty state visible
    expect(
      screen
        .getByText('No response-body artifact is in this ledger entry.', { exact: false })
        .closest('[role="status"]')
    ).toBeInTheDocument()
  })

  it.each([
    ['unavailable', 'Capture unavailable'],
    ['missing', 'Body not retained'],
    ['corrupt', 'Corrupt']
  ] as const)('renders the explicit %s body state', (contentState, title) => {
    renderPayloads([stateArtifact(contentState)])

    expect(screen.getByText(title)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^(?:View|Load) payload$/ })).not.toBeInTheDocument()
  })

  it('explains an allowlisted unavailable capture reason without exposing arbitrary detail', () => {
    renderPayloads([stateArtifact('unavailable', 'capture_memory_budget_exceeded')])

    expect(screen.getByText('Capture unavailable')).toBeInTheDocument()
    expect(screen.getByText('The body exceeded the aggregate in-memory artifact capture budget.')).toBeInTheDocument()
  })
})

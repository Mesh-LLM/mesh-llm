// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
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

function expectNaturalPayloadViewport() {
  const viewport = screen.getByRole('region', { name: 'Request payload content' })
  const scrollArea = viewport.parentElement
  expect(viewport).toHaveAttribute('tabindex', '0')
  expect(scrollArea).not.toHaveClass('h-80')
  expect(scrollArea).not.toHaveClass('sm:h-[28rem]')
  expect(scrollArea).not.toHaveClass('lg:h-[32rem]')
  expect(scrollArea).not.toHaveClass('h-64')
  expect(scrollArea?.querySelector('[data-orientation="horizontal"]')).toBeInTheDocument()
  expect(scrollArea?.querySelector('[data-orientation="vertical"]')).not.toBeInTheDocument()
  expect(viewport).not.toHaveClass('h-full')
  expect(viewport).not.toHaveClass('min-h-0')
  return viewport
}

afterEach(() => {
  api.getArtifact.mockReset()
})

describe('LogRequestPayloads', () => {
  it('displays selected inline payload content immediately without an artifact read', () => {
    // Given
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa('{"request":true}'))
    const response = availableArtifact(RESPONSE_ARTIFACT_ID, 'response_body', undefined)

    // When
    renderPayloads([request, response])

    // Then
    const pane = within(screen.getByRole('region', { name: 'Request' }))
    const paneHeader = pane.getByRole('banner')
    const payloadControl = within(paneHeader).getByRole('radiogroup', { name: 'Payload' })
    const displayToolbar = pane.getByRole('toolbar', { name: 'Display' })
    const displayLabel = within(displayToolbar).getByText('Display')
    const formatControl = within(displayToolbar).getByRole('radiogroup', { name: 'Display' })
    const jsonPayload = pane.getByRole('region', { name: 'Request JSON payload' })
    expect(within(payloadControl).getByRole('radio', { name: 'Request' })).toBeChecked()
    expect(payloadControl.parentElement).toHaveClass('min-w-0', 'w-full', 'sm:w-auto')
    expect(within(paneHeader).queryByRole('radio', { name: 'Pretty' })).not.toBeInTheDocument()
    expect(within(displayToolbar).queryByRole('radio', { name: 'Request' })).not.toBeInTheDocument()
    expect(displayToolbar).toHaveClass(
      'min-w-0',
      'flex-col',
      'items-stretch',
      'sm:flex-row',
      'sm:items-center',
      'sm:justify-between'
    )
    expect(displayLabel).toHaveClass('shrink-0')
    expect(formatControl).toHaveAttribute('aria-labelledby', displayLabel.id)
    expect(formatControl).toHaveClass('flex', 'flex-wrap', 'gap-1.5')
    expect(within(formatControl).getByRole('radio', { name: 'Pretty' })).toHaveClass('ui-control')
    expect(screen.getAllByRole('radiogroup', { name: 'Display' })).toHaveLength(1)
    expect(within(jsonPayload).queryByRole('radiogroup', { name: 'Display' })).not.toBeInTheDocument()
    expect(pane.queryByRole('button', { name: /^(?:View|Load) payload$/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Response' })).not.toBeInTheDocument()
    expect(api.getArtifact).not.toHaveBeenCalled()
  })

  it('keeps raw content inside the sole natural payload viewport', async () => {
    // Given
    const user = userEvent.setup()
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa(`{"unbroken":"${'x'.repeat(512)}"}`))

    // When
    renderPayloads([request])
    const viewport = expectNaturalPayloadViewport()
    await user.click(screen.getByRole('radio', { name: 'Raw' }))

    // Then
    expect(within(viewport).getByRole('region', { name: 'Request JSON payload' })).toHaveTextContent('x'.repeat(512))
    expect(viewport.querySelector('[data-radix-scroll-area-viewport]')).toBeNull()
  })

  it('keeps hostile plaintext inert inside the natural payload viewport', () => {
    // Given
    const hostileText = '<img src=x onerror=alert(1)><script>alert(2)</script>'
    const request = {
      ...availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa(hostileText)),
      mediaKind: 'text/plain'
    }

    // When
    const { container } = renderPayloads([request])

    // Then
    const viewport = expectNaturalPayloadViewport()
    expect(within(viewport).getByRole('region', { name: 'Request plaintext payload' })).toHaveTextContent(hostileText)
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
  })

  it('preserves one inspector-owned format selection across request and normal JSON response panes', async () => {
    // Given
    const user = userEvent.setup()
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa('{"request":true}'))
    const response = availableArtifact(RESPONSE_ARTIFACT_ID, 'response_body', btoa('{"response":true}'))
    renderPayloads([request, response])
    const formatControl = screen.getByRole('radiogroup', { name: 'Display' })

    // When
    await user.click(within(formatControl).getByRole('radio', { name: 'Raw' }))
    await user.click(screen.getByRole('radio', { name: 'Response' }))

    // Then
    const responseJson = screen.getByRole('region', { name: 'Response JSON payload' })
    expect(within(formatControl).getByRole('radio', { name: 'Raw' })).toBeChecked()
    expect(responseJson.querySelectorAll('[data-json-line]')).toHaveLength(1)
    expect(within(responseJson).queryByRole('radiogroup', { name: 'Display' })).not.toBeInTheDocument()
    expect(screen.getAllByRole('radiogroup', { name: 'Display' })).toHaveLength(1)

    // When
    await user.click(screen.getByRole('radio', { name: 'Request' }))

    // Then
    expect(screen.getByRole('radio', { name: 'Raw' })).toBeChecked()
    expect(
      screen.getByRole('region', { name: 'Request JSON payload' }).querySelectorAll('[data-json-line]')
    ).toHaveLength(1)
  })

  it('preserves raw format when the response is an event stream', async () => {
    // Given
    const user = userEvent.setup()
    const request = availableArtifact(REQUEST_ARTIFACT_ID, 'request_body', btoa('{"request":true}'))
    const response = {
      ...availableArtifact(RESPONSE_ARTIFACT_ID, 'response_body', btoa('data: {"response":true}\n\ndata: [DONE]\n\n')),
      mediaKind: 'text/event-stream'
    }
    renderPayloads([request, response])

    // When
    await user.click(screen.getByRole('radio', { name: 'Raw' }))
    await user.click(screen.getByRole('radio', { name: 'Response' }))

    // Then
    expect(screen.getByRole('radio', { name: 'Raw' })).toBeChecked()
    expect(screen.getByRole('region', { name: 'Response event stream frame 1' })).toBeInTheDocument()
    expect(
      screen
        .getByRole('region', { name: 'Response event stream frame 1 JSON data' })
        .querySelectorAll('[data-json-line]')
    ).toHaveLength(1)
    expect(screen.getAllByRole('radiogroup', { name: 'Display' })).toHaveLength(1)
  })
})

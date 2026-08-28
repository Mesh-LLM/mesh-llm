import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SharedModal, SharedModalContent } from '@/components/ui/SharedModal'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'

const queries = vi.hoisted(() => ({ summary: vi.fn() }))

vi.mock('@/features/logs/api/use-log-request-details-query', () => ({
  useLogRequestSummaryQuery: (...args: unknown[]) => queries.summary(...args)
}))

import { LogRequestInspectorHeader } from '@/features/logs/components/LogRequestInspectorHeader'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')
const REQUEST: LogRequest = {
  requestId: REQUEST_ID,
  outcome: 'completed',
  createdAt: '2026-08-08T12:00:00Z',
  terminalAt: '2026-08-08T12:00:01Z',
  route: 'chat_completions',
  model: 'Qwen3',
  provider: 'mesh',
  engine: 'skippy',
  statusCode: 200,
  source: 'durable'
}

describe('LogRequestInspectorHeader', () => {
  it('keeps compact request identity and status chrome wrap-safe', () => {
    // Given
    queries.summary.mockReturnValue({ data: REQUEST, isLoading: false, isError: false })

    // When
    render(
      <SharedModal open>
        <SharedModalContent>
          <LogRequestInspectorHeader knownRequest={REQUEST} requestId={REQUEST_ID} />
        </SharedModalContent>
      </SharedModal>
    )

    // Then
    const dialog = screen.getByRole('dialog', { name: 'Request Inspector' })
    const title = within(dialog).getByRole('heading', { name: 'Request Inspector' })
    const titleRow = title.parentElement
    const header = titleRow?.parentElement
    const descriptionId = dialog.getAttribute('aria-describedby')
    const description = descriptionId ? document.getElementById(descriptionId) : null
    if (!titleRow || !header || !description) throw new Error('Request inspector header structure is missing')

    expect(header).toHaveClass('min-w-0', 'shrink-0', 'px-4', 'pb-3', 'pt-3', 'sm:px-5', 'sm:pb-4', 'sm:pt-4.5')
    expect(titleRow).toHaveClass('min-w-0', 'flex-wrap', 'items-start')
    expect(title).toHaveClass('min-w-0', 'flex-1', 'break-words')
    expect(description).toHaveClass('pr-16', 'lg:pr-0')
    expect(within(titleRow).getByText('Completed')).toHaveClass('max-w-full', 'shrink-0')
    expect(within(header).getByText(REQUEST_ID.toString()).closest('.max-w-3xl')).toHaveClass('min-w-0')
  })
})

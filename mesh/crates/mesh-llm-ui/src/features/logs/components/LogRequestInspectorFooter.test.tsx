import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import type { LogRequestDetailTab } from '@/features/logs/lib/log-request-details'

const queries = vi.hoisted(() => ({
  summary: vi.fn(),
  events: vi.fn(),
  artifacts: vi.fn(),
  attempts: vi.fn()
}))

vi.mock('@/features/logs/api/use-log-request-details-query', () => ({
  useLogRequestSummaryQuery: (...args: unknown[]) => queries.summary(...args),
  useLogRequestEventsQuery: (...args: unknown[]) => queries.events(...args),
  useLogRequestArtifactsQuery: (...args: unknown[]) => queries.artifacts(...args),
  useLogRequestAttemptsQuery: (...args: unknown[]) => queries.attempts(...args)
}))

import { LogRequestDetails } from '@/features/logs/components/LogRequestDetails'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

function request(overrides: Partial<Pick<LogRequest, 'outcome' | 'source' | 'terminalAt'>> = {}): LogRequest {
  return {
    requestId: REQUEST_ID,
    outcome: 'completed',
    createdAt: '2026-08-08T12:00:00Z',
    terminalAt: '2026-08-08T12:00:01Z',
    route: 'chat_completions',
    model: 'Qwen3',
    provider: 'mesh',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable',
    ...overrides
  }
}

function ready<T>(data: T) {
  return { data, isLoading: false, isError: false }
}

function renderDetails(tab: LogRequestDetailTab, onClose = vi.fn()) {
  return render(<LogRequestDetails embedded onBack={onClose} onTabChange={vi.fn()} requestId={REQUEST_ID} tab={tab} />)
}

describe('Request Inspector footer', () => {
  beforeEach(() => {
    for (const query of Object.values(queries)) query.mockReset()
    queries.summary.mockReturnValue(ready(request()))
    queries.events.mockReturnValue(ready({ items: [] }))
    queries.artifacts.mockReturnValue(ready({ items: [] }))
    queries.attempts.mockReturnValue(ready({ items: [] }))
  })

  it.each(['overview', 'payloads', 'timeline', 'diagnostics'] as const)(
    'keeps one scroll body and fixed actions visible on %s',
    (tab) => {
      const view = renderDetails(tab)
      const footer = screen.getByRole('contentinfo', { name: 'Request inspector actions' })
      const scrollBodies = view.container.querySelectorAll('[data-request-inspector-scroll="body"]')
      const scrollBody = scrollBodies.item(0)

      expect(footer).toHaveClass('shrink-0')
      expect(within(footer).getByRole('button', { name: 'Close' })).toBeInTheDocument()
      expect(within(footer).getByRole('button', { name: 'Delete terminal request' })).toBeInTheDocument()
      expect(scrollBodies).toHaveLength(1)
      expect(view.container.querySelectorAll('.overflow-y-auto')).toHaveLength(1)
      expect(scrollBody).toHaveClass(
        'min-h-0',
        'flex-1',
        'overflow-y-auto',
        'overscroll-y-contain',
        'outline-none',
        'focus-visible:outline-2',
        'focus-visible:outline-offset-[-2px]',
        'focus-visible:outline-accent',
        'focus-visible:outline-solid'
      )
      expect(scrollBody).toHaveAttribute('tabindex', '0')
    }
  )

  it('always closes through the existing details close action', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    renderDetails('overview', onClose)

    const footer = screen.getByRole('contentinfo', { name: 'Request inspector actions' })
    await user.click(within(footer).getByRole('button', { name: 'Close' }))

    expect(onClose).toHaveBeenCalledOnce()
  })

  it('uses the shared responsive action strip with directly aligned equal-height controls', () => {
    renderDetails('overview')

    const footer = screen.getByRole('contentinfo', { name: 'Request inspector actions' })
    const close = within(footer).getByRole('button', { name: 'Close' })
    const deleteRequest = within(footer).getByRole('button', { name: 'Delete terminal request' })

    expect(footer).toHaveClass(
      'min-w-0',
      'shrink-0',
      'flex-col',
      'gap-2',
      'px-4',
      'py-2.5',
      'sm:flex-row',
      'sm:flex-wrap',
      'sm:items-center',
      'sm:justify-end'
    )
    expect(footer).not.toHaveClass('fixed', 'sticky')
    expect(close.parentElement).toBe(footer)
    expect(deleteRequest.parentElement).toBe(footer)
    for (const action of [close, deleteRequest]) {
      expect(action).toHaveClass('min-h-11', 'w-full', 'sm:w-auto', 'sm:min-w-24', 'lg:min-h-8')
    }
  })

  it('opens the existing audited deletion confirmation from the footer', async () => {
    const user = userEvent.setup()
    renderDetails('timeline')

    const footer = screen.getByRole('contentinfo', { name: 'Request inspector actions' })
    await user.click(within(footer).getByRole('button', { name: 'Delete terminal request' }))

    expect(screen.getByRole('dialog', { name: 'Delete terminal request?' })).toBeInTheDocument()
    expect(screen.getByText('Required audit reason')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Confirm deletion' })).toBeDisabled()
  })

  it.each([
    ['active durable', request({ outcome: 'active', terminalAt: undefined })],
    ['terminal active-source', request({ source: 'active' })]
  ])('hides deletion for a %s request while preserving Close', (_name, summary) => {
    queries.summary.mockReturnValue(ready(summary))
    renderDetails('diagnostics')

    const footer = screen.getByRole('contentinfo', { name: 'Request inspector actions' })
    expect(within(footer).getByRole('button', { name: 'Close' })).toBeInTheDocument()
    expect(within(footer).queryByRole('button', { name: 'Delete terminal request' })).not.toBeInTheDocument()
  })
})

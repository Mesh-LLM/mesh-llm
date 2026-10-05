import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import {
  LogRequestDiagnostics,
  type LogRequestDiagnosticsProps
} from '@/features/logs/components/LogRequestDiagnostics'

const LONG_VALUE = 'machine-value-without-break-opportunities'.repeat(12)
const REQUEST: LogRequest = {
  requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000001'),
  outcome: 'failed',
  createdAt: '2026-08-08T12:00:00Z',
  terminalAt: '2026-08-08T12:00:01Z',
  route: LONG_VALUE,
  model: LONG_VALUE,
  provider: LONG_VALUE,
  engine: 'skippy',
  statusCode: 502,
  source: 'durable'
}

const READY_DIAGNOSTICS: LogRequestDiagnosticsProps = {
  request: REQUEST,
  events: [],
  attempts: [],
  artifacts: [],
  requestLoading: false,
  requestError: false,
  eventsLoading: false,
  eventsError: false,
  attemptsLoading: false,
  attemptsError: false,
  artifactsLoading: false,
  artifactsError: false
}

describe('LogRequestDiagnostics layout', () => {
  it('keeps the primary loading state compact and wrap-safe', () => {
    // Given / When
    render(<LogRequestDiagnostics {...READY_DIAGNOSTICS} request={undefined} requestLoading />)

    // Then
    expect(screen.getByRole('status')).toHaveClass(
      'min-w-0',
      'break-words',
      'px-[var(--panel-x)]',
      'py-[var(--panel-y)]'
    )
  })

  it('keeps failed summary evidence and secondary notices inside the diagnostics width', () => {
    // Given / When
    const view = render(<LogRequestDiagnostics {...READY_DIAGNOSTICS} attemptsLoading eventsLoading />)

    // Then
    const diagnostics = screen.getByRole('region', { name: 'Request diagnostics' })
    const summaryState = view.container.querySelector('[data-diagnostic-state="failed"]')
    const summary = screen.getByLabelText('Diagnostic summary')
    const terminal = screen.getByRole('region', { name: 'Terminal record' })
    const secondaryStatuses = within(diagnostics)
      .getAllByRole('status')
      .filter((status) => !status.hasAttribute('data-diagnostic-state'))
    const noticePanel = secondaryStatuses[0]?.parentElement
    if (!noticePanel) throw new Error('Diagnostic query notice panel is missing')
    const queryNotices = within(noticePanel).getAllByRole('status')

    expect(diagnostics).toHaveClass('min-w-0', 'space-y-[var(--shell-normal)]')
    expect(summaryState).toHaveClass('min-w-0', 'gap-2', 'px-[var(--panel-x)]', 'py-[var(--panel-y)]', 'sm:gap-3')
    expect(summaryState?.lastElementChild).toHaveClass('max-w-full', 'shrink-0')
    expect(summary).toHaveClass('min-w-0')
    expect(within(summary).getByText(LONG_VALUE)).toHaveClass('[overflow-wrap:anywhere]')
    expect(terminal).toHaveClass('min-w-0', 'px-[var(--panel-x)]', 'py-[var(--panel-y)]')
    expect(terminal.querySelectorAll('p').item(1)).toHaveClass('[overflow-wrap:anywhere]')
    expect(queryNotices).toHaveLength(2)
    for (const notice of queryNotices) expect(notice).toHaveClass('min-w-0', 'break-words')
    expect(noticePanel).toHaveClass('min-w-0', 'gap-2', 'px-[var(--panel-x)]', 'py-[var(--panel-y)]')
  })
})

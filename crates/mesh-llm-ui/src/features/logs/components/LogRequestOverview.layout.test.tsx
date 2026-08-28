import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { LogRequestOverview } from '@/features/logs/components/LogRequestOverview'

const LONG_MODEL = 'model-without-break-opportunities'.repeat(12)
const REQUEST: LogRequest = {
  requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000001'),
  outcome: 'completed',
  createdAt: '2026-08-08T12:00:00Z',
  terminalAt: '2026-08-08T12:00:01Z',
  route: 'chat_completions',
  model: LONG_MODEL,
  provider: 'mesh',
  engine: 'skippy',
  statusCode: 200,
  source: 'durable'
}

const EMPTY_RETAINED_STATE = { items: [], loading: false, error: false }

describe('LogRequestOverview layout', () => {
  it('uses a narrow-first metric grid with compact cells and wrap-safe machine values', () => {
    // Given / When
    render(
      <LogRequestOverview
        artifacts={EMPTY_RETAINED_STATE}
        attempts={EMPTY_RETAINED_STATE}
        events={EMPTY_RETAINED_STATE}
        request={REQUEST}
      />
    )

    // Then
    const metrics = screen.getByLabelText('Request metrics')
    const statusCell = within(metrics).getByText('Status').closest('dt')?.parentElement
    if (!statusCell) throw new Error('Request status metric cell is missing')

    expect(metrics).toHaveClass('grid-cols-1', 'sm:grid-cols-2', 'lg:grid-cols-3', 'xl:grid-cols-6')
    expect(statusCell).toHaveClass('min-w-0', 'px-[var(--panel-x)]', 'py-[var(--panel-y)]', 'sm:px-4', 'sm:py-4')
    expect(within(metrics).getByText(LONG_MODEL)).toHaveClass('min-w-0', '[overflow-wrap:anywhere]')
  })
})

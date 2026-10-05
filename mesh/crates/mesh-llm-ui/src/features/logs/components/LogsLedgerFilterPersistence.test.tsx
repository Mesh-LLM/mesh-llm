import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { parseLogsLedgerSearch } from '@/features/logs/lib/log-search'

const queryState = vi.hoisted(() => ({ current: {} }))
const auditQueryState = vi.hoisted(() => ({ current: {} }))
const liveState = vi.hoisted(() => ({ current: { state: 'connected', liveRequestIds: [] } }))

vi.mock('@/features/logs/api/use-logs-ledger-query', () => ({
  useLogsLedgerQuery: () => queryState.current
}))

vi.mock('@/features/logs/api/use-logs-audit-query', () => ({
  useLogsAuditQuery: () => auditQueryState.current
}))

vi.mock('@/features/logs/api/use-logs-live-recovery', () => ({
  useLogsLiveRecovery: () => liveState.current
}))

import { LogsLedger } from '@/features/logs/components/LogsLedger'

const REQUEST_A = '00000000-0000-4000-8000-000000000001'
const REQUEST_B = '00000000-0000-4000-8000-000000000002'

function request(id: string, outcome: LogRequest['outcome'], source: LogRequest['source']): LogRequest {
  return {
    requestId: LogRequestId.parse(id),
    outcome,
    createdAt: '2026-08-04T12:00:00Z',
    terminalAt: outcome === 'active' ? undefined : '2026-08-04T12:00:01Z',
    route: 'chat_completions',
    model: 'Qwen3',
    provider: 'mesh',
    engine: 'raw_ingress',
    statusCode: outcome === 'completed' ? 200 : undefined,
    source
  }
}

function supported(rows: readonly LogRequest[]) {
  return {
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    data: { state: 'supported', value: { items: rows, nextCursor: undefined } }
  }
}

describe('LogsLedger filter persistence', () => {
  beforeEach(() => {
    queryState.current = supported([request(REQUEST_A, 'failed', 'durable')])
    auditQueryState.current = supported([])
    liveState.current = { state: 'connected', liveRequestIds: [] }
  })

  it('keeps request filters open through a polling and refetch-like parent update', async () => {
    const user = userEvent.setup()
    const props = {
      onSearchChange: vi.fn(),
      search: parseLogsLedgerSearch({})
    }
    const view = render(<LogsLedger {...props} />)
    const trigger = screen.getByRole('button', { name: 'Filter event logs' })

    await user.click(trigger)
    expect(screen.getByRole('dialog', { name: 'Event log filters' })).toBeInTheDocument()

    liveState.current = { state: 'polling', liveRequestIds: [] }
    queryState.current = {
      ...supported([request(REQUEST_A, 'completed', 'durable'), request(REQUEST_B, 'active', 'active')]),
      isFetching: true
    }
    view.rerender(<LogsLedger {...props} />)

    expect(screen.getByText('Updating')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Filter event logs' })).toBe(trigger)
    expect(trigger).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('dialog', { name: 'Event log filters' })).toBeInTheDocument()
  })
})

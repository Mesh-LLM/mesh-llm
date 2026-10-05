import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { parseLogsLedgerSearch } from '@/features/logs/lib/log-search'

const requestQuery = vi.hoisted(() => ({ current: {} }))
const auditQuery = vi.hoisted(() => ({ current: {} }))

vi.mock('@/features/logs/api/use-logs-ledger-query', () => ({
  useLogsLedgerQuery: () => requestQuery.current
}))
vi.mock('@/features/logs/api/use-logs-audit-query', () => ({
  useLogsAuditQuery: () => auditQuery.current
}))
vi.mock('@/features/logs/api/use-logs-live-recovery', () => ({
  useLogsLiveRecovery: () => ({ state: 'connected', liveRequestIds: [] })
}))

import { LogsLedger } from '@/features/logs/components/LogsLedger'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'
const ENDPOINT_ID = '9f0c4cbe8cb7a8d5d577c20e50ef03fd2f63a2e7fd9897c155823bcbb281bb04'
const REQUEST: LogRequest = {
  requestId: LogRequestId.parse(REQUEST_ID),
  outcome: 'completed',
  createdAt: '2026-08-08T12:00:00Z',
  terminalAt: '2026-08-08T12:00:01Z',
  route: 'chat_completions',
  model: 'Qwen3',
  provider: 'mesh',
  engine: 'skippy',
  statusCode: 200,
  source: 'durable',
  callerEndpointId: ENDPOINT_ID,
  callerAddr: '203.0.113.24:48712',
  callerPathType: 'remote_quic_http'
}

function supported<T>(items: readonly T[]) {
  return {
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    data: { state: 'supported', value: { items } }
  }
}

describe('request caller identity in the ledger', () => {
  beforeEach(() => {
    requestQuery.current = supported([REQUEST])
    auditQuery.current = supported([])
  })

  it('adds the caller to the Origin column as a second identity line', () => {
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    const row = screen.getByRole('row', { name: `Inspect request ${REQUEST_ID}` })
    const caller = row.querySelector('[data-log-origin-caller]')
    const path = row.querySelector('[data-log-origin-path]')
    expect(caller).toHaveTextContent('9f0c…bb04')
    expect(path).toHaveTextContent('Remote QUIC HTTP')
  })

  it.each([ENDPOINT_ID, '203.0.113.24:48712', 'remote_quic_http'])(
    'matches caller field %s in loaded-window search',
    async (query) => {
      const user = userEvent.setup()
      render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

      await user.type(screen.getByRole('textbox', { name: 'Search loaded event window' }), query)

      const table = screen.getByRole('table', { name: 'MeshLLM event logs' })
      expect(within(table).getByRole('row', { name: `Inspect request ${REQUEST_ID}` })).toBeInTheDocument()
    }
  )
})

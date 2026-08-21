import '@testing-library/jest-dom/vitest'

import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import { LogsApiError } from '@/features/logs/api/client'
import type { LogAuditEntry, LogRequest } from '@/features/logs/api/schemas'
import {
  HARNESS_LOG_FIXTURES,
  HARNESS_LOG_SCENARIO_IDS,
  HARNESS_REFERENCE_TIME
} from '@/features/logs/lib/log-fixtures'
import { parseLogsLedgerSearch } from '@/features/logs/lib/log-search'

const queryState = vi.hoisted(() => ({ current: {} }))
const auditQueryState = vi.hoisted(() => ({ current: {} }))
const liveState = vi.hoisted(() => {
  const togglePolling = vi.fn()
  return {
    current: { state: 'connected', liveRequestIds: [], pollingEnabled: true, togglePolling },
    togglePolling
  }
})
const useLogsLiveRecoveryMock = vi.hoisted(() => vi.fn())
const useLogsAuditQueryMock = vi.hoisted(() => vi.fn())
const useLogsLedgerQueryMock = vi.hoisted(() => vi.fn())

vi.mock('@/features/logs/api/use-logs-ledger-query', () => ({
  useLogsLedgerQuery: useLogsLedgerQueryMock
}))

vi.mock('@/features/logs/api/use-logs-audit-query', () => ({
  useLogsAuditQuery: useLogsAuditQueryMock
}))

vi.mock('@/features/logs/api/use-logs-live-recovery', () => ({
  useLogsLiveRecovery: useLogsLiveRecoveryMock
}))

import { LogsLedger } from '@/features/logs/components/LogsLedger'

const REQUEST_A = '00000000-0000-4000-8000-000000000001'

function request(id: string, outcome: LogRequest['outcome'], source: LogRequest['source']): LogRequest {
  return {
    requestId: LogRequestId.parse(id),
    outcome,
    createdAt: '2026-08-04T12:00:00Z',
    terminalAt: outcome === 'active' ? undefined : '2026-08-04T12:00:01Z',
    route: 'reserve',
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    statusCode: outcome === 'completed' ? 200 : undefined,
    source
  }
}

function audit(code: string, occurredAt: string, sequence: number): LogAuditEntry {
  return {
    entryId: `audit-${sequence}`,
    occurredAt,
    source: 'mesh',
    code,
    severity: 'info',
    sequence
  }
}

function supported<T>(rows: readonly T[], nextCursor?: string, incomplete = false) {
  return {
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    data: {
      state: 'supported',
      value: {
        items: rows,
        nextCursor: nextCursor ? { toString: () => nextCursor } : undefined,
        ...(incomplete ? { incomplete: true } : {})
      }
    }
  }
}

describe('LogsLedger', () => {
  beforeEach(() => {
    useLogsLedgerQueryMock.mockReset()
    useLogsLedgerQueryMock.mockImplementation(() => queryState.current)
    useLogsLiveRecoveryMock.mockReset()
    useLogsLiveRecoveryMock.mockImplementation(() => liveState.current)
    liveState.togglePolling.mockReset()
    liveState.current = {
      state: 'connected',
      liveRequestIds: [],
      pollingEnabled: true,
      togglePolling: liveState.togglePolling
    }
    queryState.current = supported(
      [request(REQUEST_A, 'failed', 'durable'), request(REQUEST_A, 'active', 'active')],
      'next-page'
    )
    auditQueryState.current = supported([])
    useLogsAuditQueryMock.mockReset()
    useLogsAuditQueryMock.mockImplementation(() => auditQueryState.current)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('shares one rolling request scope across requests, audits, and maintenance while keeping explicit and inspector state stable', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-08-04T12:00:00Z'))
    const relativeSearch = parseLogsLedgerSearch({ timeRange: '1h' })
    const view = render(<LogsLedger search={relativeSearch} onSearchChange={vi.fn()} />)

    expect(useLogsLedgerQueryMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ from: '2026-08-04T11:00:00.000Z', to: '2026-08-04T12:00:00.000Z' })
    )
    expect(useLogsAuditQueryMock).toHaveBeenLastCalledWith({
      from: '2026-08-04T11:00:00.000Z',
      to: '2026-08-04T12:00:00.000Z'
    })

    act(() => vi.advanceTimersByTime(60_000))

    const rollingScope = {
      from: '2026-08-04T11:01:00.000Z',
      to: '2026-08-04T12:01:00.000Z'
    }
    expect(useLogsLedgerQueryMock).toHaveBeenLastCalledWith(expect.objectContaining(rollingScope))
    expect(useLogsAuditQueryMock).toHaveBeenLastCalledWith(rollingScope)

    const callsBeforeInspector = useLogsLedgerQueryMock.mock.calls.length
    view.rerender(
      <LogsLedger
        search={parseLogsLedgerSearch({ ...relativeSearch, inspectType: 'audit', inspectId: 'audit-1' })}
        onSearchChange={vi.fn()}
      />
    )
    const inspectorScope = useLogsLedgerQueryMock.mock.calls.at(-1)?.[0]
    expect(useLogsLedgerQueryMock.mock.calls).toHaveLength(callsBeforeInspector + 1)
    expect(inspectorScope).toEqual(expect.objectContaining(rollingScope))

    const explicitSearch = parseLogsLedgerSearch({ from: '2026-08-04T00:00:00Z', to: '2026-08-04T01:00:00Z' })
    view.rerender(<LogsLedger search={explicitSearch} onSearchChange={vi.fn()} />)
    act(() => vi.advanceTimersByTime(60_000))

    expect(useLogsLedgerQueryMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ from: '2026-08-04T00:00:00Z', to: '2026-08-04T01:00:00Z' })
    )
  })

  it('uses resolved preset bounds consistently when a legacy URL also contains explicit bounds', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-08-04T13:00:00Z'))
    render(
      <LogsLedger
        search={parseLogsLedgerSearch({
          timeRange: '1h',
          from: '2020-01-01T00:00:00Z',
          to: '2020-01-01T01:00:00Z'
        })}
        onSearchChange={vi.fn()}
      />
    )

    const resolvedBounds = { from: '2026-08-04T12:00:00.000Z', to: '2026-08-04T13:00:00.000Z' }
    expect(useLogsLedgerQueryMock).toHaveBeenLastCalledWith(expect.objectContaining(resolvedBounds))
    expect(useLogsAuditQueryMock).toHaveBeenLastCalledWith(resolvedBounds)
    expect(screen.getByLabelText('Chart time range')).toHaveValue('1h')
    expect(screen.getByText('Total').closest('.panel-shell')).toHaveTextContent('1')
    expect(screen.getByRole('region', { name: 'Request records' })).toHaveTextContent(
      'Selected range: Last hour · retained records only'
    )
  })

  it('describes lifetime KPIs as retained records in the selected range', () => {
    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    const chartRange = screen.getByLabelText('Chart time range')
    expect(within(chartRange).getByRole('option', { name: 'Lifetime' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Request records' })).toHaveTextContent(
      'Selected range: Lifetime · retained records only'
    )
    expect(screen.getByText('Total').closest('.panel-shell')).toHaveTextContent('Retained records')
    expect(screen.getByLabelText('Retained request records over Lifetime')).toBeInTheDocument()
  })

  it('presents the seven-day ledger window truthfully in the chart selector', () => {
    render(<LogsLedger search={parseLogsLedgerSearch({ timeRange: '7d' })} onSearchChange={vi.fn()} />)

    const chartRange = screen.getByLabelText('Chart time range')
    expect(chartRange).toHaveValue('7d')
    expect(within(chartRange).getByRole('option', { name: 'Last week' })).toBeInTheDocument()
  })

  it('keeps an active row in place when it supersedes durable history and renders table pagination', async () => {
    const user = userEvent.setup()
    const onSearchChange = vi.fn()
    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={onSearchChange} />)

    expect(screen.getAllByText(REQUEST_A)).toHaveLength(1)
    expect(screen.getAllByText('active')).toHaveLength(2)
    const rowsPerPage = screen.getByRole('combobox', { name: 'Rows per page' })
    expect(rowsPerPage).toBeInTheDocument()
    expect(rowsPerPage).toHaveValue(String(20))
    expect(screen.getByRole('button', { name: 'Go to previous page' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Go to next page' })).toBeDisabled()

    await user.click(screen.getByRole('row', { name: `Inspect request ${REQUEST_A}` }))
    expect(onSearchChange).toHaveBeenCalledWith(
      expect.objectContaining({
        focusRequestId: REQUEST_A,
        inspectType: 'request',
        inspectId: REQUEST_A,
        tab: 'overview'
      })
    )
  })

  it('clears time and category filters with an accessible reset action', async () => {
    const user = userEvent.setup()
    const onSearchChange = vi.fn()
    render(
      <LogsLedger
        search={parseLogsLedgerSearch({ from: '2026-08-01T00:00:00Z', model: 'Qwen3', provider: 'reserve-a' })}
        onSearchChange={onSearchChange}
      />
    )

    await user.click(screen.getByRole('button', { name: 'Reset view' }))
    expect(onSearchChange).toHaveBeenCalledWith({})
  })

  it('applies inclusive audit bounds by instant across offsets', () => {
    // Given
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-08-04T12:00:00Z'))
    queryState.current = supported([])
    auditQueryState.current = supported([
      audit('before_lower_bound', '2026-08-04T11:30:00+01:00', 1),
      audit('at_lower_bound', '2026-08-04T12:00:00+01:00', 2),
      audit('at_upper_bound', '2026-08-04T10:00:00-02:00', 3),
      audit('after_upper_bound', '2026-08-04T11:30:00-01:00', 4)
    ])

    try {
      // When
      render(<LogsLedger search={parseLogsLedgerSearch({ timeRange: '1h' })} onSearchChange={vi.fn()} />)

      // Then
      expect(screen.getByRole('row', { name: 'Inspect operational event at_lower_bound' })).toBeInTheDocument()
      expect(screen.getByRole('row', { name: 'Inspect operational event at_upper_bound' })).toBeInTheDocument()
      expect(
        screen.queryByRole('row', { name: 'Inspect operational event before_lower_bound' })
      ).not.toBeInTheDocument()
      expect(screen.queryByRole('row', { name: 'Inspect operational event after_upper_bound' })).not.toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it('applies explicit historic from/to bounds to the bounded loaded audit window', () => {
    queryState.current = supported([])
    auditQueryState.current = supported([
      audit('before_explicit_window', '2026-08-03T23:59:59Z', 1),
      audit('at_explicit_start', '2026-08-04T01:00:00+01:00', 2),
      audit('inside_explicit_window', '2026-08-04T00:30:00Z', 3),
      audit('at_explicit_end', '2026-08-03T21:00:00-04:00', 4),
      audit('after_explicit_window', '2026-08-04T01:00:01Z', 5)
    ])

    render(
      <LogsLedger
        search={parseLogsLedgerSearch({ from: '2026-08-04T00:00:00Z', to: '2026-08-04T01:00:00Z' })}
        onSearchChange={vi.fn()}
      />
    )

    expect(useLogsAuditQueryMock).toHaveBeenCalledWith({
      from: '2026-08-04T00:00:00Z',
      to: '2026-08-04T01:00:00Z'
    })

    for (const code of ['at_explicit_start', 'inside_explicit_window', 'at_explicit_end']) {
      expect(screen.getByRole('row', { name: `Inspect operational event ${code}` })).toBeInTheDocument()
    }
    expect(
      screen.queryByRole('row', { name: 'Inspect operational event before_explicit_window' })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('row', { name: 'Inspect operational event after_explicit_window' })
    ).not.toBeInTheDocument()
    expect(screen.getByText(/events in this bounded loaded window/i)).toBeVisible()
  })

  it('discloses when audit server pagination reaches the operational safety cap', () => {
    queryState.current = supported([])
    auditQueryState.current = supported([audit('maintenance_operation', '2026-08-04T12:00:00Z', 1)], 'next', true)

    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    expect(screen.getByText('Operational window is bounded').closest('[role="status"]')).toHaveTextContent(
      'The unified table shows the first 1,000 only'
    )
  })

  it('places mesh identity, live status, and cleanup in the banner while keeping export in ledger controls', () => {
    render(<LogsLedger search={parseLogsLedgerSearch({ model: 'Qwen3' })} onSearchChange={vi.fn()} />)

    const controls = screen.getByRole('region', { name: 'Event log controls' })
    expect(within(controls).getByText(/events in this bounded loaded window/i)).toBeVisible()
    expect(within(controls).getByLabelText('Search loaded event window')).toBeVisible()
    expect(within(controls).queryByRole('heading')).not.toBeInTheDocument()
    expect(within(controls).getByRole('button', { name: 'Export view' })).toBeVisible()
    expect(within(controls).queryByRole('button', { name: 'Clean up logs' })).not.toBeInTheDocument()

    const infoBanner = screen.getByRole('region', { name: 'System logs' })
    expect(within(infoBanner).getByRole('heading', { level: 1, name: 'System logs' })).toHaveAttribute(
      'id',
      'logs-ledger-title'
    )
    expect(infoBanner).toHaveTextContent('Monitor request activity and operational events from this MeshLLM host.')
    expect(infoBanner).toHaveTextContent('Live')
    expect(infoBanner).toHaveTextContent('Local only')
    expect(within(infoBanner).getByRole('button', { name: 'Clean up logs' })).toBeVisible()
    expect(infoBanner).not.toContainElement(screen.getByRole('button', { name: 'Export view' }))
    expect(screen.queryByRole('button', { name: 'Dead-letter retry' })).not.toBeInTheDocument()
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
    expect(screen.queryByRole('region', { name: 'Log operations' })).not.toBeInTheDocument()
  })

  it('keeps banner cleanup before chart controls and ledger export in the keyboard order', async () => {
    const user = userEvent.setup()
    // A filtered search keeps the Reset control enabled so it participates in the tab order.
    render(<LogsLedger search={parseLogsLedgerSearch({ model: 'Qwen3' })} onSearchChange={vi.fn()} />)

    await user.tab()
    expect(screen.getByRole('button', { name: 'Clean up logs' })).toHaveFocus()
    await user.tab()
    expect(screen.getByLabelText('Bucket interval')).toHaveFocus()
    await user.tab()
    expect(screen.getByLabelText('Chart time range')).toHaveFocus()
    await user.tab()
    expect(screen.getByLabelText('Search loaded event window')).toHaveFocus()
    await user.tab()
    expect(screen.getByRole('button', { name: 'Reset view' })).toHaveFocus()
    await user.tab()
    expect(screen.getByRole('button', { name: /^Filter event logs/ })).toHaveFocus()
    await user.tab()
    expect(screen.getByRole('button', { name: 'Columns' })).toHaveFocus()
    await user.tab()
    expect(screen.getByRole('button', { name: 'Export view' })).toHaveFocus()
  })

  it('uses the chart selector as the only page-wide time-range control', async () => {
    const user = userEvent.setup()
    const onSearchChange = vi.fn()
    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={onSearchChange} />)

    expect(screen.queryByLabelText('Filter logs by time range')).not.toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Chart time range'), '12h')

    expect(onSearchChange).toHaveBeenCalledWith(expect.objectContaining({ timeRange: '12h' }))
  })

  it('filters the loaded page by request ID', async () => {
    const user = userEvent.setup()
    const REQUEST_B = '00000000-0000-4000-8000-000000000002'
    queryState.current = supported([
      request(REQUEST_A, 'completed', 'durable'),
      request(REQUEST_B, 'failed', 'durable')
    ])
    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    expect(screen.getAllByRole('row', { name: /Inspect request/ })).toHaveLength(2)

    await user.type(screen.getByLabelText('Search loaded event window'), REQUEST_A)

    expect(screen.getByRole('row', { name: `Inspect request ${REQUEST_A}` })).toBeInTheDocument()
    expect(screen.queryByRole('row', { name: `Inspect request ${REQUEST_B}` })).not.toBeInTheDocument()
  })

  it('excludes management self-observation from the workload ledger', () => {
    const managementRequest = { ...request(REQUEST_A, 'completed', 'durable'), route: 'management_get_status' }
    queryState.current = supported([managementRequest])
    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    expect(screen.queryByRole('row', { name: `Inspect request ${REQUEST_A}` })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /management records/i })).not.toBeInTheDocument()
  })

  it('uses document scrolling when a larger page size renders more rows', async () => {
    const user = userEvent.setup()
    queryState.current = supported(
      Array.from({ length: 50 }, (_, index) =>
        request(`00000000-0000-4000-8000-${String(index + 1).padStart(12, '0')}`, 'completed', 'durable')
      )
    )
    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    const tableRegion = screen.getByRole('region', { name: 'Scrollable event columns' })
    expect(tableRegion).toHaveClass('overflow-x-auto')
    expect(tableRegion.className).not.toMatch(/max-h-|overflow-y-/)
    expect(screen.getAllByRole('row', { name: /Inspect request/ })).toHaveLength(20)

    await user.selectOptions(screen.getByRole('combobox', { name: 'Rows per page' }), '50')

    expect(screen.getAllByRole('row', { name: /Inspect request/ })).toHaveLength(50)
    expect(screen.getByText('Page 1 of 1')).toBeVisible()
  })

  it('restores focus to the opened request after returning from the inspector', () => {
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({ focusRequestId: REQUEST_A })} />)

    expect(screen.getByRole('row', { name: `Inspect request ${REQUEST_A}` })).toHaveFocus()
  })

  it('does not steal focus while the operator filters with a retained focus request', async () => {
    const user = userEvent.setup()
    const requestB = '00000000-0000-4000-8000-000000000002'
    queryState.current = supported([request(REQUEST_A, 'completed', 'durable'), request(requestB, 'failed', 'durable')])
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({ focusRequestId: REQUEST_A })} />)
    const input = screen.getByLabelText('Search loaded event window')

    await user.click(input)
    await user.type(input, REQUEST_A)

    expect(input).toHaveFocus()
    expect(input).toHaveValue(REQUEST_A)
  })

  it('renders active fallback polling as an accessible pressed toggle', async () => {
    const user = userEvent.setup()
    liveState.current = {
      state: 'polling',
      liveRequestIds: [],
      pollingEnabled: true,
      togglePolling: liveState.togglePolling
    }
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    const pollingToggle = screen.getByRole('button', { name: 'Fallback log polling' })
    expect(
      within(screen.getByRole('region', { name: 'System logs' })).getByRole('button', {
        name: 'Fallback log polling'
      })
    ).toBe(pollingToggle)
    expect(pollingToggle).toHaveAttribute('aria-pressed', 'true')
    expect(within(pollingToggle).getByText('Polling')).toBeVisible()

    await user.click(pollingToggle)

    expect(liveState.togglePolling).toHaveBeenCalledOnce()
    expect(useLogsLiveRecoveryMock).toHaveBeenCalledWith(expect.objectContaining({ enabled: true }))
  })

  it('presents paused fallback polling with muted pressed-state semantics', () => {
    liveState.current = {
      state: 'polling',
      liveRequestIds: [],
      pollingEnabled: false,
      togglePolling: liveState.togglePolling
    }
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    const pollingToggle = screen.getByRole('button', { name: 'Fallback log polling' })
    expect(pollingToggle).toHaveAttribute('aria-pressed', 'false')
    expect(within(pollingToggle).getByText('Polling paused')).toHaveStyle({ color: 'var(--color-fg-dim)' })
    expect(screen.queryByText('Live data stale')).not.toBeInTheDocument()
  })

  it.each([
    ['connected', 'Live'],
    ['reconnecting', 'Reconnecting'],
    ['gap', 'Recovering gap'],
    ['stale', 'Live data stale']
  ])('keeps %s live status as a passive badge', (state, label) => {
    liveState.current = {
      state,
      liveRequestIds: [],
      pollingEnabled: false,
      togglePolling: liveState.togglePolling
    }
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    expect(screen.getByText(label, { exact: true })).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Fallback log polling' })).not.toBeInTheDocument()
    expect(screen.queryByText('Polling paused')).not.toBeInTheDocument()
  })

  it.each([
    ['metadata_only', true, 'Payloads · Metadata only'],
    ['redacted_artifacts', true, 'Payloads · Redacted · Ready'],
    ['redacted_artifacts', false, 'Payloads · Redacted · Unavailable'],
    ['unavailable', false, 'Payloads · Unavailable']
  ] as const)('shows the active %s payload capture state', (captureMode, artifactCaptureReady, label) => {
    render(
      <LogsLedger
        loggingStatus={{
          metadata_available: true,
          capture_mode: captureMode,
          artifact_capture_available: captureMode === 'redacted_artifacts',
          artifact_capture_ready: artifactCaptureReady
        }}
        onSearchChange={vi.fn()}
        search={parseLogsLedgerSearch({})}
      />
    )

    expect(screen.getByText(label, { exact: true })).toBeVisible()
  })

  it('keeps polling interactive while an in-flight hydration is shown as updating', () => {
    queryState.current = { ...supported([request(REQUEST_A, 'active', 'active')]), isFetching: true }
    liveState.current = {
      state: 'polling',
      liveRequestIds: [],
      pollingEnabled: false,
      togglePolling: liveState.togglePolling
    }
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    const pollingToggle = screen.getByRole('button', { name: 'Fallback log polling' })
    expect(pollingToggle).toHaveAttribute('aria-pressed', 'false')
    expect(within(pollingToggle).getByText('Updating')).toBeVisible()
  })

  it('keeps live recovery disabled when the ledger API is unsupported', () => {
    queryState.current = {
      isLoading: false,
      isError: false,
      isFetching: false,
      data: { state: 'unsupported' },
      refetch: vi.fn()
    }

    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    expect(screen.getByRole('status')).toHaveTextContent('Request window unavailable')
    expect(screen.getByRole('table', { name: 'MeshLLM event logs' })).toHaveTextContent(
      'No request or operational events are loaded yet.'
    )
    expect(useLogsLiveRecoveryMock).toHaveBeenCalledWith(expect.objectContaining({ enabled: false }))
  })

  it('renders an empty ledger as an announced state without an empty request table', () => {
    queryState.current = supported([], undefined)
    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    expect(screen.getByRole('table', { name: 'MeshLLM event logs' })).toHaveTextContent(
      'No request or operational events are loaded yet.'
    )
    expect(screen.queryByLabelText('Filter logs by time range')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Search loaded event window')).toBeInTheDocument()
  })

  it('offers a stable, labeled retry action when the logs API fails', async () => {
    const user = userEvent.setup()
    const refetch = vi.fn()
    queryState.current = { isLoading: false, isError: true, isFetching: false, data: undefined, refetch }
    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('Some log data could not be loaded')
    expect(alert).toHaveTextContent('Request history')
    expect(alert).toHaveTextContent('Unavailable')
    const retry = screen.getByRole('button', { name: 'Retry requests' })
    expect(retry).toBeEnabled()

    await user.click(retry)

    expect(refetch).toHaveBeenCalledOnce()
  })

  it('renders a composed logs skeleton while both sources are initially loading', () => {
    queryState.current = {
      isLoading: true,
      isError: false,
      isFetching: true,
      data: undefined,
      refetch: vi.fn()
    }
    auditQueryState.current = {
      isLoading: true,
      isError: false,
      isFetching: true,
      data: undefined,
      refetch: vi.fn()
    }

    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    const loadingStatus = screen.getByRole('status', { name: 'Loading system logs' })
    expect(loadingStatus).toHaveTextContent('Loading system logs')
    expect(loadingStatus.querySelector('[data-loading-region="logs-chart"]')).toBeInTheDocument()
    expect(loadingStatus.querySelector('[data-loading-region="logs-kpis"]')).toBeInTheDocument()
    expect(loadingStatus.querySelector('[data-loading-region="logs-ledger"]')).toBeInTheDocument()
    expect(loadingStatus.querySelector('[data-loading-region="logs-ledger-table"]')).toBeInTheDocument()
    expect(loadingStatus.querySelector('[data-loading-region="logs-ledger-pagination"]')).toBeInTheDocument()
    expect(loadingStatus.querySelectorAll('[data-loading-ghost-shimmer]').length).toBeGreaterThan(10)

    expect(screen.queryByRole('region', { name: 'Event log controls' })).not.toBeInTheDocument()
    expect(screen.queryByRole('table', { name: 'MeshLLM event logs' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Bucket interval')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Search loaded event window')).not.toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('keeps the usable source surface when only the other window is initially loading', () => {
    auditQueryState.current = {
      isLoading: true,
      isError: false,
      isFetching: true,
      data: undefined,
      refetch: vi.fn()
    }

    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    expect(screen.queryByRole('status', { name: 'Loading system logs' })).not.toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'MeshLLM event logs' })).toBeInTheDocument()
    expect(screen.getByLabelText(/Events over time stacked bar chart/)).toBeInTheDocument()
  })

  it('combines simultaneous source failures while preserving independent retries', async () => {
    const user = userEvent.setup()
    const refetchRequests = vi.fn()
    const refetchOperations = vi.fn()
    queryState.current = {
      isLoading: false,
      isError: true,
      isFetching: false,
      data: undefined,
      refetch: refetchRequests
    }
    auditQueryState.current = {
      isLoading: false,
      isError: true,
      isFetching: false,
      data: undefined,
      refetch: refetchOperations
    }

    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    const alert = screen.getByRole('alert')
    expect(screen.getAllByRole('alert')).toHaveLength(1)
    expect(alert).toHaveTextContent('Log data could not be loaded')
    expect(within(alert).getByRole('group', { name: 'Log data source recovery' })).toHaveTextContent(
      'Request historyUnavailable'
    )
    expect(alert).toHaveTextContent('Operational eventsUnavailable')

    await user.click(within(alert).getByRole('button', { name: 'Retry requests' }))
    expect(refetchRequests).toHaveBeenCalledOnce()
    expect(refetchOperations).not.toHaveBeenCalled()

    await user.click(within(alert).getByRole('button', { name: 'Retry operations' }))
    expect(refetchOperations).toHaveBeenCalledOnce()
  })

  it('keeps the combined recovery surface stable while one source retries', () => {
    queryState.current = {
      isLoading: false,
      isError: true,
      isFetching: true,
      data: undefined,
      refetch: vi.fn()
    }

    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('Request historyRetrying')
    expect(within(alert).getByRole('button', { name: 'Retrying…' })).toBeDisabled()
  })

  it('renders one actionable compatibility state instead of the recovery surface', () => {
    queryState.current = {
      isLoading: false,
      isError: true,
      isFetching: false,
      error: new LogsApiError(503, 'logging_schema_incompatible'),
      data: undefined,
      refetch: vi.fn()
    }
    auditQueryState.current = { ...queryState.current, refetch: vi.fn() }

    render(
      <LogsLedger
        loggingStatus={{
          metadata_available: false,
          metadata_state: 'schema_incompatible',
          schema_version: 14,
          supported_schema_version: 11,
          capture_mode: 'unavailable',
          artifact_capture_available: false,
          artifact_capture_ready: false
        }}
        search={parseLogsLedgerSearch({})}
        onSearchChange={vi.fn()}
      />
    )

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('Log database version mismatch')
    expect(alert).toHaveTextContent('Update the node to a build that supports schema v14')
    expect(alert).toHaveTextContent('Log history was left unchanged')
    expect(alert).toHaveTextContent('Database schema v14')
    expect(alert).toHaveTextContent('Runtime supports v11')
    expect(screen.queryByRole('button', { name: /Retry/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Log data source recovery' })).not.toBeInTheDocument()
    expect(screen.queryByRole('status', { name: 'Loading system logs' })).not.toBeInTheDocument()
  })

  it('uses typed HTTP compatibility details when status has not loaded yet', () => {
    queryState.current = {
      isLoading: false,
      isError: true,
      isFetching: false,
      error: new LogsApiError(503, 'logging_schema_incompatible', {
        schemaVersion: 10,
        supportedSchemaVersion: 11
      }),
      data: undefined,
      refetch: vi.fn()
    }

    render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

    expect(screen.getByRole('alert')).toHaveTextContent('This MeshLLM build cannot safely upgrade')
    expect(screen.getByText('Database schema v10')).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Retry requests' })).not.toBeInTheDocument()
  })

  it('renders the populated harness ledger across outcomes, durability states, and metadata fallbacks', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(HARNESS_REFERENCE_TIME))
    queryState.current = supported(HARNESS_LOG_FIXTURES)

    try {
      render(<LogsLedger search={parseLogsLedgerSearch({})} onSearchChange={vi.fn()} />)

      const table = screen.getByRole('table', { name: 'MeshLLM event logs' })
      for (const outcome of ['active', 'completed', 'failed', 'rejected', 'cancelled', 'dropped']) {
        expect(within(table).getAllByText(outcome).length).toBeGreaterThan(0)
      }
      expect(within(table).getAllByText('active').length).toBeGreaterThan(0)
      expect(within(table).getAllByText('durable').length).toBeGreaterThan(1)
      expect(within(table).getAllByText('—').length).toBeGreaterThan(0)
      expect(
        within(table).queryByRole('row', {
          name: `Inspect request ${HARNESS_LOG_SCENARIO_IDS.droppedCapacity.toString()}`
        })
      ).not.toBeInTheDocument()
      expect(
        within(table).queryByRole('row', {
          name: `Inspect request ${HARNESS_LOG_SCENARIO_IDS.completedActiveSource.toString()}`
        })
      ).not.toBeInTheDocument()
      expect(screen.queryByRole('button', { name: /management records/i })).not.toBeInTheDocument()
      expect(screen.getByLabelText(/Events over time stacked bar chart/)).toBeInTheDocument()
      expect(screen.queryByText('No selected events during the chart time range.')).not.toBeInTheDocument()
      expect(screen.getByRole('combobox', { name: 'Rows per page' })).toHaveValue('20')
      expect(screen.getByRole('navigation', { name: 'Loaded event rows' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Go to next page' })).toBeEnabled()
      const workloadFixtureCount = HARNESS_LOG_FIXTURES.filter(
        (row) => row.route !== 'models' && !row.route?.startsWith('management_')
      ).length
      expect(screen.getAllByText(String(workloadFixtureCount)).length).toBeGreaterThan(0)
    } finally {
      vi.useRealTimers()
    }
  })
})

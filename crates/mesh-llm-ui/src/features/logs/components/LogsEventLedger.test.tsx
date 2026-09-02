import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogAuditEntry, LogAuditSeverity, LogRequest } from '@/features/logs/api/schemas'
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

import { LogsLedger } from './LogsLedger'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'

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
  source: 'durable'
}

const AUDIT_WITHOUT_SEVERITY: LogAuditEntry = {
  entryId: 'audit-1',
  occurredAt: '2026-08-08T12:01:00Z',
  source: 'runtime',
  code: 'runtime_ready',
  sequence: 1
}

const AUDIT: LogAuditEntry = { ...AUDIT_WITHOUT_SEVERITY, severity: 'info' }

const AUDIT_SEVERITY_CASES = [
  ['info', 'lucide-info', 'var(--color-fg-dim)'],
  ['warning', 'lucide-triangle-alert', 'var(--color-warn-text)'],
  ['error', 'lucide-circle-x', 'var(--color-bad-text)']
] as const satisfies readonly (readonly [LogAuditSeverity, string, string])[]

function supported<T>(items: readonly T[]) {
  return {
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    data: { state: 'supported', value: { items } }
  }
}

function unsupported() {
  return {
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    data: { state: 'unsupported' }
  }
}

describe('unified logs event ledger', () => {
  beforeEach(() => {
    requestQuery.current = supported([REQUEST])
    auditQuery.current = supported([AUDIT])
  })

  it('identifies the ledger as system logs for this host', () => {
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    expect(screen.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
    expect(screen.getByText('Monitor request activity and operational events from this MeshLLM host.')).toBeVisible()
  })

  it('renders request and audit events in exactly one accessible bounded-window table', () => {
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    expect(screen.getAllByRole('table')).toHaveLength(1)
    const table = screen.getByRole('table', { name: 'MeshLLM event logs' })
    expect(within(table).getByText(REQUEST_ID)).toBeInTheDocument()
    expect(within(table).getByText('runtime_ready')).toBeInTheDocument()
    expect(screen.getByText(/bounded loaded window/i)).toBeInTheDocument()
    expect(screen.getByText('Scroll horizontally for all columns.')).toBeInTheDocument()
    expect(screen.queryByText('Operational audit')).not.toBeInTheDocument()
  })

  it.each(AUDIT_SEVERITY_CASES)(
    'renders %s audit state with its hidden semantic icon while preserving label and tone',
    (severity, iconClass, color) => {
      requestQuery.current = supported([])
      auditQuery.current = supported([{ ...AUDIT, severity }])
      render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

      const row = screen.getByRole('row', { name: 'Inspect operational event runtime_ready' })
      const badge = within(row).getByText(severity, { exact: true })
      const icon = badge.querySelector('svg')

      expect(icon).toBeInTheDocument()
      expect(icon).toHaveClass(iconClass)
      expect(icon).toHaveAttribute('aria-hidden', 'true')
      expect(badge).toHaveTextContent(severity)
      expect(badge).toHaveStyle({ color })
    }
  )

  it('keeps missing audit severity as a text-only muted badge', () => {
    requestQuery.current = supported([])
    auditQuery.current = supported([AUDIT_WITHOUT_SEVERITY])
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    const row = screen.getByRole('row', { name: 'Inspect operational event runtime_ready' })
    const badge = within(row).getByText('Not provided', { exact: true })

    expect(badge.querySelector('svg')).not.toBeInTheDocument()
    expect(badge).toHaveStyle({ color: 'var(--color-fg-dim)' })
  })

  it('presents ledger columns in the operator scan order', () => {
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    const table = screen.getByRole('table', { name: 'MeshLLM event logs' })
    const headers = within(table)
      .getAllByRole('columnheader')
      .map((header) => header.textContent?.trim())

    expect(headers).toEqual(['Occurred', 'Category', 'State', 'Origin', 'Event', 'Context'])
  })

  it('keeps request and audit capability windows independent', () => {
    requestQuery.current = unsupported()
    const view = render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    expect(screen.getByRole('table', { name: 'MeshLLM event logs' })).toHaveTextContent('runtime_ready')
    expect(screen.getByText(/request window unavailable/i)).toBeInTheDocument()

    requestQuery.current = supported([REQUEST])
    auditQuery.current = unsupported()
    view.rerender(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    expect(screen.getByRole('table', { name: 'MeshLLM event logs' })).toHaveTextContent(REQUEST_ID)
    expect(screen.getByText(/operational window unavailable/i)).toBeInTheDocument()
  })

  it('keeps current categories visible and omits Iroh without an authoritative category', async () => {
    const user = userEvent.setup()
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    await user.click(screen.getByRole('button', { name: /Filter event logs/ }))
    const filter = screen.getByRole('dialog', { name: 'Event log filters' })
    for (const category of ['Requests', 'System', 'QUIC', 'Gossip']) {
      expect(within(filter).getByRole('checkbox', { name: new RegExp(category, 'i') })).toBeChecked()
    }
    expect(within(filter).queryByRole('checkbox', { name: /Iroh/i })).not.toBeInTheDocument()
  })

  it('applies the category selection to both the event chart and ledger rows', () => {
    const view = render(
      <LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({ categories: ['requests'] })} />
    )

    let legend = screen.getByRole('list', { name: 'Visible event categories' })
    let table = screen.getByRole('table', { name: 'MeshLLM event logs' })
    expect(legend).toHaveTextContent('Requests1')
    expect(legend).not.toHaveTextContent('System')
    expect(within(table).getByText(REQUEST_ID)).toBeInTheDocument()
    expect(within(table).queryByText(AUDIT.code)).not.toBeInTheDocument()

    view.rerender(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({ categories: ['system'] })} />)

    legend = screen.getByRole('list', { name: 'Visible event categories' })
    table = screen.getByRole('table', { name: 'MeshLLM event logs' })
    expect(legend).toHaveTextContent('System1')
    expect(legend).not.toHaveTextContent('Requests')
    expect(within(table).queryByText(REQUEST_ID)).not.toBeInTheDocument()
    expect(within(table).getByText(AUDIT.code)).toBeInTheDocument()
  })

  it.each([
    ['request source', REQUEST.source, REQUEST_ID],
    ['audit entry ID', AUDIT.entryId, AUDIT.code],
    ['displayed timestamp', new Date(REQUEST.createdAt).toLocaleString(), REQUEST_ID]
  ])('searches the displayed %s', async (_field, query, expectedRowText) => {
    const user = userEvent.setup()
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    await user.type(screen.getByRole('textbox', { name: 'Search loaded event window' }), query)

    expect(screen.getByRole('table', { name: 'MeshLLM event logs' })).toHaveTextContent(expectedRowText)
  })

  it('omits source-history controls and paginates loaded rows at page level', async () => {
    const user = userEvent.setup()
    auditQuery.current = supported(
      Array.from({ length: 20 }, (_, index): LogAuditEntry => ({
        ...AUDIT,
        entryId: `audit-${index + 1}`,
        occurredAt: `2026-08-08T12:${String(index + 1).padStart(2, '0')}:00Z`,
        code: `runtime_event_${index + 1}`,
        sequence: index + 1
      }))
    )
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    expect(screen.queryByRole('navigation', { name: 'Loaded history pages' })).not.toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Request page' })).not.toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Operational page' })).not.toBeInTheDocument()
    for (const name of [
      'Previous request page',
      'Next request page',
      'Previous operational page',
      'Next operational page'
    ]) {
      expect(screen.queryByRole('button', { name })).not.toBeInTheDocument()
    }

    const table = screen.getByRole('table', { name: 'MeshLLM event logs' })
    expect(within(table).queryByText(REQUEST_ID)).not.toBeInTheDocument()
    expect(within(table).getAllByRole('row', { name: /Inspect operational event runtime_event_/ })).toHaveLength(10)
    expect(screen.getByRole('combobox', { name: 'Rows per page' })).toHaveValue('10')
    expect(screen.getByText('Page 1 of 3')).toBeVisible()
    const pagination = screen.getByRole('navigation', { name: 'Loaded event rows' })
    await user.click(within(pagination).getByRole('button', { name: 'Go to next page' }))

    expect(within(table).queryByText(REQUEST_ID)).not.toBeInTheDocument()
    expect(within(table).getAllByRole('row', { name: /Inspect operational event runtime_event_/ })).toHaveLength(10)
    expect(screen.getByText('Page 2 of 3')).toBeVisible()

    await user.click(within(pagination).getByRole('button', { name: 'Go to next page' }))
    expect(within(table).getByText(REQUEST_ID)).toBeVisible()
    expect(screen.getByText('Page 3 of 3')).toBeVisible()
  })

  it('makes the horizontally scrollable event columns keyboard reachable', () => {
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)

    const region = screen.getByRole('region', { name: 'Scrollable event columns' })
    expect(region).toHaveAttribute('tabindex', '0')
    expect(region).toHaveClass('overflow-x-auto')
    expect(region).not.toHaveClass('overflow-y-auto')
    expect(region).not.toHaveClass('max-h-[71rem]')
    expect(region.querySelector('[data-radix-scroll-area-viewport]')).not.toBeInTheDocument()
    const table = screen.getByRole('table', { name: 'MeshLLM event logs' })
    expect(table.parentElement).toHaveClass('overflow-visible')
    expect(table.parentElement).not.toHaveClass('overflow-auto')
  })

  it('keeps the no-match table state free of pagination controls', () => {
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({ categories: 'none' })} />)

    expect(screen.getByText('No events match this loaded window.')).toBeVisible()
    expect(screen.queryByText(/Page 0 of 1/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: 'Loaded event rows' })).not.toBeInTheDocument()
  })

  it('styles type-specific context labels with primary ink while keeping machine values dim', () => {
    render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch({})} />)
    const table = screen.getByRole('table', { name: 'MeshLLM event logs' })
    const contextFields = [
      {
        row: within(table).getByRole('row', { name: `Inspect request ${REQUEST_ID}` }),
        fields: [
          ['Model', 'Qwen3'],
          ['Route', 'chat_completions']
        ]
      },
      {
        row: within(table).getByRole('row', { name: 'Inspect operational event runtime_ready' }),
        fields: [
          ['Sequence', String(AUDIT.sequence)],
          ['Entry ID', AUDIT.entryId]
        ]
      }
    ] as const

    for (const { row, fields } of contextFields) {
      for (const [label, value] of fields) {
        expect(within(row).getByText(label, { exact: true })).toHaveClass('text-primary')
        expect(within(row).getByText(value, { exact: true })).toHaveClass('text-fg-dim')
      }
    }
    expect(within(table).queryByText('Request lifecycle')).not.toBeInTheDocument()
    expect(within(table).queryByText('Operational metadata')).not.toBeInTheDocument()
  })
})

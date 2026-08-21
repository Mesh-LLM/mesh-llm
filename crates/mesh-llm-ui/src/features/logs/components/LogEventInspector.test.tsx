import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogAuditEntry, LogRequest } from '@/features/logs/api/schemas'
import type { LogInspector } from '@/features/logs/lib/log-inspector'

const requestQueries = vi.hoisted(() => ({
  summary: vi.fn(),
  events: vi.fn(),
  artifacts: vi.fn(),
  attempts: vi.fn()
}))

vi.mock('@/features/logs/api/use-log-request-details-query', () => ({
  useLogRequestSummaryQuery: (...args: unknown[]) => requestQueries.summary(...args),
  useLogRequestEventsQuery: (...args: unknown[]) => requestQueries.events(...args),
  useLogRequestArtifactsQuery: (...args: unknown[]) => requestQueries.artifacts(...args),
  useLogRequestAttemptsQuery: (...args: unknown[]) => requestQueries.attempts(...args)
}))

import { LogEventInspector } from './LogEventInspector'

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
const AUDIT: LogAuditEntry = {
  entryId: 'audit-1',
  occurredAt: '2026-08-08T12:01:00Z',
  source: 'runtime',
  code: 'runtime_ready',
  severity: 'info',
  sequence: 7,
  contextVersion: 1,
  subjectKind: 'model',
  subjectId: 'unsloth/Qwen3.5-4B-GGUF',
  operationId: 'runtime-instance-7',
  requestId: REQUEST_ID,
  reasonCode: 'model_loaded',
  outcome: 'ready',
  durationMs: 412,
  numericSummaries: { layers: 36 }
}

function ready<T>(data: T) {
  return { data, isLoading: false, isError: false }
}

function InspectorHarness({
  audit = AUDIT,
  initialInspector
}: {
  readonly audit?: LogAuditEntry
  readonly initialInspector?: LogInspector
}) {
  const [inspector, setInspector] = useState<LogInspector | undefined>(initialInspector)
  return (
    <>
      <button type="button" onClick={() => setInspector({ type: 'audit', id: AUDIT.entryId })}>
        Open audit
      </button>
      <button type="button" onClick={() => setInspector({ type: 'request', id: REQUEST_ID })}>
        Open request
      </button>
      <LogEventInspector
        auditEntries={[audit]}
        inspector={inspector}
        onClose={() => setInspector(undefined)}
        onRequestTabChange={vi.fn()}
        requestTab="overview"
      />
    </>
  )
}

describe('LogEventInspector', () => {
  beforeEach(() => {
    requestQueries.summary.mockReset()
    requestQueries.events.mockReset()
    requestQueries.artifacts.mockReset()
    requestQueries.attempts.mockReset()
    requestQueries.summary.mockReturnValue(ready(REQUEST))
    requestQueries.events.mockReturnValue(ready({ items: [] }))
    requestQueries.artifacts.mockReturnValue(ready({ items: [] }))
    requestQueries.attempts.mockReturnValue(ready({ items: [] }))
  })

  it('opens an audit metadata dialog without touching request-detail APIs or exposing private fields', async () => {
    const user = userEvent.setup()
    render(<InspectorHarness />)

    await user.click(screen.getByRole('button', { name: 'Open audit' }))

    const dialog = screen.getByRole('dialog', { name: 'Operational event runtime_ready' })
    for (const value of [
      AUDIT.entryId,
      AUDIT.occurredAt,
      AUDIT.source,
      AUDIT.severity ?? 'Not provided',
      String(AUDIT.sequence),
      AUDIT.subjectKind ?? '',
      AUDIT.subjectId ?? '',
      AUDIT.operationId ?? '',
      AUDIT.requestId ?? '',
      AUDIT.reasonCode ?? '',
      AUDIT.outcome ?? '',
      `${AUDIT.durationMs} ms`,
      String(AUDIT.numericSummaries?.layers)
    ]) {
      expect(within(dialog).getByText(value, { exact: true })).toBeInTheDocument()
    }
    expect(within(dialog).getAllByRole('term')).toHaveLength(13)
    expect(within(dialog).getAllByRole('definition')).toHaveLength(13)
    expect(dialog).not.toHaveTextContent(/payload|destination|peer address|raw fields/i)
    expect(requestQueries.summary).not.toHaveBeenCalled()
    expect(requestQueries.events).not.toHaveBeenCalled()
    expect(requestQueries.artifacts).not.toHaveBeenCalled()
    expect(requestQueries.attempts).not.toHaveBeenCalled()
  })

  it('makes the event code the primary identity and promotes state above the metadata ledger', async () => {
    const user = userEvent.setup()
    render(<InspectorHarness />)

    await user.click(screen.getByRole('button', { name: 'Open audit' }))

    const dialog = screen.getByRole('dialog', { name: 'Operational event runtime_ready' })
    const title = within(dialog).getByRole('heading', { name: 'Operational event runtime_ready' })
    expect(title).toHaveTextContent(AUDIT.code)
    expect(title).toHaveClass(
      'text-[length:var(--density-type-headline)]',
      'font-semibold',
      'leading-5',
      'tracking-[-0.02em]',
      'text-fg'
    )

    const state = within(dialog).getByRole('heading', { name: 'Event state' })
    expect(state.parentElement).toContainElement(within(dialog).getByText('info', { exact: true }))
    expect(state.parentElement).toContainElement(within(dialog).getByText('ready', { exact: true }))
    expect(within(dialog).getByRole('heading', { name: 'Event metadata' })).toBeInTheDocument()
  })

  it('keeps the audit title a 1:1 typography match with the Request Inspector title', () => {
    const auditView = render(<InspectorHarness initialInspector={{ type: 'audit', id: AUDIT.entryId }} />)
    const auditDialog = screen.getByRole('dialog', { name: 'Operational event runtime_ready' })
    const auditTitle = within(auditDialog).getByRole('heading', { name: 'Operational event runtime_ready' })
    const auditClasses = Array.from(auditTitle.classList)

    auditView.unmount()
    render(<InspectorHarness initialInspector={{ type: 'request', id: REQUEST_ID }} />)
    const requestDialog = screen.getByRole('dialog', { name: 'Request Inspector' })
    const requestTitle = within(requestDialog).getByRole('heading', { name: 'Request Inspector' })
    // flex-1 is layout for the outcome badge sibling, not title typography.
    const requestClasses = Array.from(requestTitle.classList).filter((className) => className !== 'flex-1')

    expect(auditClasses).toEqual(requestClasses)
  })

  it('keeps unknown outcome values muted instead of inferring a state from substrings', async () => {
    const user = userEvent.setup()
    const audit = { ...AUDIT, outcome: 'unblocked' }
    render(<InspectorHarness audit={audit} />)

    await user.click(screen.getByRole('button', { name: 'Open audit' }))

    const dialog = screen.getByRole('dialog', { name: 'Operational event runtime_ready' })
    expect(within(dialog).getByText('unblocked', { exact: true })).toHaveStyle({ color: 'var(--color-fg-dim)' })
  })

  it('keeps the request frame fixed while audit content adapts with a named scroll body', () => {
    const auditView = render(<InspectorHarness initialInspector={{ type: 'audit', id: AUDIT.entryId }} />)
    const auditDialog = screen.getByRole('dialog', { name: 'Operational event runtime_ready' })

    expect(auditDialog).toHaveClass(
      'flex',
      'h-dvh',
      'w-full',
      'flex-col',
      'overflow-hidden',
      'rounded-none',
      'sm:h-auto',
      'sm:max-h-[min(calc(100dvh-4rem),50rem)]',
      'sm:w-[calc(100vw-2rem)]',
      'sm:max-w-[720px]',
      'sm:rounded-[var(--radius-lg)]'
    )
    expect(within(auditDialog).getByRole('region', { name: 'Operational event metadata' })).toHaveClass(
      'min-h-0',
      'flex-1',
      'overflow-y-auto'
    )

    auditView.unmount()
    render(<InspectorHarness initialInspector={{ type: 'request', id: REQUEST_ID }} />)
    expect(screen.getByRole('dialog', { name: 'Request Inspector' })).toHaveClass(
      'flex',
      'h-dvh',
      'w-full',
      'flex-col',
      'overflow-hidden',
      'rounded-none',
      'sm:h-[min(calc(100dvh-3rem),54rem)]',
      'sm:w-[calc(100vw-2rem)]',
      'sm:max-w-[1120px]',
      'sm:rounded-[var(--radius-lg)]'
    )
  })

  it('opens without scale motion, traps focus, closes on Escape, and restores trigger focus', async () => {
    const user = userEvent.setup()
    render(<InspectorHarness />)
    const trigger = screen.getByRole('button', { name: 'Open audit' })

    await user.click(trigger)
    const dialog = screen.getByRole('dialog', { name: 'Operational event runtime_ready' })
    expect(dialog).toHaveClass('data-[state=open]:animate-in', 'data-[state=closed]:animate-out')
    expect(dialog).toHaveClass('data-[state=open]:fade-in-0', 'data-[state=closed]:fade-out-0')
    expect(dialog).toHaveClass('data-[state=open]:zoom-in-100', 'data-[state=closed]:zoom-out-100')
    expect(within(dialog).getByRole('button', { name: 'Close inspector' })).toHaveFocus()

    await user.keyboard('{Escape}')

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })

  it('opens a request on Overview with retained metadata queries enabled', async () => {
    const user = userEvent.setup()
    render(<InspectorHarness />)

    await user.click(screen.getByRole('button', { name: 'Open request' }))

    const dialog = screen.getByRole('dialog', { name: 'Request Inspector' })
    expect(within(dialog).getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true')
    expect(
      within(dialog)
        .getAllByRole('tab')
        .map((tab) => tab.textContent)
    ).toEqual(['Overview', 'Payloads', 'Timeline', 'Diagnostics'])
    expect(requestQueries.summary).toHaveBeenCalledWith(LogRequestId.parse(REQUEST_ID))
    expect(requestQueries.events).toHaveBeenCalledWith(LogRequestId.parse(REQUEST_ID), true)
    expect(requestQueries.artifacts).toHaveBeenCalledWith(LogRequestId.parse(REQUEST_ID), true)
    expect(requestQueries.attempts).toHaveBeenCalledWith(LogRequestId.parse(REQUEST_ID), true)
  })

  it('keeps request identity, outcome, and close actions in the fixed header', async () => {
    const user = userEvent.setup()
    render(<InspectorHarness />)

    await user.click(screen.getByRole('button', { name: 'Open request' }))

    const dialog = screen.getByRole('dialog', { name: 'Request Inspector' })
    const title = within(dialog).getByRole('heading', { name: 'Request Inspector' })
    const header = title.parentElement?.parentElement
    expect(header).not.toBeNull()
    if (!header) throw new Error('Request inspector header is missing')
    expect(title).toBeInTheDocument()
    expect(within(header).getByText(REQUEST_ID)).toHaveClass('break-words')
    expect(within(header).getByRole('button', { name: 'Copy Request ID' })).toBeInTheDocument()
    const outcome = within(header).getByText('Completed')
    expect(outcome).toBeInTheDocument()
    expect(title.parentElement).toHaveClass('items-start', 'pr-16', 'lg:pr-12')
    expect(title.parentElement?.lastElementChild).toBe(outcome)
    expect(outcome).toHaveClass('shrink-0')
    expect(within(header).getByRole('button', { name: 'Close inspector' })).toHaveFocus()
  })

  it('exposes a fixed request shell with one internally scrolling body', async () => {
    const user = userEvent.setup()
    render(<InspectorHarness />)

    await user.click(screen.getByRole('button', { name: 'Open request' }))

    const dialog = screen.getByRole('dialog', { name: 'Request Inspector' })
    expect(dialog).toHaveAttribute('data-request-inspector-shell', 'fixed')
    expect(dialog).toHaveClass('flex', 'h-dvh', 'w-full', 'flex-col', 'overflow-hidden')
    const scrollBody = dialog.querySelector('[data-request-inspector-scroll="body"]')
    expect(scrollBody).toHaveClass('min-h-0', 'flex-1', 'overflow-y-auto')
  })

  it('gives embedded request details an accessible region name without a dangling heading reference', () => {
    render(<InspectorHarness initialInspector={{ type: 'request', id: REQUEST_ID }} />)

    const dialog = screen.getByRole('dialog', { name: 'Request Inspector' })
    const details = within(dialog).getByRole('region', { name: `Request details for ${REQUEST_ID}` })

    expect(details).not.toHaveAttribute('aria-labelledby')
  })
})

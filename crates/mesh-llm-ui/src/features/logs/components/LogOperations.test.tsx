import '@testing-library/jest-dom/vitest'

import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { LogCleanupPreviewRequest, LogCleanupRunRequest, LogDeleteRequest } from '@/features/logs/api/client'
import { LogAuditId, LogOperationId, LogPageCursor, LogRequestId } from '@/features/logs/api/ids'
import type { LogCleanupReceipt, LogDeleteReceipt, LogExport } from '@/features/logs/api/schemas'
import { HARNESS_LOG_FIXTURES } from '@/features/logs/lib/log-fixtures'
import type { LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'

const api = vi.hoisted(() => ({
  exportRequests: vi.fn(),
  previewCleanup: vi.fn(),
  runCleanup: vi.fn(),
  deleteRequest: vi.fn()
}))

vi.mock('@/features/logs/api/client', () => ({
  LogsApiClient: class {
    exportRequests = api.exportRequests
    previewCleanup = api.previewCleanup
    runCleanup = api.runCleanup
    deleteRequest = api.deleteRequest
  }
}))

import { LogOperations } from '@/features/logs/components/LogOperations'
import { LogRequestDeleteControl } from '@/features/logs/components/LogRequestDeleteControl'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')
const OPERATION_ID = LogOperationId.parse('00000000-0000-4000-8000-000000000002')
const AUDIT_ID = LogAuditId.parse('00000000-0000-4000-8000-000000000003')

function cleanupReceipt(
  state: LogCleanupReceipt['state'],
  options: {
    readonly failedArtifacts?: number
    readonly hasMore?: boolean
    readonly operationId?: LogOperationId
    readonly planned?: LogCleanupReceipt['planned']
  } = {}
): LogCleanupReceipt {
  const failedArtifacts = options.failedArtifacts ?? (state === 'partial' ? 1 : 0)
  const planned = options.planned ?? { requests: 3, events: 4, artifacts: 2, proxyRecords: 1, databaseRows: 10 }
  return {
    operationId: options.operationId ?? OPERATION_ID,
    auditId: AUDIT_ID,
    cutoffBefore: '2026-08-01T00:00:00Z',
    requestLimit: 100,
    scope: {
      source: 'durable',
      cutoffBefore: '2026-08-01T00:00:00Z',
      requestLimit: 100,
      from: '2026-07-01T00:00:00Z',
      to: '2026-08-01T00:00:00Z',
      route: 'reserve',
      model: 'Qwen/Qwen3',
      provider: 'reserve-a',
      engine: 'skippy',
      outcome: 'completed'
    },
    state,
    hasMore: options.hasMore ?? true,
    selectionFingerprint: 'safe-fingerprint',
    planned,
    executed: {
      requests: state === 'previewed' ? 0 : planned.requests,
      events: state === 'previewed' ? 0 : planned.events,
      artifacts: state === 'previewed' ? 0 : planned.artifacts,
      proxyRecords: state === 'previewed' ? 0 : planned.proxyRecords,
      databaseRows: state === 'previewed' ? 0 : planned.databaseRows
    },
    artifactDeletion: {
      removed: state === 'previewed' ? 0 : 1,
      failed: failedArtifacts,
      failureClass: failedArtifacts > 0 ? ('unsafe_path' as const) : undefined
    }
  }
}

function deleteReceipt(
  state: LogDeleteReceipt['state'],
  options: { readonly failedArtifacts?: number; readonly operationId?: LogOperationId } = {}
): LogDeleteReceipt {
  const failedArtifacts = options.failedArtifacts ?? (state === 'partial' ? 1 : 0)
  const receipt = {
    operationId: options.operationId ?? OPERATION_ID,
    requestId: REQUEST_ID,
    selectionFingerprint: 'safe-fingerprint',
    planned: { requests: 1, events: 2, artifacts: 2, proxyRecords: 1, databaseRows: 6 },
    executed: { requests: 1, events: 2, artifacts: 2, proxyRecords: 1, databaseRows: 6 },
    artifactDeletion: {
      removed: 1,
      failed: failedArtifacts,
      failureClass: failedArtifacts > 0 ? ('unsafe_path' as const) : undefined
    }
  }
  return state === 'pending' ? { ...receipt, state, auditId: undefined } : { ...receipt, state, auditId: AUDIT_ID }
}

function exportResult(): LogExport {
  return { items: [], nextCursor: undefined, truncated: true, retryRequired: false, artifactContentIncluded: false }
}

function requestRow(createdAt: string): LogEventLedgerRow {
  const request = HARNESS_LOG_FIXTURES[1]
  if (!request) throw new Error('Missing terminal request fixture')
  return {
    type: 'request',
    id: `request:${request.requestId.toString()}`,
    occurredAt: createdAt,
    category: 'requests',
    request: { ...request, createdAt }
  }
}

describe('LogOperations', () => {
  beforeEach(() => {
    api.exportRequests.mockReset()
    api.previewCleanup.mockReset()
    api.runCleanup.mockReset()
    api.deleteRequest.mockReset()
  })

  it('exports the current durable filter/cursor context as a bounded metadata-only download', async () => {
    const user = userEvent.setup()
    const createObjectUrl = vi.fn(() => 'blob:export')
    const revokeObjectUrl = vi.fn()
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    Object.assign(URL, { createObjectURL: createObjectUrl, revokeObjectURL: revokeObjectUrl })
    api.exportRequests.mockResolvedValue(exportResult())

    try {
      render(
        <LogOperations operation="export" query={{ cursor: LogPageCursor.parse('resume-token'), model: 'Qwen3' }} />
      )
      await user.click(screen.getByRole('button', { name: 'Export view' }))
      expect(
        screen.getByText(
          'Metadata-only export. Retained artifact payloads are never loaded or included by this control.'
        )
      ).toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Download export' })).toBeDisabled()
      await user.type(screen.getByPlaceholderText('Why is this export needed?'), 'incident review')
      await user.click(screen.getByRole('button', { name: 'Download export' }))

      await waitFor(() => expect(api.exportRequests).toHaveBeenCalledTimes(1))
      expect(api.exportRequests).toHaveBeenCalledWith(
        expect.objectContaining({
          cursor: expect.objectContaining({ toString: expect.any(Function) }),
          model: 'Qwen3'
        }),
        { reason: 'incident review', includeArtifacts: false }
      )
      expect(createObjectUrl).toHaveBeenCalledTimes(1)
      expect(revokeObjectUrl).toHaveBeenCalledWith('blob:export')
      expect(
        screen.getByText('A bounded partial export was downloaded. Narrow the retained filter context before retrying.')
      ).toBeInTheDocument()
    } finally {
      anchorClick.mockRestore()
    }
  })

  it('replaces raw scope fields with a time window and category estimate', async () => {
    const user = userEvent.setup()
    render(<LogOperations operation="cleanup" query={{ from: '2026-07-01T00:00:00Z', to: '2026-08-01T00:00:00Z' }} />)

    await user.click(screen.getByRole('button', { name: 'Clean up logs' }))
    expect(screen.getByRole('heading', { name: 'Choose logs to remove' })).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'Window start' })).toHaveAttribute('aria-valuetext')
    expect(screen.getByRole('slider', { name: 'Window end' })).toHaveAttribute('aria-valuetext')
    expect(screen.queryByLabelText('Delete terminal logs before')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Request scope')).not.toBeInTheDocument()

    const requests = screen.getByRole('button', { name: /Requests chart layer.*selected for cleanup preview/ })
    expect(requests).toBeEnabled()
    await user.click(requests)
    expect(
      screen.getByRole('button', { name: /Requests chart layer.*not selected for cleanup preview/ })
    ).toHaveAttribute('data-state', 'off')
    expect(screen.getByText(/Select Requests to include terminal request history/)).toBeInTheDocument()
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'retention cleanup')
    expect(screen.getByRole('button', { name: 'Review deletion' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /Requests chart layer.*not selected for cleanup preview/ }))
    expect(screen.getByRole('button', { name: 'Review deletion' })).toBeEnabled()
  })

  it('keeps an in-flight preview visible until its result is available', async () => {
    const user = userEvent.setup()
    let resolvePreview: ((receipt: LogCleanupReceipt) => void) | undefined
    api.previewCleanup.mockReturnValue(
      new Promise<LogCleanupReceipt>((resolve) => {
        resolvePreview = resolve
      })
    )
    render(<LogOperations operation="cleanup" query={{}} />)

    await user.click(screen.getByRole('button', { name: 'Clean up logs' }))
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'retention cleanup')
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))

    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled()
    await user.keyboard('{Escape}')
    expect(screen.getByRole('dialog', { name: 'Review log cleanup' })).toBeInTheDocument()

    await act(async () => resolvePreview?.(cleanupReceipt('previewed')))
    const reviewHeading = await screen.findByRole('heading', { name: 'Review log cleanup' })
    await waitFor(() => expect(reviewHeading).toHaveFocus())
  })

  it('keeps a prepared preview visible when the live query window advances while the dialog is open', async () => {
    const user = userEvent.setup()
    api.previewCleanup.mockResolvedValue(cleanupReceipt('previewed'))
    const view = render(
      <LogOperations operation="cleanup" query={{ from: '2026-08-01T00:00:00Z', to: '2026-08-01T01:00:00Z' }} />
    )

    await user.click(screen.getByRole('button', { name: 'Clean up logs' }))
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'retention cleanup')
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))
    await waitFor(() => expect(api.previewCleanup).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('heading', { name: 'Review log cleanup' })).toBeInTheDocument()

    // A relative timeRange (e.g. driven by an advancing clock) moves query.from/to
    // while the dialog stays open. That must not remount the dialog and discard
    // the prepared operation and preview receipt.
    view.rerender(
      <LogOperations operation="cleanup" query={{ from: '2026-08-01T00:05:00Z', to: '2026-08-01T01:05:00Z' }} />
    )

    expect(screen.getByRole('heading', { name: 'Review log cleanup' })).toBeInTheDocument()
    expect(api.previewCleanup).toHaveBeenCalledTimes(1)
  })

  it('uses the latest loaded window when records arrive before the dialog opens', async () => {
    const user = userEvent.setup()
    api.previewCleanup.mockResolvedValue(cleanupReceipt('previewed'))
    const view = render(
      <LogOperations operation="cleanup" query={{}} rows={[requestRow('2026-08-01T00:00:00.000100000Z')]} />
    )
    view.rerender(
      <LogOperations operation="cleanup" query={{}} rows={[requestRow('2026-08-01T00:00:02.000740000Z')]} />
    )

    await user.click(screen.getByRole('button', { name: 'Clean up logs' }))
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'retention cleanup')
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))

    await waitFor(() => expect(api.previewCleanup).toHaveBeenCalledTimes(1))
    expect(api.previewCleanup).toHaveBeenCalledWith(
      expect.objectContaining({
        from: '2026-08-01T00:00:02.000Z',
        to: '2026-08-01T01:00:02.001Z',
        cutoffBefore: '2026-08-01T01:00:02.001Z'
      })
    )
  })

  it('requires a fresh deletion review after cancellation, then an explicit reasoned confirmation and restores focus', async () => {
    const user = userEvent.setup()
    api.previewCleanup.mockResolvedValue(cleanupReceipt('previewed'))
    api.runCleanup.mockResolvedValue(cleanupReceipt('partial'))
    render(
      <LogOperations
        operation="cleanup"
        query={{
          cursor: LogPageCursor.parse('page-2'),
          limit: 25,
          sort: 'desc',
          status: 200,
          source: 'durable',
          from: '2026-07-01T00:00:00Z',
          to: '2026-08-01T00:00:00Z',
          route: 'reserve',
          model: 'Qwen/Qwen3',
          provider: 'reserve-a',
          engine: 'skippy',
          outcome: 'completed'
        }}
      />
    )

    const trigger = screen.getByRole('button', { name: 'Clean up logs' })
    await user.click(trigger)
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(trigger).toHaveFocus())

    await user.click(trigger)
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'retention cleanup')
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))

    await waitFor(() => expect(api.previewCleanup).toHaveBeenCalledTimes(1))
    expect(api.previewCleanup).toHaveBeenCalledWith(
      expect.objectContaining({
        cutoffBefore: '2026-08-01T00:00:00.001Z',
        requestLimit: 100,
        source: 'durable',
        from: '2026-07-01T00:00:00.000Z',
        to: '2026-08-01T00:00:00.001Z',
        route: 'reserve',
        model: 'Qwen/Qwen3',
        provider: 'reserve-a',
        engine: 'skippy',
        outcome: 'completed',
        reason: 'retention cleanup'
      })
    )
    const previewRequest = api.previewCleanup.mock.calls[0]?.[0]
    expect(previewRequest).not.toHaveProperty('cursor')
    expect(previewRequest).not.toHaveProperty('limit')
    expect(previewRequest).not.toHaveProperty('sort')
    expect(previewRequest).not.toHaveProperty('status')
    expect(screen.getByText('Operation ID')).toBeInTheDocument()
    expect(screen.getByText(OPERATION_ID.toString())).toBeInTheDocument()
    expect(screen.getByText('Audit ID')).toBeInTheDocument()
    expect(screen.getByText(AUDIT_ID.toString())).toBeInTheDocument()
    expect(screen.getByText(/terminal request groups? will be removed/)).toBeInTheDocument()
    expect(screen.getByText('Operational events stay retained.')).toBeInTheDocument()
    expect(screen.getByText(/model Qwen\/Qwen3/)).toBeInTheDocument()
    expect(screen.queryByText('/private/retention-reason')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(trigger).toHaveFocus())
    await user.click(trigger)
    expect(screen.getByRole('heading', { name: 'Choose logs to remove' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Review deletion' })).toBeDisabled()
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'second retention cleanup')
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))
    await waitFor(() => expect(api.previewCleanup).toHaveBeenCalledTimes(2))
    await user.click(screen.getByRole('button', { name: 'Delete this batch' }))
    await waitFor(() =>
      expect(api.runCleanup).toHaveBeenCalledWith({ operationId: OPERATION_ID, reason: 'second retention cleanup' })
    )
    expect(
      screen.getByText('Partial cascade: 1 artifact file(s) removed and 1 could not be removed (unsafe_path).')
    ).toBeInTheDocument()
  })

  it('turns an empty server preview into a safe adjustment state', async () => {
    const user = userEvent.setup()
    api.previewCleanup.mockResolvedValue(
      cleanupReceipt('previewed', {
        hasMore: false,
        planned: { requests: 0, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 0 }
      })
    )
    render(<LogOperations operation="cleanup" query={{}} />)

    await user.click(screen.getByRole('button', { name: 'Clean up logs' }))
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'retention cleanup')
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))

    expect(await screen.findByRole('heading', { name: 'Nothing to remove' })).toBeInTheDocument()
    expect(screen.getByText('No terminal request groups matched')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Delete this batch' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Adjust window' }))
    expect(screen.getByRole('heading', { name: 'Choose logs to remove' })).toBeInTheDocument()
    expect(screen.getByDisplayValue('retention cleanup')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('slider', { name: 'Window start' })).toHaveFocus())
  })

  it('notifies only after a cleanup run succeeds, never for its preview or failure', async () => {
    const user = userEvent.setup()
    const onMaintenanceMutationSucceeded = vi.fn()
    api.previewCleanup.mockResolvedValue(cleanupReceipt('previewed'))
    api.runCleanup
      .mockRejectedValueOnce(new Error('Cleanup unavailable'))
      .mockResolvedValueOnce(cleanupReceipt('completed'))
    render(
      <LogOperations operation="cleanup" onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded} query={{}} />
    )

    await user.click(screen.getByRole('button', { name: 'Clean up logs' }))
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'retention cleanup')
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))

    await waitFor(() => expect(api.previewCleanup).toHaveBeenCalledTimes(1))
    expect(onMaintenanceMutationSucceeded).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Delete this batch' }))
    await waitFor(() => expect(api.runCleanup).toHaveBeenCalledTimes(1))
    expect(onMaintenanceMutationSucceeded).not.toHaveBeenCalled()
    expect(screen.getByRole('status')).toHaveTextContent('Cleanup unavailable')

    await user.click(screen.getByRole('button', { name: 'Delete this batch' }))
    await waitFor(() => expect(api.runCleanup).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(onMaintenanceMutationSucceeded).toHaveBeenCalledOnce())
    expect(screen.getByRole('status')).toHaveTextContent('Log cleanup completed.')
  })

  it('retries retained cleanup artifacts with the frozen receipt operation and audit reason', async () => {
    const user = userEvent.setup()
    const reason = 'retention cleanup /private/retention-reason?token=secret'
    const onMaintenanceMutationSucceeded = vi.fn()
    api.previewCleanup.mockImplementation(async (request: LogCleanupPreviewRequest) =>
      cleanupReceipt('previewed', { operationId: request.operationId })
    )
    api.runCleanup
      .mockImplementationOnce(async (request: LogCleanupRunRequest) =>
        cleanupReceipt('partial', { failedArtifacts: 1, hasMore: true, operationId: request.operationId })
      )
      .mockImplementationOnce(async (request: LogCleanupRunRequest) =>
        cleanupReceipt('completed', { failedArtifacts: 0, hasMore: true, operationId: request.operationId })
      )
    render(
      <LogOperations operation="cleanup" onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded} query={{}} />
    )

    await user.click(screen.getByRole('button', { name: 'Clean up logs' }))
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), reason)
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))
    await waitFor(() => expect(api.previewCleanup).toHaveBeenCalledTimes(1))
    await user.click(screen.getByRole('button', { name: 'Delete this batch' }))
    await waitFor(() => expect(api.runCleanup).toHaveBeenCalledTimes(1))

    const previewOperation = api.previewCleanup.mock.calls[0]?.[0]?.operationId
    const firstRun = api.runCleanup.mock.calls[0]?.[0]
    expect(firstRun).toEqual({ operationId: previewOperation, reason })
    expect(onMaintenanceMutationSucceeded).toHaveBeenCalledOnce()
    expect(screen.getByRole('button', { name: 'Retry file removal' })).toBeInTheDocument()
    expect(screen.getByText('Additional request groups remain')).toBeInTheDocument()
    expect(screen.queryByDisplayValue(reason)).not.toBeInTheDocument()
    expect(screen.queryByText('/private/retention-reason?token=secret')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Retry file removal' }))
    await waitFor(() => expect(api.runCleanup).toHaveBeenCalledTimes(2))
    expect(api.runCleanup.mock.calls[1]?.[0]).toEqual(firstRun)
    expect(api.previewCleanup).toHaveBeenCalledTimes(1)
    expect(onMaintenanceMutationSucceeded).toHaveBeenCalledTimes(2)
    expect(screen.getByText('Additional request groups remain')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Retry file removal' })).not.toBeInTheDocument()
  })

  it('does not retry a partial cleanup without retained failed artifacts', async () => {
    const user = userEvent.setup()
    api.previewCleanup.mockResolvedValue(cleanupReceipt('previewed'))
    api.runCleanup.mockResolvedValue(cleanupReceipt('partial', { failedArtifacts: 0, hasMore: true }))
    render(<LogOperations operation="cleanup" query={{}} />)

    await user.click(screen.getByRole('button', { name: 'Clean up logs' }))
    await user.type(screen.getByPlaceholderText('Why are these logs being removed?'), 'retention cleanup')
    await user.click(screen.getByRole('button', { name: 'Review deletion' }))
    await waitFor(() => expect(api.previewCleanup).toHaveBeenCalledTimes(1))
    await user.click(screen.getByRole('button', { name: 'Delete this batch' }))
    await waitFor(() => expect(api.runCleanup).toHaveBeenCalledTimes(1))

    expect(screen.getByText('Additional request groups remain')).toBeInTheDocument()
    expect(
      screen.getByText(
        'Cleanup changed 10 records, but the server reported a partial result. Review the audit details before continuing.'
      )
    ).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Retry file removal' })).not.toBeInTheDocument()
  })

  it('retries retained deletion artifacts with the frozen receipt operation and restores focus', async () => {
    const user = userEvent.setup()
    const reason = 'incident cleanup /private/delete-reason?token=secret'
    const onMaintenanceMutationSucceeded = vi.fn()
    api.deleteRequest
      .mockImplementationOnce(async (_requestId: LogRequestId, request: LogDeleteRequest) =>
        deleteReceipt('partial', { failedArtifacts: 1, operationId: request.operationId })
      )
      .mockImplementationOnce(async (_requestId: LogRequestId, request: LogDeleteRequest) =>
        deleteReceipt('completed', { failedArtifacts: 0, operationId: request.operationId })
      )
    render(
      <LogRequestDeleteControl onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded} requestId={REQUEST_ID} />
    )

    const trigger = screen.getByRole('button', { name: 'Delete terminal request' })
    await user.click(trigger)
    await user.type(screen.getByPlaceholderText('Why remove this request?'), reason)
    await user.click(screen.getByRole('button', { name: 'Confirm deletion' }))
    await waitFor(() => expect(api.deleteRequest).toHaveBeenCalledTimes(1))

    const firstDeletion = api.deleteRequest.mock.calls[0]?.[1]
    expect(onMaintenanceMutationSucceeded).toHaveBeenCalledOnce()
    expect(screen.getByRole('button', { name: 'Retry deletion' })).toBeInTheDocument()
    expect(screen.queryByDisplayValue(reason)).not.toBeInTheDocument()
    expect(screen.queryByText('/private/delete-reason?token=secret')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Retry deletion' }))
    await waitFor(() => expect(api.deleteRequest).toHaveBeenCalledTimes(2))
    expect(api.deleteRequest.mock.calls[1]?.[1]).toEqual(firstDeletion)
    expect(onMaintenanceMutationSucceeded).toHaveBeenCalledTimes(2)
    expect(screen.queryByRole('button', { name: 'Retry deletion' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(trigger).toHaveFocus())
  })

  it('notifies only after a completed deletion succeeds, never after a failed request', async () => {
    const user = userEvent.setup()
    const onMaintenanceMutationSucceeded = vi.fn()
    api.deleteRequest
      .mockRejectedValueOnce(new Error('Deletion unavailable'))
      .mockResolvedValueOnce(deleteReceipt('completed'))
    render(
      <LogRequestDeleteControl onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded} requestId={REQUEST_ID} />
    )

    await user.click(screen.getByRole('button', { name: 'Delete terminal request' }))
    await user.type(screen.getByPlaceholderText('Why remove this request?'), 'incident cleanup')
    await user.click(screen.getByRole('button', { name: 'Confirm deletion' }))
    await waitFor(() => expect(api.deleteRequest).toHaveBeenCalledTimes(1))
    expect(onMaintenanceMutationSucceeded).not.toHaveBeenCalled()
    expect(screen.getByRole('status')).toHaveTextContent('Deletion unavailable')

    await user.click(screen.getByRole('button', { name: 'Confirm deletion' }))
    await waitFor(() => expect(api.deleteRequest).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(onMaintenanceMutationSucceeded).toHaveBeenCalledOnce())

    expect(screen.queryByRole('button', { name: 'Retry deletion' })).not.toBeInTheDocument()
  })

  it('presents an accepted deletion as pending and resumes it without claiming completion', async () => {
    const user = userEvent.setup()
    const onMaintenanceMutationSucceeded = vi.fn()
    api.deleteRequest.mockResolvedValueOnce(deleteReceipt('pending')).mockResolvedValueOnce(deleteReceipt('completed'))
    render(
      <LogRequestDeleteControl onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded} requestId={REQUEST_ID} />
    )

    await user.click(screen.getByRole('button', { name: 'Delete terminal request' }))
    await user.type(screen.getByPlaceholderText('Why remove this request?'), 'durable pending delete')
    await user.click(screen.getByRole('button', { name: 'Confirm deletion' }))

    await waitFor(() => expect(api.deleteRequest).toHaveBeenCalledOnce())
    expect(
      screen.getByText('Deletion accepted and still pending. Retry to check or resume this operation.')
    ).toBeVisible()
    expect(screen.getByText('Not assigned')).toBeInTheDocument()
    expect(screen.queryByText('Request removed.')).not.toBeInTheDocument()
    expect(onMaintenanceMutationSucceeded).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Retry deletion' }))
    await waitFor(() => expect(api.deleteRequest).toHaveBeenCalledTimes(2))
    expect(screen.getByText('Request removed.')).toBeVisible()
    expect(onMaintenanceMutationSucceeded).toHaveBeenCalledOnce()
  })

  it('places export and cleanup independently while preserving active-source restrictions', () => {
    const view = render(<LogOperations operation="export" query={{ source: 'active' }} />)
    expect(screen.getByRole('button', { name: 'Export view' })).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'Clean up logs' })).not.toBeInTheDocument()
    expect(screen.getByText('Clear source selection to export durable records.')).toBeInTheDocument()

    view.rerender(<LogOperations operation="cleanup" query={{ source: 'active' }} />)
    expect(screen.queryByRole('button', { name: 'Export view' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Clean up logs' })).toBeDisabled()
    expect(
      screen.getByText('Clear active-source or non-terminal outcome filters before removing durable logs.')
    ).toBeInTheDocument()
  })
})

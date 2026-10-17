import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import type { RecordsStatus } from '@/features/capsules/api/recordsClient'
import { CleanUpRecordsDialog } from '@/features/capsules/components/CleanUpRecordsDialog'

// The network layer only: every test goes through the real client and the
// real tool route shape (`POST /api/plugins/capsules/tools/<op>`).
function stubTool(status: number, body: unknown) {
  const fetchSpy = vi.fn(
    async (_url: string, _init?: RequestInit) =>
      new Response(typeof body === 'string' ? body : JSON.stringify(body), { status })
  )
  vi.stubGlobal('fetch', fetchSpy)
  return fetchSpy
}

const STATUS: RecordsStatus = {
  records_path: '/data/ledger',
  record_count: 8,
  head: null,
  log_id: 'capsules',
  stored_text_count: 3,
  new_history_pending: null,
  sharing: {
    record_at_completion: { value: 'counterparty', source: 'default' },
    history_segments: { value: 'prospective', source: 'default' },
    adjudications: { value: 'deliver_to_subjects', source: 'default' },
    witness: { value: null, source: 'default' }
  }
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <QueryClientProvider client={client}>
      <TooltipProvider>{children}</TooltipProvider>
    </QueryClientProvider>
  )
}

function open(props: { status?: RecordsStatus | null; sample?: boolean } = {}) {
  render(
    <CleanUpRecordsDialog
      onOpenChange={() => {}}
      open
      sample={props.sample ?? false}
      status={props.status === undefined ? STATUS : props.status}
    />,
    { wrapper }
  )
  return screen.getByRole('dialog', { name: 'Clean up records' })
}

describe('CleanUpRecordsDialog', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('turns off deleting stored text when no prompt or answer text is kept, and opens on an option that can run', () => {
    const fetchSpy = stubTool(200, {})
    const dialog = open({ status: { ...STATUS, stored_text_count: 0 } })
    const radios = within(dialog).getAllByRole('radio')
    const deleteOption = radios.find((r) => (r as HTMLInputElement).value === 'delete_stored_text') as HTMLInputElement
    expect(deleteOption).toBeDisabled()
    expect(deleteOption).not.toBeChecked()
    expect(
      within(dialog).getByText('No prompt or answer text is kept, so there is nothing to delete.')
    ).toBeInTheDocument()
    expect(within(dialog).queryByRole('button', { name: 'Delete stored text' })).not.toBeInTheDocument()
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('deletes stored text and names the record the cleanup sealed', async () => {
    const fetchSpy = stubTool(200, {
      deleted_count: 3,
      sealed: { capsule_id: 'c'.repeat(64), record_number: 9, kind: 'stored_text_deleted' }
    })
    const user = userEvent.setup()
    const dialog = open()
    await user.click(within(dialog).getByRole('button', { name: 'Delete stored text' }))
    expect(String(fetchSpy.mock.calls[0][0])).toMatch(/\/tools\/evidence_delete_stored_text$/)
    expect(await within(dialog).findByRole('status')).toHaveTextContent(
      'Deleted the stored text of 3 records. Sealed as record 9.'
    )
  })

  it('a new log runs only once the box is ticked', async () => {
    const fetchSpy = stubTool(200, {
      staged: true,
      takes_effect: 'next_start',
      sealed: { capsule_id: 'c'.repeat(64), record_number: 9, kind: 'history_closing' }
    })
    const user = userEvent.setup()
    const dialog = open()
    await user.click(within(dialog).getByRole('radio', { name: /Start a new log/ }))
    const run = within(dialog).getByRole('button', { name: 'Start a new log' })
    expect(run).toBeDisabled()
    await user.click(within(dialog).getByRole('checkbox'))
    // Re-read: once runnable, the button is no longer wrapped in its reason.
    const runnable = within(dialog).getByRole('button', { name: 'Start a new log' })
    expect(runnable).toBeEnabled()
    await user.click(runnable)
    const [url, init] = fetchSpy.mock.calls[0]
    expect(String(url)).toMatch(/\/tools\/evidence_start_new_history$/)
    expect(JSON.parse(String(init?.body))).toEqual({ confirm: true })
    expect(await within(dialog).findByRole('status')).toHaveTextContent(/next time this node starts/)
  })

  it('on sample data nothing runs, and the reason is said', () => {
    const dialog = open({ sample: true, status: null })
    expect(within(dialog).getByRole('button', { name: 'Delete stored text' })).toBeDisabled()
    expect(within(dialog).getByText('Not available on sample data.')).toBeInTheDocument()
  })

  it('a refusal from the node is shown as it came, never as success', async () => {
    // The host route answers a plugin-tool error as a non-2xx whose body is
    // the error text.
    const fetchSpy = stubTool(500, 'the records on disk end at x')
    const user = userEvent.setup()
    const dialog = open()
    await user.click(within(dialog).getByRole('radio', { name: /Rebuild the index/ }))
    await user.click(within(dialog).getByRole('button', { name: 'Rebuild index' }))
    const line = await within(dialog).findByRole('status')
    expect(line).toHaveTextContent('the records on disk end at x')
    expect(line.className).toMatch(/text-bad/)
    expect(String(fetchSpy.mock.calls[0][0])).toMatch(
      /\/api\/plugins\/capsules\/tools\/evidence_rebuild_index$/
    )
  })
})

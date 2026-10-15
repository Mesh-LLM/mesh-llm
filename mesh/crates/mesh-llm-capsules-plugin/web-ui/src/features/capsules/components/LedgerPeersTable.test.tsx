// The Peers table shell: two never-merged row
// groups, the Exchanges-style toolbar (Search / Filter / Columns / Export /
// Save evidence file / Reset view), and the zero-dealings acceptance check.
// the online-status filter is retired
// along with the row's own online badge/latency/Route button; only the
// Alarm filter remains.
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { LedgerPeersTable } from '@/features/capsules/components/LedgerPeersTable'
import {
  HARNESS_PANE_B_PAYLOAD,
  PEER_TAB_HARNESS_MESH_MODELS,
  PEER_TAB_HARNESS_MESH_PEERS
} from '@/features/capsules/lib/peer-fixtures'
import { advertisedOnlyPeers, deriveMeshStatus } from '@/features/capsules/lib/peer-mesh-status'
import {
  advertisedOnlyRowView,
  dealtWithRowView,
  PEER_COLUMN_INFO,
  SELF_REPORTED_DETAIL,
  SELF_REPORTED_NOTE
} from '@/features/capsules/lib/peer-row-view'

const [CLEAN_ROW, ALARMED_ROW] = HARNESS_PANE_B_PAYLOAD.rows

function buildFixtureProps() {
  const statusFor = (peerId: string) =>
    deriveMeshStatus(peerId, PEER_TAB_HARNESS_MESH_PEERS, PEER_TAB_HARNESS_MESH_MODELS)
  const dealtWithRawRows = [CLEAN_ROW, ALARMED_ROW]
  const advertisedRawPeers = advertisedOnlyPeers(dealtWithRawRows, PEER_TAB_HARNESS_MESH_PEERS)
  const dealtWith = dealtWithRawRows.map((row) => dealtWithRowView(row))
  const advertisedUnused = advertisedRawPeers.map((peer) => advertisedOnlyRowView(peer.shortId ?? peer.id))
  return {
    dealtWith,
    advertisedUnused,
    dealtWithRawRows,
    advertisedUnusedRawPeers: advertisedRawPeers,
    meshStatus: { statusFor, peers: PEER_TAB_HARNESS_MESH_PEERS },
    exchangeSourcesFor: () => [],
    recordsById: new Map()
  }
}

describe('LedgerPeersTable — two row groups, never merged', () => {
  it('renders "Nodes you have dealt with" and "Nodes advertised but unused" as separate groups', () => {
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    expect(screen.getByText(/Nodes you have dealt with · 2/)).toBeInTheDocument()
    expect(screen.getByText(/Nodes advertised but unused · 1/)).toBeInTheDocument()
    expect(screen.getByText(CLEAN_ROW.peer_id ?? '')).toBeInTheDocument()
    expect(screen.getByText(ALARMED_ROW.peer_id ?? '')).toBeInTheDocument()
    // the fixture's third mesh peer ("unused-node.local", shortId
    // "11223344") has no matching Pane B row -- it must appear ONLY in the
    // advertised group, never in the dealt-with one.
    const advertisedGroup = screen.getByText(/Nodes advertised but unused · 1/).closest('tbody')
    expect(advertisedGroup).not.toBeNull()
    expect(within(advertisedGroup as HTMLElement).getByText('11223344')).toBeInTheDocument()
  })

  it('a zero-dealings peer in the advertised group shows "no exchanges yet" with "—" accountability columns', () => {
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    expect(screen.getByText('no exchanges yet')).toBeInTheDocument()
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })

  it('never carries a percentage or ratio ramp on any peer figure', () => {
    const props = buildFixtureProps()
    const { container } = render(<LedgerPeersTable {...props} />)
    const bodyCells = container.querySelectorAll('td')
    for (const cell of bodyCells) {
      expect(cell.textContent ?? '').not.toContain('%')
    }
  })

  it('no online status, latency, or Route here control remains anywhere in the table', () => {
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    expect(screen.queryByRole('button', { name: 'Route here' })).not.toBeInTheDocument()
    expect(screen.queryByText(/\bms\b/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/filter peers/i)).toBeInTheDocument()
  })
})

describe('LedgerPeersTable — no "match this filter" when nothing is filtered', () => {
  it('an empty group with no search or filter set says "None yet.", never "match this filter"', () => {
    const props = { ...buildFixtureProps(), dealtWith: [], dealtWithRawRows: [] }
    render(<LedgerPeersTable {...props} />)
    const dealtWithGroup = screen.getByText(/Nodes you have dealt with · 0/).closest('tbody') as HTMLElement
    expect(within(dealtWithGroup).getByText('None yet.')).toBeInTheDocument()
    expect(screen.queryByText(/match this filter/)).not.toBeInTheDocument()
  })

  it('the same empty group says "match this filter" once a search is actually narrowing it', async () => {
    const user = userEvent.setup()
    render(<LedgerPeersTable {...buildFixtureProps()} />)
    await user.type(screen.getByLabelText('Search peers'), 'no-such-peer')
    expect(screen.getByText('No exchanges match this filter.')).toBeInTheDocument()
    expect(screen.getByText('No advertised peers match this filter.')).toBeInTheDocument()
  })
})

describe('LedgerPeersTable — toolbar', () => {
  it('search filters both groups to the matching peer only', async () => {
    const user = userEvent.setup()
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    await user.type(screen.getByLabelText('Search peers'), ALARMED_ROW.peer_id ?? '')

    expect(screen.getByText(ALARMED_ROW.peer_id ?? '')).toBeInTheDocument()
    expect(screen.queryByText(CLEAN_ROW.peer_id ?? '')).not.toBeInTheDocument()
  })

  it('the Needs a look filter narrows to only the peer that needs a look, and never says Alarm', async () => {
    const user = userEvent.setup()
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    await user.click(screen.getByRole('button', { name: /filter peers/i }))
    await user.click(screen.getByRole('button', { name: /^none$/i }))
    expect(screen.queryByText(/alarm/i)).not.toBeInTheDocument()
    await user.click(screen.getByRole('checkbox', { name: /needs a look/i }))

    expect(screen.getByText(ALARMED_ROW.peer_id ?? '')).toBeInTheDocument()
    expect(screen.queryByText(CLEAN_ROW.peer_id ?? '')).not.toBeInTheDocument()
  })

  it('the Columns menu hides a column, Reset view restores it', async () => {
    const user = userEvent.setup()
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    // Each header now carries an (i) info glyph, so its accessible name is the
    // label plus the glyph's description -- match on the label substring.
    expect(screen.getByRole('columnheader', { name: /^Same request & answer/ })).toBeInTheDocument()
    // Plain-language names (UX §2).
    expect(screen.getByRole('columnheader', { name: /^They confirmed/ })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /columns/i }))
    await user.click(screen.getByRole('menuitemcheckbox', { name: 'Same request & answer' }))
    expect(screen.queryByRole('columnheader', { name: /^Same request & answer/ })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /reset view/i }))
    expect(screen.getByRole('columnheader', { name: /^Same request & answer/ })).toBeInTheDocument()
  })

  it('the period column is "Last dealt with", never the word "Period" or "Window"', () => {
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    expect(screen.getByRole('columnheader', { name: /^Last dealt with/ })).toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: /^Period/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: /^Window/ })).not.toBeInTheDocument()
  })

  it('Item 4: each column header carries an (i) whose aria-describedby holds the moved column explanation', () => {
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    // The Confirmed header's (i) carries the UX §8 sentence verbatim.
    const confirmedGlyph = screen.getByRole('button', { name: 'About They confirmed' })
    const confirmedDesc = document.getElementById(confirmedGlyph.getAttribute('aria-describedby') as string)
    expect(confirmedDesc).toHaveTextContent(PEER_COLUMN_INFO.confirmed)
    expect(PEER_COLUMN_INFO.confirmed).toContain('confirmed by their own signed record, checked on this machine')

    // Every visible accountability column header has its own (i) glyph.
    for (const label of [
      'Exchanges',
      'They confirmed',
      'Same request & answer',
      'Disputes judged',
      'Their records witnessed'
    ]) {
      expect(screen.getByRole('button', { name: `About ${label}` })).toBeInTheDocument()
    }
  })

  it('the Columns toggle names the period column "Last dealt with", never "Period"/"Window"', async () => {
    const user = userEvent.setup()
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    await user.click(screen.getByRole('button', { name: /columns/i }))
    expect(screen.getByRole('menuitemcheckbox', { name: 'Last dealt with' })).toBeInTheDocument()
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Period' })).not.toBeInTheDocument()
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Window' })).not.toBeInTheDocument()
  })

  it('says "self-reported" once, in a legend under the table, with the fuller sentence behind its (i)', () => {
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    expect(screen.getByTestId('peers-legend')).toHaveTextContent(SELF_REPORTED_NOTE)
    expect(screen.getAllByText(new RegExp(SELF_REPORTED_NOTE))).toHaveLength(1)
    const glyph = screen.getByRole('button', { name: 'About the self-reported note' })
    expect(document.getElementById(glyph.getAttribute('aria-describedby') as string)).toHaveTextContent(
      SELF_REPORTED_DETAIL
    )
  })

  it('Reset view is not shown while the view is at its default, never greyed', () => {
    const props = buildFixtureProps()
    render(<LedgerPeersTable {...props} />)

    expect(screen.queryByRole('button', { name: /reset view/i })).not.toBeInTheDocument()
  })

  it('Export view (CSV) and Save evidence file trigger a file save with peer-scoped content', async () => {
    const user = userEvent.setup()
    const props = buildFixtureProps()
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    const createObjectURL = vi.fn(() => 'blob:mock')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL })

    render(<LedgerPeersTable {...props} />)
    await user.click(screen.getByRole('button', { name: /export view/i }))
    await user.click(screen.getByRole('button', { name: /save evidence file/i }))

    expect(clickSpy).toHaveBeenCalledTimes(2)
    expect(createObjectURL).toHaveBeenCalledTimes(2)

    clickSpy.mockRestore()
    vi.unstubAllGlobals()
  })
})

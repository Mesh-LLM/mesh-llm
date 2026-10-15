// Replaces `PeerCard.test.tsx` -- the row is a
// table row now (review §3-F: "Peers is a card, not the table").
// Accountability-only acceptance checks:
// denominator-honest adjudication, an always-visible alarm chip, and NO
// online badge / latency / Route-to-chat button anywhere in the row (moved
// to the Network tab) -- `meshStatus` is threaded through to the modal
// only.
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PeerTableRow } from '@/features/capsules/components/PeerTableRow'
import {
  HARNESS_PANE_B_PAYLOAD,
  PEER_TAB_HARNESS_EXCHANGE_SOURCES,
  PEER_TAB_HARNESS_LEDGER_RECORDS,
  PEER_TAB_HARNESS_MESH_MODELS,
  PEER_TAB_HARNESS_MESH_PEERS
} from '@/features/capsules/lib/peer-fixtures'
import { deriveMeshStatus } from '@/features/capsules/lib/peer-mesh-status'
import { advertisedOnlyRowView, dealtWithRowView, SELF_REPORTED_NOTE } from '@/features/capsules/lib/peer-row-view'
import { PEER_PAYMENTS_TOOLTIP, ROUTING_NOT_ON_THIS_PAGE } from '@/features/capsules/lib/tooltip-copy'

const [CLEAN_ROW, ALARMED_ROW] = HARNESS_PANE_B_PAYLOAD.rows

function renderInTable(ui: React.ReactElement) {
  return render(
    <table>
      <tbody>{ui}</tbody>
    </table>
  )
}

describe('PeerTableRow — dealt-with peers', () => {
  it('renders the accountability columns and no alarm chip for a clean peer whose pushed halves closed through the gate', () => {
    const view = dealtWithRowView(CLEAN_ROW)
    renderInTable(<PeerTableRow meshStatus={null} view={view} />)

    expect(screen.getByText('24')).toBeInTheDocument()
    // 16 of the 24 exchanges have a pushed half that closed through the gate.
    expect(screen.getByText('16 of 24')).toBeInTheDocument()
    expect(screen.getByText('16 · 0 differ')).toBeInTheDocument()
    expect(screen.getByText('8 of 24 · 8 corroborated')).toBeInTheDocument()
    expect(screen.getByText('not shown yet')).toBeInTheDocument()
    expect(screen.getByText('8 Sep')).toBeInTheDocument()
    // Said once in the table legend, never on each row (UX §2).
    expect(screen.queryByText(SELF_REPORTED_NOTE)).not.toBeInTheDocument()
    expect(screen.queryByText(/⚠/)).not.toBeInTheDocument()
    expect(screen.queryByText(/online/)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Route here' })).not.toBeInTheDocument()
  })

  it('names what needs a look on an alarmed peer, counted, with no generic ⚠ glyph -- and the counts agree with the gate', () => {
    const view = dealtWithRowView(ALARMED_ROW)
    const { container } = renderInTable(<PeerTableRow meshStatus={null} view={view} />)

    // UX §8: the specific thing, counted -- never a generic warning glyph.
    expect(screen.getByText('1 disagreement')).toBeInTheDocument()
    expect(screen.getByText('1 differing answer')).toBeInTheDocument()
    expect(container.textContent).not.toContain('⚠')
    // Each badge explains itself in one sentence (hover + aria-describedby).
    const badge = screen.getByText('1 disagreement').closest('[aria-describedby]') as HTMLElement
    expect(document.getElementById(badge.getAttribute('aria-describedby') as string)).toHaveTextContent(
      '1 exchange where your record and theirs disagree.'
    )
    // 9 of 14 confirmed, 1 pushed half the gate reads as CONTRADICTED.
    expect(screen.getByText('9 of 14 · 1 differ')).toBeInTheDocument()
    expect(screen.getByText('9 · 1 differ')).toBeInTheDocument()
    expect(screen.getByText('7 of 14 · 6 corroborated · 1 contradicted')).toBeInTheDocument()
  })

  it('keeps the verdict date: the differing-answer badge’s sentence names when it was sealed, from the local ledger', () => {
    const resolveTimestamp = vi.fn(() => '2026-09-08')
    const view = dealtWithRowView(ALARMED_ROW, resolveTimestamp)
    renderInTable(<PeerTableRow meshStatus={null} view={view} />)

    expect(resolveTimestamp).toHaveBeenCalledWith('cap-alarmed-adjudication-0007')
    const badge = screen.getByText('1 differing answer').closest('[aria-describedby]') as HTMLElement
    expect(document.getElementById(badge.getAttribute('aria-describedby') as string)).toHaveTextContent(
      'Latest: 2026-09-08.'
    )
  })

  it('never renders online status, latency, or a Route here control (Network tab job)', () => {
    const meshStatus = deriveMeshStatus(
      CLEAN_ROW.peer_id ?? '',
      PEER_TAB_HARNESS_MESH_PEERS,
      PEER_TAB_HARNESS_MESH_MODELS
    )
    const view = dealtWithRowView(CLEAN_ROW)
    const { container } = renderInTable(<PeerTableRow meshStatus={meshStatus} view={view} />)

    expect(screen.queryByText(/\bms\b/)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Route here' })).not.toBeInTheDocument()
    expect(container.textContent).not.toContain('you measured')
  })

  it('opens the PeerInspector modal on row click, Overview tab active by default, with no liveness line', async () => {
    const user = userEvent.setup()
    const meshStatus = deriveMeshStatus(
      CLEAN_ROW.peer_id ?? '',
      PEER_TAB_HARNESS_MESH_PEERS,
      PEER_TAB_HARNESS_MESH_MODELS
    )
    const view = dealtWithRowView(CLEAN_ROW)
    renderInTable(<PeerTableRow meshStatus={meshStatus} view={view} />)

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    await user.click(screen.getByText(CLEAN_ROW.peer_id ?? ''))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByRole('tab', { name: /overview/i })).toHaveAttribute('data-state', 'active')
    // Accountability only: the drill shows no mesh status, latency or
    // online state (the Network tab's), and no per-drill self-reported line.
    for (const liveness of ['online', 'offline', 'status unknown', 'latency unknown', 'mesh status not available']) {
      expect(within(dialog).queryByText(liveness)).not.toBeInTheDocument()
    }
    expect(within(dialog).queryByText('self-reported — not independently attested')).not.toBeInTheDocument()
  })

  it('the drill shows your dealings with them and says plainly that routing is not changed here', async () => {
    const user = userEvent.setup()
    renderInTable(<PeerTableRow meshStatus={null} view={dealtWithRowView(CLEAN_ROW)} />)
    await user.click(screen.getByText(CLEAN_ROW.peer_id ?? ''))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText('24 exchanges with you · they confirmed 16 of 24')).toBeInTheDocument()
    expect(within(dialog).getByText(ROUTING_NOT_ON_THIS_PAGE.text)).toBeInTheDocument()
    expect(within(dialog).queryByRole('button', { name: /stop routing/i })).not.toBeInTheDocument()
  })

  it('drills into the timeline with exchange sources supplied', async () => {
    const user = userEvent.setup()
    const view = dealtWithRowView(CLEAN_ROW)
    renderInTable(
      <PeerTableRow
        exchangeSources={PEER_TAB_HARNESS_EXCHANGE_SOURCES[CLEAN_ROW.peer_id ?? '']}
        meshStatus={null}
        recordsById={PEER_TAB_HARNESS_LEDGER_RECORDS}
        view={view}
      />
    )

    await user.click(screen.getByText(CLEAN_ROW.peer_id ?? ''))
    await user.click(screen.getByRole('tab', { name: /timeline/i }))
    expect(screen.getByText('Your exchanges')).toBeInTheDocument()
  })

  it('the Columns toggle hides one accountability column without hiding the others', () => {
    const view = dealtWithRowView(CLEAN_ROW)
    renderInTable(<PeerTableRow meshStatus={null} view={view} visibleColumns={new Set(['match'])} />)

    expect(screen.getByText('16 · 0 differ')).toBeInTheDocument()
    expect(screen.queryByText('24 / 24')).not.toBeInTheDocument()
    expect(screen.queryByText(/8 of 24/)).not.toBeInTheDocument()
  })
})

describe('PeerTableRow — advertised-but-unused peers (no Pane B row)', () => {
  it('shows "no exchanges yet" and "—" placeholders for every accountability column, and is not clickable', async () => {
    const user = userEvent.setup()
    const view = advertisedOnlyRowView('node:unused-peer')
    renderInTable(<PeerTableRow meshStatus={null} view={view} />)

    expect(screen.getByText('node:unused-peer')).toBeInTheDocument()
    expect(screen.getByText('no exchanges yet')).toBeInTheDocument()
    expect(screen.getByText('0')).toBeInTheDocument()
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(5)
    expect(screen.queryByText(/⚠/)).not.toBeInTheDocument()

    await user.click(screen.getByText('node:unused-peer'))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
})

describe('PeerTableRow -- payments with this peer', () => {
  const settlement = {
    paid_exchanges: 3,
    settled_payer_observed: 2,
    no_settlement_seen: 1,
    provider_book: 'not_available'
  }

  it('counts paid exchanges from your own records, with a one-sentence hover', () => {
    renderInTable(<PeerTableRow meshStatus={null} view={dealtWithRowView({ ...CLEAN_ROW, settlement })} />)
    const line = document.querySelector('[data-peer-payments]') as HTMLElement
    expect(line.textContent).toContain('3 paid · 2 settled by your wallet · 1 no payment seen')
    const describedBy = line.closest('[aria-describedby]')?.getAttribute('aria-describedby') as string
    expect(document.getElementById(describedBy)?.textContent).toBe(PEER_PAYMENTS_TOOLTIP)
  })

  it('shows no payments line for a peer with no paid exchange on record', () => {
    renderInTable(<PeerTableRow meshStatus={null} view={dealtWithRowView(CLEAN_ROW)} />)
    expect(document.querySelector('[data-peer-payments]')).toBeNull()
  })
})

// The "Their history" tab, rendered. The
// acceptance line: on the two-node capture (nothing fetched, nothing asked),
// "never asked" renders as never asked, not zero.
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { PaneBRow } from '@/features/capsules/api/sidecarTypes'
import { PeerHistoryTab } from '@/features/capsules/components/PeerHistoryTab'
import { HARNESS_PANE_B_PAYLOAD } from '@/features/capsules/lib/peer-fixtures'

const [CLEAN] = HARNESS_PANE_B_PAYLOAD.rows

/** The freeze-candidate pane-b row for `key:71eb777777777777`, as captured. */
const CAPTURED_PEER: PaneBRow = {
  ...CLEAN,
  peer_id: 'key:71eb777777777777',
  history: { state: 'NOT_CHECKED' },
  verdicts: { state: 'NOT_CHECKED' },
  asked: { state: 'absent', count: 0 },
  exchange_count: 3
}

function section(name: string): HTMLElement {
  return screen.getByRole('region', { name })
}

describe('PeerHistoryTab', () => {
  it('on the captured two-node peer, every un-asked part reads as never asked, with no zero', () => {
    render(<PeerHistoryTab row={CAPTURED_PEER} />)
    expect(within(section('Their log, as shown to you')).getByText(/^Never asked/)).toBeInTheDocument()
    expect(within(section('What others say')).getByText(/^Never asked/)).toBeInTheDocument()
    expect(within(section('Asked of you')).getByText(/^Not shown/)).toBeInTheDocument()

    const verdicts = within(section('Verdicts on their exchanges')).getByRole('table')
    const cells = within(verdicts)
      .getAllByRole('cell')
      .map((cell) => cell.textContent)
    expect(cells).toEqual(Array.from({ length: 3 }, () => ['not counted', 'not shown', 'never asked']).flat())

    for (const name of [
      'Their log, as shown to you',
      'What others say',
      'Verdicts on their exchanges',
      'Asked of you'
    ]) {
      // The (i) copy is sr-only text inside the section; strip it before the
      // no-digit check so only the visible face is tested.
      const face = [...section(name).querySelectorAll(':scope > :not(h3)')].map((el) => el.textContent).join(' ')
      expect(face, name).not.toMatch(/\d/)
    }
  })

  it('the Their log hover says "checked" only when a fetched log was checked', () => {
    render(<PeerHistoryTab row={CAPTURED_PEER} />)
    const region = section('Their log, as shown to you')
    const hovers = Array.from(region.querySelectorAll('[aria-describedby]')).map(
      (element) => document.getElementById(element.getAttribute('aria-describedby') ?? '')?.textContent ?? ''
    )
    expect(hovers.join(' ')).toContain('hasn’t asked for their log')
    expect(hovers.join(' ')).not.toContain('fetched and checked')
  })

  it('renders per-checkpoint counts, their reply to delivered verdicts, and inbound requests', () => {
    render(
      <PeerHistoryTab
        row={{
          ...CAPTURED_PEER,
          history: {
            state: 'verified',
            segment: {
              links: [
                {
                  checkpoint: { mmr_size: 7, timestamp: '2026-09-26T05:06:00Z', witnesses: [{}] },
                  leaf_counts: { exchange: 3 },
                  leaf_digests: ['f'.repeat(64)]
                }
              ]
            },
            adjudications: {
              state: 'enriched',
              delivered: { contradicted: { delivered: 1, acknowledged: 0, disputed: 1 } }
            }
          },
          asked_of_you: {
            entries: [
              {
                ts: '2026-09-26T05:10:00Z',
                path: 'evidence-request',
                requester_id: 'key:71eb777777777777',
                subject_kind: 'chain_segment',
                status: 'refused',
                reason: 'not_authorized'
              }
            ]
          }
        }}
      />
    )
    const log = section('Their log, as shown to you')
    expect(within(log).getByText('3 exchanges')).toBeInTheDocument()
    expect(within(log).getByText('lists 1 witness (not checked here)')).toBeInTheDocument()
    expect(document.body.textContent).not.toContain('f'.repeat(64))
    expect(
      within(section('Verdicts on their exchanges')).getByText('1 · 0 acknowledged · 1 disputed')
    ).toBeInTheDocument()
    const asked = section('Asked of you')
    expect(within(asked).getByText('1 request that named you · 0 answered')).toBeInTheDocument()
    expect(within(asked).getByText(/not shared under the sharing setting/)).toBeInTheDocument()
  })
})

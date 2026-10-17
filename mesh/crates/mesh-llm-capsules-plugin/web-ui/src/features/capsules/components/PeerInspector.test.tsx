import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PeerInspector } from '@/features/capsules/components/PeerInspector'
import { HARNESS_PANE_B_PAYLOAD } from '@/features/capsules/lib/peer-fixtures'

describe('PeerInspector drill tabs', () => {
  // The drill's tab strip clipped "Exchanges" off its end. The
  // four tabs now share the width and a long label wraps.
  it('shows all four tabs, sharing the width, with labels allowed to wrap', () => {
    render(<PeerInspector meshStatus={null} onClose={() => {}} open points={[]} row={HARNESS_PANE_B_PAYLOAD.rows[0]} />)
    const list = screen.getByRole('tablist', { name: 'Peer inspector sections' })
    expect(list).toHaveClass('w-full')
    const tabs = screen.getAllByRole('tab')
    expect(tabs.map((tab) => tab.textContent)).toEqual([
      'Overview',
      'Their log, as shown to you',
      'Timeline',
      'Exchanges'
    ])
    for (const tab of tabs) {
      expect(tab).toHaveClass('flex-1', 'whitespace-normal')
    }
  })
})

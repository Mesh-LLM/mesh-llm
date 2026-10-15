// Item 4: the reusable (i) info glyph. Pins the accessibility contract the
// honest-copy-behind-a-hover pattern relies on -- a focusable, non-click
// trigger whose honest sentence is wired to it by `aria-describedby` and is
// present in the DOM whether or not the tooltip is open.
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { InfoHover } from '@/features/capsules/components/InfoHover'

describe('InfoHover — the reusable (i) info glyph', () => {
  it('renders a focusable, non-checkbox button glyph named for what it describes', () => {
    render(<InfoHover describes="Confirmed by the other side" label="the honest sentence" />)
    const glyph = screen.getByRole('button', { name: 'About Confirmed by the other side' })
    expect(glyph).toBeInTheDocument()
    // A button, so it takes keyboard focus (Radix Tooltip opens on hover AND
    // focus) -- never a click-only affordance.
    expect(glyph).toHaveAttribute('type', 'button')
  })

  it('wires aria-describedby from the glyph to a node carrying the honest sentence', () => {
    render(<InfoHover describes="the chip" label="their signed half of this exchange, held here and recomputed" />)
    const glyph = screen.getByRole('button', { name: 'About the chip' })
    const descriptionId = glyph.getAttribute('aria-describedby')
    expect(descriptionId).toBeTruthy()
    const description = document.getElementById(descriptionId as string)
    expect(description).not.toBeNull()
    // The sentence is present in the DOM regardless of the tooltip open state.
    expect(description).toHaveTextContent('their signed half of this exchange, held here and recomputed')
  })

  it('moves the EXACT copy passed in, never rewriting it', () => {
    const sentence = 'not a score — recomputed from sealed records'
    render(<InfoHover describes="tile" label={sentence} />)
    const glyph = screen.getByRole('button', { name: 'About tile' })
    const description = document.getElementById(glyph.getAttribute('aria-describedby') as string)
    expect(description?.textContent).toBe(sentence)
  })
})

// @vitest-environment jsdom
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Pager } from '@/components/ui/Pager'

describe('Pager', () => {
  it('renders nothing when a single page covers the content', () => {
    // Given a pager with one page
    const { container } = render(<Pager ariaLabel="Pages" count={1} value={0} onValueChange={vi.fn()} />)

    // Then no paging affordance is offered
    expect(container).toBeEmptyDOMElement()
  })

  it('renders one dot per page and marks the active page', () => {
    // Given a pager across four pages positioned on the second
    render(<Pager ariaLabel="Pages" count={4} value={1} onValueChange={vi.fn()} />)

    // Then every page is reachable and the active one is checked
    const dots = screen.getAllByRole('radio')
    expect(dots).toHaveLength(4)
    expect(dots[1]).toBeChecked()
    expect(screen.getByRole('radiogroup', { name: 'Pages' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Page 3 of 4' })).toBeInTheDocument()
  })

  it('keeps radio hit targets separate from the visual dot sizes', () => {
    // Given a pager with active and inactive page choices
    render(<Pager ariaLabel="Pages" count={2} value={0} onValueChange={vi.fn()} />)

    // Then each radio has a 24px hit target and an aria-hidden visual child
    const radios = screen.getAllByRole('radio')
    expect(radios).toHaveLength(2)
    for (const radio of radios) {
      expect(radio).toHaveClass('size-6', 'p-0')
      expect(radio.querySelector('[aria-hidden="true"]')).toBeInTheDocument()
    }
    expect(radios[0].querySelector('[aria-hidden="true"]')).toHaveClass('h-1.5', 'w-4')
    expect(radios[1].querySelector('[aria-hidden="true"]')).toHaveClass('size-1.5')
  })

  it('renders visible page numbers only when the numbered variant is requested', () => {
    // Given a numbered pager positioned on the second of four pages
    render(<Pager ariaLabel="Pages" count={4} value={1} variant="numbered" onValueChange={vi.fn()} />)

    // Then each radio shows its page number and the active page uses the selected control treatment
    const radios = screen.getAllByRole('radio')
    expect(radios).toHaveLength(4)
    expect(radios.map((radio) => radio.textContent)).toEqual(['1', '2', '3', '4'])
    expect(radios[0].querySelector('[aria-hidden="true"]')).toHaveClass('text-fg-dim')
    expect(radios[1].querySelector('[aria-hidden="true"]')).toHaveClass('bg-accent', 'text-accent-ink')
  })

  it('gives numbered controls distinct operational geometry and surfaces', () => {
    // Given a numbered pager on its first page
    render(<Pager ariaLabel="Frames" count={3} value={0} variant="numbered" onValueChange={vi.fn()} />)

    // Then the navigator uses a wrapping grid with deliberate spacing
    const group = screen.getByRole('radiogroup', { name: 'Frames' })
    expect(group.parentElement).toHaveClass('grid', 'grid-cols-[auto_minmax(0,1fr)_auto]', 'gap-2')
    expect(group).toHaveClass('flex-wrap', 'gap-1.5')

    // And every step and direct choice is a bordered 32px operational target
    const previous = screen.getByRole('button', { name: 'Previous page' })
    const next = screen.getByRole('button', { name: 'Next page' })
    expect(previous).toHaveClass('size-10', 'border', 'border-border', 'bg-panel')
    expect(previous).toHaveClass('disabled:opacity-50')
    expect(previous).toBeDisabled()
    expect(next).toHaveClass('size-10', 'border', 'border-border', 'bg-panel')
    expect(next).toBeEnabled()

    const radios = screen.getAllByRole('radio')
    expect(radios.map((radio) => radio.textContent)).toEqual(['1', '2', '3'])
    for (const radio of radios) {
      expect(radio).toHaveClass('size-10', 'border')
    }
    expect(radios[0]).toHaveClass('border-accent', 'bg-accent', 'text-accent-ink')
    expect(radios[1]).toHaveClass('border-border', 'bg-panel', 'hover:border-border-strong', 'hover:bg-panel-strong')
  })

  it('supports direct and keyboard numbered movement with a domain status', async () => {
    // Given a controlled numbered frame pager
    const user = userEvent.setup()
    const onValueChange = vi.fn()
    const statusLabel = (index: number, count: number) => `Frame ${index + 1} of ${count}`
    const { rerender } = render(
      <Pager
        ariaLabel="Frames"
        count={3}
        statusLabel={statusLabel}
        value={0}
        variant="numbered"
        onValueChange={onValueChange}
      />
    )

    // Then the first boundary and live status are explicit
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent('Frame 1 of 3')

    // When a numbered choice is selected directly
    await user.click(screen.getByRole('radio', { name: 'Page 3 of 3' }))

    // Then the direct logical index is reported
    expect(onValueChange).toHaveBeenLastCalledWith(2)

    // When the controlled value moves to frame two and ArrowRight is pressed
    rerender(
      <Pager
        ariaLabel="Frames"
        count={3}
        statusLabel={statusLabel}
        value={1}
        variant="numbered"
        onValueChange={onValueChange}
      />
    )
    screen.getByRole('radio', { name: 'Page 2 of 3' }).focus()
    await user.keyboard('{ArrowRight}')

    // Then one logical frame is requested
    expect(onValueChange).toHaveBeenLastCalledWith(2)

    // When the controlled pager reaches the final frame
    rerender(
      <Pager
        ariaLabel="Frames"
        count={3}
        statusLabel={statusLabel}
        value={2}
        variant="numbered"
        onValueChange={onValueChange}
      />
    )

    // Then the final boundary and domain status are explicit
    expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeEnabled()
    expect(screen.getByRole('status')).toHaveTextContent('Frame 3 of 3')
  })

  it('uses zero adjacent spacing so seven targets and arrows fit the narrow contract', () => {
    // Given the maximum visible page window
    render(<Pager ariaLabel="Pages" count={84} value={41} onValueChange={vi.fn()} />)

    // Then nine 24px controls occupy 216px before zero-width visual gaps
    expect(screen.getByRole('radiogroup').parentElement).toHaveClass('gap-0')
    expect(screen.getByRole('radiogroup')).toHaveClass('gap-0')
    expect(screen.getByRole('button', { name: 'Previous page' })).toHaveClass('size-6')
    expect(screen.getByRole('button', { name: 'Next page' })).toHaveClass('size-6')
    expect(screen.getAllByRole('radio')).toHaveLength(7)
    for (const radio of screen.getAllByRole('radio')) {
      expect(radio).toHaveClass('size-6')
    }
    for (const gap of screen.getAllByTestId('pager-gap')) {
      expect(gap).toHaveClass('w-0', 'text-foreground')
    }
  })

  it('announces the current page once through a polite status', () => {
    // Given a pager with multiple pages
    const { rerender } = render(<Pager ariaLabel="Pages" count={4} value={0} onValueChange={vi.fn()} />)

    // Then the current page is available as one non-focus-moving live status
    const status = screen.getByRole('status')
    expect(status).toHaveTextContent('Page 1 of 4')
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).toHaveAttribute('aria-atomic', 'true')

    // When the controlled page changes
    rerender(<Pager ariaLabel="Pages" count={4} value={1} onValueChange={vi.fn()} />)

    // Then only the status text changes
    expect(screen.getByRole('status')).toHaveTextContent('Page 2 of 4')
  })

  it('disables the indicator transition under reduced motion', () => {
    // Given a pager with active and inactive indicators
    render(<Pager ariaLabel="Pages" count={2} value={0} onValueChange={vi.fn()} />)

    // Then indicator motion has an explicit reduced-motion override
    expect(screen.getAllByRole('radio')[0].querySelector('[aria-hidden="true"]')).toHaveClass(
      'motion-reduce:transition-none'
    )
  })

  it('bounds large page selections while retaining the current, first, and last pages', async () => {
    // Given a pager with 84 pages positioned in the middle
    const user = userEvent.setup()
    const onValueChange = vi.fn()
    render(
      <Pager
        ariaLabel="Pages"
        count={84}
        pageLabel={(index) => `Page ${index + 1} of 84`}
        value={41}
        onValueChange={onValueChange}
      />
    )

    // Then the visible page choices remain bounded and retain navigation anchors
    const radios = screen.getAllByRole('radio')
    expect(radios.length).toBeLessThanOrEqual(7)
    expect(screen.getByRole('radio', { name: 'Page 1 of 84' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Page 42 of 84' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'Page 84 of 84' })).toBeInTheDocument()

    // When the reader uses previous and next navigation
    await user.click(screen.getByRole('button', { name: 'Previous page' }))
    await user.click(screen.getByRole('button', { name: 'Next page' }))

    // Then each control advances exactly one page
    expect(onValueChange).toHaveBeenNthCalledWith(1, 40)
    expect(onValueChange).toHaveBeenNthCalledWith(2, 42)
  })

  it('keeps visible large-page choices in Radix arrow-key order', async () => {
    // Given a bounded page window focused on the current page
    const user = userEvent.setup()
    render(<Pager ariaLabel="Pages" count={84} value={41} onValueChange={vi.fn()} />)
    const current = screen.getByRole('radio', { name: 'Page 42 of 84' })

    // When the reader moves forward with the horizontal arrow key
    current.focus()
    await user.keyboard('{ArrowRight}')

    // Then focus advances to the next visible radio without adding controls
    expect(screen.getByRole('radio', { name: 'Page 43 of 84' })).toHaveFocus()
    expect(screen.getAllByRole('radio')).toHaveLength(7)
  })

  it('marks nonadjacent visible page choices with hidden visual gaps', () => {
    // Given a bounded pager around page six of a large collection
    render(<Pager ariaLabel="Pages" count={125} value={5} onValueChange={vi.fn()} />)

    // Then gaps separate nonconsecutive choices without becoming radio options or focus targets
    const gaps = screen.getAllByTestId('pager-gap')
    expect(gaps).toHaveLength(2)
    for (const gap of gaps) {
      expect(gap).toHaveAttribute('aria-hidden', 'true')
      expect(gap).not.toHaveAttribute('role')
      expect(gap).not.toHaveAttribute('tabindex')
    }
    expect(screen.getAllByRole('radio')).toHaveLength(7)
    expect(screen.getByRole('radio', { name: 'Page 1 of 125' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Page 4 of 125' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Page 8 of 125' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Page 125 of 125' })).toBeInTheDocument()
  })

  it('moves one logical page per horizontal arrow and clamps at both ends', async () => {
    // Given a large pager positioned on page six
    const user = userEvent.setup()
    const onValueChange = vi.fn()
    const { rerender } = render(<Pager ariaLabel="Pages" count={125} value={5} onValueChange={onValueChange} />)
    const current = screen.getByRole('radio', { name: 'Page 6 of 125' })

    // When the reader presses ArrowRight
    current.focus()
    await user.keyboard('{ArrowRight}')

    // Then exactly the next logical page is requested
    expect(onValueChange).toHaveBeenLastCalledWith(6)

    // When the controlled pager recomputes around page seven
    rerender(<Pager ariaLabel="Pages" count={125} value={6} onValueChange={onValueChange} />)

    // Then the selected page remains in the visible window
    expect(screen.getByRole('radio', { name: 'Page 7 of 125' })).toBeChecked()

    // When the pager is at the first page and ArrowLeft is pressed
    rerender(<Pager ariaLabel="Pages" count={125} value={0} onValueChange={onValueChange} />)
    screen.getByRole('radio', { name: 'Page 1 of 125' }).focus()
    await user.keyboard('{ArrowLeft}')
    expect(onValueChange).toHaveBeenLastCalledWith(0)

    // When the pager is at the last page and ArrowRight is pressed
    rerender(<Pager ariaLabel="Pages" count={125} value={124} onValueChange={onValueChange} />)
    screen.getByRole('radio', { name: 'Page 125 of 125' }).focus()
    await user.keyboard('{ArrowRight}')
    expect(onValueChange).toHaveBeenLastCalledWith(124)
  })

  it('moves one logical page for vertical arrow keys', async () => {
    // Given a large pager positioned on page six
    const user = userEvent.setup()
    const onValueChange = vi.fn()
    render(<Pager ariaLabel="Pages" count={125} value={5} onValueChange={onValueChange} />)
    const current = screen.getByRole('radio', { name: 'Page 6 of 125' })
    current.focus()

    // When the reader presses ArrowDown and ArrowUp
    await user.keyboard('{ArrowDown}')
    await user.keyboard('{ArrowUp}')

    // Then each key requests one adjacent logical page
    expect(onValueChange).toHaveBeenNthCalledWith(1, 6)
    expect(onValueChange).toHaveBeenNthCalledWith(2, 4)
  })

  it('steps through pages and clamps at both ends', async () => {
    // Given a pager on the first of three pages
    const user = userEvent.setup()
    const onValueChange = vi.fn()
    const { rerender } = render(<Pager ariaLabel="Pages" count={3} value={0} onValueChange={onValueChange} />)

    // Then the backwards step is unavailable and the forwards step advances
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(onValueChange).toHaveBeenCalledWith(1)

    // When the pager reaches the final page
    rerender(<Pager ariaLabel="Pages" count={3} value={2} onValueChange={onValueChange} />)

    // Then the forwards step is unavailable
    expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeEnabled()
  })

  it('selects a page directly from its dot', async () => {
    // Given a pager across three pages
    const user = userEvent.setup()
    const onValueChange = vi.fn()
    render(<Pager ariaLabel="Pages" count={3} value={0} onValueChange={onValueChange} />)

    // When the last dot is chosen
    await user.click(screen.getByRole('radio', { name: 'Page 3 of 3' }))

    // Then the pager reports that page
    expect(onValueChange).toHaveBeenCalledWith(2)
  })

  it('uses caller-supplied labels', () => {
    // Given a pager with domain labels
    render(
      <Pager
        ariaLabel="Lifecycle pages"
        count={2}
        nextLabel="Later events"
        pageLabel={(index) => `Segment ${index + 1}`}
        previousLabel="Earlier events"
        value={0}
        onValueChange={vi.fn()}
      />
    )

    // Then those labels reach assistive technology
    expect(screen.getByRole('button', { name: 'Later events' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Earlier events' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Segment 2' })).toBeInTheDocument()
  })
})

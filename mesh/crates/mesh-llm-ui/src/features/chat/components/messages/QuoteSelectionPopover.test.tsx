import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { QuoteSelectionPopover } from './QuoteSelectionPopover'

function selectText(text: string, containerAttrs: Record<string, string> = {}, start = 0, end = text.length) {
  const container = document.createElement('div')
  for (const [key, value] of Object.entries(containerAttrs)) container.setAttribute(key, value)
  const node = document.createTextNode(text)
  container.appendChild(node)
  document.body.appendChild(container)
  const range = document.createRange()
  range.setStart(node, start)
  range.setEnd(node, end)
  const selection = document.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)
  return container
}

async function settleMouseUp() {
  await act(async () => {
    fireEvent.mouseUp(document.body)
    vi.advanceTimersByTime(1)
  })
}

async function settleKeyboardSelection() {
  await act(async () => {
    document.dispatchEvent(new Event('selectionchange'))
    vi.advanceTimersByTime(150)
  })
}

describe('QuoteSelectionPopover', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    // jsdom Ranges have no layout; give every range a stable rect so the
    // popover can position.
    ;(Range.prototype as { getBoundingClientRect?: unknown }).getBoundingClientRect = () => new DOMRect(10, 10, 120, 24)
  })

  afterEach(() => {
    vi.useRealTimers()
    delete (Range.prototype as { getBoundingClientRect?: unknown }).getBoundingClientRect
    document.getSelection()?.removeAllRanges()
    document.body.replaceChildren()
  })

  it('offers the quote for a selection inside an assistant message and hands it over on click', async () => {
    selectText('The relay health line reports answered relays.', { 'data-quote-source': 'assistant' }, 4, 16)
    const onQuote = vi.fn()
    render(<QuoteSelectionPopover onQuote={onQuote} />)
    await settleMouseUp()

    const button = screen.getByRole('button', { name: /ask about this/i })
    await act(async () => {
      fireEvent.click(button)
    })

    expect(onQuote).toHaveBeenCalledTimes(1)
    expect(onQuote.mock.calls[0][0].text).toBe('relay health')
    // Consumed: popover gone, selection cleared.
    expect(screen.queryByRole('button', { name: /ask about this/i })).toBeNull()
    expect(document.getSelection()?.isCollapsed).toBe(true)
  })

  it('surfaces the affordance for keyboard-built selections once they settle', async () => {
    selectText('keyboard selected passage', { 'data-quote-source': 'assistant' })
    const onQuote = vi.fn()
    render(<QuoteSelectionPopover onQuote={onQuote} />)

    await settleKeyboardSelection()

    expect(screen.getByRole('button', { name: /ask about this/i })).not.toBeNull()
    expect(onQuote).not.toHaveBeenCalled()
  })

  it('stays hidden for selections outside assistant messages', async () => {
    selectText('plain paragraph text')
    const onQuote = vi.fn()
    render(<QuoteSelectionPopover onQuote={onQuote} />)
    await settleMouseUp()

    expect(screen.queryByRole('button', { name: /ask about this/i })).toBeNull()
    expect(onQuote).not.toHaveBeenCalled()
  })

  it('dismisses on Escape', async () => {
    selectText('selected assistant passage', { 'data-quote-source': 'assistant' })
    render(<QuoteSelectionPopover onQuote={vi.fn()} />)
    await settleMouseUp()
    expect(screen.getByRole('button', { name: /ask about this/i })).not.toBeNull()

    await act(async () => {
      fireEvent.keyDown(document, { key: 'Escape' })
    })
    expect(screen.queryByRole('button', { name: /ask about this/i })).toBeNull()
  })

  it('stays dismissed after a scroll cancels an armed keyboard read', async () => {
    selectText('selected assistant passage', { 'data-quote-source': 'assistant' })
    render(<QuoteSelectionPopover onQuote={vi.fn()} />)
    await settleMouseUp()
    expect(screen.getByRole('button', { name: /ask about this/i })).not.toBeNull()

    await act(async () => {
      document.dispatchEvent(new Event('selectionchange'))
      fireEvent.scroll(window)
      vi.advanceTimersByTime(150)
    })
    expect(screen.queryByRole('button', { name: /ask about this/i })).toBeNull()
  })

  it('does not surface the popover mid-drag; the pointer flow owns the read', async () => {
    selectText('dragged passage', { 'data-quote-source': 'assistant' })
    render(<QuoteSelectionPopover onQuote={vi.fn()} />)

    await act(async () => {
      fireEvent.mouseDown(document.body)
      document.dispatchEvent(new Event('selectionchange'))
      vi.advanceTimersByTime(150)
    })
    expect(screen.queryByRole('button', { name: /ask about this/i })).toBeNull()
  })

  it('hands over the live selection, not a stale capture, when clicked after a quick change', async () => {
    selectText('first passage', { 'data-quote-source': 'assistant' })
    const onQuote = vi.fn()
    render(<QuoteSelectionPopover onQuote={onQuote} />)
    await settleMouseUp()

    // The selection changes again and the click lands before the debounce fires.
    selectText('second passage', { 'data-quote-source': 'assistant' })
    await act(async () => {
      document.dispatchEvent(new Event('selectionchange'))
    })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /ask about this/i }))
    })

    expect(onQuote).toHaveBeenCalledTimes(1)
    expect(onQuote.mock.calls[0][0].text).toBe('second passage')
  })

  it('keeps the popover inside the right viewport edge', async () => {
    const innerWidth = window.innerWidth
    ;(Range.prototype as { getBoundingClientRect?: unknown }).getBoundingClientRect = () =>
      new DOMRect(innerWidth - 12, 10, 8, 24)
    selectText('flush right selection', { 'data-quote-source': 'assistant' })
    render(<QuoteSelectionPopover onQuote={vi.fn()} />)
    await settleMouseUp()

    const popover = screen.getByTestId('quote-selection-popover')
    const left = Number(popover.style.left.replace('px', ''))
    expect(left).toBeLessThanOrEqual(innerWidth - 8 - 150)
    expect(left).toBeGreaterThanOrEqual(8)
  })

  it('rejects a non-collapsed live selection in non-quotable text instead of falling back to the stale quote', async () => {
    selectText('original assistant passage', { 'data-quote-source': 'assistant' })
    const onQuote = vi.fn()
    render(<QuoteSelectionPopover onQuote={onQuote} />)

    await act(async () => {
      fireEvent.mouseUp(document.body)
      vi.advanceTimersByTime(1)
    })
    expect(screen.getByRole('button', { name: /ask about this/i })).not.toBeNull()

    // Move the selection into a user message (non-quotable, non-collapsed)
    selectText('user message text that cannot be quoted')
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /ask about this/i }))
    })

    // Live selection is non-collapsed and extraction failed — no stale fallback
    expect(onQuote).not.toHaveBeenCalled()
  })

  it('falls back to the captured quote when the live selection has collapsed', async () => {
    selectText('captured assistant passage', { 'data-quote-source': 'assistant' })
    const onQuote = vi.fn()
    render(<QuoteSelectionPopover onQuote={onQuote} />)

    await act(async () => {
      fireEvent.mouseUp(document.body)
      vi.advanceTimersByTime(1)
    })

    // Collapse the selection
    document.getSelection()?.removeAllRanges()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /ask about this/i }))
    })

    // Selection is collapsed — the captured quote is used
    expect(onQuote).toHaveBeenCalledTimes(1)
    expect(onQuote.mock.calls[0][0].text).toBe('captured assistant passage')
  })

  it('cancels a pending keyboard read when a pointer press starts', async () => {
    selectText('keyboard then mouse passage', { 'data-quote-source': 'assistant' })
    render(<QuoteSelectionPopover onQuote={vi.fn()} />)

    // Arm the keyboard read
    await act(async () => {
      document.dispatchEvent(new Event('selectionchange'))
    })
    // Pointer press arrives before the 150ms debounce fires
    await act(async () => {
      fireEvent.mouseDown(document.body)
      vi.advanceTimersByTime(150)
    })
    // The keyboard timer was cancelled; the popover must not appear
    expect(screen.queryByRole('button', { name: /ask about this/i })).toBeNull()
  })

  it('does not arm the keyboard read for a touch-originated selection on a hybrid device', async () => {
    selectText('touch selected passage', { 'data-quote-source': 'assistant' })
    render(<QuoteSelectionPopover onQuote={vi.fn()} />)

    // Touch gesture (pointerdown with pointerType touch, no mousedown)
    await act(async () => {
      document.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch' }))
      document.dispatchEvent(new Event('selectionchange'))
      vi.advanceTimersByTime(150)
    })
    // The popover must not appear for a touch-originated selection
    expect(screen.queryByRole('button', { name: /ask about this/i })).toBeNull()
  })

  it('arms the keyboard read again after a non-touch pointer event clears the touch flag', async () => {
    selectText('touch then mouse passage', { 'data-quote-source': 'assistant' })
    render(<QuoteSelectionPopover onQuote={vi.fn()} />)

    // Touch first — suppressed
    await act(async () => {
      document.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch' }))
      document.dispatchEvent(new Event('selectionchange'))
      vi.advanceTimersByTime(150)
    })
    expect(screen.queryByRole('button', { name: /ask about this/i })).toBeNull()

    // Mouse pointer clears the touch flag
    await act(async () => {
      document.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'mouse' }))
      document.dispatchEvent(new Event('selectionchange'))
      vi.advanceTimersByTime(150)
    })
    expect(screen.getByRole('button', { name: /ask about this/i })).not.toBeNull()
  })
})

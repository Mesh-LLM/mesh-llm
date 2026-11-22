import { useEffect, useRef, useState } from 'react'
import { TextQuote } from 'lucide-react'

import { extractQuoteFromSelection, type ComposerQuote } from '@/features/chat/lib/quote-to-ask'

type PendingQuoteSelection = {
  quote: ComposerQuote
  rect: DOMRect
}

/** Conservative size budgets for the "Ask about this" button so edge
 *  selections cannot push it out of the viewport. */
const POPOVER_WIDTH_BUDGET = 150
const POPOVER_HEIGHT_BUDGET = 40

/**
 * Floating "Ask about this" affordance for quote-to-ask (desktop-first):
 * appears near a text selection anchored inside a single assistant message
 * and hands the extracted quote to the composer. The quote is captured on
 * mouseup (or a settled keyboard selection), before the popover's own click
 * can collapse the selection.
 */
export function QuoteSelectionPopover({ onQuote }: { onQuote: (quote: ComposerQuote) => void }) {
  const [pending, setPending] = useState<PendingQuoteSelection | null>(null)
  // Latest callback without rebinding the document listeners on every render.
  const onQuoteRef = useRef(onQuote)
  useEffect(() => {
    onQuoteRef.current = onQuote
  }, [onQuote])

  useEffect(() => {
    let showing = false
    const readSelection = () => {
      const selection = document.getSelection()
      if (!selection || selection.rangeCount === 0) {
        showing = false
        setPending(null)
        return
      }
      // Quote first: no assistant-message selection means no popover, and
      // the rect read below stays untouched in DOMs without layout (jsdom).
      const quote = extractQuoteFromSelection(selection)
      if (!quote) {
        showing = false
        setPending(null)
        return
      }
      const range = selection.getRangeAt(0)
      const rect = typeof range.getBoundingClientRect === 'function' ? range.getBoundingClientRect() : null
      if (!rect || (rect.width === 0 && rect.height === 0)) {
        showing = false
        setPending(null)
        return
      }
      showing = true
      setPending({ quote, rect })
    }
    // Keyboard-built selections (shift+arrows, select-all) never fire mouseup;
    // read them once the selection has stopped changing. Dismissals cancel the
    // armed read, an in-progress pointer drag never arms it (mouseup owns that
    // flow), and touch-originated selections never arm it: the native
    // selection menu owns touch, tracked by gesture rather than device class
    // so hybrid touch-plus-hover pointers are excluded correctly too.
    let keyboardReadTimer: number | undefined
    let mouseupReadTimer: number | undefined
    let pointerDown = false
    let touchGesture = false
    const dismiss = () => {
      window.clearTimeout(keyboardReadTimer)
      window.clearTimeout(mouseupReadTimer)
      showing = false
      setPending(null)
    }
    const onPointerDown = (event: PointerEvent) => {
      touchGesture = event.pointerType === 'touch'
      if (touchGesture) window.clearTimeout(keyboardReadTimer)
    }
    const onMouseDown = () => {
      pointerDown = true
      // A pointer press owns the selection flow from here; a keyboard-armed
      // read must not fire mid-gesture.
      window.clearTimeout(keyboardReadTimer)
    }
    const onMouseUp = (event: MouseEvent) => {
      pointerDown = false
      // The popover's own button must not re-run extraction over the
      // selection it is about to consume.
      if (event.target instanceof Element && event.target.closest('[data-quote-popover]')) return
      // Let the browser settle the selection before reading it.
      mouseupReadTimer = window.setTimeout(readSelection, 0)
    }
    const onSelectionChange = () => {
      const selection = document.getSelection()
      if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
        dismiss()
        return
      }
      window.clearTimeout(keyboardReadTimer)
      if (!pointerDown && !touchGesture) keyboardReadTimer = window.setTimeout(readSelection, 150)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || !showing) return
      // Capture phase + stopPropagation: while the popover is up it owns
      // Escape, so the same keypress cannot also trigger the composer's
      // documented Escape-to-stop on a streaming response. Once dismissed,
      // the next Escape reaches the composer as before.
      event.stopPropagation()
      dismiss()
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('mousedown', onMouseDown)
    document.addEventListener('mouseup', onMouseUp)
    document.addEventListener('selectionchange', onSelectionChange)
    document.addEventListener('keydown', onKeyDown, { capture: true })
    // Capture so any scroll container (message list, sidebar) dismisses.
    window.addEventListener('scroll', dismiss, true)
    window.addEventListener('resize', dismiss)
    return () => {
      window.clearTimeout(mouseupReadTimer)
      dismiss()
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('mousedown', onMouseDown)
      document.removeEventListener('mouseup', onMouseUp)
      document.removeEventListener('selectionchange', onSelectionChange)
      document.removeEventListener('keydown', onKeyDown, { capture: true } as EventListenerOptions)
      window.removeEventListener('scroll', dismiss, true)
      window.removeEventListener('resize', dismiss)
    }
  }, [])

  if (!pending) return null
  const { quote, rect } = pending
  const left = Math.max(Math.min(rect.left, window.innerWidth - 8 - POPOVER_WIDTH_BUDGET), 8)
  const belowTop = rect.bottom + 6
  // Flip above the selection when there is no room below it, so a selection
  // at the tail of the last answer cannot cover the composer.
  const top =
    belowTop + POPOVER_HEIGHT_BUDGET > window.innerHeight ? Math.max(rect.top - 6 - POPOVER_HEIGHT_BUDGET, 8) : belowTop
  return (
    <div data-quote-popover="" data-testid="quote-selection-popover" className="fixed z-50" style={{ top, left }}>
      <button
        type="button"
        className="ui-control inline-flex items-center gap-1.5 rounded-[var(--radius)] border bg-panel px-2.5 py-1.5 font-mono text-[length:var(--density-type-caption)] text-fg-muted shadow-sm hover:text-fg"
        // Mousedown would collapse the captured selection before click fires.
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => {
          // The selection is still live here (mousedown was prevented), so
          // prefer a fresh read. If it changed to another quotable passage,
          // the user gets what is on screen, not the stale capture. If it
          // changed to non-quotable text (a user message), reject rather
          // than quoting the stale capture; fall back only when the live
          // selection has collapsed (the popover's own focus took it).
          const live = document.getSelection()
          const fresh = live ? extractQuoteFromSelection(live) : null
          if (fresh) {
            onQuoteRef.current(fresh)
          } else if (!live || live.isCollapsed) {
            onQuoteRef.current(quote)
          }
          live?.removeAllRanges()
          setPending(null)
        }}
      >
        <TextQuote aria-hidden={true} className="size-3" />
        Ask about this
      </button>
    </div>
  )
}

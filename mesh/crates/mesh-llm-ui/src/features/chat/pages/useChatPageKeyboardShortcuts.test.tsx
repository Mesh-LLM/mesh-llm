import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { useChatPageKeyboardShortcuts } from '@/features/chat/pages/useChatPageKeyboardShortcuts'

function pressO(init: KeyboardEventInit, target?: EventTarget) {
  // Browsers deliver 'O' (uppercase) while Shift is held; sending the
  // uppercase form here is what pins the toLowerCase() in the hook.
  const event = new KeyboardEvent('keydown', { key: 'O', bubbles: true, cancelable: true, ...init })
  ;(target ?? window).dispatchEvent(event)
  return event
}

describe('useChatPageKeyboardShortcuts', () => {
  it('starts a new chat on Cmd+Shift+O (macOS)', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))

    const event = pressO({ metaKey: true, shiftKey: true })

    expect(onNewChat).toHaveBeenCalledTimes(1)
    expect(event.defaultPrevented).toBe(true)
  })

  it('starts a new chat on Ctrl+Shift+O (Windows/Linux)', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))

    pressO({ ctrlKey: true, shiftKey: true })

    expect(onNewChat).toHaveBeenCalledTimes(1)
  })

  it('matches on the physical key when the layout types a non-Latin letter (Cyrillic)', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))

    // On a Cyrillic layout the QWERTY-O position reports key 'щ', code 'KeyO'.
    pressO({ key: 'щ', code: 'KeyO', metaKey: true, shiftKey: true })

    expect(onNewChat).toHaveBeenCalledTimes(1)
  })

  it('does not create a phantom chord on alt-Latin layouts (Colemak y at the QWERTY-O position)', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))

    // Colemak's QWERTY-O position types 'y' (code 'KeyO'); the physical-key
    // fallback must not apply, or Cmd+Shift+Y would spawn a chat.
    pressO({ key: 'y', code: 'KeyO', metaKey: true, shiftKey: true })

    expect(onNewChat).not.toHaveBeenCalled()
  })

  it('fires while a text field is focused, matching ChatGPT parity (drafts are preserved)', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))
    const textarea = document.body.appendChild(document.createElement('textarea'))

    pressO({ metaKey: true, shiftKey: true }, textarea)

    expect(onNewChat).toHaveBeenCalledTimes(1)
    textarea.remove()
  })

  it('ignores lookalike chords', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))

    pressO({})
    pressO({ shiftKey: true })
    pressO({ metaKey: true })
    pressO({ ctrlKey: true })
    pressO({ metaKey: true, shiftKey: true, altKey: true })

    expect(onNewChat).not.toHaveBeenCalled()
  })

  it('does not fire while a dialog is open', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))
    const dialog = document.body.appendChild(document.createElement('div'))
    dialog.setAttribute('role', 'dialog')
    const field = dialog.appendChild(document.createElement('textarea'))

    pressO({ metaKey: true, shiftKey: true }, field)

    expect(onNewChat).not.toHaveBeenCalled()
    dialog.remove()
  })

  it('does not fire while an alertdialog is open (delete-confirmation uses role="alertdialog")', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))
    const dialog = document.body.appendChild(document.createElement('div'))
    dialog.setAttribute('role', 'alertdialog')
    const field = dialog.appendChild(document.createElement('textarea'))

    pressO({ metaKey: true, shiftKey: true }, field)

    expect(onNewChat).not.toHaveBeenCalled()
    dialog.remove()
  })

  it('does not fire when an earlier handler already claimed the event', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))
    window.addEventListener('keydown', (event) => event.preventDefault(), { once: true, capture: true })

    pressO({ metaKey: true, shiftKey: true })

    expect(onNewChat).not.toHaveBeenCalled()
  })

  it('fires once for a held chord, ignoring key auto-repeat', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))

    pressO({ metaKey: true, shiftKey: true })
    pressO({ metaKey: true, shiftKey: true, repeat: true })
    pressO({ metaKey: true, shiftKey: true, repeat: true })

    expect(onNewChat).toHaveBeenCalledTimes(1)
  })

  it('does not fire during an IME composition', () => {
    const onNewChat = vi.fn()
    renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))

    pressO({ metaKey: true, shiftKey: true, isComposing: true })

    expect(onNewChat).not.toHaveBeenCalled()
  })

  it('removes the listener on unmount', () => {
    const onNewChat = vi.fn()
    const { unmount } = renderHook(() => useChatPageKeyboardShortcuts({ onNewChat }))

    unmount()
    pressO({ metaKey: true, shiftKey: true })

    expect(onNewChat).not.toHaveBeenCalled()
  })
})

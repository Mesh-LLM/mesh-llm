import { useEffect } from 'react'

type UseChatPageKeyboardShortcutsParams = {
  onNewChat: () => void
}

// ChatGPT-style new-chat chord: Cmd+Shift+O on macOS, Ctrl+Shift+O elsewhere.
// Fires even while the composer is focused (per-conversation drafts are
// preserved by the handler), but never inside a dialog — a system-prompt or
// delete-confirmation modal owns the keyboard while it is open.
export function useChatPageKeyboardShortcuts({ onNewChat }: UseChatPageKeyboardShortcutsParams) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return
      // Holding the chord auto-repeats keydown; without this guard one held
      // press spawns an empty conversation per repeat tick. An active IME
      // composition owns the keyboard likewise.
      if (event.repeat || event.isComposing) return

      const dialogTarget =
        event.target instanceof Element ? event.target.closest('[role="dialog"], [role="alertdialog"]') : null
      if (dialogTarget) return

      // `key` keeps the binding on the letter O for Dvorak/Colemak. `code`
      // rescues non-Latin layouts where the QWERTY-O position types something
      // else (Cyrillic щ) — but only when the typed letter is non-Latin, or
      // alt-Latin layouts would get a phantom chord on that position
      // (Colemak y, Dvorak r).
      const matchesChordKey = event.key.toLowerCase() === 'o' || (event.code === 'KeyO' && !/^[a-z]$/i.test(event.key))
      if (matchesChordKey && event.shiftKey && (event.metaKey || event.ctrlKey) && !event.altKey) {
        event.preventDefault()
        onNewChat()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onNewChat])
}

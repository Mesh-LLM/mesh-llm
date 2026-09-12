import { useCallback, useState } from 'react'
import { useClipboardCopy, type ClipboardCopyState } from '@/lib/useClipboardCopy'

export type ShareOutcome = 'shared' | 'copied' | 'dismissed' | 'failed'

type NavigatorWithShare = Navigator & {
  share?: (data: { title?: string; text?: string }) => Promise<void>
}

function nativeShare(): NavigatorWithShare['share'] | undefined {
  if (typeof navigator === 'undefined') return undefined
  const candidate = (navigator as NavigatorWithShare).share
  return typeof candidate === 'function' ? candidate.bind(navigator) : undefined
}

export function shareSheetAvailable() {
  return nativeShare() !== undefined
}

/**
 * Hand an invite to the platform share sheet when one exists, and fall back to
 * the clipboard everywhere else. A dismissed sheet is not a failure and must
 * not silently copy instead: the user chose to cancel.
 */
export function useShareInvite() {
  const { copyText } = useClipboardCopy()
  const [outcome, setOutcome] = useState<ShareOutcome | null>(null)

  const share = useCallback(
    async (text: string, title?: string): Promise<ShareOutcome> => {
      const sheet = nativeShare()
      if (sheet) {
        try {
          await sheet({ title, text })
          setOutcome('shared')
          return 'shared'
        } catch (error) {
          const dismissed = error instanceof DOMException && error.name === 'AbortError'
          const result: ShareOutcome = dismissed ? 'dismissed' : 'failed'
          setOutcome(result)
          return result
        }
      }

      const copied = await copyText(text)
      const result: ShareOutcome = copied ? 'copied' : 'failed'
      setOutcome(result)
      return result
    },
    [copyText]
  )

  return { share, outcome, canUseShareSheet: shareSheetAvailable() }
}

export function shareButtonLabel(outcome: ShareOutcome | null, canUseShareSheet: boolean): string {
  if (outcome === 'shared') return 'Shared'
  if (outcome === 'copied') return 'Copied'
  if (outcome === 'failed') return 'Failed'
  return canUseShareSheet ? 'Share' : 'Copy invite'
}

export type { ClipboardCopyState }

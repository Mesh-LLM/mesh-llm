import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, Copy } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function CopyRequestIdButton({ requestId }: { readonly requestId: string }) {
  const [copied, setCopied] = useState(false)
  const resetTimer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(resetTimer.current), [])

  const handleCopy = useCallback(() => {
    if (!navigator.clipboard) return
    void navigator.clipboard.writeText(requestId).then(
      () => {
        setCopied(true)
        resetTimer.current = window.setTimeout(() => setCopied(false), 1500)
      },
      () => setCopied(false)
    )
  }, [requestId])

  return (
    <Button
      aria-label={`Copy request ID ${requestId}`}
      className="ui-control-ghost h-6 w-6 shrink-0 rounded-[var(--radius-sm)] text-fg-faint hover:text-foreground"
      onClick={handleCopy}
      size="icon"
      title="Copy request ID"
      type="button"
      variant="ghost"
    >
      {copied ? <Check className="size-3" aria-hidden="true" /> : <Copy className="size-3" aria-hidden="true" />}
    </Button>
  )
}

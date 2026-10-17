// The plugin page's React root: the providers the console app used to supply
// around the fork's `/capsules` route, then the Evidence tab itself.
//
// The console routes `/plugins/capsules/evidence` to this page; the
// per-row deep link that used to be `/capsules/exchange/<key>` is now
// `?focusExchangeKey=<key>` on that route, read once at mount.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { LedgerPageContent } from '@/features/capsules/pages/LedgerPage'
import { DataModeProvider } from '@/lib/data-mode'

export const FOCUS_EXCHANGE_PARAM = 'focusExchangeKey'

export function focusExchangeKeyFrom(search: string): string | undefined {
  const value = new URLSearchParams(search).get(FOCUS_EXCHANGE_PARAM)
  return value ? value : undefined
}

export function EvidencePage({ focusExchangeKey }: { focusExchangeKey?: string }) {
  const [queryClient] = useState(() => new QueryClient())
  return (
    <QueryClientProvider client={queryClient}>
      {/* The page reads live plugin data; the console's own data-mode toggle
          is not part of the plugin contract, so it is not consulted. */}
      <DataModeProvider initialMode="live" persist={false}>
        <TooltipProvider>
          <LedgerPageContent focusExchangeKey={focusExchangeKey} />
        </TooltipProvider>
      </DataModeProvider>
    </QueryClientProvider>
  )
}

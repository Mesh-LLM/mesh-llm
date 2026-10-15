// The Ledger's exchange_id column cell: the short id
// plus an "open in Logs" affordance, split into its own file so the column
// definitions module (`ExchangeColumns.tsx`) stays a components-only export
// boundary for fast refresh.
import { ArrowUpRight } from 'lucide-react'
import { navigateHost } from '@/plugin-host/host'
import { SEE_IN_LOGS_NOT_ON_THIS_PAGE } from '@/features/capsules/lib/tooltip-copy'

/** A digest-keyed grouping fallback (`exchange_key_for`, host side) is not a
 *  real exchange join-key -- `open in Logs` must never link on one, since no
 *  Logs row can ever carry a `digest:` value as its `exchangeId`. */
function isRealExchangeId(exchangeKey: string): boolean {
  return !exchangeKey.startsWith('digest:')
}

/** "see in Logs" from an exchange row. Opening Logs at one exchange
 *  needs a host hook this page doesn't have yet (Logs carrying the exchange
 *  id), so the row says so in plain text; nothing shows for a digest-keyed
 *  row, which no Logs request can carry. */
export function SeeInLogsLink({ exchangeKey }: { exchangeKey: string }) {
  if (!isRealExchangeId(exchangeKey)) return null
  return (
    <span className="text-fg-faint" data-see-in-logs="plain">
      {SEE_IN_LOGS_NOT_ON_THIS_PAGE}
    </span>
  )
}

export function ExchangeIdCell({ exchangeKey }: { exchangeKey: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="font-mono text-foreground" title={exchangeKey}>
        {exchangeKey.slice(0, 8)}
      </span>
      {isRealExchangeId(exchangeKey) ? (
        <button
          aria-label={`Open exchange ${exchangeKey} in Logs`}
          className="ui-control-ghost rounded-[var(--radius)] p-0.5 text-fg-dim hover:text-foreground"
          onClick={(event) => {
            event.stopPropagation()
            navigateHost(`/logs?focusExchangeId=${encodeURIComponent(exchangeKey)}`)
          }}
          type="button"
        >
          <ArrowUpRight aria-hidden="true" className="size-3" />
        </button>
      ) : null}
    </span>
  )
}

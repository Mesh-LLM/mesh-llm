// One row of the Peers table (replaces the old
// `PeerCard` -- review §3-F: "Peers is a card, not the table"). Each
// accountability column gets its own cell, never blended into one
// paragraph. A row with exchange history opens the same `PeerInspector`
// modal PeerCard used to; an "advertised but unused" row (no `PaneBRow` at
// all) has nothing to drill into, so it isn't clickable.
//
// No online badge, latency, or
// Route-to-chat button here -- those are Network's job. `meshStatus` is
// still threaded through to `PeerInspector` (the deep-dive modal keeps its
// own operational Overview tab), just no longer rendered on the row.
import { useMemo, useState } from 'react'
import { TableCell, TableRow } from '@/components/ui/table'
import { StatusBadge } from '@/components/ui/StatusBadge'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import { PeerInspector } from '@/features/capsules/components/PeerInspector'
import { buildTimelinePoints, type PeerExchangeSource } from '@/features/capsules/lib/peer-exchange-timeline'
import type { PeerMeshStatus } from '@/features/capsules/lib/peer-mesh-status'
import {
  ALL_PEER_TABLE_COLUMNS,
  type PeerTableColumnKey,
  type PeerTableRowView
} from '@/features/capsules/lib/peer-row-view'
import { HoverChip } from '@/features/capsules/components/HoverChip'
import { InfoHover } from '@/features/capsules/components/InfoHover'
import { PEER_ALIAS_TOOLTIP, PEER_PAYMENTS_TOOLTIP, PEER_TWINS_TOOLTIP } from '@/features/capsules/lib/tooltip-copy'

export type PeerTableRowProps = {
  view: PeerTableRowView
  meshStatus: PeerMeshStatus | null
  exchangeSources?: readonly PeerExchangeSource[]
  recordsById?: ReadonlyMap<string, CapsuleRecord>
  /** Columns toggle (the Peers table toolbar) -- Peer/Alarm are
   *  never hideable, only the six accountability columns are. */
  visibleColumns?: ReadonlySet<PeerTableColumnKey>
}

const EMPTY_SOURCES: readonly PeerExchangeSource[] = []
const EMPTY_RECORDS: ReadonlyMap<string, CapsuleRecord> = new Map()

export function PeerTableRow({
  view,
  meshStatus,
  exchangeSources = EMPTY_SOURCES,
  recordsById = EMPTY_RECORDS,
  visibleColumns = ALL_PEER_TABLE_COLUMNS
}: PeerTableRowProps) {
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const points = useMemo(
    () => (view.row ? buildTimelinePoints(view.row, exchangeSources, recordsById) : []),
    [view.row, exchangeSources, recordsById]
  )
  const inspectable = view.row !== null
  const openInspector = () => setInspectorOpen(true)

  return (
    <>
      <TableRow
        aria-label={inspectable ? `Open peer inspector for ${view.displayId}` : undefined}
        className={
          inspectable
            ? 'cursor-pointer align-top outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent'
            : 'align-top'
        }
        onClick={inspectable ? openInspector : undefined}
        onKeyDown={
          inspectable
            ? (event) => {
                if (event.target !== event.currentTarget || (event.key !== 'Enter' && event.key !== ' ')) return
                event.preventDefault()
                openInspector()
              }
            : undefined
        }
        tabIndex={inspectable ? 0 : undefined}
      >
        <TableCell>
          <div className="flex flex-col items-start gap-1.5">
            <span className="truncate font-mono text-sm font-medium text-foreground">{view.displayId}</span>
            {/* D3: the peer's other
               id spaces render as ALIASES on this one row -- signing key ·
               node · endpoint -- never as extra peer rows. */}
            {view.aliasLine ? (
              <span className="inline-flex items-center gap-1">
                <span className="font-mono text-fg-faint text-xs">{view.aliasLine}</span>
                <InfoHover census="peer:aliases" describes="the peer's ids" label={PEER_ALIAS_TOOLTIP} />
              </span>
            ) : null}
            {/* UX §2: "self-reported" is true of every row, so it is said
               once in the legend under the table, not on each row. */}
            {view.identityNote ? <span className="text-fg-faint text-xs">{view.identityNote}</span> : null}
            {/* Visible while collapsed (chooser-v2 §3-F), but as the specific
               thing, counted, with one sentence on hover -- never a generic
               warning glyph (UX §8). */}
            {view.payments ? (
              <HoverChip census="peer:payments" label={PEER_PAYMENTS_TOOLTIP}>
                <span className="text-xs text-fg-dim" data-peer-payments="true">
                  Payments: {view.payments}
                </span>
              </HoverChip>
            ) : null}
            {view.twins ? (
              <HoverChip census="peer:twins" label={PEER_TWINS_TOOLTIP}>
                <span className="text-xs text-fg-dim" data-peer-twins="true">
                  {view.twins}
                </span>
              </HoverChip>
            ) : null}
            {view.attention.map((item) => (
              <HoverChip census={`peer_attention:${item.key}`} key={item.key} label={item.tooltip}>
                <span>
                  <StatusBadge size="caption" tone={item.tone}>
                    {item.label}
                  </StatusBadge>
                </span>
              </HoverChip>
            ))}
          </div>
        </TableCell>
        {visibleColumns.has('exchanges') ? (
          <TableCell className="text-xs text-fg-dim">{view.exchangeCount}</TableCell>
        ) : null}
        {visibleColumns.has('confirmed') ? (
          <TableCell className="text-xs text-fg-dim">{view.confirmedByOtherSide}</TableCell>
        ) : null}
        {visibleColumns.has('match') ? <TableCell className="text-xs text-fg-dim">{view.match}</TableCell> : null}
        {visibleColumns.has('adjudication') ? (
          <TableCell className="text-xs text-fg-dim">{view.adjudicationCompact}</TableCell>
        ) : null}
        {visibleColumns.has('witness') ? (
          <TableCell className="text-xs text-fg-dim">{view.witnessCompact}</TableCell>
        ) : null}
        {visibleColumns.has('period') ? <TableCell className="text-xs text-fg-dim">{view.period}</TableCell> : null}
      </TableRow>
      {view.row ? (
        <PeerInspector
          meshStatus={meshStatus}
          onClose={() => setInspectorOpen(false)}
          open={inspectorOpen}
          points={points}
          row={view.row}
        />
      ) : null}
    </>
  )
}

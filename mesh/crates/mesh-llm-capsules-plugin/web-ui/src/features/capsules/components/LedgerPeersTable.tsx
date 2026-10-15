// The Peers table -- review §3-F: "table with the
// Exchanges toolbar (Search / Filter / Columns / Export / Save evidence
// file / Reset view)"; two row groups, "Nodes you have dealt with" /
// "Nodes advertised but unused", NEVER merged into one sorted list (they're
// two separate `<TableBody>` sections fed by two separate arrays -- no
// shared sort key could accidentally interleave them). No column here is
// sortable (R-B: no sort/no default order/no ranking by outcome) -- headers
// are plain text, not `DataTableColumnHeader`, so there is no sort control
// to wire up in the first place.
//
// Accountability-only: the online/
// offline status filter is retired along with the row's own online badge,
// latency, and Route-to-chat button -- that's Network's job now. The Alarm
// filter stays (a sealed contradiction/failed-verification IS an
// accountability fact).
import { useMemo, useState } from 'react'
import { Columns3, RotateCcw, Search as SearchIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/DropdownMenu'
import { FilterPopover, type FilterValueOption } from '@/components/ui/FilterPopover'
import { Input } from '@/components/ui/input'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import { PeerTableRow } from '@/features/capsules/components/PeerTableRow'
import type { PeerExchangeSource } from '@/features/capsules/lib/peer-exchange-timeline'
import { peerEvidenceBundle, peerRowsToCsv } from '@/features/capsules/lib/peer-export'
import type { PeerMeshStatusIndex } from '@/features/capsules/lib/peer-mesh-status'
import { saveTextFile } from '@/features/capsules/lib/exchange-export'
import type { PaneBRow } from '@/features/capsules/api/sidecarTypes'
import {
  ALL_PEER_TABLE_COLUMNS,
  PEER_COLUMN_INFO,
  SELF_REPORTED_DETAIL,
  SELF_REPORTED_NOTE,
  type PeerTableColumnKey,
  type PeerTableRowView
} from '@/features/capsules/lib/peer-row-view'
import { InfoHover } from '@/features/capsules/components/InfoHover'
import type { SplitPeer } from '@/features/capsules/lib/split-stage'
import { ADVERTISED_UNUSED_TOOLTIP, PEER_NEEDS_A_LOOK_LABEL, SPLIT_TOOLTIPS } from '@/features/capsules/lib/tooltip-copy'
import type { Peer } from '@/features/app-tabs/types'

// Plain-language column names (UX §2). One label drives both the header and
// the Columns dropdown -- never "Period"/"Window" (bookkeeping vocabulary).
const COLUMN_LABELS: Record<PeerTableColumnKey, string> = {
  exchanges: 'Exchanges',
  confirmed: 'They confirmed',
  match: 'Same request & answer',
  adjudication: 'Disputes judged',
  witness: 'Their records witnessed',
  period: 'Last dealt with'
}
const ALL_COLUMNS: PeerTableColumnKey[] = [...ALL_PEER_TABLE_COLUMNS]

type AlarmFilterValue = 'has_alarm' | 'clean'
type PeerFilterKey = 'alarm'

const ALL_ALARM_VALUES: AlarmFilterValue[] = ['has_alarm', 'clean']

function filterOptionLabel(value: string): string {
  switch (value) {
    case 'has_alarm':
      return 'Needs a look'
    case 'clean':
      return 'Clean'
    default:
      return value
  }
}

function alarmValue(view: PeerTableRowView): AlarmFilterValue {
  return view.alarm.present ? 'has_alarm' : 'clean'
}

function matchesSearch(view: PeerTableRowView, query: string): boolean {
  if (!query) return true
  return view.displayId.toLowerCase().includes(query)
}

function GroupHeaderRow({
  label,
  count,
  columnCount,
  tooltip
}: {
  label: string
  count: number
  columnCount: number
  tooltip?: string
}) {
  return (
    <TableRow className="border-border-soft bg-panel-strong/40 hover:bg-panel-strong/40">
      <TableCell className="type-caption font-mono text-fg-faint" colSpan={columnCount}>
        <span className="inline-flex items-center gap-1">
          {label} · {count}
          {tooltip ? <InfoHover census={`peer_group:${label}`} describes={label} label={tooltip} /> : null}
        </span>
      </TableCell>
    </TableRow>
  )
}

export type LedgerPeersTableProps = {
  dealtWith: readonly PeerTableRowView[]
  advertisedUnused: readonly PeerTableRowView[]
  dealtWithRawRows: readonly PaneBRow[]
  advertisedUnusedRawPeers: readonly Peer[]
  meshStatus: PeerMeshStatusIndex
  exchangeSourcesFor: (peerId: string) => readonly PeerExchangeSource[]
  recordsById: ReadonlyMap<string, CapsuleRecord>
  /** Nodes that ran part of a request this node sent to a split's
   *  coordinator: a third group, never merged with the other two. */
  throughSplit?: readonly SplitPeer[]
}

export function LedgerPeersTable({
  dealtWith,
  advertisedUnused,
  dealtWithRawRows,
  advertisedUnusedRawPeers,
  meshStatus,
  exchangeSourcesFor,
  recordsById,
  throughSplit = []
}: LedgerPeersTableProps) {
  const [search, setSearch] = useState('')
  const [alarmFilter, setAlarmFilter] = useState<Set<string>>(new Set(ALL_ALARM_VALUES))
  const [visibleColumns, setVisibleColumns] = useState<Set<PeerTableColumnKey>>(new Set(ALL_COLUMNS))

  const trimmedSearch = search.trim().toLowerCase()
  const allRows = useMemo(() => [...dealtWith, ...advertisedUnused], [dealtWith, advertisedUnused])

  function passesFilters(view: PeerTableRowView): boolean {
    return alarmFilter.has(alarmValue(view)) && matchesSearch(view, trimmedSearch)
  }

  const visibleDealtWith = useMemo(
    () => dealtWith.filter(passesFilters),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `passesFilters` closes over the same two state values already listed
    [dealtWith, alarmFilter, trimmedSearch]
  )
  const visibleAdvertised = useMemo(
    () => advertisedUnused.filter(passesFilters),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `passesFilters` closes over the same two state values already listed
    [advertisedUnused, alarmFilter, trimmedSearch]
  )
  const visibleRows = useMemo(() => [...visibleDealtWith, ...visibleAdvertised], [visibleDealtWith, visibleAdvertised])
  const visibleThroughSplit = throughSplit.filter((peer) => peer.nodeId.toLowerCase().includes(trimmedSearch))

  const alarmOptions: FilterValueOption[] = ALL_ALARM_VALUES.map((value) => ({
    value,
    count: allRows.filter((r) => alarmValue(r) === value).length
  }))
  const activeFilterGroups = alarmFilter.size < ALL_ALARM_VALUES.length ? 1 : 0
  // An empty group only says "match this filter" when a search or filter is
  // actually narrowing it; with nothing set, the group is simply empty.
  const narrowed = trimmedSearch !== '' || activeFilterGroups > 0

  const columnCount = 1 + visibleColumns.size

  function resetView() {
    setSearch('')
    setAlarmFilter(new Set(ALL_ALARM_VALUES))
    setVisibleColumns(new Set(ALL_COLUMNS))
  }

  const viewIsDefault =
    search === '' && alarmFilter.size === ALL_ALARM_VALUES.length && visibleColumns.size === ALL_COLUMNS.length

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-soft pb-2">
        <p className="type-caption font-mono text-fg-dim">
          {visibleRows.length === allRows.length ? visibleRows.length : `${visibleRows.length} of ${allRows.length}`}{' '}
          shown
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <SearchIcon
              aria-hidden="true"
              className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-fg-faint"
            />
            <Input
              aria-label="Search peers"
              className="ui-control h-8 w-52 rounded-[var(--radius)] border-border-soft pl-8 text-[length:var(--density-type-caption)]"
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Peer ID…"
              value={search}
            />
          </div>
          <FilterPopover<PeerFilterKey>
            activeFilterGroups={activeFilterGroups}
            categories={[{ key: 'alarm', label: PEER_NEEDS_A_LOOK_LABEL }]}
            contentLabel="Peer filters"
            formatOptionLabel={filterOptionLabel}
            id="ledger-peers-filters"
            itemLabel="peers"
            onClear={() => setAlarmFilter(new Set(ALL_ALARM_VALUES))}
            onSelectAll={() => setAlarmFilter(new Set(ALL_ALARM_VALUES))}
            onSelectNone={() => setAlarmFilter(new Set())}
            onValueChange={(_key, value, checked) => {
              setAlarmFilter((prev) => {
                const next = new Set(prev)
                if (checked) next.add(value)
                else next.delete(value)
                return next
              })
            }}
            optionsByCategory={{ alarm: alarmOptions }}
            selectedValuesByCategory={{ alarm: alarmFilter }}
            title="Peer filters"
            totalCount={allRows.length}
            triggerLabel="Filter peers"
            visibleCount={visibleRows.length}
          />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
                size="sm"
                variant="outline"
              >
                <Columns3 aria-hidden="true" className="size-3.5" />
                Columns
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Toggle columns</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {ALL_COLUMNS.map((column) => (
                <DropdownMenuCheckboxItem
                  checked={visibleColumns.has(column)}
                  key={column}
                  onCheckedChange={(checked) =>
                    setVisibleColumns((prev) => {
                      const next = new Set(prev)
                      if (checked) next.add(column)
                      else next.delete(column)
                      return next
                    })
                  }
                >
                  {COLUMN_LABELS[column]}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
            onClick={() => saveTextFile('mesh-peers-view.csv', peerRowsToCsv(visibleRows), 'text/csv')}
            size="sm"
            type="button"
            variant="outline"
          >
            Export view (CSV)
          </Button>
          <Button
            className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
            onClick={() =>
              saveTextFile(
                'mesh-peers-evidence.json',
                peerEvidenceBundle(dealtWithRawRows, advertisedUnusedRawPeers),
                'application/json'
              )
            }
            size="sm"
            type="button"
            variant="outline"
          >
            Save evidence file
          </Button>
          {/* Only once there is a view to reset: never a greyed control. */}
          {viewIsDefault ? null : (
            <Button
              className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
              onClick={resetView}
              size="sm"
              type="button"
              variant="outline"
            >
              <RotateCcw aria-hidden="true" className="size-3.5" />
              Reset view
            </Button>
          )}
        </div>
      </div>

      <Table aria-label="Peers">
        <TableHeader className="bg-panel-strong">
          <TableRow className="border-border-soft hover:bg-panel-strong">
            <TableHead className="type-label h-9 px-3 text-fg-faint">Peer / identity</TableHead>
            {ALL_COLUMNS.map((column) =>
              visibleColumns.has(column) ? (
                <TableHead className="type-label h-9 px-3 text-fg-faint" key={column}>
                  <span className="inline-flex items-center gap-1">
                    {/* The period column's header is the real date window across
                       the visible rows (e.g. `22–26 Sep`), never the word
                       "Period"; every other column keeps its plain label. */}
                    {COLUMN_LABELS[column]}
                    {/* Terse header on the face; what the column means and what
                       evidence backs it live behind the (i). */}
                    <InfoHover
                      census={`peer_column:${column}`}
                      describes={COLUMN_LABELS[column]}
                      label={PEER_COLUMN_INFO[column]}
                    />
                  </span>
                </TableHead>
              ) : null
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          <GroupHeaderRow columnCount={columnCount} count={visibleDealtWith.length} label="Nodes you have dealt with" />
          {visibleDealtWith.length === 0 ? (
            <TableRow className="border-border-soft">
              <TableCell className="h-16 text-center text-xs text-fg-dim" colSpan={columnCount}>
                {narrowed ? 'No exchanges match this filter.' : 'None yet.'}
              </TableCell>
            </TableRow>
          ) : (
            visibleDealtWith.map((view) => (
              <PeerTableRow
                exchangeSources={exchangeSourcesFor(view.displayId)}
                key={view.key}
                meshStatus={meshStatus.statusFor(view.displayId)}
                recordsById={recordsById}
                view={view}
                visibleColumns={visibleColumns}
              />
            ))
          )}
        </TableBody>
        <TableBody>
          <GroupHeaderRow
            columnCount={columnCount}
            count={visibleAdvertised.length}
            label="Nodes advertised but unused"
            tooltip={ADVERTISED_UNUSED_TOOLTIP}
          />
          {visibleAdvertised.length === 0 ? (
            <TableRow className="border-border-soft">
              <TableCell className="h-16 text-center text-xs text-fg-dim" colSpan={columnCount}>
                {narrowed ? 'No advertised peers match this filter.' : 'None yet.'}
              </TableCell>
            </TableRow>
          ) : (
            visibleAdvertised.map((view) => (
              <PeerTableRow
                key={view.key}
                meshStatus={meshStatus.statusFor(view.displayId)}
                view={view}
                visibleColumns={visibleColumns}
              />
            ))
          )}
        </TableBody>
        {visibleThroughSplit.length > 0 ? (
          <TableBody data-peer-group="through-split">
            <GroupHeaderRow
              columnCount={columnCount}
              count={visibleThroughSplit.length}
              label="Nodes that served you through a split"
            />
            {visibleThroughSplit.map((peer) => (
              <TableRow className="border-border-soft" key={peer.nodeId}>
                <TableCell className="font-mono text-xs text-foreground" colSpan={columnCount}>
                  <span className="inline-flex items-center gap-2">
                    {peer.nodeId}
                    <span className="text-fg-dim">
                      the coordinator says it ran part of {peer.splits} request{peer.splits === 1 ? '' : 's'} you sent it
                    </span>
                    <InfoHover census="peers:through_split" describes="this group" label={SPLIT_TOOLTIPS.throughSplit} />
                  </span>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        ) : null}
      </Table>
      {/* One legend line under the table: every peer name here is what that
         peer says about itself (UX §2), said once rather than on each row. */}
      <p className="mt-2 inline-flex items-center gap-1 text-fg-faint text-xs" data-testid="peers-legend">
        Peer names are {SELF_REPORTED_NOTE}.
        <InfoHover census="peer:self_reported" describes="the self-reported note" label={SELF_REPORTED_DETAIL} />
      </p>
    </div>
  )
}

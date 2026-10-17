// Ledger tab — four sections (Served / Peers / Exchanges /
// Integrity) replacing the prior four sub-tabs. Data comes from this plugin's
// own `panes/*` routes through the console host's plugin-scoped fetch
// (`api/sidecarClient.ts`) -- the user never configures anything; honest
// absent states where data is unavailable.
//
// Security boundary: NO user-visible strings may name internal tooling,
// internal item IDs, or any branded service name. Comments are exempt.
import { Fragment, type ReactElement, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeftRight, FolderOpen, Search as SearchIcon, ShieldCheck, Trash2, Users } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { EmptyState } from '@/components/ui/EmptyState'
import { FilterPopover, type FilterValueOption } from '@/components/ui/FilterPopover'
import { InfoBanner } from '@/components/ui/InfoBanner'
import { Input } from '@/components/ui/input'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { navigateHost } from '@/plugin-host/host'
import { TooltipProvider } from '@/components/ui/tooltip'
import { TabPanel } from '@/components/ui/TabPanel'
import { fetchCapsuleLedger } from '@/features/capsules/api/client'
import { askForRecord } from '@/features/capsules/api/evidenceRequestClient'
import { judgeAskReply, type AskOutcome, type AskTarget } from '@/features/capsules/lib/ask-for-record'
import type { CapsuleRecord, JsonRecord } from '@/features/capsules/api/types'
import {
  PaneFetchError,
  fetchPaneA,
  fetchPaneB,
  fetchPaneCList
} from '@/features/capsules/api/sidecarClient'
import { servedAskedCoverage } from '@/features/capsules/lib/exchanges-served-view'
import { LedgerPeersTable } from '@/features/capsules/components/LedgerPeersTable'
import { InfoHover } from '@/features/capsules/components/InfoHover'
import { ExchangeStreamRow } from '@/features/capsules/components/ExchangeStreamRow'
import { TwinBracket } from '@/features/capsules/components/TwinBracket'
import { peersThroughSplit } from '@/features/capsules/lib/split-stage'
import {
  buildExchangeCounterpartyIndex,
  buildExchangeLedgerRows,
  exchangeStateFilterValue,
  type ExchangeLedgerRow
} from '@/features/capsules/lib/exchange-ledger'
import { buildRailSegments, sortStreamByTime } from '@/features/capsules/lib/exchange-stream'
import {
  dayHeaderIndices,
  dayKeyForRow,
  dayTalliesByKey,
  dayTallyLine
} from '@/features/capsules/lib/exchange-day-groups'
import { EVIDENCE_SOURCE_LABEL, evidenceSource } from '@/features/capsules/lib/evidence-source'
import { exchangeTextNotice, heroStatusLine } from '@/features/capsules/lib/your-records'
import { useRecordsStatus } from '@/features/capsules/lib/use-your-records'
import { CleanUpRecordsDialog } from '@/features/capsules/components/CleanUpRecordsDialog'
import { YourRecordsDialog } from '@/features/capsules/components/YourRecordsDialog'
import { WitnessList } from '@/features/capsules/components/WitnessList'
import { formatExchangeTimestamp } from '@/features/capsules/lib/local-time'
import {
  exceptionsFirstLine,
  exceptionsFirstTally,
  exchangesHeadline
} from '@/features/capsules/lib/exceptions-first-line'
import {
  LEDGER_PAGE_SIZE,
  exchangeRowDomId,
  flattenPage,
  fullRangeLabel,
  groupStreamRows,
  indexGroupsWithinPage,
  isTwinBracketGroup,
  pageBoundaryContinuity,
  pageIndexForGroupKey,
  pageRowRange,
  paginateGroups,
  windowBannerHeadline
} from '@/features/capsules/lib/exchange-pages'
import { deriveRightCellState, LEDGER_STATE_FILTER_VALUES } from '@/features/capsules/lib/exchange-row-state'
import {
  exchangeEvidenceBundle,
  exchangeRowsToCsv,
  integrityEvidenceBundle,
  saveTextFile
} from '@/features/capsules/lib/exchange-export'
import {
  buildRegistrationCopy,
  buildSetupSteps,
  CAPTURE_BOUNDARY_FACT,
  CHAIN_BAR_INFO,
  chainStripCaption,
  checkpointCoverageByRecord,
  checkpointRegistration,
  coveredRecordCount,
  continuityFact,
  identityFact,
  ownerLinked,
  sealedBreakdownText,
  INTEGRITY_TILE_INFO,
  RETENTION_FACT,
  witnessKeyProblems,
  witnessRestartNote,
  witnessRows,
  witnessTileNote,
  type SetupStep
} from '@/features/capsules/lib/integrity-view'
import { HARNESS_PANE_A_PAYLOAD, HARNESS_PANE_C_PAYLOAD } from '@/features/capsules/lib/exchange-fixtures'
import {
  HARNESS_PANE_B_PAYLOAD,
  PEER_TAB_HARNESS_EXCHANGE_SOURCES,
  PEER_TAB_HARNESS_LEDGER_RECORDS,
  PEER_TAB_HARNESS_MESH_MODELS,
  PEER_TAB_HARNESS_MESH_PEERS
} from '@/features/capsules/lib/peer-fixtures'
import { livePeerExchangeSources, type PeerExchangeSource } from '@/features/capsules/lib/peer-exchange-timeline'
import { advertisedOnlyPeers, usePeerMeshStatusIndex } from '@/features/capsules/lib/peer-mesh-status'
import {
  advertisedOnlyRowView,
  dealtWithRowView,
  peerDisplayId,
  peerSortKey,
  sortPeerRows,
  unattributedExchangesLine
} from '@/features/capsules/lib/peer-row-view'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { useDataMode } from '@/lib/data-mode'
import { HoverChip } from '@/features/capsules/components/HoverChip'
import {
  HERO_DESCRIPTION_AFTER_LINK,
  HERO_DESCRIPTION_BEFORE_LINK,
  HERO_DESCRIPTION_LINK_TEXT,
  CONTINUITY_TOOLTIP,
  SAVE_EVIDENCE_FILE_TOOLTIP,
  SETUP_STEPS_LABEL,
  SETUP_STEPS_TOOLTIP,
  DOCS_URL,
  HERO_TOOLTIPS,
  EXCHANGE_TEXT_DELETE_NOW,
  CLOSE_CARD_COUNTS_TOOLTIP,
  CLOSE_CARD_TOOLTIP,
  SAMPLE_DATA_UNAVAILABLE,
  SETTLEMENT_REFUSED_NOTE
} from '@/features/capsules/lib/tooltip-copy'
import type { PaymentsPresence } from '@/features/capsules/api/sidecarTypes'
import { NOT_ADJUDICATED_LINES } from '@/features/capsules/lib/referee-row'
import {
  settlementCloseCounts,
  settlementCloseLine,
  type SettlementCloseCounts,
  unjoinedSettlementText,
  providerSettlementText
} from '@/features/capsules/lib/settlement-view'

// ---------------------------------------------------------------------------
// Error helper — honest fetch-failure messages, never "set the URL"
// ---------------------------------------------------------------------------

/** Returns a user-visible diagnostic string for a failed pane fetch. Never
 *  blames the user for missing configuration -- there is none to set. */
function describePaneError(error: unknown): string {
  if (error instanceof PaneFetchError && error.status === 503) {
    return "This plugin's evidence service isn't running."
  }
  if (error instanceof PaneFetchError && error.status === 404) {
    return "This plugin build doesn't serve its evidence panes yet."
  }
  return "Couldn't load accountability data from this host right now."
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

type LedgerTab = 'peers' | 'exchanges' | 'integrity'

// ---------------------------------------------------------------------------
// Exchanges served / asked strip strip (pane-a's card.served_summary) — sits above the
// Exchanges records table. Retired the standalone tab that held it;
// this is the quantity answer, records are the list below it.
// ---------------------------------------------------------------------------

/** p2 item 5: a control that can't act right now says why on hover (and to
 *  screen readers), never greys out silently. `reason === null` renders the
 *  control untouched. The wrapper takes the hover because a disabled button
 *  gets no pointer events. */
function DisabledReason({ reason, children }: { reason: string | null; children: ReactElement }) {
  if (reason === null) return children
  return (
    <HoverChip census="action:disabled_reason" label={reason}>
      <span className="inline-flex">{children}</span>
    </HoverChip>
  )
}

function ExchangesServedAskedHeader({ card }: { card: JsonRecord | null | undefined }) {
  const coverage = servedAskedCoverage(card)

  // No served-summary fold exists yet (`card` null, or a present card with
  // no witnessed range) -- render nothing rather than a "no data" line
  // sitting above real rows. The coverage statement is a later item, once
  // the fold is wired and `coverage.kind` can actually be `'verified'`.
  if (coverage.kind === 'absent') {
    return null
  }
  if (coverage.kind === 'failed') {
    return (
      <div className="rounded border border-border-soft bg-panel-strong/40 px-3 py-2 text-xs text-amber-500">
        {coverage.headline}
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-1 rounded border border-border-soft bg-panel-strong/40 px-3 py-2">
      <p className="text-sm font-medium text-foreground">Served: {coverage.servedText}</p>
      <p className="text-xs text-fg-dim">{coverage.askedText}</p>
      <p className="text-xs text-fg-faint">{coverage.statement}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Peers section (pane-b)
// ---------------------------------------------------------------------------

function PeersSection({ recordsById }: { recordsById: Map<string, CapsuleRecord> }) {
  const { mode } = useDataMode()
  const harnessMode = mode === 'harness'
  const query = useQuery({
    queryKey: ['ledger', 'pane-b', mode],
    queryFn: () => (harnessMode ? Promise.resolve(HARNESS_PANE_B_PAYLOAD) : fetchPaneB()),
    refetchInterval: 15_000,
    retry: false
  })
  // Shares its cache with ExchangesSection's own pane-c query (same
  // queryKey) -- the per-exchange timeline (Phase 2) joins this peer's
  // exchange_ids against the same list, never a second fetch.
  const paneCQuery = useQuery({
    queryKey: ['ledger', 'pane-c'],
    queryFn: () => fetchPaneCList(),
    refetchInterval: 15_000,
    retry: false,
    enabled: !harnessMode
  })
  const meshStatus = usePeerMeshStatusIndex(PEER_TAB_HARNESS_MESH_PEERS, PEER_TAB_HARNESS_MESH_MODELS)
  const effectiveRecordsById = harnessMode ? PEER_TAB_HARNESS_LEDGER_RECORDS : recordsById

  const resolveTimestamp = (capsuleId: string): string | null => effectiveRecordsById.get(capsuleId)?.timestamp ?? null

  if (query.isLoading) return <p className="text-sm text-muted-foreground">Loading…</p>
  if (query.isError) {
    return <p className="text-sm text-amber-500">{describePaneError(query.error)}</p>
  }

  // No card/row is ever rendered for a row with no counterparty identity
  // (design chooser-v1 §7 S1.1) -- those exchanges are rolled into the
  // headline count below instead of a synthetic "unknown peer" row.
  const allRawRows = query.data?.rows ?? []
  const dealtWithRawRows = allRawRows.filter((row) => peerDisplayId(row) !== null)
  // Counted from the SAME rows Exchanges lists (pane C), so the sentence
  // "They appear under Exchanges" is true by construction: never a pane B
  // residual that Exchanges does not show.
  const unattributedExchangeCount = (paneCQuery.data?.rows ?? []).filter((row) => !row.counterparty).length

  // Closest-first (latency asc); any row carrying an alarm floats to top.
  const sortedDealtWithRawRows = sortPeerRows(dealtWithRawRows, (row) =>
    peerSortKey(row, meshStatus.statusFor(peerDisplayId(row) ?? '')?.latencyMs ?? null)
  )
  // "Nodes advertised but unused" -- mesh-known
  // peers Pane B carries no row for at all (no exchange has ever
  // happened). A different group, never merged with the rows above.
  const advertisedPeers = advertisedOnlyPeers(dealtWithRawRows, meshStatus.peers)

  const sourcesByPeerId = new Map<string, readonly PeerExchangeSource[]>()
  for (const row of sortedDealtWithRawRows) {
    const peerId = peerDisplayId(row) ?? ''
    sourcesByPeerId.set(
      peerId,
      harnessMode
        ? (PEER_TAB_HARNESS_EXCHANGE_SOURCES[peerId] ?? [])
        : livePeerExchangeSources(row, paneCQuery.data?.rows ?? [])
    )
  }

  const dealtWithViews = sortedDealtWithRawRows.map((row) =>
    dealtWithRowView(row, resolveTimestamp, harnessMode ? [] : (paneCQuery.data?.rows ?? []))
  )
  const advertisedViews = advertisedPeers.map((peer) => advertisedOnlyRowView(peer.shortId ?? peer.id))

  // §3E -- the role-aware
  // headline needs the real advertised/dealt-with counts, which only this
  // section's data pipeline has (mesh status + pane-b rows); the call
  // site's own intro sentence stays static prose above it. `role !== 'you'`
  // excludes this node's own self entry, same exclusion `advertisedOnlyPeers`
  // already applies to the table's second row group.
  const advertisedCount = meshStatus.peers.filter((peer) => peer.role !== 'you').length
  const dealtWithCount = dealtWithViews.length

  if (dealtWithViews.length === 0 && advertisedViews.length === 0) {
    return (
      <EmptyState
        description={`${advertisedCount} peers advertised on the mesh so far · ${dealtWithCount} you have dealt with on the record. Once you ask another node for an answer, or serve one to a peer, this tab will show what they've shown you and whether it holds up.`}
        hint={
          <Button
            className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
            onClick={() => navigateHost('/chat')}
            size="sm"
            type="button"
            variant="outline"
          >
            Open Chat to start one
          </Button>
        }
        icon={<Users aria-hidden="true" className="size-10" strokeWidth={1.4} />}
        title="No peers recorded yet"
      />
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {/* L3.1/§3E — the role-aware headline, two counts, never a ratio. */}
      <p className="text-sm font-medium text-foreground">
        {advertisedCount} peer{advertisedCount === 1 ? '' : 's'} advertised · {dealtWithCount} you have dealt with on
        the record.
      </p>
      {unattributedExchangeCount > 0 ? (
        <p className="text-sm text-foreground">{unattributedExchangesLine(unattributedExchangeCount)}</p>
      ) : null}
      <LedgerPeersTable
        advertisedUnused={advertisedViews}
        advertisedUnusedRawPeers={advertisedPeers}
        dealtWith={dealtWithViews}
        dealtWithRawRows={sortedDealtWithRawRows}
        exchangeSourcesFor={(peerId) => sourcesByPeerId.get(peerId) ?? []}
        meshStatus={meshStatus}
        recordsById={effectiveRecordsById}
        throughSplit={peersThroughSplit((paneCQuery.data?.rows ?? []).flatMap((row) => (row.split ? [row.split] : [])))}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Exchanges section (pane-c)
// ---------------------------------------------------------------------------

type ExchangeFilterKey = 'role' | 'checks' | 'state'

const ALL_ROLE_VALUES = ['SERVED', 'ASKED']
const ALL_CHECKS_VALUES = ['clean', 'exception']
// v3 §2a: "useful filters are states, not qualities". `twins` isn't a
// right-cell state -- it's bracket membership, which no payload carries
// until B6 -- see `rowMatchesStateFilter` below.
// `local`: a served exchange with no other side (`isLocalOnly`), kept out of `open`.
const ALL_STATE_FILTER_VALUES = [...LEDGER_STATE_FILTER_VALUES, 'local', 'twins']

function exchangeFilterOptionLabel(value: string): string {
  switch (value) {
    case 'SERVED':
      return 'Served'
    case 'ASKED':
      return 'Asked'
    case 'clean':
      return 'Clean'
    case 'exception':
      return 'Exception'
    case 'closed':
      return 'Closed'
    case 'contradicted':
      return 'Contradicted'
    case 'asked_no_reply':
      return 'Asked, no reply'
    case 'open':
      return 'Open'
    case 'local':
      return 'This node only'
    case 'twins':
      return 'Twins only'
    default:
      return value
  }
}

/** `twins` is bracket membership, not a right-cell state -- orthogonal to
 *  (not a replacement for) the state categories, so it's OR'd in as an
 *  extra way for a row to match rather than folded into
 *  `exchangeStateFilterValue`'s single bucket. With every value selected by
 *  default (today's default), this changes nothing; deselecting every
 *  state except `twins` is what makes this "Twins only" -- exactly what
 *  wires up now that rows can actually carry a
 *  bracket id. */
function rowMatchesStateFilter(row: ExchangeLedgerRow, selected: ReadonlySet<string>): boolean {
  if (selected.has(exchangeStateFilterValue(row))) return true
  return row.twinBracketId !== null && selected.has('twins')
}

function ExchangesSection({
  recordsById,
  nodePubKeyPem,
  onGoToIntegrity,
  sampleData = false,
  requesterStartedDate,
  focusExchangeKey
}: {
  recordsById: Map<string, CapsuleRecord>
  nodePubKeyPem: string | null
  /** §3E -- "Get the other side’s record" and
   *  "Register a checkpoint" beside the headline both land on the Integrity
   *  tab's setup checklist, the one place either step actually exists today
   *  (`SetupChecklist` — neither has a wired end-to-end action yet, same
   *  honest-stub discipline as `handleAskForHalf` below). Deep-linking to
   *  the specific checklist row is left to the drill-paths work. */
  onGoToIntegrity: () => void
  /** A replayed fixture run (`evidenceSource` === 'sample'): actions that
   *  need a live node are disabled with the reason. */
  sampleData?: boolean
  requesterStartedDate?: string | null
  focusExchangeKey?: string
}) {
  const { mode } = useDataMode()
  const harnessMode = mode === 'harness'
  // Distinct queryKey from the top-level/Integrity pane-a query (no `mode`
  // suffix there) — this one's queryFn branches on harness mode, and a
  // shared key with a non-branching queryFn would race for cache ownership.
  const servedSummaryQuery = useQuery({
    queryKey: ['ledger', 'pane-a', 'served-summary', mode],
    queryFn: () => (harnessMode ? Promise.resolve(HARNESS_PANE_A_PAYLOAD) : fetchPaneA()),
    refetchInterval: 15_000,
    retry: false
  })
  // p2 item 3: the owner link from the node's own status -- the SAME
  // `ownerLinked` derivation Integrity's step 2 reads, so the checks panel's
  // binding fact and Integrity can never disagree. Disabled in harness mode,
  // same gating as IntegritySection.
  const ownerStatusQuery = useStatusQuery({ enabled: !harnessMode })
  const nodeOwnerLinked = ownerStatusQuery.data ? ownerLinked(ownerStatusQuery.data.owner ?? null) : null
  // Look finding 2: per-record checkpoint coverage from the SAME card figure
  // the Integrity chain strip shades, so a row's checks panel can never say
  // "no checkpoint covers this record" while Integrity says all are covered.
  const checkpointCoverage = useMemo(() => {
    const card = servedSummaryQuery.data?.card ?? null
    const covered = typeof card?.covered_leaf_count === 'number' ? card.covered_leaf_count : null
    return checkpointCoverageByRecord(
      (servedSummaryQuery.data?.rows ?? []).map((r) => r.capsule_id),
      covered
    )
  }, [servedSummaryQuery.data])
  // Same queryKey as PeersSection's own pane-c query (no `mode` suffix) --
  // that one calls the identical `fetchPaneCList()` in live mode and is
  // simply `enabled: false` in harness mode, so both share ONE cache entry
  // / one real fetch, same discipline PeersSection's own comment documents.
  const query = useQuery({
    queryKey: ['ledger', 'pane-c'],
    queryFn: () => (harnessMode ? Promise.resolve(HARNESS_PANE_C_PAYLOAD) : fetchPaneCList()),
    refetchInterval: 15_000,
    retry: false
  })
  // Same queryKey + queryFn SHAPE as PeersSection's own pane-b query (both
  // branch on harness mode identically) -- a symmetric pair shares one
  // cache entry safely; an asymmetric pair is a race (it once hit the
  // pane-c key).
  const paneBQuery = useQuery({
    queryKey: ['ledger', 'pane-b', mode],
    queryFn: () => (harnessMode ? Promise.resolve(HARNESS_PANE_B_PAYLOAD) : fetchPaneB()),
    refetchInterval: 15_000,
    retry: false
  })

  const counterpartyIndex = useMemo(
    () => buildExchangeCounterpartyIndex(paneBQuery.data?.rows ?? []),
    [paneBQuery.data]
  )
  const allRows = useMemo(
    () => buildExchangeLedgerRows(query.data?.rows ?? [], counterpartyIndex, recordsById),
    [query.data, counterpartyIndex, recordsById]
  )

  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<Set<string>>(new Set(ALL_ROLE_VALUES))
  const [checksFilter, setChecksFilter] = useState<Set<string>>(new Set(ALL_CHECKS_VALUES))
  const [stateFilter, setStateFilter] = useState<Set<string>>(new Set(ALL_STATE_FILTER_VALUES))
  // the row's action cell must never open a detail surface of its own. The
  // remaining row actions (view a statement, ask them to state their
  // content) have no carrier yet, so they stay a no-op -- honest absence,
  // never a fabricated result.
  const handleAskForHalf = useCallback((_row: ExchangeLedgerRow) => {}, [])

  // "Ask them for their record": one evidence request to the other side over
  // the mesh (this plugin's `mesh_evidence_request` tool), judged in
  // `ask-for-record.ts`. The row shows "Asked …" at once, then whatever
  // their reply proves. A saved sample can't ask anyone.
  const [askOutcomes, setAskOutcomes] = useState<ReadonlyMap<string, AskOutcome>>(() => new Map())
  const handleAskForRecord = useCallback((row: ExchangeLedgerRow, target: AskTarget) => {
    const askedAt = new Date().toISOString()
    const record = row.raw.mine.record as { effect?: { request_digest?: unknown } } | undefined
    const requestDigest = typeof record?.effect?.request_digest === 'string' ? record.effect.request_digest : null
    const settle = (outcome: AskOutcome) =>
      setAskOutcomes((previous) => new Map(previous).set(row.exchangeKey, outcome))
    settle({ kind: 'asking', at: askedAt })
    void askForRecord(target.peerId, target)
      .then((reply) => judgeAskReply(reply, requestDigest, askedAt))
      .then(settle)
  }, [])

  // windowed paging (v3 §2a) state. `pageIndex`
  // is the source of truth; render/handlers read `safePageIndex` so a
  // filter change that shrinks the page count never indexes past the end
  // for one render before the reset effect below catches up.
  const [pageIndex, setPageIndex] = useState(0)
  const [focusedRowIndex, setFocusedRowIndex] = useState(0)
  const [highlightedKey, setHighlightedKey] = useState<string | null>(null)
  const [checksExpandedKey, setChecksExpandedKey] = useState<string | null>(null)
  const [contentExpandedKey, setContentExpandedKey] = useState<string | null>(null)
  // the row's own `▸/▾ content` / `▸/▾
  // checks` toggles (the `o`/`c` keyboard shortcuts below drive the same
  // state for the keyboard-focused row).
  const handleToggleContent = useCallback(
    (row: ExchangeLedgerRow) => setContentExpandedKey((prev) => (prev === row.exchangeKey ? null : row.exchangeKey)),
    []
  )
  const handleToggleChecks = useCallback(
    (row: ExchangeLedgerRow) => setChecksExpandedKey((prev) => (prev === row.exchangeKey ? null : row.exchangeKey)),
    []
  )
  const searchInputRef = useRef<HTMLInputElement>(null)
  const handledFocusKeyRef = useRef<string | null>(null)

  const trimmedSearch = search.trim().toLowerCase()
  const visibleRows = useMemo(
    () =>
      allRows.filter((row) => {
        if (!roleFilter.has(row.roleTag)) return false
        if (!checksFilter.has(row.hasIssue ? 'exception' : 'clean')) return false
        if (!rowMatchesStateFilter(row, stateFilter)) return false
        if (!trimmedSearch) return true
        return `${row.exchangeKey} ${row.counterparty ?? ''}`.toLowerCase().includes(trimmedSearch)
      }),
    [allRows, roleFilter, checksFilter, stateFilter, trimmedSearch]
  )

  // L-N — one time-ordered append-only stream; nothing but time (and the
  // exchangeKey tie-break) reorders it. Filtering above may drop rows, but
  // never reorders the ones that remain.
  const streamRows = useMemo(() => sortStreamByTime(visibleRows), [visibleRows])
  // Built over the FULL stream (not a page) so cross-page continuity
  // (v3 §2a "a rail crossing a boundary shows '…session continues' on both
  // pages") can be detected -- see `pageBoundaryContinuity`.
  const railSegments = useMemo(() => buildRailSegments(streamRows), [streamRows])
  // Design §3A sticky day headers -- tallies always over the full filtered
  // stream (never a page fragment), so a header states the whole day's
  // count even when that day's rows straddle a page boundary.
  const dayTallies = useMemo(() => dayTalliesByKey(streamRows), [streamRows])

  // v3 §2a: "twin brackets never split" -- adjacent rows sharing a real
  // twin-bracket id become one atomic group;
  // every other row is still its own atomic group (see `groupStreamRows`),
  // and the packer below never splits a group across a page.
  const groups = useMemo(() => groupStreamRows(streamRows), [streamRows])
  const pages = useMemo(() => paginateGroups(groups, LEDGER_PAGE_SIZE), [groups])
  const safePageIndex = Math.min(pageIndex, Math.max(pages.length - 1, 0))
  const currentPage = useMemo(() => pages[safePageIndex] ?? [], [pages, safePageIndex])
  const currentPageRows = useMemo(() => flattenPage(currentPage), [currentPage])
  const indexedGroups = useMemo(() => indexGroupsWithinPage(currentPage), [currentPage])
  // Boundary indices computed over THIS page's rows -- a day that started on
  // the previous page still gets a header here (the boundary is "differs
  // from the row before it", and index 0 always differs from "no previous
  // row").
  const dayHeaderAt = useMemo(() => dayHeaderIndices(currentPageRows), [currentPageRows])
  // No twin bracket on any row: no client marked a pair, so no pair is
  // ever adjudicated (nothing is guessed from timing). Said once, here.
  const twinsUnmarkedLine =
    query.data?.referee && !query.data.referee.twins_marked ? NOT_ADJUDICATED_LINES.host_does_not_mark_twins : null
  const { start: pageStart, end: pageEnd } = useMemo(() => pageRowRange(pages, safePageIndex), [pages, safePageIndex])
  const continuity = useMemo(
    () => pageBoundaryContinuity(railSegments, pageStart, pageEnd),
    [railSegments, pageStart, pageEnd]
  )
  const fullRangeText = fullRangeLabel(streamRows)
  const hasContradiction = streamRows.some((row) => row.rightCellState.kind === 'contradicted')
  // Exceptions-first line (below) describes the FULL set, not the current
  // filter/search view -- its own range must match that same full set,
  // never the narrower `streamRows` range above.
  const allRowsRangeText = fullRangeLabel(allRows)

  // Filters/search changing the result set (not the page itself) resets to
  // page 0 -- a stale page index into a re-shaped result set would show the
  // wrong rows.
  const filterSignature = `${trimmedSearch}|${[...roleFilter].sort().join(',')}|${[...checksFilter].sort().join(',')}|${[...stateFilter].sort().join(',')}`
  const isFirstFilterRender = useRef(true)
  useEffect(() => {
    if (isFirstFilterRender.current) {
      isFirstFilterRender.current = false
      return
    }
    setPageIndex(0)
    setFocusedRowIndex(0)
  }, [filterSignature])

  // v3 §2a per-row deep link: jump to the page containing `focusExchangeKey`,
  // highlight it, and expand its checks inline -- once per distinct key
  // (guarded by the ref) so a background refetch never silently re-expands a
  // panel the reader already collapsed.
  useEffect(() => {
    if (!focusExchangeKey || focusExchangeKey === handledFocusKeyRef.current) return
    const targetPage = pageIndexForGroupKey(pages, focusExchangeKey)
    if (targetPage === null) return
    handledFocusKeyRef.current = focusExchangeKey
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional deep-link (external URL) to state sync, guarded above to run once per key
    setPageIndex(targetPage)
    setHighlightedKey(focusExchangeKey)
    setChecksExpandedKey(focusExchangeKey)
    const rowIndexOnPage = flattenPage(pages[targetPage]).findIndex((row) => row.exchangeKey === focusExchangeKey)
    setFocusedRowIndex(rowIndexOnPage >= 0 ? rowIndexOnPage : 0)
  }, [focusExchangeKey, pages])

  // Keeps the keyboard cursor (j/k) and any deep-link/next-contradiction
  // jump visible without relying on ref plumbing into each row.
  useEffect(() => {
    const row = currentPageRows[focusedRowIndex]
    if (!row) return
    document.getElementById(exchangeRowDomId(row.exchangeKey))?.scrollIntoView({ block: 'nearest' })
  }, [focusedRowIndex, currentPageRows])

  // v3 §2a: "one control -- Next contradiction ▸ -- that moves to the next
  // CONTRADICTED row regardless of page." Searches forward from the
  // keyboard cursor's position in the full filtered stream, wrapping to the
  // first contradiction found if none sit later in the stream.
  const jumpToNextContradiction = useCallback(() => {
    const currentGlobalIndex = pageStart + focusedRowIndex
    const contradictions = streamRows
      .map((row, index) => ({ row, index }))
      .filter(({ row }) => row.rightCellState.kind === 'contradicted')
    if (contradictions.length === 0) return
    const next = contradictions.find(({ index }) => index > currentGlobalIndex) ?? contradictions[0]
    const targetPage = pageIndexForGroupKey(pages, next.row.exchangeKey)
    if (targetPage === null) return
    setPageIndex(targetPage)
    setHighlightedKey(next.row.exchangeKey)
    const rowIndexOnPage = flattenPage(pages[targetPage]).findIndex((row) => row.exchangeKey === next.row.exchangeKey)
    setFocusedRowIndex(rowIndexOnPage >= 0 ? rowIndexOnPage : 0)
  }, [streamRows, pages, pageStart, focusedRowIndex])

  // Keyboard map (v3 §2a): j/k row, o toggle content (v3
  // §3's real inline toggle -- one populated side, one
  // empty side, flipping with role), c toggle checks (v3
  // §4's real inline toggle), / search, n next contradiction,
  // Escape closes whichever expansion is open
  // (there is no modal left to close, so Escape's job is the two per-row
  // toggles). Ignored while typing in a text field so normal typing (e.g.
  // "close" in the search box) never fires a shortcut -- Escape is the one
  // exception, checked before that guard, since closing an expansion should
  // work even if the search box happens to hold focus.
  useEffect(() => {
    function isTypingTarget(target: EventTarget | null): boolean {
      return (
        target instanceof HTMLElement &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      )
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (event.key === '/') {
        if (isTypingTarget(event.target)) return
        event.preventDefault()
        searchInputRef.current?.focus()
        return
      }
      if (event.key === 'Escape') {
        if (checksExpandedKey === null && contentExpandedKey === null) return
        event.preventDefault()
        setChecksExpandedKey(null)
        setContentExpandedKey(null)
        return
      }
      if (isTypingTarget(event.target) || currentPageRows.length === 0) return
      switch (event.key) {
        case 'j':
          event.preventDefault()
          setFocusedRowIndex((i) => Math.min(i + 1, currentPageRows.length - 1))
          break
        case 'k':
          event.preventDefault()
          setFocusedRowIndex((i) => Math.max(i - 1, 0))
          break
        case 'o': {
          event.preventDefault()
          const row = currentPageRows[focusedRowIndex]
          if (row) setContentExpandedKey((prev) => (prev === row.exchangeKey ? null : row.exchangeKey))
          break
        }
        case 'c': {
          event.preventDefault()
          const row = currentPageRows[focusedRowIndex]
          if (row) setChecksExpandedKey((prev) => (prev === row.exchangeKey ? null : row.exchangeKey))
          break
        }
        case 'n':
          event.preventDefault()
          jumpToNextContradiction()
          break
        default:
          break
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [currentPageRows, focusedRowIndex, jumpToNextContradiction, checksExpandedKey, contentExpandedKey])

  const servedAskedHeader = <ExchangesServedAskedHeader card={servedSummaryQuery.data?.card ?? null} />

  if (query.isLoading) {
    return (
      <div className="flex flex-col gap-2">
        {servedAskedHeader}
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    )
  }
  if (query.isError) {
    return (
      <div className="flex flex-col gap-2">
        {servedAskedHeader}
        <p className="text-sm text-amber-500">{describePaneError(query.error)}</p>
      </div>
    )
  }

  // L3.7 / §3F — Requester empty state, as an invitation rather than a
  // table header over nothing.
  if (!query.data || query.data.row_count === 0) {
    return (
      <div className="flex flex-col gap-2">
        {servedAskedHeader}
        <EmptyState
          description="Once you ask another node for an answer, or serve one to a peer, each exchange appears here as a two-sided record — your sealed record and theirs, as they send it."
          hint={
            <div className="flex flex-col items-center gap-2">
              <Button
                className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
                onClick={() => navigateHost('/chat')}
                size="sm"
                type="button"
                variant="outline"
              >
                Open Chat to start one
              </Button>
              {requesterStartedDate ? <span>Your node started keeping its half on {requesterStartedDate}.</span> : null}
            </div>
          }
          icon={<ArrowLeftRight aria-hidden="true" className="size-10" strokeWidth={1.4} />}
          title="No exchanges yet"
        />
      </div>
    )
  }

  // L3.8 — Two counts — never a ratio. Same predicate as the Integrity
  // section's own CLOSED-BY-OTHER-SIDE count (`deriveRightCellState(row).
  // kind === 'closed'`), not the old `theirs.state !== 'absent' &&
  // !r.unilateral` (`unilateral` is always `true` off this native route
  // today, so that read was always 0 by coincidence, not by evidence).
  const total = query.data.row_count
  const confirmed = query.data.rows.filter((r) => deriveRightCellState(r).kind === 'closed').length
  const disagreements = query.data.rows.filter((r) => deriveRightCellState(r).kind === 'contradicted').length
  const exceptionsTally = exceptionsFirstTally(allRows)

  const roleOptions: FilterValueOption[] = ALL_ROLE_VALUES.map((value) => ({
    value,
    count: allRows.filter((r) => r.roleTag === value).length
  }))
  const checksOptions: FilterValueOption[] = ALL_CHECKS_VALUES.map((value) => ({
    value,
    count: allRows.filter((r) => (r.hasIssue ? 'exception' : 'clean') === value).length
  }))
  const stateOptions: FilterValueOption[] = ALL_STATE_FILTER_VALUES.map((value) => ({
    value,
    // `twins` always counts 0 today -- honest, not a bug (see
    // `rowMatchesStateFilter`).
    count: value === 'twins' ? 0 : allRows.filter((r) => exchangeStateFilterValue(r) === value).length
  }))
  const activeFilterGroups =
    (roleFilter.size < ALL_ROLE_VALUES.length ? 1 : 0) +
    (checksFilter.size < ALL_CHECKS_VALUES.length ? 1 : 0) +
    (stateFilter.size < ALL_STATE_FILTER_VALUES.length ? 1 : 0)

  return (
    <div className="flex flex-col gap-2">
      {servedAskedHeader}
      {/* L3.1 — Exchanges section headline */}
      <p className="text-sm text-fg-dim">
        Each exchange is a pair of sealed records — yours and theirs. Their record normally arrives when the exchange
        finishes. If it doesn’t, you can ask for it.
      </p>
      {/* UX §3 — ONE headline: count, confirmed, disagreements. Registration
         is an Integrity fact and lives there (and in the hero line). */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-medium text-foreground" data-testid="exchanges-headline">
          {exchangesHeadline(total, confirmed, disagreements, allRowsRangeText)}
        </p>
        <div className="flex items-center gap-2">
          {/* p2 item 5: both land on Integrity's setup steps, which a saved
             sample can't act on -- disabled there, with the reason on hover,
             never a click that silently goes nowhere useful. */}
          <DisabledReason reason={sampleData ? SAMPLE_DATA_UNAVAILABLE : SETUP_STEPS_TOOLTIP}>
            <Button
              className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
              disabled={sampleData}
              onClick={onGoToIntegrity}
              size="sm"
              type="button"
              variant="outline"
            >
              {SETUP_STEPS_LABEL}
            </Button>
          </DisabledReason>
        </div>
      </div>
      {/* Exceptions-first line — only when something needs attention; the
         all-clear case would just repeat the headline's count. */}
      {exceptionsTally.needingAttention > 0 ? (
        <p className="text-sm text-fg-dim">{exceptionsFirstLine(exceptionsTally, allRowsRangeText)}</p>
      ) : null}
      {twinsUnmarkedLine ? (
        <p className="type-caption text-fg-faint" data-testid="twins-unmarked">
          Side-by-side check: {twinsUnmarkedLine}
        </p>
      ) : null}

      {/* D4(c): no "N shown" counter
         here -- the window banner below already states "Showing X of Y
         exchanges"; the same counter twice is noise, not information. The
         filter popover still reports visible/total while filtering. */}
      <div className="flex flex-wrap items-center justify-end gap-2 border-b border-border-soft pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <SearchIcon
              aria-hidden="true"
              className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-fg-faint"
            />
            <Input
              aria-label="Search exchanges"
              className="ui-control h-8 w-52 rounded-[var(--radius)] border-border-soft pl-8 text-[length:var(--density-type-caption)]"
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Exchange ID or counterparty…"
              ref={searchInputRef}
              value={search}
            />
          </div>
          <FilterPopover<ExchangeFilterKey>
            activeFilterGroups={activeFilterGroups}
            categories={[
              { key: 'role', label: 'Your role' },
              { key: 'checks', label: 'Checks' },
              { key: 'state', label: 'State' }
            ]}
            contentLabel="Exchange filters"
            formatOptionLabel={exchangeFilterOptionLabel}
            id="exchange-ledger-filters"
            itemLabel="exchanges"
            onClear={() => {
              setRoleFilter(new Set(ALL_ROLE_VALUES))
              setChecksFilter(new Set(ALL_CHECKS_VALUES))
              setStateFilter(new Set(ALL_STATE_FILTER_VALUES))
            }}
            onSelectAll={(key) => {
              if (key === 'role') setRoleFilter(new Set(ALL_ROLE_VALUES))
              else if (key === 'checks') setChecksFilter(new Set(ALL_CHECKS_VALUES))
              else setStateFilter(new Set(ALL_STATE_FILTER_VALUES))
            }}
            onSelectNone={(key) => {
              if (key === 'role') setRoleFilter(new Set())
              else if (key === 'checks') setChecksFilter(new Set())
              else setStateFilter(new Set())
            }}
            onValueChange={(key, value, checked) => {
              const setFilter = key === 'role' ? setRoleFilter : key === 'checks' ? setChecksFilter : setStateFilter
              setFilter((prev) => {
                const next = new Set(prev)
                if (checked) next.add(value)
                else next.delete(value)
                return next
              })
            }}
            optionsByCategory={{ role: roleOptions, checks: checksOptions, state: stateOptions }}
            selectedValuesByCategory={{ role: roleFilter, checks: checksFilter, state: stateFilter }}
            title="Exchange filters"
            totalCount={allRows.length}
            triggerLabel="Filter exchanges"
            visibleCount={visibleRows.length}
          />
          {/* Only when there is one to jump to: a greyed control that can
             never act is noise (p2 item 1). */}
          {hasContradiction ? (
            <Button
              className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
              onClick={jumpToNextContradiction}
              size="sm"
              type="button"
              variant="outline"
            >
              Next contradiction ▸
            </Button>
          ) : null}
          {/* Two distinct actions, never collapsed: a CSV of the current
             view vs. the portable evidence bundle (full records). */}
          <Button
            className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
            onClick={() => saveTextFile('mesh-exchanges-view.csv', exchangeRowsToCsv(visibleRows), 'text/csv')}
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
                'mesh-exchanges-evidence.json',
                exchangeEvidenceBundle(visibleRows.map((visibleRow) => visibleRow.raw)),
                'application/json'
              )
            }
            size="sm"
            type="button"
            variant="outline"
          >
            Save evidence file
          </Button>
        </div>
      </div>

      {/* v3 §2 — one time-ordered, append-only stream (L-N), replacing the
         flat table + `Confirmed` chip column with a real two-sided row per
         exchange. Toggle ① content (§3) is
         the `o` keyboard toggle below; a row click (or its in-cell action)
         still opens the existing full-detail inspector modal, a separate
         concern. */}
      {streamRows.length === 0 ? (
        <p aria-label="Exchange records" className="py-6 text-center text-sm text-fg-dim" role="status">
          No exchanges match this filter.
        </p>
      ) : (
        <>
          {/* v3 §2a — bounded-window banner (L-K): the full-range count and
             span are computed over `streamRows` (the full filtered set),
             never just the current page. */}
          {/* UX §3: hidden when every row is already on screen -- a banner
             over five rows saying "5 of 5" is chrome, not information. */}
          {pages.length > 1 ? (
            <div className="flex flex-col gap-1 rounded border border-border-soft bg-panel-strong/40 px-3 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="type-caption font-mono text-fg-dim">
                  {windowBannerHeadline(currentPageRows.length, streamRows.length, fullRangeText)}
                </p>
                <div className="flex items-center gap-1">
                  <Button
                    aria-label="Older exchanges"
                    className="ui-control h-7 gap-1 rounded-[var(--radius)] px-2 text-[length:var(--density-type-caption)]"
                    disabled={safePageIndex >= pages.length - 1}
                    onClick={() => setPageIndex((prev) => Math.min(prev + 1, pages.length - 1))}
                    size="sm"
                    type="button"
                    variant="outline"
                  >
                    ← Older
                  </Button>
                  <Button
                    aria-label="Newer exchanges"
                    className="ui-control h-7 gap-1 rounded-[var(--radius)] px-2 text-[length:var(--density-type-caption)]"
                    disabled={safePageIndex <= 0}
                    onClick={() => setPageIndex((prev) => Math.max(prev - 1, 0))}
                    size="sm"
                    type="button"
                    variant="outline"
                  >
                    Newer →
                  </Button>
                </div>
              </div>
              <p className="type-caption text-fg-faint">
                Not shown here: exchanges outside this range. Counts above are for the full range.
              </p>
            </div>
          ) : null}

          {/* Look finding 5: ONE page scroll, no nested scroll box. The
             enclosing Card clips with `overflow-clip` (not `overflow-hidden`),
             which keeps its rounded corners without becoming a scroll
             container, so this header pins against the page scroll itself. */}
          <div className="rounded border border-border-soft" data-testid="exchange-list">
            <div className="sticky top-0 z-10 grid grid-cols-2 gap-0 border-b border-border-soft bg-panel px-3 py-1.5">
              <p className="type-caption font-mono text-fg-faint">YOUR RECORD</p>
              <p className="type-caption font-mono text-fg-faint">THEIR RECORD, AS GIVEN TO YOU</p>
            </div>

            <div aria-label="Exchange records" className="flex flex-col divide-y divide-border-soft" role="list">
              {continuity.continuesFromPreviousPage ? (
                <p className="pl-3 pt-2 type-caption font-mono text-fg-faint">…session continues</p>
              ) : null}
              {indexedGroups.map(({ group, startIndex }) => {
                const rowElements = group.rows.map((row, offset) => {
                  const index = startIndex + offset
                  // Design §3A sticky day headers -- rendered right before
                  // the first row of each calendar day this page shows.
                  const dayTally = dayHeaderAt.has(index) ? dayTallies.get(dayKeyForRow(row)) : undefined
                  return (
                    <Fragment key={row.exchangeKey}>
                      {dayTally ? (
                        <div
                          className="sticky top-8 z-[5] border-b border-border-soft bg-panel px-3 py-1 type-caption font-mono text-fg-dim"
                          data-day-header={dayKeyForRow(row)}
                        >
                          {dayTallyLine(dayTally)}
                        </div>
                      ) : null}
                      <ExchangeStreamRow
                        checksExpanded={checksExpandedKey === row.exchangeKey}
                        contentExpanded={contentExpandedKey === row.exchangeKey}
                        focused={index === focusedRowIndex}
                        highlighted={highlightedKey === row.exchangeKey}
                        localRecord={
                          row.raw.mine.capsule_id ? (recordsById.get(row.raw.mine.capsule_id) ?? null) : null
                        }
                        checkpointCovered={
                          row.raw.mine.capsule_id ? (checkpointCoverage.get(row.raw.mine.capsule_id) ?? null) : null
                        }
                        nodePubKeyPem={nodePubKeyPem}
                        onAction={handleAskForHalf}
                        askOutcome={askOutcomes.get(row.exchangeKey) ?? null}
                        onAskForRecord={harnessMode || sampleData ? undefined : handleAskForRecord}
                        ownerLinked={nodeOwnerLinked}
                        onToggleChecks={handleToggleChecks}
                        onToggleContent={handleToggleContent}
                        rail={railSegments[pageStart + index]}
                        row={row}
                      />
                    </Fragment>
                  )
                })
                if (isTwinBracketGroup(group)) {
                  // group.rows[0].twinBracketId is guaranteed non-null here
                  // -- that's exactly what made `groupStreamRows` bracket
                  // these rows together in the first place.
                  const bracketId = group.rows[0].twinBracketId as string
                  return (
                    <TwinBracket
                      bracketId={bracketId}
                      key={group.groupKey}
                      rows={group.rows}
                    >
                      {rowElements}
                    </TwinBracket>
                  )
                }
                return rowElements
              })}
              {continuity.continuesToNextPage ? (
                <p className="pl-3 pb-2 type-caption font-mono text-fg-faint">…session continues</p>
              ) : null}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Integrity section (pane-a card field + pane-c first-person exchange
// outcomes), drawn as a chain strip. Replaces the old
// two-sentence paragraph (which could say "intact and registered with 0
// witnesses" AND "no integrity fields available" about the same node) with
// a chain strip + four first-person stat cards. Per v2 §6/R-D these counts
// describe THIS node's own chain, never a peer, so the full Logs stat-card
// prominence treatment is allowed here.
// ---------------------------------------------------------------------------

/** The chain strip: a bar of this node's sealed entries, shading the
 *  checkpoint-covered range when a checkpoint exists. */
function ChainStrip({
  sealedCount,
  checkpointCount,
  coveredLeafCount
}: {
  sealedCount: number
  checkpointCount: number | null
  coveredLeafCount: number | null
}) {
  const hasCheckpoint = checkpointCount !== null && checkpointCount > 0
  // Shade the covered-LEAF range, not the checkpoint-LINE count. Fall back to
  // no shading when the host did not report a covered-leaf count -- never shade
  // a wrong width off the line count.
  const coveredCount = hasCheckpoint && coveredLeafCount !== null ? Math.min(coveredLeafCount, sealedCount) : 0
  const coveredPct = sealedCount > 0 ? (coveredCount / sealedCount) * 100 : 0

  return (
    <div className="flex flex-col gap-1">
      <p className="type-caption inline-flex items-center gap-1 font-mono text-fg-faint">
        Your chain
        {/* The full checkpoint-coverage explanation moves behind the (i); the
           caption below stays terse. */}
        <InfoHover census="integrity:chain_strip" describes="the chain coverage bar" label={CHAIN_BAR_INFO} />
      </p>
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs text-fg-faint">1</span>
        <div
          className="relative h-4 flex-1 overflow-hidden rounded bg-panel-strong/40"
          role="img"
          aria-label={`Sealed chain, ${sealedCount} entries`}
        >
          {sealedCount > 0 ? (
            <>
              <div className="absolute inset-y-0 left-0 bg-foreground/70" style={{ width: `${coveredPct}%` }} />
              <div className="absolute inset-y-0 bg-foreground/25" style={{ left: `${coveredPct}%`, right: 0 }} />
            </>
          ) : null}
        </div>
        <span className="font-mono text-xs text-fg-faint">{sealedCount}</span>
      </div>
      {/* Three-state absence handling + leaf pluralization live in
         `chainStripCaption` (integrity-view.ts) so both are unit-tested. The
         leaf figure is the covered-leaf count, NOT the checkpoint-line count. */}
      <p className="type-caption text-fg-dim">{chainStripCaption(sealedCount, checkpointCount, coveredLeafCount)}</p>
    </div>
  )
}

type IntegrityStatCardProps = {
  readonly label: string
  readonly value: number
  readonly tone?: 'default' | 'bad'
  /** The honest explanation of what this tile counts and what evidence backs
   *  it -- moved off the tile face and behind the (i) glyph. */
  readonly info: string
  /** One plain line under the value -- what the number is made of, or why
   *  it is what it is (UX §4 "off — your choice"). */
  readonly sub?: string
}

/** One first-person stat card. A zero renders in the exact same size/weight
 *  as any other value -- never muted, never apologetic (Accept criterion). */
function IntegrityStatCard({ label, value, tone = 'default', info, sub }: IntegrityStatCardProps) {
  const valueColor = tone === 'bad' && value > 0 ? 'var(--color-bad)' : 'var(--color-foreground)'
  return (
    <div className="panel-shell min-w-0 rounded-[var(--radius-lg)] border border-border bg-panel px-[var(--panel-x)] py-[var(--panel-y)]">
      <span className="type-label inline-flex min-w-0 items-center gap-1 text-fg-faint">
        <span className="truncate">{label}</span>
        <InfoHover census={`integrity_tile:${label}`} describes={label} label={info} />
      </span>
      <div
        className="mt-[var(--panel-y,12px)] font-mono text-[length:var(--density-type-headline)] font-semibold leading-none tracking-tight"
        style={{ color: valueColor }}
      >
        {value}
      </div>
      {sub ? <p className="mt-1.5 type-caption text-fg-dim">{sub}</p> : null}
    </div>
  )
}

/** The setup checklist (ledger-ux-from-
 *  the-user §6) -- three steps in value order, replacing the incoherent
 *  "registered with 0 witnesses" line a new node used to show. Each step
 *  states what it buys and what it does not; once done, the explanatory
 *  sentence drops and only the status word remains. */
function SetupChecklist({ steps }: { steps: readonly SetupStep[] }) {
  return (
    <div className="flex flex-col gap-3 rounded border border-border-soft bg-panel-strong/40 p-3">
      {steps.map((step, index) => (
        <div className="flex flex-col gap-0.5" key={step.key}>
          <p className="type-caption font-mono text-fg-faint">
            {index + 1} · {step.title} — <span className="font-medium text-foreground">{step.status}</span>
          </p>
          {step.body ? <p className="type-caption text-fg-dim">{step.body}</p> : null}
        </div>
      ))}
    </div>
  )
}

/** The Close card: agreed periods need a Close record neither side has
 *  sealed yet, so it says "none yet". The counts so far over inference and
 *  payment sit under their own heading, never under "Agreed periods" --
 *  counts only, never an amount. */
function CloseCard({
  counts,
  payments,
  unjoined
}: {
  counts: SettlementCloseCounts
  payments: PaymentsPresence | undefined
  unjoined: string | null
}) {
  return (
    <div
      className="panel-shell flex flex-col gap-1 rounded-[var(--radius-lg)] border border-border bg-panel px-[var(--panel-x)] py-[var(--panel-y)]"
      data-testid="close-card"
    >
      <span className="type-label inline-flex items-center gap-1 text-fg-faint">
        <span>Agreed periods</span>
        <InfoHover census="integrity:close_card" describes="Agreed periods" label={CLOSE_CARD_TOOLTIP} />
      </span>
      <p className="text-sm text-foreground" data-close-agreed="true">
        none yet
      </p>
      <span className="type-label mt-2 inline-flex items-center gap-1 text-fg-faint" data-close-counts-heading="true">
        <span>Not in an agreed period yet</span>
        <InfoHover
          census="integrity:close_card_counts"
          describes="Not in an agreed period yet"
          label={CLOSE_CARD_COUNTS_TOOLTIP}
        />
      </span>
      <p className="type-caption text-fg-dim" data-close-inference="true">
        So far: {counts.exchanges} exchanges · {counts.closed} confirmed by the other side
      </p>
      <p className="type-caption text-fg-dim" data-close-settlement="true">
        {settlementCloseLine(counts, payments)}
      </p>
      {unjoined ? <p className="type-caption text-fg-faint">{unjoined}</p> : null}
      {payments === 'on' ? (
        <p className="type-caption text-fg-faint" data-close-refused-note="true">
          {SETTLEMENT_REFUSED_NOTE}
        </p>
      ) : null}
    </div>
  )
}

function IntegritySection() {
  const { mode } = useDataMode()
  const harnessMode = mode === 'harness'
  const query = useQuery({
    queryKey: ['ledger', 'pane-a'],
    queryFn: () => fetchPaneA(),
    refetchInterval: 15_000,
    retry: false
  })
  // Same queryKey + queryFn shape as ExchangesSection's own pane-c query --
  // needed here for the CLOSED BY THE OTHER SIDE / CONTRADICTED cards,
  // first-person counts derived from this node's own exchange records.
  const paneCQuery = useQuery({
    queryKey: ['ledger', 'pane-c'],
    queryFn: () => (harnessMode ? Promise.resolve(HARNESS_PANE_C_PAYLOAD) : fetchPaneCList()),
    refetchInterval: 15_000,
    retry: false
  })
  // Owner identity is real, live data -- already wired for the Network
  // dashboard (`NodeSidebar.tsx`) -- not part of the still-unwired pane-a
  // `card`. Disabled in harness mode, same gating as `usePeerMeshStatusIndex`.
  const statusQuery = useStatusQuery({ enabled: !harnessMode })

  if (query.isLoading) return <p className="text-sm text-muted-foreground">Loading…</p>
  if (query.isError) {
    return <p className="text-sm text-amber-500">{describePaneError(query.error)}</p>
  }

  const card = query.data?.card
  const rows = query.data?.rows ?? []
  const paneCRows = paneCQuery.data?.rows ?? []
  const owner = statusQuery.data?.owner ?? null

  // Extract from the card if present
  const checkpointCount = typeof card?.checkpoint_count === 'number' ? card.checkpoint_count : null
  // The covered-LEAF count (host-inverted from the checkpoint's MMR node count),
  // distinct from `checkpoint_count` (the number of checkpoint lines). The chain
  // strip caption/shading read THIS, so a 1-line checkpoint over 8 leaves reads
  // "8 leaves", never "1 leaves".
  const coveredLeafCount = typeof card?.covered_leaf_count === 'number' ? card.covered_leaf_count : null
  const continuity = typeof card?.continuity === 'string' ? card.continuity : null

  // L4.2 — owner-added-later headline
  const ownerAddedAt = typeof card?.owner_added_at === 'string' ? card.owner_added_at : null
  const ownerCardIndex = typeof card?.owner_card_index === 'number' ? card.owner_card_index : null

  const sealedCount = rows.length
  // Finding 3: Integrity counts every record this node sealed; Exchanges
  // counts exchanges. The difference is the records that note a record
  // received from the other side -- said on the tile so 8 and 5 reconcile.
  const receivedNoteCount = rows.filter((row) => row.kind === 'counterparty_half_citation').length
  // Payment records are this node's own log entries too, but not exchanges.
  const paymentRecordCount = rows.filter((row) => row.kind === 'settlement_observation').length
  // §7.5: a block or unblock is sealed too, but it is not an exchange.
  const routingChoiceCount = rows.filter((row) => row.kind === 'local_routing_choice').length
  const ownExchangeCount = sealedCount - receivedNoteCount - paymentRecordCount - routingChoiceCount
  // Same predicate the Exchanges stream badge uses (`deriveRightCellState`,
  // called with no live fetch state here -- Integrity has no per-row peer
  // fetch to draw on) so the two sections can never again show
  // contradictory counts (honesty
  // finding 1/8: the old `theirs.state !== 'absent' && !row.unilateral`
  // read every row's `unilateral` flag, which `capsule_panes_native.rs`
  // currently always sets `true` -- coincidentally always 0 here while
  // Exchanges asserted CLOSED on the same rows from an unrelated bug).
  const closedByOtherSideCount = paneCRows.filter((row) => deriveRightCellState(row).kind === 'closed').length
  const contradictedCount = paneCRows.filter((row) => deriveRightCellState(row).kind === 'contradicted').length
  // The Close card counts over the same rows with the same gate.
  const closeCounts = settlementCloseCounts(paneCRows, (row) => deriveRightCellState(row).kind === 'closed')
  const unjoinedPayments =
    [
      unjoinedSettlementText(
        paneCQuery.data?.settlement_unjoined,
        paneCQuery.data?.settlement_missing_exchange_id ?? 0
      ),
      providerSettlementText(paneCQuery.data?.settlement_provider_records)
    ]
      .filter(Boolean)
      .join(' ') || null

  const setupSteps = buildSetupSteps(card ?? null, owner, closedByOtherSideCount)
  const registrationCopy = buildRegistrationCopy(card ?? null)
  const registration = checkpointRegistration(card ?? null)

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-2">
        <div>
          <CardTitle className="text-sm">Chain integrity</CardTitle>
          {/* L4.2 — owner-added-later notice */}
          {ownerAddedAt && ownerCardIndex !== null ? (
            <p className="text-xs text-fg-dim mt-1">
              Owner bound from {ownerAddedAt} (card #{ownerCardIndex}); earlier records are unowned.
            </p>
          ) : null}
        </div>
        {/* ledger-ux-from-the-user §7 — the one hand-over affordance, on
           every range, same as the Exchanges table's own evidence export. */}
        <HoverChip census="integrity:save_evidence_file" label={SAVE_EVIDENCE_FILE_TOOLTIP}>
          <Button
            className="ui-control h-8 shrink-0 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
            onClick={() =>
              saveTextFile(
                'mesh-integrity-evidence.json',
                integrityEvidenceBundle(rows, card ?? null),
                'application/json'
              )
            }
            size="sm"
            type="button"
            variant="outline"
          >
            Save evidence file
          </Button>
        </HoverChip>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 pt-0 text-sm text-fg-dim">
        <ChainStrip checkpointCount={checkpointCount} coveredLeafCount={coveredLeafCount} sealedCount={sealedCount} />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <IntegrityStatCard
            info={INTEGRITY_TILE_INFO.sealed}
            label="Sealed"
            sub={sealedBreakdownText(ownExchangeCount, receivedNoteCount, routingChoiceCount, paymentRecordCount)}
            value={sealedCount}
          />
          <IntegrityStatCard
            info={INTEGRITY_TILE_INFO.sharedWithWitness}
            label="Shared with a witness"
            sub={witnessTileNote(registration.witnessState)}
            value={registration.witnessedCheckpointCount}
          />
          <IntegrityStatCard
            info={INTEGRITY_TILE_INFO.confirmedByOtherSide}
            label="Confirmed by the other side"
            value={closedByOtherSideCount}
          />
          <IntegrityStatCard
            info={INTEGRITY_TILE_INFO.contradicted}
            label="Disagreements"
            tone="bad"
            value={contradictedCount}
          />
        </div>

        <WitnessList
          note={witnessRestartNote(card ?? null)}
          problems={witnessKeyProblems(card ?? null)}
          rows={witnessRows(card ?? null)}
        />

        <CloseCard counts={closeCounts} payments={paneCQuery.data?.payments} unjoined={unjoinedPayments} />

        <SetupChecklist steps={setupSteps} />

        {/* Finding 7 -- each thing said once: an unwitnessed checkpoint is
           already stated by step 1's status and the witness tile, so the
           summary line renders only once a witness actually holds one. */}
        {registrationCopy ? (
          <div className="flex flex-col gap-0.5">
            {registration.registered ? <p>{registrationCopy.witnessSummary}</p> : null}
            {registrationCopy.registeredNoLaterThan ? (
              <p className="type-caption text-fg-faint">{registrationCopy.registeredNoLaterThan}</p>
            ) : null}
          </div>
        ) : null}

        <p className="inline-flex flex-wrap items-center gap-1">
          {continuity ? (
            <span>
              Continuity: <span className="font-medium text-foreground">{continuity}</span>
            </span>
          ) : (
            <span>{continuityFact(checkpointCount)}</span>
          )}
          <InfoHover census="integrity:continuity" describes="Continuity" label={CONTINUITY_TOOLTIP} />
        </p>

        {/* Once-per-node facts -- engineering detail, so behind a Details
           disclosure (UX §4); never repeated per exchange row. */}
        <details className="border-t border-border-soft pt-3" data-testid="integrity-details">
          <summary className="cursor-pointer select-none text-xs text-fg-dim">Details</summary>
          <div className="mt-2 flex flex-col gap-1">
            <p>{RETENTION_FACT}</p>
            <p>{CAPTURE_BOUNDARY_FACT}</p>
            <p>{identityFact(owner)}</p>
          </div>
        </details>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export function LedgerPageContent({ focusExchangeKey }: { focusExchangeKey?: string } = {}) {
  // `focusExchangeKey` (bridged from the
  // page URL's `?focusExchangeKey=` by `EvidencePage.tsx`, kept out
  // of this component so it stays router-free and testable without a
  // `RouterProvider`) auto-selects the Exchanges tab when a per-row deep
  // link is followed.
  const [activeTab, setActiveTab] = useState<LedgerTab>(focusExchangeKey ? 'exchanges' : 'peers')
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional prop-to-state sync for a followed deep link
    if (focusExchangeKey) setActiveTab('exchanges')
  }, [focusExchangeKey])

  // Fetch the local ledger once for all sections that need in-browser recompute
  const ledgerQuery = useQuery({
    queryKey: ['capsules', 'ledger'],
    queryFn: fetchCapsuleLedger,
    refetchInterval: 15_000
  })

  // Shares its cache with IntegritySection's own pane-a query (same
  // queryKey) -- used here only to drive the header's connectivity badge.
  const paneAStatusQuery = useQuery({
    queryKey: ['ledger', 'pane-a'],
    queryFn: () => fetchPaneA(),
    refetchInterval: 15_000,
    retry: false
  })

  const recordsById = useMemo(() => {
    const map = new Map<string, CapsuleRecord>()
    for (const record of ledgerQuery.data?.records ?? []) {
      if (record.capsule_id) map.set(record.capsule_id, record)
    }
    return map
  }, [ledgerQuery.data])

  const nodePubKeyPem = ledgerQuery.data?.nodePubKeyPem ?? null
  const sidecarConnected = paneAStatusQuery.isSuccess
  // Look finding 6: a replayed fixture run reads "Sample data", never "Live".
  const source = evidenceSource(sidecarConnected, import.meta.env.VITE_EVIDENCE_FIXTURES as string | undefined)

  // The hero line reads the SAME counts as the Integrity tiles (same queries,
  // shared cache; same `deriveRightCellState` predicate), so the hero and
  // Integrity can't disagree. Ported from the console's hero.
  const { mode } = useDataMode()
  const harnessMode = mode === 'harness'
  const paneCHeroQuery = useQuery({
    queryKey: ['ledger', 'pane-c'],
    queryFn: () => (harnessMode ? Promise.resolve(HARNESS_PANE_C_PAYLOAD) : fetchPaneCList()),
    refetchInterval: 15_000,
    retry: false
  })
  const heroLine =
    paneAStatusQuery.isSuccess && paneCHeroQuery.isSuccess
      ? heroStatusLine({
          records: paneAStatusQuery.data.rows.length,
          confirmed: paneCHeroQuery.data.rows.filter((row) => deriveRightCellState(row).kind === 'closed').length,
          disagreements: paneCHeroQuery.data.rows.filter((row) => deriveRightCellState(row).kind === 'contradicted')
            .length,
          witness: checkpointRegistration(paneAStatusQuery.data.card ?? null).witnessState
        })
      : null
  const recordsStatus = useRecordsStatus({ sample: source === 'sample' })
  // Said plainly, and for as long as it is true: this node keeps exchange text.
  const exchangeText = exchangeTextNotice(recordsStatus)
  const [cleanUpOpen, setCleanUpOpen] = useState(false)
  const [recordsOpen, setRecordsOpen] = useState(false)
  const paneACard = paneAStatusQuery.data?.card ?? null
  const paneARows = paneAStatusQuery.data?.rows ?? []

  return (
    <TooltipProvider delayDuration={250} skipDelayDuration={120}>
      <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-[calc(var(--shell-normal)*2)]">
        <InfoBanner
          description={
            <>
              {HERO_DESCRIPTION_BEFORE_LINK}
              <a
                className="underline underline-offset-2 hover:text-foreground"
                href={DOCS_URL}
                rel="noreferrer"
                target="_blank"
              >
                {HERO_DESCRIPTION_LINK_TEXT}
              </a>
              {HERO_DESCRIPTION_AFTER_LINK}
            </>
          }
          leadingIcon={<ShieldCheck aria-hidden="true" className="size-4" />}
          status={
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1">
                <StatusBadge
                  dot
                  size="caption"
                  tone={source === 'live' ? 'good' : source === 'sample' ? 'warn' : 'muted'}
                >
                  {EVIDENCE_SOURCE_LABEL[source]}
                </StatusBadge>
                <InfoHover
                  census={`hero:${source}`}
                  describes={`the ${EVIDENCE_SOURCE_LABEL[source]} chip`}
                  label={HERO_TOOLTIPS[source]}
                />
              </span>
            </div>
          }
          title="Evidence"
          titleId="evidence-title"
          titleLevel="h1"
          annotation={
            heroLine || exchangeText ? (
              <>
                {heroLine ? (
                  <p className="type-caption mt-1 text-fg-dim" data-testid="hero-status-line">
                    {heroLine}
                  </p>
                ) : null}
                {exchangeText ? (
                  <p className="type-caption mt-1 text-foreground" data-testid="hero-exchange-text-notice" role="note">
                    {exchangeText.text}
                    {exchangeText.offerDelete ? (
                      <>
                        {' '}
                        <button
                          className="underline underline-offset-2 hover:text-accent"
                          data-testid="hero-exchange-text-delete"
                          onClick={() => setCleanUpOpen(true)}
                          type="button"
                        >
                          {EXCHANGE_TEXT_DELETE_NOW}
                        </button>
                      </>
                    ) : null}
                  </p>
                ) : null}
              </>
            ) : null
          }
          action={
            <div className="flex flex-wrap items-center gap-2">
              <HoverChip census="hero:your_records" label={HERO_TOOLTIPS.yourRecords}>
                <Button
                  className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
                  data-testid="hero-your-records"
                  onClick={() => setRecordsOpen(true)}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <FolderOpen aria-hidden="true" className="size-3.5" />
                  Your records
                </Button>
              </HoverChip>
              <Button
                className="ui-control-destructive h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
                onClick={() => setCleanUpOpen(true)}
                size="sm"
                type="button"
                variant="outline"
              >
                <Trash2 aria-hidden="true" className="size-3.5" />
                Clean up records
              </Button>
            </div>
          }
        />
        <YourRecordsDialog
          checkpointNoLaterThan={
            typeof paneACard?.registered_no_later_than === 'string'
              ? formatExchangeTimestamp(paneACard.registered_no_later_than)
              : null
          }
          coveredRecords={coveredRecordCount(paneACard)}
          onExport={() =>
            saveTextFile('mesh-evidence.json', integrityEvidenceBundle(paneARows, paneACard), 'application/json')
          }
          onOpenChange={setRecordsOpen}
          open={recordsOpen}
          recordCount={paneARows.length}
          sample={source === 'sample'}
          status={recordsStatus}
        />
        <CleanUpRecordsDialog
          onOpenChange={setCleanUpOpen}
          open={cleanUpOpen}
          sample={source === 'sample'}
          status={recordsStatus}
        />

        <Card className="overflow-clip rounded-[var(--radius-lg)] border-border bg-panel p-4 shadow-none">
          <TabPanel<LedgerTab>
            ariaLabel="Evidence sections"
            onValueChange={setActiveTab}
            stretchTabs={false}
            contentClassName="px-0 pt-4"
            tabs={[
              {
                value: 'peers',
                label: 'Peers',
                content: (
                  <div>
                    {/* the second line is
                       the axis test itself: true only right now → Network;
                       true because sealed → Evidence. */}
                    <p className="text-sm text-fg-dim">
                      What have the nodes you've dealt with shown you — and does it hold up?
                    </p>
                    <p className="mb-3 text-sm text-fg-dim">
                      Accountability only. Liveness, latency and routing are in the{' '}
                      {/* A real link (so open-in-new-tab still works), but a
                         plain click goes through the console's own navigation
                         rather than reloading the whole console. */}
                      <a
                        className="underline underline-offset-2 hover:text-foreground"
                        href="/"
                        onClick={(event) => {
                          if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
                            return
                          event.preventDefault()
                          navigateHost('/')
                        }}
                      >
                        Network tab
                      </a>
                      .
                    </p>
                    <PeersSection recordsById={recordsById} />
                  </div>
                )
              },
              {
                value: 'exchanges',
                label: 'Exchanges',
                content: (
                  <ExchangesSection
                    focusExchangeKey={focusExchangeKey}
                    nodePubKeyPem={nodePubKeyPem}
                    onGoToIntegrity={() => setActiveTab('integrity')}
                    sampleData={source === 'sample'}
                    recordsById={recordsById}
                  />
                )
              },
              {
                value: 'integrity',
                label: 'Integrity',
                content: <IntegritySection />
              }
            ]}
            value={activeTab}
          />
        </Card>
      </div>
    </TooltipProvider>
  )
}

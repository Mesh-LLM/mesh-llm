// Toggle ② the security view (v3 §4) --
// renders INLINE under the row, keeping the two-sided columns, never a
// modal (v3 §4 "Where"). Exact block order: IDENTITY -> HEADER -> WHAT IT
// COMMITS TO -> CHECKS -> raw (raw bytes last).
import { CHECK_SOURCE_LEGEND, CHECK_SOURCE_WORDS } from '@/features/capsules/lib/tooltip-copy'
import { useState, type ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import { cn } from '@/lib/cn'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import { checkRowDomId } from '@/features/capsules/lib/exchange-pages'
import type { PeerRecomputeState, RecomputedIdentity } from '@/features/capsules/lib/recompute-identity'
import { toneForState } from '@/features/capsules/lib/assurance-tone'
import { WHAT_ACTUALLY_HAPPENED_GROUP, WHAT_NODE_SAID_GROUP } from '@/features/capsules/lib/nine-properties'
import {
  buildChecksRows,
  buildCommitsToRows,
  buildHeaderRows,
  buildIdentityRow,
  sharedTheirsDetail,
  theirsFetchable,
  type ChecksRow,
  type ChecksSideCell
} from '@/features/capsules/lib/security-checks-view'
import { ChipExplanationPopover } from '@/features/capsules/components/ChipExplanationPopover'
import { EVIDENCE_FILE_SENTENCE, exchangeEvidenceBundle, saveTextFile } from '@/features/capsules/lib/exchange-export'

function badgeToneFor(state: string): StatusBadgeTone {
  const tone = toneForState(state)
  return tone === 'neutral' ? 'muted' : (tone as StatusBadgeTone)
}

function BlockHeading({ children }: { children: string }) {
  return (
    <p className="type-caption font-mono uppercase tracking-wide text-fg-faint" data-block-heading={children}>
      {children}
    </p>
  )
}

function TwoCol({ yours, theirs }: { yours: ReactNode; theirs: ReactNode | null }) {
  return (
    <div className="grid grid-cols-2 gap-x-3">
      <div className="min-w-0">{yours}</div>
      <div className="min-w-0">{theirs}</div>
    </div>
  )
}

function ChecksCell({
  cell,
  propertyKey,
  factKey,
  showDetail = true
}: {
  cell: ChecksSideCell
  propertyKey: string
  factKey?: 'binding' | 'authority'
  /** False when this sentence is already the column note (said once). */
  showDetail?: boolean
}) {
  // L-M: recomputed-here and taken-from-the-source must never render
  // identically -- distinct class + a distinct data attribute so a test can
  // assert the two classes differ, not just eyeball it.
  return (
    <p
      // No coloured bar (it read as a selection cursor); the source
      // is said in words beside any result.
      className={cn('flex items-baseline gap-1.5 text-xs', cell.recomputed ? 'text-foreground' : 'text-fg-dim')}
      data-detail={cell.detail}
      data-recomputed={cell.recomputed ? 'true' : 'false'}
      data-source={cell.recomputed ? 'recomputed-in-browser' : 'from-sidecar'}
    >
      {/* v3 §4: every chip opens the four-part explanation. p2 item 2
         (supersedes L-L's "inline, never on hover"): the chip stands
         alone; its sentence moves word for word into the chip's tooltip. */}
      <ChipExplanationPopover
        cell={cell}
        detail={showDetail ? cell.detail : undefined}
        factKey={factKey}
        propertyKey={propertyKey}
      >
        <StatusBadge size="caption" tone={badgeToneFor(cell.state)}>
          {cell.label}
        </StatusBadge>
      </ChipExplanationPopover>
      {cell.state === 'PASS' || cell.state === 'FAIL' ? (
        <span className="type-caption text-fg-faint" data-check-source-words="true">
          · {cell.recomputed ? CHECK_SOURCE_WORDS.here : CHECK_SOURCE_WORDS.node}
        </span>
      ) : null}
    </p>
  )
}

/** The CHECKS column header: blank · YOURS · THEIRS, with the repeated THEIRS
 *  sentence under it once. */
function ChecksColumnHeader({ theirsNote }: { theirsNote: string | null }) {
  return (
    <div className="grid grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)_minmax(0,1fr)] gap-x-3">
      <span />
      <p className="type-caption font-mono uppercase tracking-wide text-fg-faint">yours</p>
      <div>
        <p className="type-caption font-mono uppercase tracking-wide text-fg-faint">theirs</p>
        {theirsNote ? (
          <p className="type-caption text-fg-faint" data-theirs-column-note="true">
            {theirsNote}
          </p>
        ) : null}
      </div>
      <p className="col-span-3 type-caption text-fg-faint" data-check-source-legend="true">
        {CHECK_SOURCE_LEGEND}
      </p>
    </div>
  )
}

/** One check: name -> your chip -> their chip (p2 item 2). */
function ChecksLine({
  id,
  name,
  yours,
  theirs,
  highlighted
}: {
  id?: string
  name: string
  yours: ReactNode
  theirs: ReactNode
  highlighted: boolean
}) {
  return (
    <div
      className={cn(
        'grid grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)_minmax(0,1fr)] items-baseline gap-x-3 rounded',
        highlighted && 'ring-1 ring-accent/60 bg-accent/10'
      )}
      data-check-line={name}
      id={id}
    >
      <p className="type-caption text-fg-faint">{name}</p>
      <div className="min-w-0">{yours}</div>
      <div className="min-w-0">{theirs}</div>
    </div>
  )
}

export type SecurityChecksViewProps = {
  row: ExchangeLedgerRow
  identity: RecomputedIdentity
  localRecord: CapsuleRecord | null
  /** Lifted to `ExchangeStreamRow` (mounted for every visible row, not just
   *  an expanded one) so a fetch this panel triggers can also flip that
   *  row's always-visible status badge once it resolves -- see
   *  `exchange-row-state.ts`'s `deriveRightCellState`. Optional only so a
   *  caller that hasn't wired a live fetch degrades to "not fetched" (never
   *  a fabricated match) -- `ExchangeStreamRow` always supplies a real one. */
  theirsRecompute?: PeerRecomputeState
  /** The property key the row's chip strip just jumped to
   * -- briefly rings the matching
   *  block so the jump is visible, not just scrolled-to. `null`/absent when
   *  no chip jump is pending (the panel opened some other way). */
  highlightedPropertyKey?: string | null
  /** The row's check results, computed once by `ExchangeStreamRow` and
   *  shared with its chip strip so the two can never disagree (look finding
   *  1). Computed here only when a caller doesn't supply them. */
  checksRows?: readonly ChecksRow[]
}

const NOT_FETCHED_DEFAULT: PeerRecomputeState = {
  status: 'not_fetched',
  idMatch: null,
  signatureOk: null,
  peerRecord: null,
  fetch: () => {}
}

export function SecurityChecksView({
  row,
  identity,
  localRecord,
  theirsRecompute = NOT_FETCHED_DEFAULT,
  highlightedPropertyKey = null,
  checksRows: suppliedChecksRows
}: SecurityChecksViewProps) {
  const [rawMode, setRawMode] = useState(false)
  const canFetchTheirs = theirsFetchable(row.raw) !== null

  const identityRow = buildIdentityRow(row.raw, identity, theirsRecompute)
  const headerRows = buildHeaderRows(row.raw, localRecord, theirsRecompute)
  const commitsToRows = buildCommitsToRows(row.raw, localRecord, theirsRecompute)
  const checksRows = suppliedChecksRows ?? buildChecksRows(row.raw, identity, theirsRecompute)
  const captureCoverageRow = checksRows.find((r) => r.key === 'capture_coverage')
  const nodeSaidRows = checksRows.filter((r) => r.key !== 'capture_coverage' && r.group === WHAT_NODE_SAID_GROUP)
  const actuallyHappenedRows = checksRows.filter((r) => r.group === WHAT_ACTUALLY_HAPPENED_GROUP)
  const theirsColumnNote = sharedTheirsDetail(checksRows)

  async function copyBoth() {
    const payload = JSON.stringify({ mine: row.raw.mine, theirs: row.raw.theirs }, null, 2)
    try {
      await navigator.clipboard.writeText(payload)
    } catch {
      // clipboard access denied/unavailable -- no crash, nothing else to do here.
    }
  }

  return (
    <div
      aria-label={`Security checks for ${row.exchangeKey}`}
      className="flex flex-col gap-3 rounded border border-border-soft bg-panel px-3 py-3"
      data-security-view="expanded"
      onClick={(event) => event.stopPropagation()}
      role="region"
    >
      <div className="flex items-center justify-end gap-2">
        {/* The row's own `▾ checks` toggle (`ExchangeStreamRow`) is this
           panel's only header label -- not repeated here, so there is one
           source of the "is this expanded" fact, not two. */}
        <div className="flex items-center gap-2">
          <div className="flex overflow-hidden rounded border border-border-soft" role="group">
            <Button
              aria-pressed={!rawMode}
              className="ui-control h-6 rounded-none px-2 text-[length:var(--density-type-caption)]"
              onClick={() => setRawMode(false)}
              size="sm"
              type="button"
              variant={rawMode ? 'ghost' : 'outline'}
            >
              readable
            </Button>
            <Button
              aria-pressed={rawMode}
              className="ui-control h-6 rounded-none px-2 text-[length:var(--density-type-caption)]"
              onClick={() => setRawMode(true)}
              size="sm"
              type="button"
              variant={rawMode ? 'outline' : 'ghost'}
            >
              raw
            </Button>
          </div>
          <Button
            className="ui-control h-6 gap-1 px-2 text-[length:var(--density-type-caption)]"
            onClick={copyBoth}
            size="sm"
            type="button"
            variant="outline"
          >
            ⧉ copy both
          </Button>
        </div>
      </div>

      {rawMode ? (
        <pre className="max-h-64 overflow-auto rounded border border-border-soft bg-background p-2 font-mono text-xs text-fg-dim">
          {JSON.stringify({ mine: row.raw.mine, theirs: row.raw.theirs }, null, 2)}
        </pre>
      ) : (
        <>
          <div className="flex flex-col gap-1.5">
            <BlockHeading>identity</BlockHeading>
            <TwoCol
              theirs={
                identityRow.theirs ? (
                  <p className="font-mono text-xs text-foreground">
                    {identityRow.theirs.value}
                    <span className="block text-fg-faint">{identityRow.theirs.note}</span>
                  </p>
                ) : null
              }
              yours={
                <p className="font-mono text-xs text-foreground">
                  {identityRow.yours.value}
                  <span className="block text-fg-faint">{identityRow.yours.note}</span>
                </p>
              }
            />
            {/* piece 4: a real peer-asserted join
               key exists but this browser has not fetched it yet -- an
               explicit action, never an automatic background fetch (a mesh
               call is not a free local read). Witness-level recompute only:
               the button never claims a verdict, just that this browser will
               go look. */}
            {canFetchTheirs && theirsRecompute.status !== 'fetching' ? (
              <Button
                className="ui-control h-6 w-fit gap-1 px-2 text-[length:var(--density-type-caption)]"
                data-peer-fetch-action="mesh_ledger_fetch"
                onClick={() => theirsRecompute.fetch()}
                size="sm"
                type="button"
                variant="outline"
              >
                fetch peer capsule &amp; recompute here
              </Button>
            ) : null}
          </div>

          <div className="flex flex-col gap-1.5">
            <BlockHeading>header</BlockHeading>
            {headerRows.map((headerRow) => (
              <div className="flex flex-col gap-0.5" key={headerRow.label}>
                <p className="type-caption text-fg-faint">{headerRow.label}</p>
                <TwoCol
                  theirs={
                    headerRow.theirs ? (
                      <p className="text-xs text-foreground">
                        {headerRow.theirs.value}
                        {headerRow.theirs.note ? (
                          <span className="ml-1 text-fg-faint">{headerRow.theirs.note}</span>
                        ) : null}
                      </p>
                    ) : null
                  }
                  yours={<p className="text-xs text-foreground">{headerRow.yours.value}</p>}
                />
              </div>
            ))}
          </div>

          <div className="flex flex-col gap-1.5">
            <BlockHeading>what it commits to</BlockHeading>
            {commitsToRows.map((commitsRow) => (
              <div className="flex flex-col gap-0.5" key={commitsRow.label}>
                <p className="type-caption text-fg-faint">{commitsRow.label}</p>
                <TwoCol
                  theirs={
                    commitsRow.theirs ? (
                      <p className="truncate font-mono text-xs text-foreground">
                        {commitsRow.theirs.value}
                        <span className="ml-1 text-fg-faint">{commitsRow.theirs.note}</span>
                      </p>
                    ) : null
                  }
                  yours={<p className="truncate font-mono text-xs text-foreground">{commitsRow.yours}</p>}
                />
              </div>
            ))}
          </div>

          <div className="flex flex-col gap-1.5">
            <BlockHeading>checks</BlockHeading>

            {/* Two labelled groups, no counts (v3 §4 / tab-design v2.1) -- this
                node's own nine claims, then the one axis that isn't this
                node's claim at all. Never merged back into one flat list. */}
            <p
              className="type-caption font-mono uppercase tracking-wide text-fg-faint"
              data-check-group={WHAT_NODE_SAID_GROUP}
            >
              {WHAT_NODE_SAID_GROUP}
            </p>
            <ChecksColumnHeader theirsNote={theirsColumnNote} />
            {nodeSaidRows.map((checkRow) =>
              checkRow.facts ? (
                checkRow.facts.map((fact) => (
                  <ChecksLine
                    highlighted={highlightedPropertyKey === checkRow.key}
                    id={fact.factLabel === 'binding' ? checkRowDomId(row.exchangeKey, checkRow.key) : undefined}
                    key={`${checkRow.key}:${fact.factLabel}`}
                    name={`${checkRow.label} · ${fact.factLabel}`}
                    theirs={null}
                    yours={
                      <ChecksCell
                        cell={fact.cell}
                        factKey={fact.factLabel as 'binding' | 'authority'}
                        propertyKey={checkRow.key}
                      />
                    }
                  />
                ))
              ) : (
                <ChecksLine
                  highlighted={highlightedPropertyKey === checkRow.key}
                  id={checkRowDomId(row.exchangeKey, checkRow.key)}
                  key={checkRow.key}
                  name={checkRow.label}
                  theirs={
                    checkRow.theirs ? (
                      <ChecksCell
                        cell={checkRow.theirs}
                        propertyKey={checkRow.key}
                        showDetail={checkRow.theirs.detail !== theirsColumnNote}
                      />
                    ) : null
                  }
                  yours={checkRow.yours ? <ChecksCell cell={checkRow.yours} propertyKey={checkRow.key} /> : null}
                />
              )
            )}
            {captureCoverageRow ? (
              <p className="text-xs text-fg-dim">
                <span className="text-fg-faint">{captureCoverageRow.label}</span> {captureCoverageRow.singleLine}
              </p>
            ) : null}

            <p
              className="type-caption font-mono uppercase tracking-wide text-fg-faint"
              data-check-group={WHAT_ACTUALLY_HAPPENED_GROUP}
            >
              {WHAT_ACTUALLY_HAPPENED_GROUP}
            </p>
            {actuallyHappenedRows.map((checkRow) => (
              <ChecksLine
                highlighted={highlightedPropertyKey === checkRow.key}
                id={checkRowDomId(row.exchangeKey, checkRow.key)}
                key={checkRow.key}
                name={checkRow.label}
                theirs={
                  checkRow.theirs ? (
                    <ChecksCell
                      cell={checkRow.theirs}
                      propertyKey={checkRow.key}
                      showDetail={checkRow.theirs.detail !== theirsColumnNote}
                    />
                  ) : null
                }
                yours={checkRow.yours ? <ChecksCell cell={checkRow.yours} propertyKey={checkRow.key} /> : null}
              />
            ))}
          </div>
        </>
      )}

      {/* "Save evidence file" acts on the
         current scope (v3 §4) -- from a row it saves that exchange. p2 item
         1: the disabled "open in Logs" shell is dropped until the Ledger and
         Logs sides share one id (the exchange_id join key); no inert control. */}
      <div className="flex flex-col gap-1.5 border-t border-border-soft pt-2">
        <div className="flex flex-wrap items-center gap-2">
          <Button
            className="ui-control h-7 gap-1 rounded-[var(--radius)] px-2 text-[length:var(--density-type-caption)]"
            onClick={() =>
              saveTextFile(
                `mesh-exchange-${row.exchangeKey}-evidence.json`,
                exchangeEvidenceBundle([row.raw]),
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
        <p className="text-xs text-fg-faint">{EVIDENCE_FILE_SENTENCE}</p>
      </div>
    </div>
  )
}

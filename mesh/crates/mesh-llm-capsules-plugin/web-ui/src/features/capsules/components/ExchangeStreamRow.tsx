// One row of the two-sided Ledger stream.
// v3 §2's anatomy: a role marker (`●` asked / `◐` served -- no rail, L-O), a
// left session rail when this row and the previous one share a session
// (L-N), and the two-sided record itself -- `YOUR RECORD │ THEIR RECORD, AS
// GIVEN TO YOU`. The column labels themselves now live in the sticky header
// above the stream (v3 §2a), not repeated per row.
// v3 §2/§3/§4 -- the row carries its own two
// explicit toggles, `▸/▾ content` and `▸/▾ checks`, right below the summary
// row. There is no modal and no whole-row click target: the two toggles are
// the entire detail surface.
import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import { cn } from '@/lib/cn'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import { SeeInLogsLink } from '@/features/capsules/components/ExchangeIdCell'
import { ExchangeRowChips } from '@/features/capsules/components/ExchangeRowChips'
import { SecurityChecksView } from '@/features/capsules/components/SecurityChecksView'
import {
  ProviderSettlementStrip,
  SettlementEntries,
  SettlementStrip
} from '@/features/capsules/components/SettlementRow'
import { buildChecksRows } from '@/features/capsules/lib/security-checks-view'
import {
  theirContentAction,
  theirContentText,
  yourContentFixedText
} from '@/features/capsules/lib/exchange-content-state'
import { isServedForPay, type ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import {
  askIsOffered,
  askReplyNote,
  askTarget,
  stateAfterAsk,
  type AskOutcome,
  type AskTarget
} from '@/features/capsules/lib/ask-for-record'
import { entryRowChipMark, entryRowChipPropertyKey } from '@/features/capsules/lib/entry-row-chips'
import { checkRowDomId, exchangeRowDomId } from '@/features/capsules/lib/exchange-pages'
import {
  askForRecordIsDue,
  bracketStrip,
  bracketStripText,
  closedPropertyCellItems,
  deriveRightCellState,
  isAlarmState,
  isAskAction,
  pushedHalfRecompute,
  rightCellAction,
  rightCellDetail,
  type RightCellStateKind,
  rightCellStatusLabel,
  rightCellText,
  rowStatusLabel
} from '@/features/capsules/lib/exchange-row-state'
import { InfoHover } from '@/features/capsules/components/InfoHover'
import type { RailSegment } from '@/features/capsules/lib/exchange-stream'
import { usePeerLedgerRecompute, useRecomputedIdentity } from '@/features/capsules/lib/recompute-identity'
import {
  durationText,
  pocBlock,
  rowModelName,
  servingProvenance,
  tokenFlowText
} from '@/features/capsules/lib/serving-provenance'
import { shortId } from '@/features/capsules/lib/short-id'
import { StageStrip } from '@/features/capsules/components/StageStrip'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { useClipboardCopy } from '@/lib/useClipboardCopy'
import { formatExchangeTimestamp } from '@/features/capsules/lib/local-time'
import { RowVerdictLine } from '@/features/capsules/components/RowVerdictLine'
import { rowCombinationText } from '@/features/capsules/lib/row-combination'
import {
  CLOSED_FROM_FETCH_NOT_SAVED,
  LOCAL_ONLY_TOOLTIP,
  OWN_RECORD_FAILS_WARNING,
  SERVED_FOR_PAY_NO_PAYER_TOOLTIP
} from '@/features/capsules/lib/tooltip-copy'

/** The gated cell text (Do (2)) when an ask
 *  action's row carries no recorded counterparty. This splits by which of two
 *  distinct truths holds -- the old single "nothing to ask yet" implied a
 *  future action the reader can't take and hid which situation they were in:
 *   - a SERVED row: this node served locally, there is no remote counterparty
 *     on the other side to ask at all -> `Local — no other side`;
 *   - a SERVED row this node served for pay: another node paid for it, but
 *     its id is not recorded -> `Served for pay — payer not recorded`;
 *   - an ASKED row: a remote exchange happened but the peer is unknown/
 *     unrecorded (no counterparty identity field exists on the record yet; the
 *     join comes from Pane B, per-peer) -> `Other side: not known`.
 *  Each is a stated fact, never a deficit or a "not yet". */
const NO_OTHER_SIDE_TEXT = 'Local — no other side'
const OTHER_SIDE_NOT_KNOWN_TEXT = 'Other side: not known'
const PAYER_NOT_RECORDED_TEXT = 'Served for pay — payer not recorded'

/** Wall-clock time for the ask timeout, re-read every 30 s so a waiting row
 *  offers the ask once the timeout passes -- never `Date.now()` in render. */
function useNowMs(): number {
  const [nowMs, setNowMs] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 30_000)
    return () => window.clearInterval(timer)
  }, [])
  return nowMs
}

function roleText(roleTag: string): string {
  // `You served`/`You asked` are stated in words on every row -- never
  // inferred from position or colour alone (v3 §2).
  return roleTag === 'SERVED' ? 'You served' : 'You asked'
}

/** Design §3A: "one colour per state ... taken from the console palette,
 *  nowhere else." CLOSED gets the Logs card's COMPLETED green, CONTRADICTED
 *  its FAILED red, refused/absent the `Local only` amber -- every other OPEN
 *  variant stays muted/no colour (L-A: "an open row is never styled as a
 *  problem"). Distinct from `isAlarmState`, which only gates the dot. */
function rowStateTone(kind: RightCellStateKind): StatusBadgeTone {
  switch (kind) {
    case 'closed':
      return 'good'
    case 'contradicted':
      return 'bad'
    case 'open_refused':
    case 'open_absent':
      return 'warn'
    default:
      return 'muted'
  }
}

/** A short/copyable identifier (design §3A) -- `role="link"`, not `button`, for the same reason
 *  `ExchangeRowChips` isn't: this jumps/copies rather than performing a row
 *  action, so it doesn't inflate the row's `getAllByRole('button')` count.
 *  Copy state via `useClipboardCopy` (`CopyInstructionRow`'s own hook) --
 *  never a second, hand-rolled clipboard try/catch. */
function CopyableId({ label, value }: { label: string; value: string }) {
  const { copyState, copyText } = useClipboardCopy()
  function copy() {
    void copyText(value)
  }
  return (
    <span className="inline-flex items-center gap-1">
      <span className="text-fg-faint">{label}</span>
      <span
        aria-label={`${copyStateLabel(copyState)} ${label} id ${value}`}
        className="ui-control-ghost cursor-pointer text-foreground"
        onClick={copy}
        onKeyDown={(event) => {
          if (event.key !== 'Enter' && event.key !== ' ') return
          event.preventDefault()
          copy()
        }}
        role="link"
        tabIndex={0}
        title={value}
      >
        {shortId(value)}
      </span>
      {copyState === 'copied' ? <span className="text-fg-faint">copied</span> : null}
    </span>
  )
}

export type ExchangeStreamRowProps = {
  row: ExchangeLedgerRow
  rail: RailSegment
  /** Keyboard nav cursor (j/k) -- transient, not persisted anywhere. */
  focused?: boolean
  /** The per-row deep-link target (v3 §2a "opening its page with the row
   *  expanded and highlighted") -- persists until a different row is
   *  focused via deep link, independent of keyboard focus. */
  highlighted?: boolean
  /** `c` keyboard toggle -- reveals this row's Checks column value inline.
   *  Real per-row Checks toggle ② (v3 §4) is a later batch; this is the
   *  honest minimum today's data already supports (`checksText`, the same
   *  field the CSV export already uses). */
  checksExpanded?: boolean
  /** `o` keyboard toggle -- reveals this row's content cells inline
   *  (v3 §3): one populated side, one
   *  empty side, flipping with `Your role` (L-F). */
  contentExpanded?: boolean
  /** This row's own local capsule record --
   *  needed to recompute `content_binding`/`producer_signature` in-browser
   *  for the security view. `null` when the record hasn't been fetched
   *  (never a stand-in for a trusted result). */
  localRecord?: CapsuleRecord | null
  nodePubKeyPem?: string | null
  /** Whether the latest checkpoint covers this row's own record (look
   *  finding 2, `checkpointCoverageByRecord`); `null` = not reported. */
  checkpointCovered?: boolean | null
  /** p2 item 3: this node's owner link (`ownerLinked`, the same derivation
   *  Integrity's step 2 reads); `null` = not known. */
  ownerLinked?: boolean | null
  /** Toggle ① -- flips `contentExpanded` for this row (the `▸/▾ content` control). */
  onToggleContent: (row: ExchangeLedgerRow) => void
  /** Toggle ② -- flips `checksExpanded` for this row (the `▸/▾ checks` control). */
  onToggleChecks: (row: ExchangeLedgerRow) => void
  onAction: (row: ExchangeLedgerRow) => void
  /** What "Ask them for their record" produced for this row, if it asked. */
  askOutcome?: AskOutcome | null
  /** Sends the ask. Absent where the page can't ask (the row offers none). */
  onAskForRecord?: (row: ExchangeLedgerRow, target: AskTarget) => void
}

export function ExchangeStreamRow({
  row,
  rail,
  focused = false,
  highlighted = false,
  checksExpanded = false,
  contentExpanded = false,
  localRecord = null,
  nodePubKeyPem = null,
  checkpointCovered = null,
  ownerLinked = null,
  onToggleContent,
  onToggleChecks,
  onAction,
  askOutcome = null,
  onAskForRecord
}: ExchangeStreamRowProps) {
  // Lifted here (not `SecurityChecksView`, which only mounts once `▸
  // checks` is expanded) so a fetch this hook's `.fetch()` triggers can
  // also flip this row's ALWAYS-VISIBLE status badge the moment it
  // resolves, not just the expanded panel's own cells
  // (finding 1). The hook
  // itself is cheap and a no-op until `.fetch()` is called (rules of
  // hooks require it run unconditionally, same as `useRecomputedIdentity`
  // below).
  // A live fetch wins when this browser ran one; otherwise the pushed half the
  // pane already correlated (its capsule_id recomputed before the query
  // resolved). Either way the SAME gate below judges it, and the checks panel
  // shows the same evidence.
  const peerFetch = usePeerLedgerRecompute(row.raw)
  const theirsRecompute = peerFetch.status === 'found' ? peerFetch : (pushedHalfRecompute(row.raw) ?? peerFetch)
  const gateState = deriveRightCellState(row.raw, theirsRecompute, localRecord)
  // Once this row has asked the other side for its record, their reply (or
  // its absence) is the state: a record goes through the same gate as a
  // pushed one (`ask-for-record.ts`), a signed refusal reads as one.
  const state = askOutcome ? stateAfterAsk(row.raw, askOutcome, localRecord) : gateState
  // Whether the plugin verified their reply, and if not, why: said on the row.
  const askNote = askOutcome ? askReplyNote(askOutcome) : null
  const alarm = isAlarmState(state)
  // Whom to ask, and how to name the exchange, from our own record of it.
  const askFor = askTarget(localRecord ?? (row.raw.mine.record as CapsuleRecord | undefined) ?? null)
  // Do (2): an ask action with no recorded
  // counterparty renders no button at all, with the text branching on WHICH
  // truth holds -- a SERVED row has no remote other side to ask; an ASKED row
  // has one, but it's unknown/unrecorded. Never a single "nothing to ask yet"
  // that hides the difference.
  const gatedByCounterparty = isAskAction(state.kind) && !row.counterparty && !askFor
  // A served exchange with no other side anywhere: LOCAL, not OPEN, since
  // nothing of theirs can arrive (`isLocalOnly`). An ask this row sent would
  // name another side, so it ends this.
  const localOnly = row.localOnly && !askOutcome
  const servedForPay = isServedForPay(row.raw)
  const gatedText =
    row.roleTag !== 'SERVED' ? OTHER_SIDE_NOT_KNOWN_TEXT : servedForPay ? PAYER_NOT_RECORDED_TEXT : NO_OTHER_SIDE_TEXT
  const cellText = gatedByCounterparty ? gatedText : rightCellText(state)
  // UX §3: with push on, their record normally arrives when the exchange
  // finishes -- the ask is offered only once that has had time to happen.
  const nowMs = useNowMs()
  const askWaiting = isAskAction(state.kind) && !askForRecordIsDue(row.timestamp, nowMs)
  // The ask is offered past the timeout on a row still waiting for their
  // record, when we know whom to ask; never while an ask is in flight. An ask
  // kind that can't be offered renders no button rather than a dead one.
  const askOffered =
    onAskForRecord !== undefined &&
    askOutcome?.kind !== 'asking' &&
    askIsOffered(state.kind, askFor, row.timestamp, nowMs)
  const action = askOffered
    ? state.kind === 'open_asked'
      ? 'Ask again'
      : 'Ask them for their record'
    : gatedByCounterparty || askWaiting || isAskAction(state.kind)
      ? null
      : rightCellAction(state)
  // Look finding 1: the chip strip on the collapsed row must show the same
  // results as the panel, so this node's own record is recomputed for every
  // rendered row, not only an expanded one. It is a local hash + signature
  // verify over bytes already in the browser -- no fetch.
  const identity = useRecomputedIdentity(localRecord, nodePubKeyPem)
  // ONE set of check results for both the strip and the panel; the gate's
  // verdict feeds the outcome row so the badge, strip and panel agree.
  const checksRows = buildChecksRows(row.raw, identity, theirsRecompute, {
    gateKind: state.kind,
    checkpointCovered,
    ownerLinked
  })
  // Confirmed from a record this page fetched, not one the node holds: say
  // it isn't saved, so a reload reopening the row is never a surprise.
  const closedFromFetch =
    state.kind === 'closed' &&
    (askOutcome?.kind === 'record' || (peerFetch.status === 'found' && !pushedHalfRecompute(row.raw)))
  // Your own copy failed "words match" or "signed". Whatever
  // the badge says (Confirmed included), the row says so.
  const ownRecordFails = entryRowChipMark(checksRows, 'content') === '✗' || entryRowChipMark(checksRows, 'sig') === '✗'
  // The bracket strip, drawn in words (UX §3): `Yours ● sealed —— Theirs ●
  // same`. Same state the badge renders.
  const bracket = bracketStrip(row.raw, state)
  const strip = bracketStripText(localOnly ? { ...bracket, theirs: { glyph: '○', word: 'no other side' } } : bracket)
  // One hover on the strip for what the badge and the chips say together.
  const combination = rowCombinationText({
    state: state.kind,
    roleTag: row.roleTag,
    servedForPay,
    yoursHeld: row.raw.mine.state !== 'absent',
    marks: {
      content: entryRowChipMark(checksRows, 'content'),
      sig: entryRowChipMark(checksRows, 'sig'),
      inclusion: entryRowChipMark(checksRows, 'inclusion'),
      registered: entryRowChipMark(checksRows, 'registered')
    },
    refusedAtTheDoor: row.raw.theirs.evidence_outcome === 'claims_refused',
    late: askForRecordIsDue(row.timestamp, nowMs)
  })
  // UX §3 "Left: the event in words" -- role, peer, model, tokens, duration,
  // all read from this row's own record (the fetched `localRecord`, else the
  // body the pane sent with the pair). Any field the record doesn't carry is
  // left out, never a placeholder. The ids move into the expansion.
  const ownRecord = localRecord ?? (row.raw.mine.record as CapsuleRecord | undefined) ?? null
  const provenance = ownRecord ? servingProvenance(ownRecord) : null
  // A requester's own record often carries no model details, usage or
  // latency; the other side's sealed record does. Ours first, theirs
  // labelled as theirs -- never a count presented as our own.
  const theirRecord = (row.raw.theirs.record as CapsuleRecord | undefined) ?? null
  const theirProvenance = theirRecord ? servingProvenance(theirRecord) : null
  const modelRef = provenance?.model ?? theirProvenance?.model ?? null
  const modelIdentity = rowModelName(provenance, theirProvenance)
  const ownTokens = provenance ? tokenFlowText(provenance.promptTokens, provenance.completionTokens) : null
  const theirTokens = theirProvenance
    ? tokenFlowText(theirProvenance.promptTokens, theirProvenance.completionTokens)
    : null
  const tokens = ownTokens ?? (theirTokens ? `${theirTokens} (their count)` : null)
  const ownDuration = ownRecord ? durationText(pocBlock(ownRecord).latency_ms) : null
  const theirDuration = theirRecord ? durationText(pocBlock(theirRecord).latency_ms) : null
  const duration = ownDuration ?? (theirDuration ? `${theirDuration} (their measure)` : null)
  const tone = rowStateTone(state.kind)

  // Chip-strip -> checks-panel jump (§3A "each chip a link into the expansion's matching check"). A chip
  // click on an already-collapsed panel must expand it first and wait for
  // `SecurityChecksView` to actually mount before a `getElementById` lookup
  // can find anything -- `pendingScrollKeyRef` carries the target across
  // that one render; a chip click while already expanded scrolls straight
  // away.
  const pendingScrollKeyRef = useRef<string | null>(null)
  const [highlightedCheckKey, setHighlightedCheckKey] = useState<string | null>(null)

  function scrollToCheck(propertyKey: string) {
    document.getElementById(checkRowDomId(row.exchangeKey, propertyKey))?.scrollIntoView({ block: 'nearest' })
    setHighlightedCheckKey(propertyKey)
    window.setTimeout(() => {
      setHighlightedCheckKey((prev) => (prev === propertyKey ? null : prev))
    }, 1500)
  }

  function handleChipActivate(propertyKey: string) {
    if (checksExpanded) {
      scrollToCheck(propertyKey)
      return
    }
    pendingScrollKeyRef.current = propertyKey
    onToggleChecks(row)
  }

  useEffect(() => {
    if (!checksExpanded || !pendingScrollKeyRef.current) return
    const propertyKey = pendingScrollKeyRef.current
    pendingScrollKeyRef.current = null
    scrollToCheck(propertyKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- scrollToCheck closes over this render's row/state; re-running on checksExpanded alone is the intended trigger
  }, [checksExpanded])

  return (
    <div className="flex flex-col" id={exchangeRowDomId(row.exchangeKey)}>
      {rail.isSegmentStart && row.sessionId ? (
        <p className="pl-3 pt-2 type-caption font-mono text-fg-faint">session {row.sessionId}</p>
      ) : null}
      <div
        aria-current={highlighted ? 'true' : undefined}
        aria-label={`Exchange ${row.exchangeKey}`}
        className={cn(
          'flex flex-col gap-2 border-l-2 py-3 pl-3 pr-1',
          rail.hasRail ? 'border-accent/50' : 'border-transparent',
          focused && 'ring-1 ring-inset ring-accent/70',
          highlighted && 'bg-[color-mix(in_oklab,var(--color-accent)_10%,transparent)]'
        )}
        data-focused={focused ? 'true' : undefined}
        data-highlighted={highlighted ? 'true' : undefined}
        data-right-cell-state={state.kind}
        data-role-tag={row.roleTag}
        role="group"
      >
        <div className="grid grid-cols-2 gap-0 rounded border border-border-soft">
          <div className="flex flex-col gap-1 border-r border-border-soft px-3 py-2">
            <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-fg-dim">
              {row.raw.referee_call ? (
                <span
                  className="rounded border border-border-soft px-1 text-[10px] uppercase tracking-wide text-fg-dim"
                  data-referee-call="true"
                >
                  referee answer
                </span>
              ) : null}
              <time className="font-mono tabular-nums" dateTime={row.timestamp ?? undefined}>
                {formatExchangeTimestamp(row.timestamp)}
              </time>
            </p>
            {/* Real text separators (not CSS gaps), so a copy or a screen
               reader gets "You asked key:… · model · 46 → 2 tokens". */}
            <p className="text-xs text-foreground" data-event-line="true">
              <span className="font-medium">{roleText(row.roleTag)}</span>{' '}
              {row.counterparty ? (
                <span className="font-mono">{row.counterparty}</span>
              ) : (
                <span className="text-fg-dim">counterparty not recorded</span>
              )}
              {modelIdentity ? (
                <>
                  <span className="text-fg-faint"> · </span>
                  <span className="font-mono" title={modelRef ?? undefined}>
                    {modelIdentity}
                  </span>
                </>
              ) : null}
              {tokens ? (
                <>
                  <span className="text-fg-faint"> · </span>
                  <span className="tabular-nums">{tokens}</span>
                </>
              ) : null}
              {duration ? (
                <>
                  <span className="text-fg-faint"> · </span>
                  <span className="tabular-nums">{duration}</span>
                </>
              ) : null}
            </p>
            <RowVerdictLine row={row.raw} />
          </div>
          <div className="flex flex-col gap-1.5 px-3 py-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="inline-flex items-center gap-1">
                <p className="font-mono text-[11px] text-fg-faint" data-bracket-strip="true">
                  {strip}
                </p>
                <InfoHover census="row:combination" describes="this row as a whole" label={combination} side="left" />
              </span>
              <span className="inline-flex items-center gap-1" data-row-tone={tone}>
                {/* L-A/L-B: only CONTRADICTED gets the alarm dot; the tone
                   (§3A "one colour per state") still varies with CLOSED/
                   refused/absent, never just alarm-vs-muted. */}
                <StatusBadge dot={alarm} size="caption" tone={localOnly ? 'muted' : tone}>
                  {localOnly ? 'LOCAL' : rowStatusLabel(state, row.raw.theirs.in_their_log)}
                </StatusBadge>
                {/* Terse state on the face; the fuller story behind the (i). */}
                <InfoHover
                  census={localOnly ? 'row_state:local' : `row_state:${state.kind}`}
                  describes={localOnly ? 'the LOCAL state' : `the ${rightCellStatusLabel(state)} state`}
                  label={
                    localOnly
                      ? LOCAL_ONLY_TOOLTIP
                      : servedForPay && gatedByCounterparty
                        ? SERVED_FOR_PAY_NO_PAYER_TOOLTIP
                        : rightCellDetail(state)
                  }
                  side="left"
                />
              </span>
            </div>
            <p className="text-xs text-foreground" data-right-cell-text="true">
              {cellText}
            </p>
            {askNote ? (
              <p
                className={cn('text-xs', askNote.verified ? 'text-foreground' : 'text-fg-dim')}
                data-ask-reply-verified={askNote.verified ? 'true' : 'false'}
                role="note"
              >
                {askNote.text}
              </p>
            ) : null}
            {closedFromFetch ? (
              <p className="text-xs text-fg-dim" data-closed-from-fetch="true" role="note">
                {CLOSED_FROM_FETCH_NOT_SAVED}
              </p>
            ) : null}
            {ownRecordFails ? (
              <p className="text-xs text-bad" data-own-record-fails="true" role="note">
                {OWN_RECORD_FAILS_WARNING}
              </p>
            ) : null}
            {action ? (
              <Button
                className="ui-control h-7 w-fit gap-1 rounded-[var(--radius)] px-2 text-[length:var(--density-type-caption)]"
                onClick={() =>
                  // Compare opens this row's checks at "their record",
                  // where yours and theirs sit side by side. Other actions ask
                  // the page (a counterparty request, a statement to view).
                  state.kind === 'contradicted'
                    ? handleChipActivate(entryRowChipPropertyKey('theirs'))
                    : askOffered && askFor && onAskForRecord
                      ? onAskForRecord(row, askFor)
                      : onAction(row)
                }
                size="sm"
                type="button"
                variant="outline"
              >
                {action}
              </Button>
            ) : null}
          </div>
        </div>
        <ExchangeRowChips checks={checksRows} onChipActivate={handleChipActivate} />
        {row.raw.split ? <StageStrip split={row.raw.split} /> : null}
        {/* Payments sit beside the inference state, never inside it: CLOSED
           is about the two records of the exchange; settlement is this
           node's own payment records for it. */}
        <SettlementStrip settlement={row.raw.settlement} />
        <ProviderSettlementStrip requester={row.raw.counterparty} settlement={row.raw.provider_settlement} />
        {/* v3 §2's row footer: two independent
           disclosure toggles, never a modal. Always present, regardless of
           the right-cell state. */}
        <div className="flex items-center gap-3 text-xs text-fg-dim">
          <button
            aria-expanded={contentExpanded}
            className="ui-control-ghost font-mono"
            onClick={() => onToggleContent(row)}
            type="button"
          >
            {contentExpanded ? 'What was said ▾' : 'What was said ▸'}
          </button>
          <button
            aria-expanded={checksExpanded}
            className="ui-control-ghost font-mono"
            onClick={() => onToggleChecks(row)}
            type="button"
          >
            {checksExpanded ? 'How we checked ▾' : 'How we checked ▸'}
          </button>
        </div>
        {contentExpanded ? (
          <div className="grid grid-cols-2 gap-0 rounded border border-border-soft" data-content-toggle="expanded">
            <div
              className="flex flex-col gap-1 border-r border-border-soft px-3 py-2"
              data-your-content-state={row.contentToggleState.your.kind}
            >
              {row.contentToggleState.your.kind === 'populated' ? (
                <>
                  <p className="text-xs text-foreground">
                    <span className="text-fg-faint">You asked</span> · {row.raw.mine.text ?? '—'}
                  </p>
                  {row.raw.mine.reply_text ? (
                    <p className="text-xs text-foreground">
                      <span className="text-fg-faint">They streamed back</span> · {row.raw.mine.reply_text}
                    </p>
                  ) : null}
                </>
              ) : (
                <p className="text-xs text-foreground">{yourContentFixedText(row.contentToggleState.your)}</p>
              )}
            </div>
            <div
              className="flex flex-col gap-1.5 px-3 py-2"
              data-their-content-state={row.contentToggleState.their.kind}
            >
              <p className="text-xs text-fg-dim">{theirContentText(row.contentToggleState.their)}</p>
              {theirContentAction(row.contentToggleState.their) ? (
                <Button
                  className="ui-control h-7 w-fit gap-1 rounded-[var(--radius)] px-2 text-[length:var(--density-type-caption)]"
                  onClick={(event) => {
                    event.stopPropagation()
                    onAction(row)
                  }}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  {theirContentAction(row.contentToggleState.their)}
                </Button>
              ) : null}
            </div>
          </div>
        ) : null}
        {checksExpanded ? (
          // UX §3: the engineer's facts lead the expansion -- the CLOSED
          // per-property cells (each restating a fact the gate established),
          // then the full ids that used to sit on the row face.
          <div className="flex flex-wrap items-center gap-1.5 font-mono text-xs" data-expansion-head="true">
            {state.kind === 'closed'
              ? closedPropertyCellItems(row.raw).map((cell) => (
                  <span
                    className="inline-flex items-center gap-1 rounded border border-border-soft px-1.5 py-0.5 text-[11px] text-fg-dim"
                    data-closed-property-cell={cell.key}
                    key={cell.key}
                  >
                    {cell.label}
                    <InfoHover census={`closed_cell:${cell.key}`} describes={cell.label} label={cell.tooltip} />
                  </span>
                ))
              : null}
            <CopyableId label="exch" value={row.exchangeKey} />
            <SeeInLogsLink exchangeKey={row.exchangeKey} />
            <span className="text-fg-faint">·</span>
            {(row.raw.mine.capsule_id ?? row.raw.mine.text) ? (
              <CopyableId label="rec" value={(row.raw.mine.capsule_id ?? row.raw.mine.text) as string} />
            ) : (
              <span className="inline-flex items-center gap-1">
                <span className="text-fg-faint">rec</span>
                <span>—</span>
              </span>
            )}
          </div>
        ) : null}
        {checksExpanded ? <SettlementEntries settlement={row.raw.settlement} /> : null}
        {checksExpanded ? (
          <SettlementEntries perspective="provider" settlement={row.raw.provider_settlement} />
        ) : null}
        {checksExpanded ? (
          <SecurityChecksView
            checksRows={checksRows}
            highlightedPropertyKey={highlightedCheckKey}
            identity={identity}
            localRecord={localRecord}
            row={row}
            theirsRecompute={theirsRecompute}
          />
        ) : null}
      </div>
    </div>
  )
}

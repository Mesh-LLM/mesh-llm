// The TWIN bracket wrapper (v3 §5): a header
// naming the bracket, the two (or more) real two-sided rows it brackets
// (passed as `children`, already-rendered `ExchangeStreamRow`s -- this
// component adds chrome around them, it does not re-derive their content),
// and a COMPARISON block below.
//
// OBSERVE-ONLY (item 4): no verdict renders here, ever -- not PASS, not
// FAIL, not "identical." The header status is always the honest
// "no verdict" state, and the COMPARISON block states plainly that
// adjudication is not yet enabled. This stays true even once real
// parameters/digests are wired (see `twin-bracket.ts`) -- computing an
// equality verdict from a digest match is a deliberate product decision, not a
// default this component reaches on its own.
import { useState } from 'react'
import ReactDiffViewer from 'react-diff-viewer-continued'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import { saveTextFile } from '@/features/capsules/lib/exchange-export'
import { InfoHover } from '@/features/capsules/components/InfoHover'
import { VerdictRecordDialog } from '@/features/capsules/components/VerdictRecordDialog'
import { HoverChip } from '@/features/capsules/components/HoverChip'
import { notAdjudicatedLine, twinRefereeRow } from '@/features/capsules/lib/referee-row'
import { TWIN_NO_VERDICT_TOOLTIP, TWIN_TOOLTIPS, TWIN_VERDICT_TOOLTIPS } from '@/features/capsules/lib/tooltip-copy'
import {
  TWIN_ANSWER_LABEL,
  twinAnswerState,
  twinComparisonParametersLine,
  TWIN_PAIR_SENTENCE,
  twinResponseTexts,
  twinVerdict
} from '@/features/capsules/lib/twin-bracket'

const diffViewerStyles = {
  diffContainer: {
    background: 'var(--color-panel)',
    border: '1px solid var(--color-border-soft)',
    borderRadius: 'var(--radius)',
    minWidth: '100%',
    width: '100%',
    fontSize: 'var(--density-type-caption)',
    color: 'var(--color-foreground)'
  },
  titleBlock: {
    background: 'var(--color-panel-strong)',
    color: 'var(--color-fg-dim)',
    borderBottom: '1px solid var(--color-border-soft)'
  }
}

const TWIN_VERDICT_TONE = {
  corroborated: 'good',
  contradicted: 'bad',
  inconclusive: 'muted',
  not_comparable: 'muted'
} as const

function twinVerdictLabel(verdict: NonNullable<ReturnType<typeof twinVerdict>['verdict']>): string {
  switch (verdict.kind) {
    case 'corroborated':
      return 'referee: corroborated'
    case 'contradicted':
      return `referee: contradicted ${verdict.party.slice(0, 10)}…`
    case 'inconclusive':
      return 'referee: inconclusive'
    case 'not_comparable':
      return 'not comparable'
  }
}

export type TwinBracketProps = {
  bracketId: string
  rows: readonly ExchangeLedgerRow[]
  children: React.ReactNode
}

export function TwinBracket({ bracketId, rows, children }: TwinBracketProps) {
  const [compareOpen, setCompareOpen] = useState(false)
  const [verdictOpen, setVerdictOpen] = useState(false)
  const parametersLine = twinComparisonParametersLine(rows)
  const disclosure = TWIN_PAIR_SENTENCE
  const [textA, textB] = twinResponseTexts(rows)
  const canCompare = textA !== null && textB !== null
  const answer = twinAnswerState(rows)
  const { verdict, capsuleId } = twinVerdict(rows)
  const notAdjudicated = notAdjudicatedLine(twinRefereeRow(rows)) ?? 'Not adjudicated yet.'

  const handleSave = () => {
    saveTextFile(
      `twin-bracket-${bracketId}.json`,
      JSON.stringify(
        rows.map((row) => row.raw),
        null,
        2
      ),
      'application/json'
    )
  }

  return (
    <div
      aria-label={`Twin comparison ${bracketId}`}
      className="flex flex-col border-2 border-[var(--color-accent)]/40"
      data-testid={`twin-bracket-${bracketId}`}
      role="group"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-soft bg-panel-strong/60 px-3 py-1.5">
        <p className="type-caption inline-flex items-center gap-1 font-medium text-fg-dim">
          Side-by-side check · <span className="font-mono">{bracketId.length > 12 ? `${bracketId.slice(0, 12)}…` : bracketId}</span> · same request, two machines
          <InfoHover census="twin:header" describes="the side-by-side check" label={TWIN_TOOLTIPS.header} />
        </p>
        <span className="inline-flex items-center gap-1">
          {/* A checked fact (the two sealed answer-text digests), never a verdict. */}
          <HoverChip census={`twin:answer:${answer}`} label={TWIN_TOOLTIPS[answer]}>
            <span className="inline-flex" tabIndex={0}>
              <StatusBadge size="caption" tone={answer === 'different' ? 'warn' : 'muted'}>
                {TWIN_ANSWER_LABEL[answer]}
              </StatusBadge>
            </span>
          </HoverChip>
          {verdict === null ? (
            <>
              <StatusBadge size="caption" tone="muted">
                not adjudicated
              </StatusBadge>
              <InfoHover census="twin:no_verdict" describes="the not adjudicated badge" label={TWIN_NO_VERDICT_TOOLTIP} />
            </>
          ) : (
            <HoverChip census={`twin:verdict:${verdict.kind}`} label={TWIN_VERDICT_TOOLTIPS[verdict.kind]}>
              <span className="inline-flex" tabIndex={0}>
                <StatusBadge size="caption" tone={TWIN_VERDICT_TONE[verdict.kind]}>
                  {twinVerdictLabel(verdict)}
                </StatusBadge>
              </span>
            </HoverChip>
          )}
        </span>
      </div>

      <div className="flex flex-col divide-y divide-border-soft">{children}</div>

      <div className="flex flex-col gap-1.5 border-t border-border-soft bg-panel-strong/40 px-3 py-2">
        <p className="type-caption text-fg-dim">What was compared</p>
        {parametersLine ? (
          <p className="type-caption inline-flex items-center gap-1 text-fg-faint">
            {parametersLine}
            <InfoHover census="twin:parameters" describes="the settings line" label={TWIN_TOOLTIPS.parameters} />
          </p>
        ) : null}
        {/* OBSERVE-ONLY -- this line NEVER renders "identical"/"differs" or
           any other computed verdict, only the honest state of adjudication
           itself. */}
        {verdict === null ? (
          <p className="type-caption text-fg-faint">{notAdjudicated}</p>
        ) : (
          <p className="type-caption flex flex-wrap items-center gap-x-2 text-fg-faint">
            {/* Not "signed": the signature is checked only when the record is
               opened (View the verdict), so the line doesn't claim it. */}
            <span>Referee's verdict: {capsuleId ? `${capsuleId.slice(0, 12)}…` : 'id not recorded'}</span>
            {capsuleId ? (
              <>
                <button
                  className="underline underline-offset-2 hover:text-foreground"
                  onClick={() => setVerdictOpen(true)}
                  type="button"
                >
                  View the verdict
                </button>
                {verdictOpen ? (
                  <VerdictRecordDialog capsuleId={capsuleId} onOpenChange={setVerdictOpen} open={verdictOpen} />
                ) : null}
              </>
            ) : null}
          </p>
        )}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="type-caption text-fg-faint">{disclosure}</p>
          <div className="flex items-center gap-1">
            <HoverChip
              census="twin:compare"
              label={canCompare ? TWIN_TOOLTIPS.compare : 'Neither side has answer text to compare yet.'}
            >
              <Button
                disabled={!canCompare}
                onClick={() => setCompareOpen((open) => !open)}
                size="sm"
                type="button"
                variant="outline"
              >
                {compareOpen ? 'Compare ▾' : 'Compare ▸'}
              </Button>
            </HoverChip>
            <HoverChip census="twin:save" label={TWIN_TOOLTIPS.save}>
              <Button onClick={handleSave} size="sm" type="button" variant="outline">
                Save both records
              </Button>
            </HoverChip>
          </div>
        </div>
        {compareOpen && canCompare ? (
          <div className="mt-1 overflow-hidden rounded border border-border-soft">
            <ReactDiffViewer
              leftTitle="First answer"
              newValue={textB ?? ''}
              oldValue={textA ?? ''}
              rightTitle="Second answer"
              splitView
              styles={diffViewerStyles}
            />
          </div>
        ) : null}
      </div>
    </div>
  )
}

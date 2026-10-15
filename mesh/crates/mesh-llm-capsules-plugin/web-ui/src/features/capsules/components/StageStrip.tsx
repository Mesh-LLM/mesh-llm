// The stage strip under a split request's row (design §7.2): one cell per
// stage, the hand-offs line, and the request's terminal state beside it.
// It is added beneath the row and never replaces the row's own state.
import { HoverChip } from '@/features/capsules/components/HoverChip'
import { InfoHover } from '@/features/capsules/components/InfoHover'
import type { SplitRowJson } from '@/features/capsules/lib/split-stage'
import { splitView, stageRowHeadline } from '@/features/capsules/lib/split-stage'
import { HANDOFF_TOOLTIPS, SPLIT_TOOLTIPS, STAGE_CELL_TOOLTIPS } from '@/features/capsules/lib/tooltip-copy'

const HANDOFF_TONE: Record<string, string> = {
  agree: 'text-foreground',
  gap: 'text-fg-dim',
  break: 'text-[var(--color-danger)]',
  malformed: 'text-[var(--color-danger)]'
}

export function StageStrip({ split }: { split: SplitRowJson }) {
  if (split.viewer === 'stage') {
    return (
      <p className="type-caption font-mono text-fg-dim" data-split-stage-row="true">
        {stageRowHeadline(split.stage_block)}
      </p>
    )
  }
  const view = splitView(split)
  if (view === null) return null
  return (
    <div
      aria-label="Split stages"
      className="flex flex-col gap-1 rounded border border-border-soft px-3 py-2 font-mono text-xs"
      data-split-strip="true"
      role="group"
    >
      <p className="text-fg-dim">{view.headline}</p>
      <p className="inline-flex flex-wrap items-center gap-x-2 text-foreground">
        {/* Each cell's node is the coordinator's assignment, not a node the
            stage record itself is linked to: the strip says so on its face. */}
        <span className="text-fg-faint" data-stage-cells-source="coordinator">
          per coordinator:
        </span>
        {view.cells.map((cell, i) => (
          <HoverChip census={`split:cell:${cell.state}`} key={cell.stageIndex} label={STAGE_CELL_TOOLTIPS[cell.state]}>
            <span data-stage-cell={cell.state} tabIndex={0}>
              {i > 0 ? <span className="text-fg-faint">· </span> : null}
              {cell.nodeId} {cell.layers}
              {cell.mark ? ` ${cell.mark}` : ''}
              {cell.state !== 'ok' ? <span className="text-fg-dim"> {cell.words}</span> : null}
            </span>
          </HoverChip>
        ))}
        <InfoHover census="split:stage_cell" describes="the stage cells" label={SPLIT_TOOLTIPS.stageCell} />
      </p>
      <p className="inline-flex flex-wrap items-center gap-x-2">
        <span className="text-fg-faint">hand-offs:</span>
        {view.handoffs.map((item) => (
          <HoverChip census={`split:handoff:${item.state}`} key={item.label} label={HANDOFF_TOOLTIPS[item.state]}>
            <span className={HANDOFF_TONE[item.state]} data-handoff={item.state} tabIndex={0}>
              {item.label} {item.words}
              {item.state === 'agree' ? ' ✓' : ''}
            </span>
          </HoverChip>
        ))}
        <span className="text-fg-faint">·</span>
        <span data-split-terminal-state={view.terminalState}>request: {view.terminalState.replace(/_/g, ' ')}</span>
        <InfoHover census="split:handoffs" describes="the hand-offs line" label={SPLIT_TOOLTIPS.handoffs} />
      </p>
    </div>
  )
}

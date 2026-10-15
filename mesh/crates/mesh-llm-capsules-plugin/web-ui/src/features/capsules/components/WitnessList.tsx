// One line per witness the operator named (and any that holds a receipt):
// its name, what it holds, and why a receipt does not count yet. A witness
// counts only once its receipt checked; the plugin never names a default
// witness and contacts none until the operator adds one.
import type { WitnessRow } from '@/features/capsules/lib/integrity-view'

const TONE_COLOR: Record<WitnessRow['tone'], string> = {
  good: 'var(--color-good-text)',
  pending: 'var(--color-warn)',
  bad: 'var(--color-bad-text)',
  muted: 'var(--color-fg-faint)'
}

export function WitnessList({
  rows,
  note,
  problems = []
}: {
  rows: WitnessRow[]
  note?: string | null
  problems?: readonly string[]
}) {
  if (rows.length === 0 && !note && problems.length === 0) return null
  return (
    <div className="flex flex-col gap-1.5" data-testid="witness-list">
      <span className="type-label text-fg-faint">Witnesses</span>
      <ul className="flex flex-col gap-1">
        {rows.map((row) => (
          <li className="flex min-w-0 flex-wrap items-baseline gap-x-2" key={row.url || row.name}>
            <span className="font-medium text-foreground" title={row.url}>
              {row.name}
            </span>
            <span style={{ color: TONE_COLOR[row.tone] }}>{row.status}</span>
            {row.detail ? <span className="type-caption text-fg-faint">{row.detail}</span> : null}
            {row.key ? <span className="type-caption text-fg-faint">· {row.key}</span> : null}
          </li>
        ))}
      </ul>
      {note ? <p className="type-caption text-fg-faint" data-testid="witness-restart-note">{note}</p> : null}
      {problems.map((problem) => (
        <p className="type-caption" data-testid="witness-key-problem" key={problem} style={{ color: TONE_COLOR.bad }}>
          {problem}
        </p>
      ))}
    </div>
  )
}

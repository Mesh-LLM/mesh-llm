// Phase 2 — the expanded card's one timeline.
// Three aligned lanes over real time: their-chain spine (a single labeled
// band -- see the file-level note on why this isn't per-event marks),
// your exchanges two-sided (served above the line, requested below), and
// an adjudication lane directly under the requested points only. Recharts'
// bucketed `EventsOverTimeChart` doesn't compose here -- it aggregates
// counts into time buckets, but this view needs one addressable point per
// exchange at its own timestamp, so lanes render as plain positioned
// elements (percentage-of-range `left`) instead of a bar chart.
import type { PeerTimelinePoint } from '@/features/capsules/lib/peer-exchange-timeline'
import { theirChainSummary } from '@/features/capsules/lib/peer-row-view'
import type { PaneBRow } from '@/features/capsules/api/sidecarTypes'

export type PeerTimelineProps = {
  row: PaneBRow
  points: readonly PeerTimelinePoint[]
  onSelectPoint: (point: PeerTimelinePoint) => void
}

const RECONCILIATION_TONE: Record<PeerTimelinePoint['reconciliation'], 'good' | 'bad' | 'neutral'> = {
  verified: 'good',
  failed: 'bad',
  missing: 'neutral',
  unknown: 'neutral'
}

const VERDICT_LABEL: Record<string, string> = {
  corroborated: 'corroborated',
  contradicted: 'contradicted',
  inconclusive: 'inconclusive'
}

function toneClass(tone: 'good' | 'bad' | 'neutral' | 'warn'): string {
  if (tone === 'good') return 'bg-[var(--color-good)]'
  if (tone === 'bad') return 'bg-[var(--color-bad)]'
  if (tone === 'warn') return 'bg-[var(--color-warn)]'
  return 'bg-[var(--color-fg-faint)]'
}

function timeRange(points: readonly PeerTimelinePoint[]): { start: number; end: number } {
  const times = points
    .map((point) => (point.timestamp ? new Date(point.timestamp).getTime() : NaN))
    .filter((t) => !Number.isNaN(t))
  if (times.length === 0) return { start: 0, end: 1 }
  const start = Math.min(...times)
  const end = Math.max(...times)
  return { start, end: end > start ? end : start + 1 }
}

function leftPercent(point: PeerTimelinePoint, range: { start: number; end: number }): number {
  if (!point.timestamp) return 50
  const t = new Date(point.timestamp).getTime()
  return ((t - range.start) / (range.end - range.start)) * 100
}

export function PeerTimeline({ row, points, onSelectPoint }: PeerTimelineProps) {
  const chain = theirChainSummary(row)
  const range = timeRange(points)
  const requestedPoints = points.filter((point) => point.direction === 'requested')

  return (
    <div className="flex flex-col gap-3 border-t border-border-soft pt-3">
      {/* Lane 1: their-chain spine -- a single labeled band, honest about
         what it can and can't show (see file-level note). */}
      <div className="rounded border border-border-soft bg-panel-strong/50 px-3 py-2">
        <p className="text-xs font-medium text-fg-dim">Their log, as shown to you</p>
        <p className="mt-0.5 text-xs text-fg-faint">{chain.text}</p>
        <p className="mt-1 text-[11px] text-fg-faint">We never fetch what a peer holds about other peers.</p>
      </div>

      {points.length === 0 ? (
        <p className="text-xs text-fg-faint">No individually detailed exchanges on this view yet.</p>
      ) : (
        <>
          {/* Lane 2: your exchanges, two-sided -- served above the line,
             requested below. */}
          <div>
            <p className="mb-1 text-xs font-medium text-fg-dim">Your exchanges</p>
            <div className="relative h-16 rounded border border-border-soft bg-panel-strong/30">
              <div aria-hidden="true" className="absolute inset-x-0 top-1/2 h-px bg-border-soft" />
              {points.map((point) => (
                <button
                  aria-label={`${point.direction === 'served' ? 'Served' : 'Requested'} exchange ${point.exchangeId}, ${point.reconciliation}`}
                  className={`absolute size-2.5 -translate-x-1/2 rounded-full ${toneClass(RECONCILIATION_TONE[point.reconciliation])} ${
                    point.direction === 'served' ? 'top-2' : 'bottom-2'
                  }`}
                  key={point.exchangeId}
                  onClick={() => onSelectPoint(point)}
                  style={{ left: `${leftPercent(point, range)}%` }}
                  title={`${point.exchangeId} · ${point.direction} · ${point.reconciliation}`}
                  type="button"
                />
              ))}
            </div>
            <div className="mt-1 flex justify-between text-[10px] text-fg-faint">
              <span>↑ served-by-you</span>
              <span>↓ requested-by-you</span>
            </div>
          </div>

          {/* Lane 3: adjudication, aligned under requested points only --
             every requested point gets a marker, NOT_CHECKED included, so
             "no-verdict" is always visible rather than silently blank. */}
          <div>
            <p className="mb-1 text-xs font-medium text-fg-dim">Adjudication (requested only)</p>
            <div className="relative h-6 rounded border border-border-soft bg-panel-strong/30">
              {requestedPoints.map((point) => {
                const verdict = point.adjudication?.verdict
                const tone: 'good' | 'bad' | 'warn' | 'neutral' =
                  verdict === 'corroborated'
                    ? 'good'
                    : verdict === 'contradicted'
                      ? 'bad'
                      : verdict === 'inconclusive'
                        ? 'warn'
                        : 'neutral'
                return (
                  <button
                    aria-label={`Adjudication for ${point.exchangeId}: ${verdict ? VERDICT_LABEL[verdict] : 'not checked'}`}
                    className={`absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full ${
                      verdict ? toneClass(tone) : 'border border-fg-faint bg-transparent'
                    }`}
                    key={point.exchangeId}
                    onClick={() => onSelectPoint(point)}
                    style={{ left: `${leftPercent(point, range)}%` }}
                    title={
                      verdict ? `${point.exchangeId}: ${VERDICT_LABEL[verdict]}` : `${point.exchangeId}: not checked`
                    }
                    type="button"
                  />
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

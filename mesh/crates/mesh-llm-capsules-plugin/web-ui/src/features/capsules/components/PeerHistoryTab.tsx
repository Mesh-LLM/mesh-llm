// The peer drill's "Their history" tab: your
// dealings with them, their log as shown to you, what others say, verdicts
// on their exchanges by provenance, and what they asked of you. Rendering only -- every
// state and count comes from `their-history-view.ts`.
import type { ReactNode } from 'react'
import type { PaneBRow } from '@/features/capsules/api/sidecarTypes'
import { InfoHover } from '@/features/capsules/components/InfoHover'
import {
  askedOfYouView,
  dealingsText,
  othersSayView,
  PROVENANCE_LABELS,
  theirLogView,
  verdictRows,
  type ProvenanceKey
} from '@/features/capsules/lib/their-history-view'
import { PEER_HISTORY_TOOLTIPS, THEIR_LOG_TOOLTIPS } from '@/features/capsules/lib/tooltip-copy'

const PROVENANCES: ProvenanceKey[] = ['byYou', 'deliveredToThem', 'byOthers']

function Section({
  title,
  census,
  tooltip,
  children
}: {
  title: string
  census: keyof typeof PEER_HISTORY_TOOLTIPS
  tooltip: string
  children: ReactNode
}) {
  return (
    <section aria-label={title} className="flex flex-col gap-1.5">
      <h3 className="inline-flex items-center gap-1 text-xs font-medium text-foreground">
        {title}
        <InfoHover census={`peer_history:${census}`} describes={title} label={tooltip} />
      </h3>
      {children}
    </section>
  )
}

function TheirLog({ row }: { row: PaneBRow }) {
  const view = theirLogView(row)
  return (
    <Section census="theirLog" title="Their log, as shown to you" tooltip={THEIR_LOG_TOOLTIPS[view.kind]}>
      <p className="text-xs text-fg-dim">{view.text}</p>
      {view.kind === 'shown' && view.checkpoints.length > 0 ? (
        <ul className="flex flex-col gap-1" data-testid="their-log-checkpoints">
          {view.checkpoints.map((checkpoint) => (
            <li className="rounded border border-border-soft px-2.5 py-1.5 text-xs text-fg-dim" key={checkpoint.size}>
              <div className="flex flex-wrap items-center gap-1.5 text-fg-faint">
                <span className="font-mono">log size {checkpoint.size}</span>
                <span aria-hidden="true">·</span>
                <span>{checkpoint.time}</span>
                <span aria-hidden="true">·</span>
                <span>
                  {checkpoint.witnessEntries === 0
                    ? 'lists no witness'
                    : `lists ${checkpoint.witnessEntries} witness${checkpoint.witnessEntries === 1 ? '' : 'es'} (not checked here)`}
                </span>
              </div>
              <div className="mt-0.5">
                {checkpoint.counts.length === 0
                  ? 'nothing new in this checkpoint'
                  : checkpoint.counts.map((entry) => entry.text).join(' · ')}
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </Section>
  )
}

function Verdicts({ row }: { row: PaneBRow }) {
  const rows = verdictRows(row)
  return (
    <Section census="verdicts" title="Verdicts on their exchanges" tooltip={PEER_HISTORY_TOOLTIPS.verdicts}>
      <table aria-label="Verdicts on their exchanges, by who stands behind them" className="w-full text-left text-xs">
        <thead className="text-fg-faint">
          <tr>
            <th className="py-1 pr-2 font-normal">Verdict</th>
            {PROVENANCES.map((key) => (
              <th className="py-1 pr-2 font-normal" key={key} scope="col">
                {PROVENANCE_LABELS[key]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="text-fg-dim">
          {rows.map((verdictRow) => (
            <tr className="border-t border-border-soft" key={verdictRow.verdict}>
              <th className="py-1 pr-2 font-normal text-foreground" scope="row">
                {verdictRow.verdict}
              </th>
              {PROVENANCES.map((key) => {
                const cell = verdictRow[key]
                return (
                  <td className={`py-1 pr-2 ${cell.known ? '' : 'text-fg-faint'}`} key={key}>
                    {cell.text}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  )
}

function AskedOfYou({ row }: { row: PaneBRow }) {
  const view = askedOfYouView(row)
  return (
    <Section census="askedOfYou" title="Asked of you" tooltip={PEER_HISTORY_TOOLTIPS.askedOfYou}>
      <p className="text-xs text-fg-dim">{view.text}</p>
      {view.kind === 'shown' && view.lines.length > 0 ? (
        <table aria-label="Requests this peer made of you" className="w-full text-left text-xs">
          <thead className="text-fg-faint">
            <tr>
              <th className="py-1 pr-2 font-normal">When</th>
              <th className="py-1 pr-2 font-normal">Who, as they named themselves</th>
              <th className="py-1 pr-2 font-normal">Asked for</th>
              <th className="py-1 pr-2 font-normal">Outcome</th>
            </tr>
          </thead>
          <tbody className="text-fg-dim">
            {view.lines.map((line, index) => (
              <tr className="border-t border-border-soft" key={`${line.when}-${index}`}>
                <td className="py-1 pr-2">{line.when}</td>
                <td className="py-1 pr-2 font-mono">{line.who}</td>
                <td className="py-1 pr-2">{line.subject}</td>
                <td className="py-1 pr-2">
                  {line.outcome}
                  {line.reason ? <span className="text-fg-faint"> — {line.reason}</span> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </Section>
  )
}

export function PeerHistoryTab({ row }: { row: PaneBRow }) {
  const others = othersSayView(row)
  return (
    <div className="flex flex-col gap-4 text-sm">
      <Section census="dealings" title="Your dealings with them" tooltip={PEER_HISTORY_TOOLTIPS.dealings}>
        <p className="text-xs text-fg-dim">{dealingsText(row)}</p>
      </Section>
      <TheirLog row={row} />
      <Section census="othersSay" title="What others say" tooltip={PEER_HISTORY_TOOLTIPS.othersSay}>
        <p className="text-xs text-fg-dim">{others.text}</p>
      </Section>
      <Verdicts row={row} />
      <AskedOfYou row={row} />
    </div>
  )
}

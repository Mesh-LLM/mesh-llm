// The Peers drill's "Referee verdicts about them": the four counts side by
// side, each opening the list of verdict records it is made of, and each
// record opening the referee's signed verdict (`VerdictRecordDialog`).
import { useState } from 'react'
import type { PaneBRow, RefereeVerdictBucketKey } from '@/features/capsules/api/sidecarTypes'
import { VerdictRecordDialog } from '@/features/capsules/components/VerdictRecordDialog'
import { shortId } from '@/features/capsules/lib/adjudication-view'
import { refereeVerdictCountItems } from '@/features/capsules/lib/referee-verdict-counts'
import { REFEREE_VERDICTS_ABOUT_THEM } from '@/features/capsules/lib/tooltip-copy'

export function RefereeVerdictCounts({ row }: { row: PaneBRow }) {
  const [openBucket, setOpenBucket] = useState<RefereeVerdictBucketKey | null>(null)
  const [openRecord, setOpenRecord] = useState<string | null>(null)
  const items = refereeVerdictCountItems(row)
  const listed = items?.find((item) => item.key === openBucket) ?? null

  return (
    <section aria-label={REFEREE_VERDICTS_ABOUT_THEM.sectionTitle} className="flex flex-col gap-2">
      <h3 className="text-xs font-medium text-foreground">{REFEREE_VERDICTS_ABOUT_THEM.sectionTitle}</h3>
      {items === null ? (
        <p>{REFEREE_VERDICTS_ABOUT_THEM.notShown}</p>
      ) : (
        <>
          <p className="text-xs text-fg-faint">{REFEREE_VERDICTS_ABOUT_THEM.explainer}</p>
          <ul aria-label="Verdict counts" className="flex flex-wrap gap-1.5">
            {items.map((item) => (
              <li key={item.key}>
                <button
                  aria-expanded={openBucket === item.key}
                  className="rounded border border-border-soft px-2 py-1 text-xs text-fg-dim enabled:hover:bg-panel-strong disabled:cursor-default"
                  data-testid={`referee-verdicts-${item.key}`}
                  disabled={item.count === 0}
                  onClick={() => setOpenBucket(openBucket === item.key ? null : item.key)}
                  title={REFEREE_VERDICTS_ABOUT_THEM.bucketTooltip[item.key]}
                  type="button"
                >
                  {item.label} <span className="font-mono text-foreground">{item.count}</span>
                </button>
              </li>
            ))}
          </ul>
          {listed ? (
            <ul aria-label={`${listed.label} verdicts`} className="flex flex-col gap-1">
              {listed.verdictCapsuleIds.map((id) => (
                <li key={id}>
                  <button
                    className="font-mono text-xs text-fg-dim underline-offset-2 hover:underline"
                    onClick={() => setOpenRecord(id)}
                    type="button"
                  >
                    Verdict record {shortId(id)}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </>
      )}
      {openRecord ? (
        <VerdictRecordDialog
          capsuleId={openRecord}
          onOpenChange={(open) => {
            if (!open) setOpenRecord(null)
          }}
          open
        />
      ) : null}
    </section>
  )
}

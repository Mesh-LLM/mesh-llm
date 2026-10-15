// Phase 2 — per-exchange drill-down (the "Log
// Request Inspector" equivalent for a peer's exchange). Reuses the
// Logs feature's generic modal shell (`SharedModal*`) -- that chrome
// composes cleanly across features; the log-specific tab/detail content
// underneath it does not, so this inspector's body is its own.
import { StatusPill } from '@/components/ui/status-pill'
import {
  SharedModal,
  SharedModalBody,
  SharedModalContent,
  SharedModalDescription,
  SharedModalHeader,
  SharedModalTitle
} from '@/components/ui/SharedModal'
import { toneForState } from '@/features/capsules/lib/assurance-tone'
import type { PeerTimelinePoint } from '@/features/capsules/lib/peer-exchange-timeline'
import { formatExchangeTimestamp } from '@/features/capsules/lib/local-time'

export type PeerExchangeInspectorProps = {
  point: PeerTimelinePoint | null
  onClose: () => void
}

const VERDICT_STATE: Record<string, string> = {
  corroborated: 'verified',
  contradicted: 'failed',
  inconclusive: 'NOT_CHECKED'
}

export function PeerExchangeInspector({ point, onClose }: PeerExchangeInspectorProps) {
  return (
    <SharedModal onOpenChange={(open) => !open && onClose()} open={point !== null}>
      {point ? (
        <SharedModalContent className="max-w-lg">
          <SharedModalHeader>
            <SharedModalTitle className="font-mono">{point.exchangeId}</SharedModalTitle>
            <SharedModalDescription>
              {formatExchangeTimestamp(point.timestamp)} ·{' '}
              {point.direction === 'requested' ? 'you → them' : 'them → you'}
            </SharedModalDescription>
          </SharedModalHeader>
          <SharedModalBody className="flex flex-col gap-3 px-5 py-4 text-sm text-fg-dim">
            <div>
              <p className="text-xs font-medium text-fg-faint">Your capsule</p>
              <p className="font-mono text-xs">{point.mineCapsuleId ?? '—'}</p>
              <p className="text-[11px] text-fg-faint">recorded — digest only, not decoded on this view</p>
            </div>
            <div>
              <p className="text-xs font-medium text-fg-faint">Their answer (their capsule reference)</p>
              <p className="font-mono text-xs">{point.theirsCapsuleId ?? '—'}</p>
              <p className="text-[11px] text-fg-faint">digest only, not decoded on this view</p>
            </div>
            <div>
              <StatusPill
                dot
                label={`pair reconciliation: ${point.reconciliation}`}
                tone={toneForState(
                  point.reconciliation === 'verified'
                    ? 'verified'
                    : point.reconciliation === 'failed'
                      ? 'failed'
                      : 'NOT_CHECKED'
                )}
              />
            </div>

            {point.direction === 'requested' ? (
              <div className="rounded border border-border-soft bg-panel-strong/40 p-3">
                <p className="mb-1 text-xs font-medium text-fg-dim">Adjudication</p>
                {point.adjudication ? (
                  <div className="flex flex-col gap-1">
                    <StatusPill
                      dot
                      label={point.adjudication.verdict}
                      tone={toneForState(VERDICT_STATE[point.adjudication.verdict] ?? 'NOT_CHECKED')}
                    />
                    <p className="text-xs">margin_tau: {point.adjudication.marginTau ?? 'not recorded'}</p>
                    <p className="text-xs">
                      referee: {point.adjudication.refereeId ?? 'no independent referee (local twin only)'}
                    </p>
                    <p className="font-mono text-[11px] text-fg-faint">{point.adjudication.adjudicationCapsuleId}</p>
                  </div>
                ) : (
                  <p className="text-xs">
                    <StatusPill dot label="NOT_CHECKED" tone={toneForState('NOT_CHECKED')} /> — no adjudication sealed
                    for this exchange. A no-verdict, a referee-unreachable case, and an owner-absent case all look
                    identical from this ledger, so none of them are guessed at.
                  </p>
                )}
              </div>
            ) : (
              <p className="text-xs text-fg-faint">
                They requested this one — adjudication only applies to exchanges you requested.
              </p>
            )}
          </SharedModalBody>
        </SharedModalContent>
      ) : null}
    </SharedModal>
  )
}

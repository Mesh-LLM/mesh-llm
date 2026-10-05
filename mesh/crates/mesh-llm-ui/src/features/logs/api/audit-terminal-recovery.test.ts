import { describe, expect, it } from 'vitest'
import {
  beginAuditTerminalRecovery,
  completeAuditTerminalEof,
  completeAuditTerminalHydration,
  type AuditTerminalRecovery
} from './audit-terminal-recovery'

type AuditSource = { readonly id: string }
type PendingHydrationPhase = 'hydrating' | 'awaiting_hydration'
type SettledHydrationPhase = 'awaiting_eof' | 'failed' | 'replaced'
type TerminalRecoveryPhase = 'failed' | 'replaced'

const ORIGINAL_SOURCE: AuditSource = { id: 'original' }
const REPLACEMENT_SOURCE: AuditSource = { id: 'replacement' }
const PENDING_HYDRATION_PHASES: readonly PendingHydrationPhase[] = ['hydrating', 'awaiting_hydration']
const SETTLED_HYDRATION_PHASES: readonly SettledHydrationPhase[] = ['awaiting_eof', 'failed', 'replaced']
const TERMINAL_RECOVERY_PHASES: readonly TerminalRecoveryPhase[] = ['failed', 'replaced']
const RECOVERIES: readonly AuditTerminalRecovery<AuditSource>[] = [
  { phase: 'hydrating', source: ORIGINAL_SOURCE },
  { phase: 'awaiting_eof', source: ORIGINAL_SOURCE },
  { phase: 'awaiting_hydration', source: ORIGINAL_SOURCE },
  { phase: 'failed', source: ORIGINAL_SOURCE },
  { phase: 'replaced', source: ORIGINAL_SOURCE }
]

describe('beginAuditTerminalRecovery', () => {
  it('starts hydration when no terminal recovery exists', () => {
    const transition = beginAuditTerminalRecovery(undefined, ORIGINAL_SOURCE)

    expect(transition).toEqual({
      recovery: { phase: 'hydrating', source: ORIGINAL_SOURCE },
      shouldReconnect: false,
      shouldMarkStale: false
    })
  })

  it('starts hydration for a different source', () => {
    const transition = beginAuditTerminalRecovery(
      { phase: 'awaiting_eof', source: ORIGINAL_SOURCE },
      REPLACEMENT_SOURCE
    )

    expect(transition).toEqual({
      recovery: { phase: 'hydrating', source: REPLACEMENT_SOURCE },
      shouldReconnect: false,
      shouldMarkStale: false
    })
  })

  it.each(RECOVERIES)('ignores a duplicate begin for the same source from $phase', (recovery) => {
    const transition = beginAuditTerminalRecovery(recovery, ORIGINAL_SOURCE)

    expect(transition).toEqual({ recovery, shouldReconnect: false, shouldMarkStale: false })
    expect(transition.recovery).toBe(recovery)
  })
})

describe('completeAuditTerminalHydration', () => {
  it('waits for EOF when hydration succeeds first', () => {
    const transition = completeAuditTerminalHydration({ phase: 'hydrating', source: ORIGINAL_SOURCE }, true)

    expect(transition).toEqual({
      recovery: { phase: 'awaiting_eof', source: ORIGINAL_SOURCE },
      shouldReconnect: false,
      shouldMarkStale: false
    })
  })

  it('replaces the source when hydration succeeds after EOF', () => {
    const transition = completeAuditTerminalHydration({ phase: 'awaiting_hydration', source: ORIGINAL_SOURCE }, true)

    expect(transition).toEqual({
      recovery: { phase: 'replaced', source: ORIGINAL_SOURCE },
      shouldReconnect: true,
      shouldMarkStale: false
    })
  })

  it('preserves an awaiting EOF recovery after duplicate hydration success', () => {
    const recovery: AuditTerminalRecovery<AuditSource> = { phase: 'awaiting_eof', source: ORIGINAL_SOURCE }

    const transition = completeAuditTerminalHydration(recovery, true)

    expect(transition).toEqual({ recovery, shouldReconnect: false, shouldMarkStale: false })
    expect(transition.recovery).toBe(recovery)
  })

  it.each(PENDING_HYDRATION_PHASES)('marks hydration failure stale from %s', (phase) => {
    const recovery: AuditTerminalRecovery<AuditSource> = { phase, source: ORIGINAL_SOURCE }

    const transition = completeAuditTerminalHydration(recovery, false)

    expect(transition).toEqual({
      recovery: { phase: 'failed', source: ORIGINAL_SOURCE },
      shouldReconnect: false,
      shouldMarkStale: true
    })
  })

  it.each(SETTLED_HYDRATION_PHASES)('ignores late hydration failure from %s', (phase) => {
    const recovery: AuditTerminalRecovery<AuditSource> = { phase, source: ORIGINAL_SOURCE }

    const transition = completeAuditTerminalHydration(recovery, false)

    expect(transition).toEqual({ recovery, shouldReconnect: false, shouldMarkStale: false })
    expect(transition.recovery).toBe(recovery)
  })

  it.each(TERMINAL_RECOVERY_PHASES)('ignores late hydration success from %s', (phase) => {
    const recovery: AuditTerminalRecovery<AuditSource> = { phase, source: ORIGINAL_SOURCE }

    const transition = completeAuditTerminalHydration(recovery, true)

    expect(transition).toEqual({ recovery, shouldReconnect: false, shouldMarkStale: false })
    expect(transition.recovery).toBe(recovery)
  })
})

describe('completeAuditTerminalEof', () => {
  it('waits for hydration when EOF arrives first', () => {
    const transition = completeAuditTerminalEof({ phase: 'hydrating', source: ORIGINAL_SOURCE })

    expect(transition).toEqual({
      recovery: { phase: 'awaiting_hydration', source: ORIGINAL_SOURCE },
      shouldReconnect: false,
      shouldMarkStale: false
    })
  })

  it('replaces the source when EOF arrives after hydration', () => {
    const transition = completeAuditTerminalEof({ phase: 'awaiting_eof', source: ORIGINAL_SOURCE })

    expect(transition).toEqual({
      recovery: { phase: 'replaced', source: ORIGINAL_SOURCE },
      shouldReconnect: true,
      shouldMarkStale: false
    })
  })

  it('preserves recovery when EOF repeats while hydration is pending', () => {
    const recovery: AuditTerminalRecovery<AuditSource> = {
      phase: 'awaiting_hydration',
      source: ORIGINAL_SOURCE
    }

    const transition = completeAuditTerminalEof(recovery)

    expect(transition).toEqual({ recovery, shouldReconnect: false, shouldMarkStale: false })
    expect(transition.recovery).toBe(recovery)
  })

  it('keeps a failed recovery stale after late EOF', () => {
    const recovery: AuditTerminalRecovery<AuditSource> = { phase: 'failed', source: ORIGINAL_SOURCE }

    const transition = completeAuditTerminalEof(recovery)

    expect(transition).toEqual({ recovery, shouldReconnect: false, shouldMarkStale: true })
    expect(transition.recovery).toBe(recovery)
  })

  it('ignores late EOF after the source was replaced', () => {
    const recovery: AuditTerminalRecovery<AuditSource> = { phase: 'replaced', source: ORIGINAL_SOURCE }

    const transition = completeAuditTerminalEof(recovery)

    expect(transition).toEqual({ recovery, shouldReconnect: false, shouldMarkStale: false })
    expect(transition.recovery).toBe(recovery)
  })
})

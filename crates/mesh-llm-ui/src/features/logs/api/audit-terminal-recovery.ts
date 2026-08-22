type AuditTerminalRecoveryPhase = 'hydrating' | 'awaiting_eof' | 'awaiting_hydration' | 'failed' | 'replaced'

export type AuditTerminalRecovery<Source> = {
  readonly phase: AuditTerminalRecoveryPhase
  readonly source: Source
}

export type AuditTerminalRecoveryTransition<Source> = {
  readonly recovery: AuditTerminalRecovery<Source>
  readonly shouldReconnect: boolean
  readonly shouldMarkStale: boolean
}

function remain<Source>(
  recovery: AuditTerminalRecovery<Source>,
  shouldMarkStale = false
): AuditTerminalRecoveryTransition<Source> {
  return { recovery, shouldReconnect: false, shouldMarkStale }
}

function advance<Source>(
  recovery: AuditTerminalRecovery<Source>,
  phase: AuditTerminalRecoveryPhase,
  shouldReconnect = false
): AuditTerminalRecoveryTransition<Source> {
  return {
    recovery: { phase, source: recovery.source },
    shouldReconnect,
    shouldMarkStale: false
  }
}

function fail<Source>(recovery: AuditTerminalRecovery<Source>): AuditTerminalRecoveryTransition<Source> {
  return { recovery: { phase: 'failed', source: recovery.source }, shouldReconnect: false, shouldMarkStale: true }
}

function unreachablePhase(phase: never): never {
  throw new RangeError(`Unknown audit terminal recovery phase: ${phase}`)
}

export function beginAuditTerminalRecovery<Source>(
  recovery: AuditTerminalRecovery<Source> | undefined,
  source: Source
): AuditTerminalRecoveryTransition<Source> {
  if (recovery?.source === source) return remain(recovery)
  return {
    recovery: { phase: 'hydrating', source },
    shouldReconnect: false,
    shouldMarkStale: false
  }
}

export function completeAuditTerminalHydration<Source>(
  recovery: AuditTerminalRecovery<Source>,
  succeeded: boolean
): AuditTerminalRecoveryTransition<Source> {
  const phase = recovery.phase
  switch (phase) {
    case 'hydrating':
      return succeeded ? advance(recovery, 'awaiting_eof') : fail(recovery)
    case 'awaiting_hydration':
      return succeeded ? advance(recovery, 'replaced', true) : fail(recovery)
    case 'awaiting_eof':
    case 'failed':
    case 'replaced':
      return remain(recovery)
    default:
      return unreachablePhase(phase)
  }
}

export function completeAuditTerminalEof<Source>(
  recovery: AuditTerminalRecovery<Source>
): AuditTerminalRecoveryTransition<Source> {
  const phase = recovery.phase
  switch (phase) {
    case 'hydrating':
      return advance(recovery, 'awaiting_hydration')
    case 'awaiting_eof':
      return advance(recovery, 'replaced', true)
    case 'failed':
      return remain(recovery, true)
    case 'awaiting_hydration':
    case 'replaced':
      return remain(recovery)
    default:
      return unreachablePhase(phase)
  }
}

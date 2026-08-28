import * as v from 'valibot'
import { BOOLEAN_SUMMARY_TOKENS, REDACTED_SUMMARY_TOKENS, STATIC_SUMMARY_TOKENS } from './command-summary-vocabulary'
import { SUMMARY_DESCRIPTORS, type SummaryDescriptor } from './command-summary-descriptors'

const CONTROL_CHARACTER = /\p{Cc}/u

function hasValidPort(value: string): boolean {
  if (!/^[0-9]+$/.test(value)) return false
  const port = Number(value)
  return port >= 0 && port <= 65535
}

const BACKEND_VALUES = ['metal', 'cuda', 'hip', 'intel'] as const
const MODE_VALUES = ['disabled', 'metrics', 'enforce'] as const
const GLOBAL_REDACTED_TOKENS = ['--join', '--root-relay', '--relay-auth'] as const

function isBackendValue(value: string): boolean {
  return BACKEND_VALUES.some((candidate) => candidate === value)
}

function isModeValue(value: string): boolean {
  return MODE_VALUES.some((candidate) => candidate === value)
}

function isAllowedRedactedToken(descriptor: SummaryDescriptor, token: string): boolean {
  return descriptor.redacted.includes(token) || GLOBAL_REDACTED_TOKENS.some((candidate) => candidate === token)
}

function matchesPath(tokens: readonly string[], descriptor: SummaryDescriptor): boolean {
  return (
    descriptor.path.length <= tokens.length &&
    descriptor.path.every((token, index) => {
      return STATIC_SUMMARY_TOKENS.has(token) && tokens[index] === token
    })
  )
}

function hasDuplicate(seen: readonly string[], token: string): boolean {
  return seen.includes(token)
}

function validateDescriptor(tokens: readonly string[], descriptor: SummaryDescriptor): boolean {
  if (!matchesPath(tokens, descriptor)) return false

  let index = descriptor.path.length
  if (descriptor.raw === 'backend' || descriptor.raw === 'mode') {
    const rawOption = descriptor.raw === 'backend' ? '--backend' : '--mode'
    const rawValue = tokens[index + 1]
    if (tokens[index] !== rawOption || rawValue === undefined) return false
    if (descriptor.raw === 'backend' && !isBackendValue(rawValue)) return false
    if (descriptor.raw === 'mode' && !isModeValue(rawValue)) return false
    index += 2
  }

  let phase: 'booleans' | 'port' | 'redacted' = 'booleans'
  let portSeen = false
  const seenBooleans: string[] = []
  const seenRedacted: string[] = []
  const seenTokens: string[] = []
  let globalPhase = false
  let lastGlobalRank = 0
  while (index < tokens.length) {
    const token = tokens[index]
    if (token === undefined) return false
    if (BOOLEAN_SUMMARY_TOKENS.has(token)) {
      if (phase !== 'booleans' || !descriptor.booleans.includes(token) || hasDuplicate(seenBooleans, token))
        return false
      seenBooleans.push(token)
      seenTokens.push(token)
      index += 1
      continue
    }
    if (token === '--port') {
      if (phase === 'redacted' || portSeen || !descriptor.hasPort) return false
      const rawValue = tokens[index + 1]
      if (rawValue === undefined || !hasValidPort(rawValue)) return false
      portSeen = true
      phase = 'port'
      index += 2
      continue
    }
    if (REDACTED_SUMMARY_TOKENS.has(token)) {
      const isGlobal = GLOBAL_REDACTED_TOKENS.some((candidate) => candidate === token)
      if (globalPhase && !isGlobal) return false
      if (isGlobal) {
        const globalRank = GLOBAL_REDACTED_TOKENS.findIndex((candidate) => candidate === token) + 1
        if (globalRank <= lastGlobalRank) return false
        globalPhase = true
        lastGlobalRank = globalRank
      }
      if (!isAllowedRedactedToken(descriptor, token) || hasDuplicate(seenRedacted, token)) return false
      if (tokens[index + 1] !== '[REDACTED]') return false
      seenRedacted.push(token)
      seenTokens.push(token)
      phase = 'redacted'
      index += 2
      continue
    }
    return false
  }
  return !descriptor.conflicts.some((pair) => pair.every((flag) => seenTokens.includes(flag)))
}

export function isSafeCommandSummary(value: string): boolean {
  const tokens = value.split(' ')
  if (
    Array.from(value).length === 0 ||
    Array.from(value).length > 256 ||
    CONTROL_CHARACTER.test(value) ||
    tokens.some((token) => token.length === 0 || /\s/u.test(token))
  )
    return false
  return tokens.length <= 32 && SUMMARY_DESCRIPTORS.some((descriptor) => validateDescriptor(tokens, descriptor))
}

export const commandSummarySchema = v.pipe(v.string(), v.minLength(1), v.maxLength(256), v.check(isSafeCommandSummary))

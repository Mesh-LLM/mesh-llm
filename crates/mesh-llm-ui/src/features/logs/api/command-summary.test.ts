import { describe, expect, it } from 'vitest'
import { safeParse } from 'valibot'
import { commandSummarySchema, isSafeCommandSummary } from './command-summary'
import { SUMMARY_DESCRIPTORS } from './command-summary-descriptors'

describe('command summary grammar', () => {
  it('accepts every parsed producer raw-option context', () => {
    const validSummaries = [
      'mesh-llm status --port 41731',
      'mesh-llm load --port 41731 name [REDACTED]',
      'mesh-llm goose --port 41731 --model [REDACTED]',
      'mesh-llm doctor split --json --port 41731 --model-ref [REDACTED]',
      'mesh-llm gpus run-benchmark --backend cuda --json',
      'mesh-llm runtime guardrails --mode metrics --json --port 41731',
      'mesh-llm runtime bootstrap --json --port 41731',
      'mesh-llm runtime remote --json --port 41731 --endpoint [REDACTED]',
      'mesh-llm runtime remote-model --json --port 41731 --endpoint [REDACTED] --model [REDACTED]',
      'mesh-llm runtime apply-config --json --port 41731 --endpoint [REDACTED] --expected-revision [REDACTED] --config [REDACTED]'
    ]

    for (const summary of validSummaries) {
      expect(isSafeCommandSummary(summary), summary).toBe(true)
      expect(safeParse(commandSummarySchema, summary).success, summary).toBe(true)
    }
  })

  it('accepts ordinary static and redacted summaries', () => {
    expect(isSafeCommandSummary('mesh-llm runtime load name [REDACTED]')).toBe(true)
    expect(isSafeCommandSummary('mesh-llm models list --json')).toBe(true)
  })

  it('rejects private values, controls, bounds, and deep malformed prefixes', () => {
    const malformedSummaries = [
      'mesh-llm load private-value',
      `mesh-llm load\u0001name [REDACTED]`,
      `mesh-llm ${new Array<string>(32).fill('load').join(' ')}`,
      'mesh-llm load unload status discover rotate-key setup --port 1234',
      'mesh-llm gpus run-benchmark --backend rocm',
      'mesh-llm runtime guardrails --mode strict',
      'mesh-llm load name [REDACTED] --port nope'
    ]

    for (const summary of malformedSummaries) {
      expect(isSafeCommandSummary(summary), summary).toBe(false)
      expect(safeParse(commandSummarySchema, summary).success, summary).toBe(false)
    }
  })

  it('rejects inserted safe tokens and impossible raw-option ordering', () => {
    const malformedSummaries = [
      'mesh-llm gpus --draft run-benchmark --backend cuda',
      'mesh-llm gpus run-benchmark model [REDACTED] --backend cuda',
      'mesh-llm gpus --json run-benchmark --backend cuda',
      'mesh-llm doctor --json split --port 41731',
      'mesh-llm runtime guardrails --mode metrics --port 41731 --json',
      'mesh-llm runtime bootstrap --port 41731 --json',
      'mesh-llm runtime remote --port 41731 --json --endpoint [REDACTED]'
    ]

    for (const summary of malformedSummaries) {
      expect(isSafeCommandSummary(summary), summary).toBe(false)
    }
  })

  it('rejects non-canonical whole-command shapes', () => {
    const malformedSummaries = [
      '   ',
      'mesh-llm models list --json --json',
      'mesh-llm models --json list',
      'mesh-llm load name [REDACTED] name [REDACTED]',
      'mesh-llm gpus run-benchmark --backend cuda --json --json',
      'mesh-llm load --port 41731 --port 41732',
      'mesh-llm load --port 41731 name [REDACTED] --json',
      'mesh-llm load name [REDACTED] status',
      'mesh-llm models nonsense',
      'mesh-llm load --json name [REDACTED]',
      'mesh-llm runtime status name [REDACTED]',
      'mesh-llm models list name [REDACTED] --json'
    ]

    for (const summary of malformedSummaries) {
      expect(isSafeCommandSummary(summary), summary).toBe(false)
      expect(safeParse(commandSummarySchema, summary).success, summary).toBe(false)
    }
  })

  it('rejects non-canonical ASCII whitespace', () => {
    const malformedSummaries = [
      ' mesh-llm models list',
      'mesh-llm models list ',
      'mesh-llm  models list',
      'mesh-llm\tmodels list',
      'mesh-llm models\nlist'
    ]

    for (const summary of malformedSummaries) {
      expect(isSafeCommandSummary(summary), summary).toBe(false)
    }
  })

  it('rejects conflicting boolean pairs', () => {
    const malformedSummaries = [
      'mesh-llm setup --service --no-service',
      'mesh-llm setup --no-service --service',
      'mesh-llm uninstall --purge-config --keep-config',
      'mesh-llm uninstall --keep-config --purge-config',
      'mesh-llm auth init --no-passphrase --keychain',
      'mesh-llm auth init --keychain --no-passphrase'
    ]

    for (const summary of malformedSummaries) {
      expect(isSafeCommandSummary(summary), summary).toBe(false)
    }
  })

  it('rejects each speculative option when benchmark speculative tuning is disabled', () => {
    const speculativeOptions = [
      '--speculative-types',
      '--spec-draft-models',
      '--spec-draft-max-tokens',
      '--spec-draft-min-tokens',
      '--spec-draft-acceptance-threshold',
      '--spec-draft-split-probability',
      '--spec-ngram-min',
      '--spec-ngram-max'
    ] as const

    for (const option of speculativeOptions) {
      const summary = `mesh-llm benchmark tune --no-speculative-tune ${option} [REDACTED]`
      expect(isSafeCommandSummary(summary), option).toBe(false)
    }
  })

  it('accepts only ASCII decimal u16 port values', () => {
    for (const port of ['0', '1', '65535']) {
      expect(isSafeCommandSummary(`mesh-llm status --port ${port}`), port).toBe(true)
    }

    for (const port of ['+1', '-1', '65536', '1.0', '١']) {
      expect(isSafeCommandSummary(`mesh-llm status --port ${port}`), port).toBe(false)
    }
  })

  it('accepts only the redacted global relay suffix shape', () => {
    expect(isSafeCommandSummary('mesh-llm load name [REDACTED] --root-relay [REDACTED]')).toBe(true)
    for (const summary of [
      'mesh-llm load name [REDACTED] --relay private-relay',
      'mesh-llm load name [REDACTED] --root-relay [REDACTED] value',
      'mesh-llm load name [REDACTED] --relay-auth private-token',
      'mesh-llm load --root-relay [REDACTED] name [REDACTED]',
      'mesh-llm load name [REDACTED] --relay-auth [REDACTED] --root-relay [REDACTED]'
    ]) {
      expect(isSafeCommandSummary(summary), summary).toBe(false)
    }
  })

  it('accepts every descriptor option set in canonical phase order', () => {
    for (const descriptor of SUMMARY_DESCRIPTORS) {
      const tokens = [...descriptor.path]
      if (descriptor.raw === 'backend') tokens.push('--backend', 'cuda')
      if (descriptor.raw === 'mode') tokens.push('--mode', 'metrics')
      tokens.push(...descriptor.booleans)
      if (descriptor.hasPort) tokens.push('--port', '41731')
      for (const marker of descriptor.redacted) tokens.push(marker, '[REDACTED]')
      const summary = tokens.join(' ')
      if (descriptor.conflicts.some((pair) => pair.every((flag) => tokens.includes(flag)))) {
        expect(isSafeCommandSummary(summary), summary).toBe(false)
        continue
      }
      if (tokens.length <= 32 && Array.from(summary).length <= 256) {
        expect(isSafeCommandSummary(summary), summary).toBe(true)
      } else {
        for (const marker of [...descriptor.booleans, ...descriptor.redacted]) {
          const single = [...descriptor.path]
          if (descriptor.raw === 'backend') single.push('--backend', 'cuda')
          if (descriptor.raw === 'mode') single.push('--mode', 'metrics')
          single.push(marker)
          if (descriptor.redacted.includes(marker)) single.push('[REDACTED]')
          expect(isSafeCommandSummary(single.join(' ')), marker).toBe(true)
        }
      }
    }
  })
})

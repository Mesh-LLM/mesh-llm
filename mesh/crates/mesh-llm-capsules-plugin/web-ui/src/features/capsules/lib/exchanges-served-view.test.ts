import { describe, expect, it } from 'vitest'
import { servedAskedCoverage, ASKED_ABSENT_TEXT } from '@/features/capsules/lib/exchanges-served-view'

describe('servedAskedCoverage', () => {
  it('is absent for a null card, never a crash or an invented number', () => {
    const result = servedAskedCoverage(null)
    expect(result.kind).toBe('absent')
  })

  it('is absent when the served_summary block is missing', () => {
    const result = servedAskedCoverage({})
    expect(result.kind).toBe('absent')
  })

  it('is absent (not verified) when the block reports NOT_CHECKED / no witnessed checkpoint', () => {
    const result = servedAskedCoverage({
      served_summary: {
        state: 'NOT_CHECKED',
        text: 'no witnessed checkpoint yet -- nothing served falls inside a witnessed range',
        source: 'self_derived'
      }
    })
    expect(result.kind).toBe('absent')
  })

  it('is failed, never silently verified, when the summary fails its own recompute+match', () => {
    const result = servedAskedCoverage({
      served_summary: {
        state: 'failed',
        text: 'served summary failed its own recompute+match: mismatch',
        source: 'self_derived',
        served_summary: { derivation: { by_model: {} } }
      }
    })
    expect(result.kind).toBe('failed')
    if (result.kind === 'failed') {
      expect(result.headline).toMatch(/recompute\+match/)
    }
  })

  it('is verified with a full coverage statement when witnessed', () => {
    const result = servedAskedCoverage({
      served_summary: {
        state: 'verified',
        text: 'llama-2-7b: 12 served',
        source: 'self_derived',
        served_summary: {
          selection: { from_entry: 1, to_entry: 42, covered_entries: 42 },
          coverage: {
            checkpoint_root: 'abcdef0123456789',
            mmr_size: 64,
            witnesses: ['https://witness-a.example', 'https://witness-b.example'],
            witnessed: true
          }
        }
      }
    })
    expect(result.kind).toBe('verified')
    if (result.kind === 'verified') {
      expect(result.servedText).toBe('llama-2-7b: 12 served')
      expect(result.askedText).toBe(ASKED_ABSENT_TEXT)
      expect(result.statement).toMatch(/range-complete through entry 42/)
      expect(result.statement).toMatch(/reconciled against 2 witnesses/)
      expect(result.statement).not.toMatch(/not reconciled/)
    }
  })

  it('never fabricates a witness count when not reconciled', () => {
    const result = servedAskedCoverage({
      served_summary: {
        state: 'verified',
        text: '0 exchanges served in the witnessed range',
        source: 'self_derived',
        served_summary: {
          selection: { from_entry: 0, to_entry: 0, covered_entries: 0 },
          coverage: { checkpoint_root: 'abc', mmr_size: 1, witnesses: [], witnessed: false }
        }
      }
    })
    expect(result.kind).toBe('verified')
    if (result.kind === 'verified') {
      expect(result.statement).toMatch(/not reconciled/)
      expect(result.statement).not.toMatch(/reconciled against/)
    }
  })

  it('never reports a property count as a score', () => {
    const result = servedAskedCoverage({
      served_summary: {
        state: 'verified',
        text: 'llama-2-7b: 12 served',
        source: 'self_derived',
        served_summary: {
          selection: { covered_entries: 12 },
          coverage: { checkpoint_root: 'abc', mmr_size: 8, witnesses: ['w1'], witnessed: true }
        }
      }
    })
    expect(result.kind).toBe('verified')
    if (result.kind === 'verified') {
      expect(result.servedText + result.askedText + result.statement).not.toMatch(/\d+\/\d+/)
    }
  })
})

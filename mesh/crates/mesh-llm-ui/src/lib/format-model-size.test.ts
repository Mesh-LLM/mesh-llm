import { describe, expect, it } from 'vitest'
import { formatModelSizeGB } from '@/lib/format-model-size'

describe('formatModelSizeGB', () => {
  it('rounds long fractional sizes to one decimal place', () => {
    expect(formatModelSizeGB(2.912109728)).toBe('2.9 GB')
  })

  it('keeps integer values whole', () => {
    expect(formatModelSizeGB(4)).toBe('4 GB')
  })

  it('drops the trailing decimal when rounding lands on a whole number', () => {
    expect(formatModelSizeGB(2.95)).toBe('3 GB')
    expect(formatModelSizeGB(3.04)).toBe('3 GB')
  })

  it('keeps sub-gigabyte model sizes in decimal GB', () => {
    expect(formatModelSizeGB(0.6)).toBe('0.6 GB')
  })

  it('returns Unknown for missing or non-positive values', () => {
    expect(formatModelSizeGB(undefined)).toBe('Unknown')
    expect(formatModelSizeGB(null)).toBe('Unknown')
    expect(formatModelSizeGB(0)).toBe('Unknown')
    expect(formatModelSizeGB(-1)).toBe('Unknown')
    expect(formatModelSizeGB(Number.NaN)).toBe('Unknown')
    expect(formatModelSizeGB(Number.POSITIVE_INFINITY)).toBe('Unknown')
  })
})

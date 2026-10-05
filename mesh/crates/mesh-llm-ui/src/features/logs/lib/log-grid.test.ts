import { describe, expect, it } from 'vitest'
import { trailingRowSpanClass } from '@/features/logs/lib/log-grid'

describe('trailingRowSpanClass', () => {
  it('leaves full rows untouched', () => {
    expect(trailingRowSpanClass(6, 0, 2, 'sm')).toBeUndefined()
    expect(trailingRowSpanClass(6, 5, 2, 'sm')).toBeUndefined()
    expect(trailingRowSpanClass(9, 0, 3, 'xl')).toBeUndefined()
    expect(trailingRowSpanClass(9, 8, 3, 'xl')).toBeUndefined()
    expect(trailingRowSpanClass(8, 7, 4, 'lg')).toBeUndefined()
  })

  it('spans the trailing cell across the whole final row when one cell remains', () => {
    expect(trailingRowSpanClass(7, 6, 2, 'sm')).toBe('sm:col-span-2')
    expect(trailingRowSpanClass(7, 6, 3, 'xl')).toBe('xl:col-span-3')
    expect(trailingRowSpanClass(5, 4, 4, 'lg')).toBe('lg:col-span-4')
  })

  it('splits extra columns across the trailing row cells', () => {
    // 7 cells in a 3-column grid: rows of 3 + trailing row of 1 spans 3.
    expect(trailingRowSpanClass(7, 6, 3, 'xl')).toBe('xl:col-span-3')
    // 8 cells in a 3-column grid: trailing row of 2 → spans 2 and 1 (natural).
    expect(trailingRowSpanClass(8, 6, 3, 'xl')).toBe('xl:col-span-2')
    expect(trailingRowSpanClass(8, 7, 3, 'xl')).toBeUndefined()
    // 6 cells in a 4-column grid: trailing row of 2 → each spans 2.
    expect(trailingRowSpanClass(6, 4, 4, 'lg')).toBe('lg:col-span-2')
    expect(trailingRowSpanClass(6, 5, 4, 'lg')).toBe('lg:col-span-2')
  })

  it('returns undefined for out-of-range indexes and malformed inputs', () => {
    expect(trailingRowSpanClass(7, 7, 2, 'sm')).toBeUndefined()
    expect(trailingRowSpanClass(7, -1, 2, 'sm')).toBeUndefined()
    expect(trailingRowSpanClass(7, 0, 1, 'sm')).toBeUndefined()
    expect(trailingRowSpanClass(6, 5, 6, 'xl')).toBeUndefined()
    expect(trailingRowSpanClass(0, 0, 2, 'sm')).toBeUndefined()
  })

  it('keeps the earlier trailing cells at natural width when only the last needs a span', () => {
    // 9 cells in a 2-column grid: trailing row of 1 → only index 8 spans.
    expect(trailingRowSpanClass(9, 8, 2, 'sm')).toBe('sm:col-span-2')
    expect(trailingRowSpanClass(9, 7, 2, 'sm')).toBeUndefined()
    // 10 cells in a 4-column grid: trailing row of 2 → both span 2? No: 4/2 → 2+2.
    expect(trailingRowSpanClass(10, 8, 4, 'lg')).toBe('lg:col-span-2')
    expect(trailingRowSpanClass(10, 9, 4, 'lg')).toBe('lg:col-span-2')
  })
})

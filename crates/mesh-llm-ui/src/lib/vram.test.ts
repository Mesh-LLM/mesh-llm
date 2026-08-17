import { describe, expect, it } from 'vitest'

import {
  allocatableVramBytes,
  formatRatedVramBytes,
  gpuAllocatableVramGB,
  gpuRatedVramGB,
  gpuReservedVramGB,
  gpuSystemReportedVramGB,
  ratedVramGBFromBytes
} from '@/lib/vram'

describe('VRAM accounting utilities', () => {
  it('maps binary-sized driver totals to the user-facing rated class', () => {
    expect(ratedVramGBFromBytes(32 * 1024 ** 3)).toBe(32)
    expect(formatRatedVramBytes(32 * 1024 ** 3)).toBe('32 GB')
  })

  it('preserves decimal-sized totals when the source already reports rated bytes', () => {
    expect(ratedVramGBFromBytes(24_000_000_000)).toBe(24)
  })

  it('maps near-decimal reported totals to the rated class', () => {
    expect(ratedVramGBFromBytes(32_359_738_368)).toBe(32)
  })

  it('keeps system-reported and rated capacity separate for calculations', () => {
    const gpu = {
      name: 'RTX 5090',
      vram_bytes: 32 * 1024 ** 3,
      reserved_bytes: 512 * 1024 ** 2
    }

    expect(gpuRatedVramGB(gpu)).toBe(32)
    expect(gpuSystemReportedVramGB(gpu)).toBeCloseTo(34.36, 2)
    expect(gpuReservedVramGB(gpu)).toBeCloseTo(0.54, 2)
    expect(gpuAllocatableVramGB(gpu)).toBeCloseTo(33.82, 2)
  })

  it('subtracts reserved memory from allocatable bytes with saturation', () => {
    expect(allocatableVramBytes(1_000, 400)).toBe(600)
    expect(allocatableVramBytes(1_000, 1_400)).toBe(0)
  })
})

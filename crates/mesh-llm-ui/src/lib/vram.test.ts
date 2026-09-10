import { describe, expect, it } from 'vitest'

import {
  allocatableVramBytes,
  formatRatedVramBytes,
  gpuAllocatableVramGB,
  gpuRatedVramGB,
  gpuReservedVramGB,
  gpuSystemReportedVramGB,
  meshAdvertisedVramGB,
  meshRatedVramGB,
  nodeAdvertisedVramGB,
  nodeRatedVramGB,
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

  it('prefers the advertised capacity a node announces to the mesh over its GPU inventory', () => {
    const node = {
      vram_gb: 44.02970624,
      gpus: [
        { rated_vram_gb: 32, vram_bytes: 34_190_917_632, reserved_bytes: 514_850_816 },
        { rated_vram_gb: 10, vram_bytes: 10_737_418_240, reserved_bytes: 383_778_816 }
      ]
    }

    expect(nodeAdvertisedVramGB(node)).toBe(44.02970624)
    expect(nodeRatedVramGB(node)).toBe(42)
  })

  it('falls back to allocatable inventory, then the rated class, when nothing is advertised', () => {
    const withReserve = { vram_gb: 0, gpus: [{ vram_bytes: 32_000_000_000, reserved_bytes: 1_000_000_000 }] }
    expect(nodeAdvertisedVramGB(withReserve)).toBe(31)

    const ratedOnly = { gpus: [{ rated_vram_gb: 24 }] }
    expect(nodeAdvertisedVramGB(ratedOnly)).toBe(24)

    expect(nodeAdvertisedVramGB({ vram_gb: 0, gpus: [] })).toBeNull()
  })

  it('reads legacy my_vram_gb when vram_gb is absent', () => {
    expect(nodeAdvertisedVramGB({ my_vram_gb: 12.5 })).toBe(12.5)
  })

  it('sums advertised and rated capacity across the local node and peers', () => {
    const mesh = {
      vram_gb: 115.448725504,
      gpus: [{ rated_vram_gb: 128, vram_bytes: 115_448_725_504 }],
      peers: [{ vram_gb: 44.02970624, gpus: [{ rated_vram_gb: 32 }, { rated_vram_gb: 10 }] }]
    }

    expect(meshAdvertisedVramGB(mesh)).toBeCloseTo(159.478, 3)
    expect(meshRatedVramGB(mesh)).toBe(170)
    expect(meshRatedVramGB({ vram_gb: 5, peers: [] })).toBeNull()
  })
})

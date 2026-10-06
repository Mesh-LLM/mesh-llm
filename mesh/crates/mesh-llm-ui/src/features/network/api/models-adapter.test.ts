import { describe, expect, it } from 'vitest'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'
import type { MeshModelRaw } from '@/lib/api/types'

describe('adaptModelsToSummary', () => {
  it.each(['Readable model', '', '   ', undefined])(
    'preserves routing identity with display name %j',
    (displayName) => {
      const name = 'gguf:0123456789abcdef'
      const [model] = adaptModelsToSummary([
        { name, display_name: displayName, status: 'warm', size_gb: 1, node_count: 1 }
      ])
      expect(model.name).toBe(name)
      expect(model.fullId).toBe(name)
      expect(model.displayName).toBe(displayName?.trim() || undefined)
    }
  )

  it('accepts public API model rows without a nested capabilities object', () => {
    const models: MeshModelRaw[] = [
      {
        name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
        status: 'warm',
        size_gb: 4.4,
        node_count: 1,
        quantization: 'Q4_K_M',
        moe: false,
        vision: false
      }
    ]

    expect(adaptModelsToSummary(models)).toEqual([
      expect.objectContaining({
        name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
        status: 'warm',
        size: '4.4 GB',
        context: 'Unknown',
        ctxMaxK: undefined,
        moe: false,
        vision: false
      })
    ])
  })

  it('rounds overprecise model sizes for display', () => {
    const models: MeshModelRaw[] = [
      {
        name: 'Qwen3.5-4B-UD-Q4_K_XL',
        status: 'warm',
        size_gb: 2.912109728,
        node_count: 1,
        quantization: 'Q4_K_XL',
        moe: false,
        vision: false
      }
    ]

    expect(adaptModelsToSummary(models)[0]).toEqual(
      expect.objectContaining({
        size: '2.9 GB',
        sizeGB: 2.912109728
      })
    )
  })

  it('maps mesh_vram_gb through to meshVramGB', () => {
    const models: MeshModelRaw[] = [
      { name: 'Bonsai-27B-GGUF:Q1_0', status: 'warm', size_gb: 14.2, node_count: 1, mesh_vram_gb: 61.7 }
    ]

    expect(adaptModelsToSummary(models)[0]).toEqual(expect.objectContaining({ meshVramGB: 61.7 }))
  })

  it('leaves meshVramGB undefined when the backend does not report it', () => {
    const models: MeshModelRaw[] = [{ name: 'Bonsai-27B-GGUF:Q1_0', status: 'warm', size_gb: 14.2, node_count: 1 }]

    expect(adaptModelsToSummary(models)[0].meshVramGB).toBeUndefined()
  })

  it('maps fit_label through to fitLabel', () => {
    const models: MeshModelRaw[] = [
      {
        name: 'Bonsai-27B-GGUF:Q1_0',
        status: 'warm',
        size_gb: 14.2,
        node_count: 1,
        fit_label: 'Possible with tradeoffs'
      }
    ]

    expect(adaptModelsToSummary(models)[0]).toEqual(expect.objectContaining({ fitLabel: 'Possible with tradeoffs' }))
  })

  it('leaves fitLabel undefined when the backend does not report it', () => {
    const models: MeshModelRaw[] = [{ name: 'Bonsai-27B-GGUF:Q1_0', status: 'warm', size_gb: 14.2, node_count: 1 }]

    expect(adaptModelsToSummary(models)[0].fitLabel).toBeUndefined()
  })

  it('prefers nested capabilities when available', () => {
    const models: MeshModelRaw[] = [
      {
        name: 'Qwen3-VL-8B-Q4_K_M',
        status: 'cold',
        size_gb: 5,
        node_count: 0,
        capabilities: { moe: true, vision: true },
        quantization: 'Q4_K_M',
        context_length: 128_000,
        moe: false,
        vision: false
      }
    ]

    expect(adaptModelsToSummary(models)[0]).toEqual(
      expect.objectContaining({
        status: 'offline',
        context: '128K',
        ctxMaxK: 128,
        moe: true,
        vision: true
      })
    )
  })
})

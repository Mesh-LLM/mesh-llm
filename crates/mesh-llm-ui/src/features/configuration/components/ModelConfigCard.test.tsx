import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ModelConfigCard } from '@/features/configuration/components/ModelConfigCard'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

const node: ConfigNode = {
  id: 'node-test',
  hostname: 'test-node',
  region: 'lab',
  status: 'online',
  cpu: 'test cpu',
  ramGB: 64,
  gpus: [{ idx: 1, name: 'RTX 6000 Pro', totalGB: 48 }],
  placement: 'separate'
}

const model: ConfigModel = {
  id: 'llama70',
  name: 'Llama-3.3-70B-Q4_K_M',
  family: 'llama',
  paramsB: 70,
  quant: 'Q4_K_M',
  sizeGB: 40.3,
  diskGB: 40.3,
  ctxMaxK: 256,
  layers: 80,
  heads: 64,
  embed: 8192,
  tokenizer: 'llama',
  moe: false,
  vision: false,
  tags: []
}

const assign: ConfigAssign = {
  id: 'assign-llama',
  modelId: model.id,
  nodeId: node.id,
  containerIdx: 1,
  ctx: 16384,
  config: { slots: 4, cacheTypeK: 'q8_0', cacheTypeV: 'q4_0' }
}

describe('ModelConfigCard', () => {
  it('edits selected model runtime and asset settings', () => {
    const onConfigChange = vi.fn()

    render(
      <ModelConfigCard
        assign={assign}
        node={node}
        models={[model]}
        modelPlacementOptions={{
          cacheTypeK: ['f16', 'q8_0', 'q4_0', 'q5_0'],
          cacheTypeV: ['f16', 'q8_0', 'q4_0', 'q5_1']
        }}
        containerFreeGB={8}
        onCtxChange={vi.fn()}
        onConfigChange={onConfigChange}
        onRemove={vi.fn()}
      />
    )

    expect(screen.getByRole('heading', { name: model.name })).toBeInTheDocument()
    expect(screen.getAllByText('16K ctx').length).toBeGreaterThan(0)
    expect(screen.getByText('GPU 1 · RTX 6000 Pro')).toBeInTheDocument()
    expect(screen.getByText('Headroom')).toBeInTheDocument()
    expect(screen.getByText('Memory')).toHaveClass('font-semibold', 'uppercase', 'text-foreground')
    expect(screen.getByText('Model')).toHaveClass('font-semibold', 'uppercase', 'text-foreground')
    expect(screen.queryByText(/^max ≈/i)).not.toBeInTheDocument()
    expect(screen.getByText('Runtime').compareDocumentPosition(screen.getByText('Context'))).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING
    )
    expect(screen.getByText('Runtime')).toHaveClass('font-semibold', 'uppercase', 'text-foreground')

    fireEvent.click(screen.getByRole('radio', { name: '5 slots' }))
    expect(onConfigChange).toHaveBeenLastCalledWith({ slots: 5, cacheTypeK: 'q8_0', cacheTypeV: 'q4_0' })

    fireEvent.click(screen.getByRole('radio', { name: '32 slots' }))
    expect(onConfigChange).toHaveBeenLastCalledWith({ slots: 32, cacheTypeK: 'q8_0', cacheTypeV: 'q4_0' })

    // Open advanced controls to reveal Split mode, mmproj, Flash attention, Cache types
    fireEvent.click(screen.getByRole('button', { name: 'Toggle advanced controls' }))

    for (const title of ['Placement', 'Assets', 'Tuning']) {
      expect(screen.getByText(title)).toHaveClass('font-semibold', 'uppercase', 'text-foreground')
    }

    fireEvent.click(within(screen.getByRole('radiogroup', { name: 'Split mode' })).getByRole('radio', { name: 'Row' }))
    expect(onConfigChange).toHaveBeenLastCalledWith({
      slots: 4,
      splitMode: 'row',
      cacheTypeK: 'q8_0',
      cacheTypeV: 'q4_0'
    })

    fireEvent.change(screen.getByLabelText('mmproj'), { target: { value: '/models/mmproj.gguf' } })
    expect(onConfigChange).toHaveBeenLastCalledWith({
      slots: 4,
      mmproj: '/models/mmproj.gguf',
      cacheTypeK: 'q8_0',
      cacheTypeV: 'q4_0'
    })

    fireEvent.click(
      within(screen.getByRole('radiogroup', { name: 'Flash attention' })).getByRole('radio', { name: 'On' })
    )
    expect(onConfigChange).toHaveBeenLastCalledWith({
      slots: 4,
      flashAttention: 'enabled',
      cacheTypeK: 'q8_0',
      cacheTypeV: 'q4_0'
    })

    fireEvent.change(screen.getByLabelText('Cache type K'), {
      target: { value: 'f16' }
    })
    expect(onConfigChange).toHaveBeenLastCalledWith({ slots: 4, cacheTypeK: 'f16', cacheTypeV: 'q4_0' })

    fireEvent.change(screen.getByLabelText('Cache type V'), {
      target: { value: 'q5_1' }
    })
    expect(onConfigChange).toHaveBeenLastCalledWith({ slots: 4, cacheTypeK: 'q8_0', cacheTypeV: 'q5_1' })
  })
})

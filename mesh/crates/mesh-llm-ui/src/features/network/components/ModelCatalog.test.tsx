// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ModelSummary } from '@/features/app-tabs/types'
import { ModelCatalog } from '@/features/network/components/ModelCatalog'

afterEach(() => {
  cleanup()
})

function buildModel(overrides: Partial<ModelSummary> = {}): ModelSummary {
  return {
    name: 'Qwen3.5-4B-UD',
    family: 'Qwen',
    size: '2.9 GB',
    context: '32K',
    status: 'ready',
    tags: [],
    nodeCount: 1,
    fullId: 'Qwen/Qwen3.5-4B-UD-Q4_K_XL',
    quant: 'Q4_K_XL',
    sizeGB: 2.912109728,
    ctxMaxK: 32,
    moe: false,
    vision: false,
    ...overrides
  }
}

describe('ModelCatalog', () => {
  it('shows the display name without exposing a routing hash, and selects the original model', () => {
    const model = buildModel({ name: 'gguf:0123456789abcdef', fullId: 'gguf:0123456789abcdef', displayName: 'Gemma 4' })
    const onSelect = vi.fn()
    render(<ModelCatalog models={[model]} onSelect={onSelect} />)
    fireEvent.click(screen.getByRole('button', { name: /View Gemma 4 model/ }))
    expect(onSelect).toHaveBeenCalledWith(model)
    expect(screen.queryByText(model.name)).not.toBeInTheDocument()
  })

  it('rounds model sizes on catalog rows', () => {
    render(<ModelCatalog models={[buildModel()]} />)

    expect(screen.getByText('2.9 GB')).toBeInTheDocument()
    expect(screen.queryByText('2.912109728 GB')).not.toBeInTheDocument()
  })
})

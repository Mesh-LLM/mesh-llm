// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

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
  it('rounds model sizes on catalog rows', () => {
    render(<ModelCatalog models={[buildModel()]} />)

    expect(screen.getByText('2.9 GB')).toBeInTheDocument()
    expect(screen.queryByText('2.912109728 GB')).not.toBeInTheDocument()
  })
})

// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import type { MeshModel } from '@/features/app-shell/lib/status-types'
import { ModelSidebar } from '@/features/dashboard/components/details/ModelSidebar'

afterEach(() => {
  cleanup()
})

describe('ModelSidebar', () => {
  it('rounds file size labels for models', () => {
    render(
      <TooltipProvider>
        <Sheet open>
          <SheetContent>
            <ModelSidebar activePeers={[]} model={buildModel()} onOpenNode={vi.fn()} />
          </SheetContent>
        </Sheet>
      </TooltipProvider>
    )

    expect(screen.getAllByText('2.9 GB').length).toBeGreaterThan(0)
    expect(screen.queryByText('2.912109728 GB')).not.toBeInTheDocument()
  })
})

function buildModel(): MeshModel {
  return {
    name: 'Qwen3.5-4B-UD',
    status: 'warm',
    node_count: 1,
    mesh_vram_gb: 7.25,
    size_gb: 2.912109728,
    quantization: 'Q4_K_XL',
    source_file: 'Qwen3.5-4B-UD-Q4_K_XL.gguf',
    multimodal: false,
    vision: false,
    audio: false,
    reasoning: false,
    tool_use: false,
    active_nodes: []
  }
}

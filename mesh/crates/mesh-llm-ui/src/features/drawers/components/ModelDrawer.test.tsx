import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ModelDrawer } from '@/features/drawers/components/ModelDrawer'
import type { ModelSummary, Peer } from '@/features/app-tabs/types'

const MODEL: ModelSummary = {
  name: 'lmstudio-community/Bonsai-27B-GGUF:Q1_0',
  family: 'lmstudio-community',
  size: '14.2 GB',
  context: '128K',
  status: 'warm',
  tags: [],
  nodeCount: 1,
  fullId: 'lmstudio-community/Bonsai-27B-GGUF:Q1_0',
  meshVramGB: 61.7
}

function peerHosting(id: string, vramGB: number): Peer {
  return {
    id,
    hostname: id,
    region: 'tor-1',
    status: 'online',
    hostedModels: [MODEL.name],
    sharePct: 100,
    latencyMs: 282,
    loadPct: 10,
    vramGB
  }
}

describe('ModelDrawer', () => {
  it('shows real mesh VRAM alongside file size instead of conflating them', () => {
    render(<ModelDrawer open model={MODEL} peers={[peerHosting('peer-1', 48)]} onClose={() => {}} />)

    expect(screen.getByText('Mesh VRAM')).toBeInTheDocument()
    expect(screen.getByText('61.7 GB')).toBeInTheDocument()
    expect(screen.getByText('File size')).toBeInTheDocument()
    expect(screen.getByText('14.2 GB')).toBeInTheDocument()
  })

  it('shows Unknown quant instead of a fabricated fallback when quant is missing', () => {
    render(<ModelDrawer open model={MODEL} peers={[]} onClose={() => {}} />)

    expect(screen.getByText('Quant')).toBeInTheDocument()
    expect(screen.getByText('Unknown')).toBeInTheDocument()
    expect(screen.queryByText('Q4_K_XL')).not.toBeInTheDocument()
  })

  it('shows each peer’s own VRAM in the active peers table', () => {
    render(<ModelDrawer open model={MODEL} peers={[peerHosting('peer-1', 24), peerHosting('peer-2', 48)]} onClose={() => {}} />)

    expect(screen.getByText('24 GB')).toBeInTheDocument()
    expect(screen.getByText('48 GB')).toBeInTheDocument()
    expect(screen.queryByText('14.2 GB', { selector: 'span' })).not.toBeInTheDocument()
  })
})

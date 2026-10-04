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

  it('derives quant from the colon tag in the model name when quant metadata is missing', () => {
    render(<ModelDrawer open model={MODEL} peers={[]} onClose={() => {}} />)

    expect(screen.getByText('Quant')).toBeInTheDocument()
    expect(screen.getByText('Q1_0')).toBeInTheDocument()
  })

  it('shows Unknown quant when the name has no colon tag and quant metadata is missing', () => {
    const model: ModelSummary = { ...MODEL, name: 'Hermes-2-Pro-Mistral-7B', fullId: 'Hermes-2-Pro-Mistral-7B' }
    render(<ModelDrawer open model={model} peers={[]} onClose={() => {}} />)

    expect(screen.getByText('Quant')).toBeInTheDocument()
    expect(screen.getByText('Unknown')).toBeInTheDocument()
  })

  it('shows Unknown quant when the colon suffix is not a quant tag', () => {
    const model: ModelSummary = { ...MODEL, name: 'gguf:0123456789abcdef', fullId: 'gguf:0123456789abcdef' }
    render(<ModelDrawer open model={model} peers={[]} onClose={() => {}} />)

    expect(screen.getByText('Quant')).toBeInTheDocument()
    expect(screen.getByText('Unknown')).toBeInTheDocument()
    expect(screen.queryByText('0123456789abcdef')).not.toBeInTheDocument()
  })

  it('prefers explicit quant metadata over the colon tag', () => {
    const model: ModelSummary = { ...MODEL, quant: 'Q4_K_M' }
    render(<ModelDrawer open model={model} peers={[]} onClose={() => {}} />)

    expect(screen.getByText('Q4_K_M')).toBeInTheDocument()
    expect(screen.queryByText('Q1_0')).not.toBeInTheDocument()
  })

  it('shows the backend fit verdict instead of a hardcoded Fits badge', () => {
    const model: ModelSummary = { ...MODEL, fitLabel: 'Likely fits' }
    render(<ModelDrawer open model={model} peers={[]} onClose={() => {}} />)

    expect(screen.getByText('Suitable for this node')).toBeInTheDocument()
    expect(screen.queryByText('Fits')).not.toBeInTheDocument()
  })

  it('falls back to Check fit when the backend reports no fit verdict', () => {
    render(<ModelDrawer open model={MODEL} peers={[]} onClose={() => {}} />)

    expect(screen.getByText('Check fit')).toBeInTheDocument()
    expect(screen.queryByText('Fits')).not.toBeInTheDocument()
  })

  it('shows each peer’s own VRAM in the active peers table', () => {
    render(<ModelDrawer open model={MODEL} peers={[peerHosting('peer-1', 24), peerHosting('peer-2', 48)]} onClose={() => {}} />)

    expect(screen.getByText('24 GB')).toBeInTheDocument()
    expect(screen.getByText('48 GB')).toBeInTheDocument()
    expect(screen.queryByText('14.2 GB', { selector: 'span' })).not.toBeInTheDocument()
  })
})

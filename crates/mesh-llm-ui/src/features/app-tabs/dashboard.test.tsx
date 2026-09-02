import { describe, expect, it, vi } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DashboardPage } from '@/features/network/pages/DashboardPage'
import { buildDashboardMeshNodes } from '@/features/network/lib/dashboard-mesh-nodes'
import { MESH_NODES, PEERS } from '@/features/app-tabs/data'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import {
  getMeshElement,
  getMeshNodeContextHighlight,
  getMeshNodeCore,
  getMeshNodeCoreOverlay,
  DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT,
  nearestNodeDistance,
  placementCentroid,
  placementClusterRadius,
  distanceFrom,
  render
} from './app-tabs-test-support'

describe('dashboard app tab', () => {
  it('renders the network component composition', () => {
    render(<DashboardPage />)
    expect(screen.getByRole('heading', { name: /your private mesh/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /model catalog/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /connected peers/i })).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /view carrack node/i }).length).toBeGreaterThan(0)
  })

  it('copies the dashboard connect command to the clipboard', async () => {
    const user = userEvent.setup()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })

    render(<DashboardPage />)

    await user.click(screen.getByRole('button', { name: 'Copy' }))

    expect(writeText).toHaveBeenCalledWith('mesh-llm --auto --join <mesh-invite-token>')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument())
  })

  it('places newly joined dashboard peers with the shared clustered mesh rule', () => {
    const joinedPeer: Peer = {
      id: 'p4',
      hostname: 'new-worker',
      region: 'iad-1',
      status: 'online',
      hostedModels: [],
      sharePct: 12,
      latencyMs: 2.4,
      loadPct: 18,
      role: 'peer',
      version: '0.64.0',
      vramGB: 24,
      toksPerSec: 11.2
    }
    const meshNodes = buildDashboardMeshNodes([...PEERS, joinedPeer], 'joined-peer-placement-test')
    const repeatedMeshNodes = buildDashboardMeshNodes([...PEERS, joinedPeer], 'joined-peer-placement-test')
    const joinedNode = meshNodes.find((node) => node.peerId === joinedPeer.id)
    const repeatedJoinedNode = repeatedMeshNodes.find((node) => node.peerId === joinedPeer.id)
    const baseCentroid = placementCentroid(MESH_NODES)

    expect(joinedNode).toBeDefined()
    expect(repeatedJoinedNode).toBeDefined()
    expect(joinedNode?.x).not.toBe(0)
    expect(joinedNode?.y).not.toBe(0)
    expect(joinedNode?.renderKind).toBe('worker')
    expect(joinedNode?.meshState).toBe('standby')
    expect(joinedNode?.vramGB).toBe(24)

    for (const pinnedNode of MESH_NODES) {
      const generatedNode = meshNodes.find((node) => node.peerId === pinnedNode.peerId || node.id === pinnedNode.id)

      expect(generatedNode?.x).toBe(pinnedNode.x)
      expect(generatedNode?.y).toBe(pinnedNode.y)
    }

    expect(nearestNodeDistance(joinedNode as MeshNode, MESH_NODES)).toBeLessThanOrEqual(
      DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT + 0.01
    )
    expect(distanceFrom(joinedNode as MeshNode, baseCentroid)).toBeLessThanOrEqual(
      placementClusterRadius(MESH_NODES, 0) + 0.01
    )
    expect(repeatedJoinedNode?.x).toBe(joinedNode?.x)
    expect(repeatedJoinedNode?.y).toBe(joinedNode?.y)
  })

  it('opens dashboard drawers from lists and MeshViz node popovers from mesh clicks', async () => {
    const user = userEvent.setup()
    render(<DashboardPage />)

    const gemmaModelRow = screen.getByRole('button', { name: /view gemma-4-26b-a4b-it-ud model/i })
    await user.click(gemmaModelRow)
    expect(gemmaModelRow).toHaveAttribute('data-active', 'true')
    let drawer = screen.getByRole('dialog')
    expect(within(drawer).getByText(/availability/i)).toBeInTheDocument()
    expect(within(drawer).getAllByText(/64k/i).length).toBeGreaterThan(0)
    expect(within(drawer).getByText(/files/i)).toBeInTheDocument()
    expect(within(drawer).getByText(/active peers/i)).toBeInTheDocument()

    await user.click(within(drawer).getByRole('button', { name: /close/i }))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expect(gemmaModelRow).not.toHaveAttribute('data-active')
    const carrackNodeButton = screen.getByRole('button', { name: 'View CARRACK node' })
    await user.hover(carrackNodeButton)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    expect(carrackNodeButton).not.toHaveAttribute('data-context-open')
    expect(getMeshNodeContextHighlight('self')).toHaveClass('opacity-0')
    const selfCoreFill = getMeshNodeCore('self').style.color
    expect(getMeshNodeCoreOverlay('self')).toHaveClass('opacity-0')

    await user.click(carrackNodeButton)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(carrackNodeButton).toHaveAttribute('data-context-open', 'true')
    expect(getMeshNodeContextHighlight('self')).toHaveClass('opacity-100', 'duration-150')
    expect(getMeshNodeContextHighlight('self').style.background).toContain('currentcolor')
    expect(getMeshNodeContextHighlight('self').style.background).not.toContain('--color-accent')
    expect(getMeshNodeCore('self').style.color).toBe(selfCoreFill)
    const radarPing = getMeshElement('.mesh-radar-ping')
    expect(radarPing.style.color).toContain('oklch(')
    expect(radarPing.style.color).not.toContain('--color-accent')
    expect(getMeshNodeCoreOverlay('self')).toHaveClass('opacity-45', 'duration-150')
    expect(getMeshNodeCoreOverlay('self').style.backgroundColor).toContain('currentcolor')
    expect(getMeshNodeCoreOverlay('self').style.backgroundColor).not.toContain('--color-accent')

    let popover = await screen.findByRole('tooltip')
    expect(within(popover).getByText(/CARRACK/i)).toBeInTheDocument()
    expect(within(popover).getByText(/990232e1c1/i)).toBeInTheDocument()
    expect(within(popover).getByText(/VRAM/i)).toBeInTheDocument()

    const lemonyCoreFill = getMeshNodeCore('lemony').style.color
    await user.click(screen.getByRole('button', { name: 'View LEMONY-28 node' }))
    popover = await screen.findByRole('tooltip')
    expect(getMeshNodeContextHighlight('self')).toHaveClass('opacity-0')
    expect(getMeshNodeContextHighlight('lemony')).toHaveClass('opacity-100', 'duration-150')
    expect(getMeshNodeCore('self').style.color).toBe(selfCoreFill)
    expect(getMeshNodeCore('lemony').style.color).toBe(lemonyCoreFill)
    expect(getMeshNodeCoreOverlay('self')).toHaveClass('opacity-0')
    expect(getMeshNodeCoreOverlay('lemony')).toHaveClass('opacity-45', 'duration-150')
    expect(within(popover).getByText(/lemony-28/i)).toBeInTheDocument()
    expect(within(popover).getByText(/e5c42cc0ad/i)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /View lemony-28 node, peer ID p2/i }))
    drawer = screen.getByRole('dialog')
    expect(within(drawer).getByText(/node metadata/i)).toBeInTheDocument()
    expect(within(drawer).getByText(/hosted models/i)).toBeInTheDocument()
    expect(within(drawer).getByText(/hardware/i)).toBeInTheDocument()
    expect(within(drawer).getByRole('heading', { name: /ownership/i })).toBeInTheDocument()
    expect(within(drawer).getByText(/gemma-4-26B-A4B-it-UD/i)).toBeInTheDocument()
  })
})

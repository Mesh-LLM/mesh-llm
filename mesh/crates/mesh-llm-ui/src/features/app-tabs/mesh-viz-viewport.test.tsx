import { describe, expect, it } from 'vitest'
import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MeshViz } from '@/features/network/components/MeshViz'
import { MESH_NODES } from '@/features/app-tabs/data'
import type { MeshNode } from '@/features/app-tabs/types'
import { env } from '@/lib/env'
import {
  applyMeshVizInteraction,
  EXPANDED_BOUNDARY_MESH_NODES,
  HUGE_BOUNDARY_MESH_NODES,
  getMeshCanvas,
  getMeshLinkPairs,
  getNodeButton,
  linkDegree,
  linkKey,
  nonClientLinkDegree,
  openDebugMenu,
  openTrafficDebugMenu,
  expectMeshNodeCoreFill,
  OVERSIZED_MESH_NODES,
  pixelValue,
  render,
  setFullscreenElement,
  setMeshCanvasSize,
  triggerMeshResize,
  triggerMeshResizeInAct
} from './app-tabs-test-support'

describe('MeshViz viewport behavior', () => {
  it('keeps MeshViz viewport interactions stable across wheel, pan, resize, and reset', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)
    const initialTop = pixelValue(lemony29.style.top)

    await applyMeshVizInteraction(() => {
      fireEvent.wheel(canvas, { deltaY: 0, clientX: 240, clientY: 210 })
    })
    expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft, 4)
    expect(pixelValue(lemony29.style.top)).toBeCloseTo(initialTop, 4)

    await applyMeshVizInteraction(() => {
      fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
      fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 150, clientY: 130 })
      fireEvent.pointerUp(canvas, { pointerId: 1 })
    })

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft + 50, 1))
    expect(pixelValue(lemony29.style.top)).toBeGreaterThan(initialTop + 20)
    expect(pixelValue(lemony29.style.top)).toBeLessThanOrEqual(initialTop + 30)

    setMeshCanvasSize(801, 420)
    await triggerMeshResizeInAct()
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(initialLeft + 40))

    await userEvent.click(screen.getByRole('button', { name: /reset view/i }))
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(initialLeft + 10))
  })

  it('pinch-zooms MeshViz on touch pointers', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, pointerType: 'touch', clientX: 350, clientY: 210 })
    fireEvent.pointerDown(canvas, { button: 0, pointerId: 2, pointerType: 'touch', clientX: 450, clientY: 210 })
    fireEvent.pointerMove(canvas, { pointerId: 2, pointerType: 'touch', clientX: 550, clientY: 210 })
    fireEvent.pointerUp(canvas, { pointerId: 2, pointerType: 'touch' })
    fireEvent.pointerUp(canvas, { pointerId: 1, pointerType: 'touch' })

    await waitFor(() => expect(Math.abs(pixelValue(lemony29.style.left) - initialLeft)).toBeGreaterThan(20))
  })

  it('connects clients to their closest non-client without consuming non-client link capacity', () => {
    const sparseNodes: MeshNode[] = [
      { id: 'client-a', label: 'CLIENT A', x: 0, y: 0, status: 'online', renderKind: 'client', client: true },
      { id: 'client-b', label: 'CLIENT B', x: 0, y: 20, status: 'online', renderKind: 'client', client: true },
      { id: 'host', label: 'HOST', x: 50, y: 50, status: 'online', host: true, renderKind: 'worker' },
      { id: 'worker-1', label: 'WORKER 1', x: 40, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-2', label: 'WORKER 2', x: 60, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-3', label: 'WORKER 3', x: 50, y: 40, status: 'online', renderKind: 'worker' },
      { id: 'worker-4', label: 'WORKER 4', x: 50, y: 60, status: 'online', renderKind: 'worker' },
      { id: 'worker-5', label: 'WORKER 5', x: 90, y: 90, status: 'online', renderKind: 'worker' }
    ]

    render(<MeshViz nodes={sparseNodes} selfId="host" height={420} />)

    const pairs = getMeshLinkPairs()
    const pairKeys = pairs.map(linkKey)
    const clientIds = new Set(['client-a', 'client-b'])

    expect(pairKeys).toContain('client-a::worker-1')
    expect(pairKeys).toContain('client-b::worker-1')
    expect(linkDegree(pairs, 'client-a')).toBe(1)
    expect(linkDegree(pairs, 'client-b')).toBe(1)
    expect(pairKeys).not.toContain('client-a::client-b')
    expect(linkDegree(pairs, 'worker-1')).toBeGreaterThan(3)

    for (const node of sparseNodes.filter((meshNode) => !meshNode.client)) {
      expect(nonClientLinkDegree(pairs, node.id, clientIds)).toBeLessThanOrEqual(3)
    }
  })

  it('connects a host to its closest local backbone neighbors', () => {
    const nearestNodes: MeshNode[] = [
      { id: 'host', label: 'HOST', x: 50, y: 50, status: 'online', host: true, renderKind: 'worker' },
      { id: 'worker-1', label: 'WORKER 1', x: 40, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-2', label: 'WORKER 2', x: 60, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-3', label: 'WORKER 3', x: 50, y: 40, status: 'online', renderKind: 'worker' },
      { id: 'worker-4', label: 'WORKER 4', x: 50, y: 60, status: 'online', renderKind: 'worker' },
      { id: 'worker-5', label: 'WORKER 5', x: 90, y: 90, status: 'online', renderKind: 'worker' }
    ]

    render(<MeshViz nodes={nearestNodes} selfId="host" height={420} />)

    const pairs = getMeshLinkPairs()
    const pairKeys = pairs.map(linkKey)

    expect(linkDegree(pairs, 'host')).toBe(3)
    expect(pairKeys).toContain('host::worker-1')
    expect(pairKeys).toContain('host::worker-2')
    expect(pairKeys).toContain('host::worker-3')
    expect(pairKeys).not.toContain('host::worker-4')
    expect(pairKeys).not.toContain('host::worker-5')
  })

  it('keeps host and worker backbone links when nearby clients attach to the same node', () => {
    const crowdedClientNodes: MeshNode[] = [
      { id: 'host', label: 'HOST', x: 50, y: 50, status: 'online', host: true, renderKind: 'worker' },
      { id: 'client-1', label: 'CLIENT 1', x: 48, y: 49, status: 'online', renderKind: 'client', client: true },
      { id: 'client-2', label: 'CLIENT 2', x: 49, y: 48, status: 'online', renderKind: 'client', client: true },
      { id: 'client-3', label: 'CLIENT 3', x: 51, y: 52, status: 'online', renderKind: 'client', client: true },
      { id: 'worker-1', label: 'WORKER 1', x: 46, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-2', label: 'WORKER 2', x: 54, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-3', label: 'WORKER 3', x: 50, y: 46, status: 'online', renderKind: 'worker' },
      { id: 'worker-4', label: 'WORKER 4', x: 80, y: 80, status: 'online', renderKind: 'worker' }
    ]

    render(<MeshViz nodes={crowdedClientNodes} selfId="host" height={420} />)

    const pairs = getMeshLinkPairs()
    const pairKeys = pairs.map(linkKey)
    const clientIds = new Set(['client-1', 'client-2', 'client-3'])

    expect(pairKeys).toContain('client-1::host')
    expect(pairKeys).toContain('client-2::host')
    expect(pairKeys).toContain('client-3::host')
    expect(pairKeys).toContain('host::worker-1')
    expect(pairKeys).toContain('host::worker-2')
    expect(pairKeys).toContain('host::worker-3')
    expect(linkDegree(pairs, 'host')).toBeGreaterThan(3)
    expect(nonClientLinkDegree(pairs, 'host', clientIds)).toBe(3)
  })

  it('prevents page text selection while panning MeshViz', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()

    await waitFor(() => expect(getNodeButton('View LEMONY-29 node').style.left).not.toBe(''))

    const pointerDownWasNotCanceled = fireEvent.pointerDown(canvas, {
      button: 0,
      pointerId: 1,
      clientX: 100,
      clientY: 100,
      cancelable: true
    })

    expect(pointerDownWasNotCanceled).toBe(false)
    await waitFor(() => expect(document.body.style.userSelect).toBe('none'))
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 130, clientY: 120, cancelable: true })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(document.body.style.userSelect).toBe(''))
  })

  it('limits MeshViz pan travel to the node bounds plus the dead-zone', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 2100, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(initialLeft + 160))

    await userEvent.click(screen.getByRole('button', { name: /reset view/i }))
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft, 1))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 2, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 2, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 2 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(initialLeft - 160))
  })

  it('allows MeshViz bounds to reach fullscreen edges while panning', async () => {
    setMeshCanvasSize(1600, 900)
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={900} />)
    const canvas = getMeshCanvas()
    const carrack = getNodeButton('View CARRACK node')
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(carrack.style.left)).toBeGreaterThan(0))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 2100, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(carrack.style.left)).toBeGreaterThan(1450))
    expect(pixelValue(carrack.style.left)).toBeLessThan(1530)

    await userEvent.click(screen.getByRole('button', { name: /reset view/i }))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 2, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 2, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 2 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(130))
    expect(pixelValue(lemony29.style.left)).toBeGreaterThan(70)
  })

  it('lowers the MeshViz max zoom-out limit as the graph boundary grows', async () => {
    render(<MeshViz nodes={EXPANDED_BOUNDARY_MESH_NODES} selfId="self" height={420} />)
    const westEdge = getNodeButton('View WEST EDGE node')
    const eastEdge = getNodeButton('View EAST EDGE node')

    await waitFor(() => {
      expect(screen.getByTestId('mesh-max-zoom-label')).toHaveTextContent('Max Zoom: 0.40')
      expect(pixelValue(westEdge.style.left)).toBeGreaterThanOrEqual(48)
      expect(pixelValue(eastEdge.style.left)).toBeLessThanOrEqual(752)
    })
  })

  it('keeps deriving the MeshViz max zoom-out limit for very large graph boundaries', async () => {
    render(<MeshViz nodes={HUGE_BOUNDARY_MESH_NODES} selfId="self" height={420} />)

    await waitFor(() => expect(screen.getByTestId('mesh-max-zoom-label')).toHaveTextContent('Max Zoom: 0.08'))
  })

  it('only shows the MeshViz max zoom label in debug mode', () => {
    env.isDevelopment = false

    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expect(screen.queryByTestId('mesh-max-zoom-label')).not.toBeInTheDocument()
  })

  it('reclamps oversized MeshViz bounds to preserve viewport intersection when the canvas shrinks', async () => {
    const user = userEvent.setup()
    setMeshCanvasSize(1600, 900)
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={900} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(130))

    await act(async () => {
      setMeshCanvasSize(800, 420)
      triggerMeshResize()
    })

    await waitFor(() => {
      const nodeBoundsBox = screen.getByTestId('mesh-node-bounds-box')
      const boundsX = Number(nodeBoundsBox.getAttribute('x'))
      const boundsY = Number(nodeBoundsBox.getAttribute('y'))
      const boundsWidth = Number(nodeBoundsBox.getAttribute('width'))
      const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

      expect(boundsX).toBeLessThanOrEqual(800)
      expect(boundsX + boundsWidth).toBeGreaterThanOrEqual(0)
      expect(boundsY).toBeLessThanOrEqual(420)
      expect(boundsY + boundsHeight).toBeGreaterThanOrEqual(0)
    })
  })

  it('resets MeshViz to a centered fit when exiting fullscreen', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)

    await act(async () => {
      setFullscreenElement(canvas)
      setMeshCanvasSize(1600, 900)
      triggerMeshResize()
      fireEvent(document, new Event('fullscreenchange'))
    })

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(140))

    await act(async () => {
      setMeshCanvasSize(800, 420)
      triggerMeshResize()
      setFullscreenElement(null)
      fireEvent(document, new Event('fullscreenchange'))
    })

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft, 1))
  })

  it('lets oversized MeshViz bounds pan until the inner bounds still touch the viewport', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={OVERSIZED_MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony = getNodeButton('View LEMONY-28 node')

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))
    await waitFor(() => expect(pixelValue(lemony.style.top)).toBeGreaterThan(0))

    for (let index = 0; index < 12; index += 1) {
      await user.click(screen.getByRole('button', { name: /zoom in/i }))
    }

    await waitFor(() => expect(pixelValue(lemony.style.top)).toBeLessThan(0))
    const centeredNodeBoundsBox = screen.getByTestId('mesh-centered-bounds-box')
    const centeredBoundsY = Number(centeredNodeBoundsBox.getAttribute('y'))
    const centeredBoundsHeight = Number(centeredNodeBoundsBox.getAttribute('height'))

    expect(centeredBoundsHeight).toBeGreaterThan(420)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 100, clientY: 2100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })

    await waitFor(() => {
      const nodeBoundsBox = screen.getByTestId('mesh-centered-bounds-box')
      const boundsY = Number(nodeBoundsBox.getAttribute('y'))
      const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

      expect(boundsY).toBeGreaterThan(centeredBoundsY + 100)
      expect(boundsY).toBeLessThanOrEqual(420)
      expect(boundsY + boundsHeight).toBeGreaterThanOrEqual(0)
    })
  })

  it('lets oversized MeshViz bounds pan while preserving a viewport intersection', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={OVERSIZED_MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony = getNodeButton('View LEMONY-28 node')

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))
    await waitFor(() => expect(pixelValue(lemony.style.top)).toBeGreaterThan(0))
    const focusX = pixelValue(lemony.style.left)
    const focusY = pixelValue(lemony.style.top) + 18

    for (let index = 0; index < 12; index += 1) {
      fireEvent.wheel(canvas, { deltaY: -100, clientX: focusX, clientY: focusY, cancelable: true })
    }

    await waitFor(() => expect(pixelValue(lemony.style.top) + 18).toBeGreaterThanOrEqual(0))
    expect(pixelValue(lemony.style.top) + 18).toBeLessThanOrEqual(420)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 100, clientY: -2100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })

    await waitFor(() => {
      const nodeBoundsBox = screen.getByTestId('mesh-centered-bounds-box')
      const boundsY = Number(nodeBoundsBox.getAttribute('y'))
      const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

      expect(boundsY).toBeLessThanOrEqual(420)
      expect(boundsY + boundsHeight).toBeGreaterThanOrEqual(-0.5)
    })
  })

  it('drops stale zoom focus when the focused edge node leaves the mesh', async () => {
    const user = userEvent.setup()
    const nodesWithoutLemony = MESH_NODES.filter((node) => node.id !== 'lemony')
    const { rerender } = render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony = getNodeButton('View LEMONY-28 node')

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))
    await waitFor(() => expect(pixelValue(lemony.style.top)).toBeGreaterThan(0))
    const focusX = pixelValue(lemony.style.left)
    const focusY = pixelValue(lemony.style.top) + 18

    for (let index = 0; index < 12; index += 1) {
      fireEvent.wheel(canvas, { deltaY: -100, clientX: focusX, clientY: focusY, cancelable: true })
    }

    rerender(<MeshViz nodes={nodesWithoutLemony} selfId="self" height={420} />)

    await waitFor(() => {
      const nodeBoundsBox = screen.getByTestId('mesh-node-bounds-box')
      const boundsY = Number(nodeBoundsBox.getAttribute('y'))
      const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

      expect(boundsY).toBeGreaterThanOrEqual(0)
      expect(boundsY + boundsHeight).toBeLessThanOrEqual(420)
    })
  })

  it('lets zoomed MeshViz centered bounds pan naturally until the hard viewport limit', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={OVERSIZED_MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))

    for (let index = 0; index < 12; index += 1) {
      await user.click(screen.getByRole('button', { name: /zoom in/i }))
    }

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 100, clientY: 2100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })

    const nodeBoundsBox = screen.getByTestId('mesh-centered-bounds-box')
    const boundsY = Number(nodeBoundsBox.getAttribute('y'))
    const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

    expect(boundsHeight).toBeGreaterThan(420)
    expect(boundsY).toBeLessThanOrEqual(420)
    expect(boundsY).toBeGreaterThan(410)
    expect(boundsY + boundsHeight).toBeGreaterThanOrEqual(0)
  })

  it('transitions MeshViz to recalculated bounds when an edge node is removed', async () => {
    const edgeNodes = [
      ...MESH_NODES,
      { ...MESH_NODES[1], id: 'edge-node', peerId: 'edge-peer', label: 'EDGE', x: 94, y: 76 }
    ]
    const { rerender } = render(<MeshViz nodes={edgeNodes} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(80))

    rerender(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    const exitingEdge = getNodeButton('View EDGE node')
    expect(exitingEdge).toBeInTheDocument()
    expect(exitingEdge).toHaveAttribute('data-node-lifecycle', 'leaving')
    expect(pixelValue(lemony29.style.left)).toBeLessThan(90)
    expect(pixelValue(lemony29.style.top)).toBeGreaterThan(0)
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /view edge node/i })).not.toBeInTheDocument()
    })
    expect(pixelValue(lemony29.style.left)).toBeLessThan(200)
  })

  it('toggles a development overlay for MeshViz bounds and dead-zone visualization', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    await openTrafficDebugMenu(user)
    expect(screen.getByText('Visuals')).toBeInTheDocument()
    const boundsToggle = screen.getByRole('button', { name: /debug boundaries/i })
    expect(boundsToggle).toHaveAttribute('aria-pressed', 'false')
    expect(screen.queryByTestId('mesh-node-bounds-box')).not.toBeInTheDocument()
    expect(screen.queryByTestId('mesh-pan-dead-zone-box')).not.toBeInTheDocument()
    expect(screen.queryByTestId('mesh-centered-bounds-box')).not.toBeInTheDocument()

    await user.click(boundsToggle)

    const nodeBoundsBox = screen.getByTestId('mesh-node-bounds-box')
    const centeredBoundsBox = screen.getByTestId('mesh-centered-bounds-box')

    expect(nodeBoundsBox).toBeInTheDocument()
    expect(screen.getByTestId('mesh-pan-dead-zone-box')).toBeInTheDocument()
    expect(centeredBoundsBox).toBeInTheDocument()
    expect(nodeBoundsBox).toHaveAttribute('stroke', 'color-mix(in oklab, var(--color-good) 78%, transparent)')
    expect(centeredBoundsBox).toHaveAttribute('stroke', 'color-mix(in oklab, var(--color-warn) 82%, transparent)')
    expect(centeredBoundsBox).toHaveAttribute('stroke-dasharray', '6 5')

    const nodeBoundsX = Number(nodeBoundsBox.getAttribute('x'))
    const nodeBoundsY = Number(nodeBoundsBox.getAttribute('y'))
    const nodeBoundsWidth = Number(nodeBoundsBox.getAttribute('width'))
    const nodeBoundsHeight = Number(nodeBoundsBox.getAttribute('height'))
    const centeredBoundsX = Number(centeredBoundsBox.getAttribute('x'))
    const centeredBoundsY = Number(centeredBoundsBox.getAttribute('y'))
    const centeredBoundsWidth = Number(centeredBoundsBox.getAttribute('width'))
    const centeredBoundsHeight = Number(centeredBoundsBox.getAttribute('height'))

    expect(centeredBoundsWidth).toBeCloseTo(nodeBoundsWidth / 2)
    expect(centeredBoundsHeight).toBeCloseTo(nodeBoundsHeight / 2)
    expect(centeredBoundsX).toBeCloseTo(nodeBoundsX + nodeBoundsWidth / 4)
    expect(centeredBoundsY).toBeCloseTo(nodeBoundsY + nodeBoundsHeight / 4)

    await openTrafficDebugMenu(user)
    expect(screen.getByRole('button', { name: /debug boundaries/i })).toHaveAttribute('aria-pressed', 'true')
  })

  it('toggles MeshViz between line and dot grid styles from the debug visuals menu', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expectMeshNodeCoreFill('self', 'oklch(0.66 0.22 28)', '18%')
    expect(screen.getByTestId('mesh-viz-line-grid')).toHaveAttribute(
      'stroke',
      'color-mix(in oklab, var(--color-foreground) 7.2%, transparent)'
    )
    expect(screen.queryByTestId('mesh-viz-dot-grid')).not.toBeInTheDocument()
    expect(screen.queryByTestId('mesh-viz-accent-dot-grid')).not.toBeInTheDocument()

    await openDebugMenu(user)
    const gridStyleToggle = screen.getByRole('button', { name: /toggle grid style \(lines\)/i })
    expect(gridStyleToggle).toHaveAttribute('aria-pressed', 'false')
    expect(gridStyleToggle).toHaveAttribute('aria-keyshortcuts', 'Control+G')
    expect(gridStyleToggle).toHaveTextContent('Ctrl+G')

    await user.click(gridStyleToggle)

    expect(screen.queryByTestId('mesh-viz-line-grid')).not.toBeInTheDocument()
    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.64 0.025 252 / 9%)')
    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('cx', '0')
    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('cy', '0')
    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('r', '1.35')
    const accentDot = screen.getByTestId('mesh-viz-accent-dot-grid')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.72 0.115 220 / 13%)')
    expect(Number(accentDot.getAttribute('cx'))).toBeGreaterThan(0)
    expect(accentDot.getAttribute('cx')).toBe(accentDot.getAttribute('cy'))
    expect(accentDot).toHaveAttribute('r', '1.25')
    const tertiaryDot = screen.getByTestId('mesh-viz-tertiary-dot-grid')
    expect(tertiaryDot).toHaveAttribute('fill', 'oklch(0.76 0.105 72 / 7%)')
    expect(tertiaryDot).toHaveAttribute('cx', '0')
    expect(tertiaryDot.getAttribute('cy')).toBe(accentDot.getAttribute('cy'))
    expect(tertiaryDot).toHaveAttribute('r', '0.85')

    await openDebugMenu(user)
    const activeGridStyleToggle = screen.getByRole('button', { name: /toggle grid style \(dots\)/i })
    expect(activeGridStyleToggle).toHaveAttribute('aria-pressed', 'true')
    expect(activeGridStyleToggle).toHaveAttribute('aria-keyshortcuts', 'Control+G')
    const dotThemeOptions = screen.getByRole('group', { name: /dot theme options/i })
    expect(dotThemeOptions).toHaveAttribute('aria-keyshortcuts', 'Control+C')
    const dotThemeLabel = screen.getByTestId('mesh-viz-dot-theme-label')
    expect(dotThemeLabel).toHaveTextContent('Dot Theme')
    expect(dotThemeLabel).toHaveClass('text-[length:var(--density-type-caption)]', 'text-foreground')
    expect(dotThemeLabel).not.toHaveClass('font-mono')
    expect(screen.getByRole('button', { name: /cycle dot theme/i })).toHaveTextContent('Ctrl+C')
    const ashSignalSwatch = screen.getByRole('button', { name: /dot theme 1: ash signal/i })
    const coolTraceSwatch = screen.getByRole('button', { name: /dot theme 2: cool trace/i })
    const warmTraceSwatch = screen.getByRole('button', { name: /dot theme 3: warm trace/i })

    expect(ashSignalSwatch).toHaveAttribute('aria-pressed', 'true')
    expect(coolTraceSwatch).toHaveAttribute('aria-pressed', 'false')
    expect(warmTraceSwatch).toHaveAttribute('aria-pressed', 'false')
    expect(ashSignalSwatch).toHaveClass('border-transparent', 'bg-panel-strong/45')
    expect(coolTraceSwatch).toHaveClass('border-transparent')
    expect(coolTraceSwatch).not.toHaveClass('border-foreground')
    expect(ashSignalSwatch.firstElementChild).toBe(screen.getByTestId('mesh-viz-dot-theme-1-index'))
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-1')).toHaveAttribute(
      'data-color-value',
      'oklch(0.64 0.025 252)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-1')).toHaveClass('opacity-100')
    expect(screen.getByTestId('mesh-viz-dot-theme-2-color-1')).toHaveClass('opacity-45')
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-2')).toHaveAttribute(
      'data-color-value',
      'oklch(0.72 0.115 220)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-3')).toHaveAttribute(
      'data-color-value',
      'oklch(0.76 0.105 72)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-4')).toHaveAttribute(
      'data-color-value',
      'oklch(0.66 0.22 28)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-index')).toHaveClass('text-foreground', 'opacity-100')
    expect(screen.getByTestId('mesh-viz-dot-theme-2-index')).toHaveClass('text-fg-faint', 'opacity-55')

    await user.click(coolTraceSwatch)

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.62 0.024 252 / 8%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.74 0.12 190 / 12%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.72 0.13 275 / 10%)')
    expectMeshNodeCoreFill('self', 'oklch(0.8 0.12 28)', '18%')
    expectMeshNodeCoreFill('lemony', 'oklch(0.74 0.12 190)', '14%')

    await openDebugMenu(user)
    expect(screen.getByRole('button', { name: /dot theme 2: cool trace/i })).toHaveAttribute('aria-pressed', 'true')
    await user.click(screen.getByRole('button', { name: /dot theme 3: warm trace/i }))

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.62 0.024 252 / 8%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.78 0.125 74 / 10%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.70 0.12 260 / 12%)')
    expectMeshNodeCoreFill('self', 'oklch(0.76 0.12 155)', '18%')

    await openDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /dot theme 1: ash signal/i }))

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.64 0.025 252 / 9%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.72 0.115 220 / 13%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.76 0.105 72 / 7%)')
    expectMeshNodeCoreFill('self', 'oklch(0.66 0.22 28)', '18%')
  })
})

import { createRef } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MeshViz, type MeshVizHandle } from '@/features/network/components/MeshViz'
import { MESH_VIZ_DOT_COLOR_SCHEMES } from '@/features/network/lib/mesh-viz-dot-color-schemes'
import { MESH_NODES } from '@/features/app-tabs/data'
import type { MeshNode } from '@/features/app-tabs/types'
import {
  applyMeshVizInteraction,
  createMatchMedia,
  DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT,
  DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT,
  debugNodeCoordinates,
  distanceFrom,
  expectMeshNodeCoreFill,
  fireWindowKeyDownInAct,
  getMeshCanvas,
  getMeshElement,
  getMeshNodeCore,
  getMeshNodeLabel,
  getMeshPackets,
  getNodeButton,
  nearestNodeDistance,
  openAddDebugNodesMenu,
  openDebugMenu,
  openRemoveDebugNodesMenu,
  pixelValue,
  placementCentroid,
  placementClusterRadius,
  render,
  setFullscreenElement,
  setMeshCanvasSize,
  triggerMeshResize
} from './app-tabs-test-support'

describe('MeshViz debug and visual behavior', () => {
  it('keeps MeshViz palette colors independent from global accent tokens', () => {
    const paletteColors = Object.values(MESH_VIZ_DOT_COLOR_SCHEMES).flatMap((schemes) =>
      schemes.flatMap((scheme) => [...scheme.colors, ...scheme.nodeColors])
    )

    expect(paletteColors).not.toHaveLength(0)
    for (const color of paletteColors) {
      expect(color).not.toContain('var(')
      expect(color).not.toContain('--color-accent')
    }
  })

  it('uses light-mode MeshViz dot color schemes by index', async () => {
    const user = userEvent.setup()
    document.documentElement.dataset.theme = 'light'
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    fireEvent.keyDown(window, { key: 'g', ctrlKey: true })

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.52 0.022 252 / 12%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.54 0.12 220 / 12%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.58 0.18 28 / 9%)')

    await openDebugMenu(user)
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-1')).toHaveAttribute(
      'data-color-value',
      'oklch(0.68 0.018 252)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-2')).toHaveAttribute(
      'data-color-value',
      'oklch(0.54 0.12 220)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-3')).toHaveAttribute(
      'data-color-value',
      'oklch(0.58 0.18 28)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-4')).toHaveAttribute(
      'data-color-value',
      'oklch(0.48 0.01 252)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-2-color-4')).toHaveAttribute(
      'data-color-value',
      'oklch(0.48 0.01 252)'
    )
    expect(screen.getByRole('button', { name: /dot theme 1: paper signal/i })).toHaveAttribute('aria-pressed', 'true')

    await user.click(screen.getByRole('button', { name: /dot theme 2: field trace/i }))

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.55 0.02 252 / 11%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.53 0.13 145 / 13%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.50 0.12 265 / 10%)')
    expectMeshNodeCoreFill('self', 'oklch(0.48 0.01 252)', '18%')
    expectMeshNodeCoreFill('lemony', 'oklch(0.53 0.13 145)', '14%')

    await openDebugMenu(user)
    expect(screen.getByTestId('mesh-viz-dot-theme-3-color-4')).toHaveAttribute(
      'data-color-value',
      'oklch(0.48 0.01 252)'
    )
    await user.click(screen.getByRole('button', { name: /dot theme 3: amber trace/i }))

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.55 0.02 252 / 11%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.56 0.13 74 / 11%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.50 0.12 245 / 9%)')
    expectMeshNodeCoreFill('lemony', 'oklch(0.56 0.13 74)', '14%')
    expectMeshNodeCoreFill('self', 'oklch(0.48 0.01 252)', '18%')
  })

  it('adds separately tracked DEBUG nodes from the MeshViz debug menu', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expect(screen.queryByRole('button', { name: /view debug-/i })).not.toBeInTheDocument()

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))

    expect(screen.getByText(/3 nodes \+ 1 debug/i)).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(1)

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug worker/i }))
    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug host/i }))

    expect(screen.getByText(/3 nodes \+ 3 debug/i)).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(3)
    expect(screen.getByRole('button', { name: /view debug-client-1 node/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /view debug-worker-2 node/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /view debug-host-3 node/i })).toBeInTheDocument()
  })

  it('starts initial MeshViz nodes as present while preserving join animation for later nodes', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expect(getNodeButton('View CARRACK node')).toHaveAttribute('data-node-lifecycle', 'present')
    expect(getNodeButton('View LEMONY-28 node')).toHaveAttribute('data-node-lifecycle', 'present')
    expect(getNodeButton('View LEMONY-29 node')).toHaveAttribute('data-node-lifecycle', 'present')

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))

    expect(screen.getByRole('button', { name: /view debug-client-1 node/i })).toHaveAttribute(
      'data-node-lifecycle',
      'entering'
    )
  })

  it('fades MeshViz node labels at dense counts and reveals the hovered label', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const labelLayer = screen.getByTestId('mesh-node-label-layer')

    expect(labelLayer).toHaveClass('z-[40]')
    expect(labelLayer).toContainElement(getMeshNodeLabel('self'))
    expect(labelLayer.compareDocumentPosition(getMeshNodeCore('self')) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy()
    expect(getMeshNodeLabel('self')).toHaveClass('opacity-100')

    fireEvent.keyDown(window, { key: '1', ctrlKey: true })
    fireEvent.keyDown(window, { key: '2', ctrlKey: true })
    fireEvent.keyDown(window, { key: '3', ctrlKey: true })

    await waitFor(() => expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(3))
    expect(getMeshNodeLabel('self')).toHaveClass('opacity-100', 'duration-[500ms]')
    expect(getMeshNodeLabel('debug-client-1')).toHaveClass('opacity-100')

    fireEvent.keyDown(window, { key: '1', ctrlKey: true })
    fireEvent.keyDown(window, { key: '2', ctrlKey: true })

    await waitFor(() => expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(5))
    expect(Number(getMeshNodeLabel('self').style.opacity)).toBeGreaterThan(0)
    await waitFor(() => expect(getMeshNodeLabel('self')).toHaveClass('opacity-0'))
    expect(getMeshNodeLabel('self')).toHaveClass('absolute', 'top-full')
    expect(getMeshNodeLabel('self')).toHaveClass('duration-[500ms]')
    expect(getMeshNodeLabel('debug-client-1')).toHaveClass('opacity-0', 'duration-[500ms]')

    await user.hover(getNodeButton('View CARRACK node'))

    await waitFor(() => expect(getMeshNodeLabel('self')).toHaveClass('opacity-100', 'duration-[300ms]'))
    expect(getMeshNodeLabel('debug-client-1')).toHaveClass('opacity-0')
  })

  it('removes DEBUG nodes from the nested MeshViz debug menu', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))
    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug worker/i }))
    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug host/i }))

    expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(3)

    await openRemoveDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug worker/i }))

    expect(screen.getByText(/3 nodes \+ 2 debug/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /view debug-client-1 node/i })).toBeInTheDocument()
    const removedWorkerNode = screen.getByRole('button', { name: /view debug-worker-2 node/i })
    expect(removedWorkerNode).toHaveAttribute('data-node-lifecycle', 'leaving')
    expect(removedWorkerNode).toBeDisabled()
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /view debug-worker-2 node/i })).not.toBeInTheDocument()
    })
    expect(screen.getByRole('button', { name: /view debug-host-3 node/i })).toBeInTheDocument()

    await openRemoveDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug host/i }))

    expect(screen.getByText(/3 nodes \+ 1 debug/i)).toBeInTheDocument()
    await waitFor(() => expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(1))
    expect(screen.getByRole('button', { name: /view debug-client-1 node/i })).toBeInTheDocument()
  })

  it('places new DEBUG nodes deterministically inside a sparse cluster envelope', async () => {
    const user = userEvent.setup()
    const { unmount } = render(
      <MeshViz meshId="deterministic-test-mesh" nodes={MESH_NODES} selfId="self" height={420} />
    )

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug worker/i }))

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug host/i }))

    const firstRunCoordinates = debugNodeCoordinates()
    const placementNodes: Array<Pick<MeshNode, 'x' | 'y'>> = [...MESH_NODES]
    const baseCentroid = placementCentroid(MESH_NODES)

    expect(firstRunCoordinates).toHaveLength(3)

    for (const [index, coordinate] of firstRunCoordinates.entries()) {
      const nearestDistance = nearestNodeDistance(coordinate, placementNodes)

      expect(nearestNodeDistance(coordinate, placementNodes)).toBeLessThanOrEqual(
        DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT + 0.01
      )
      expect(nearestDistance).toBeGreaterThanOrEqual(DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT - 0.01)
      expect(distanceFrom(coordinate, baseCentroid)).toBeLessThanOrEqual(
        placementClusterRadius(MESH_NODES, index) + 0.01
      )
      placementNodes.push(coordinate)
    }

    unmount()

    const repeatUser = userEvent.setup()
    render(<MeshViz meshId="deterministic-test-mesh" nodes={MESH_NODES} selfId="self" height={420} />)

    await openAddDebugNodesMenu(repeatUser)
    await repeatUser.click(screen.getByRole('menuitem', { name: /debug client/i }))

    await openAddDebugNodesMenu(repeatUser)
    await repeatUser.click(screen.getByRole('menuitem', { name: /debug worker/i }))

    await openAddDebugNodesMenu(repeatUser)
    await repeatUser.click(screen.getByRole('menuitem', { name: /debug host/i }))

    expect(debugNodeCoordinates()).toEqual(firstRunCoordinates)
  })

  it('keeps growing DEBUG node placement sparse but clustered', () => {
    render(<MeshViz meshId="clustered-debug-mesh" nodes={MESH_NODES} selfId="self" height={420} />)

    for (let index = 0; index < 12; index += 1) {
      fireEvent.keyDown(window, { key: '1', ctrlKey: true })
      fireEvent.keyDown(window, { key: '2', ctrlKey: true })
      fireEvent.keyDown(window, { key: '3', ctrlKey: true })
    }

    const coordinates = debugNodeCoordinates()
    const placementNodes: Array<Pick<MeshNode, 'x' | 'y'>> = [...MESH_NODES]
    const baseCentroid = placementCentroid(MESH_NODES)
    const finalClusterRadius = placementClusterRadius(MESH_NODES, coordinates.length - 1)

    expect(coordinates).toHaveLength(36)
    for (const [index, coordinate] of coordinates.entries()) {
      const nearestDistance = nearestNodeDistance(coordinate, placementNodes)

      expect(nearestDistance).toBeLessThanOrEqual(DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT + 0.01)
      expect(distanceFrom(coordinate, baseCentroid)).toBeLessThanOrEqual(
        placementClusterRadius(MESH_NODES, index) + 0.01
      )
      placementNodes.push(coordinate)
    }

    expect(
      coordinates.some(
        (coordinate) => nearestNodeDistance(coordinate, MESH_NODES) >= DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT - 0.01
      )
    ).toBe(true)
    expect(coordinates.every((coordinate) => distanceFrom(coordinate, baseCentroid) <= finalClusterRadius + 0.01)).toBe(
      true
    )
  })

  it('re-fits MeshViz after topology coordinates change even after manual panning', async () => {
    const { rerender } = render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 150, clientY: 130 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft + 50, 1))
    const pannedLeft = pixelValue(lemony29.style.left)
    const shiftedNodes = MESH_NODES.map((node) => (node.id === 'lemony-29' ? { ...node, x: 82, y: 82 } : node))

    rerender(<MeshViz nodes={shiftedNodes} selfId="self" height={420} />)

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(pannedLeft + 100))
  })

  it('eases MeshViz out to include a newly added node outside the current viewport', async () => {
    const { rerender } = render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))

    fireEvent.wheel(canvas, { deltaY: -100, clientX: 400, clientY: 210 })

    const joinedNode: MeshNode = {
      id: 'joined-outside-view',
      label: 'JOINED',
      subLabel: 'JOINED PEER',
      status: 'online',
      role: 'peer',
      renderKind: 'worker',
      x: 160,
      y: 82
    }

    rerender(<MeshViz nodes={[...MESH_NODES, joinedNode]} selfId="self" height={420} />)

    const joinedButton = getNodeButton('View JOINED node')

    await waitFor(() => {
      expect(pixelValue(joinedButton.style.left)).toBeGreaterThanOrEqual(0)
      expect(pixelValue(joinedButton.style.left)).toBeLessThanOrEqual(800)
      expect(pixelValue(joinedButton.style.top)).toBeGreaterThanOrEqual(0)
      expect(pixelValue(joinedButton.style.top)).toBeLessThanOrEqual(420)
    })
  })

  it('honors reduced-motion preferences for MeshViz animations', async () => {
    window.matchMedia = createMatchMedia(true)
    const meshRef = createRef<MeshVizHandle>()

    render(<MeshViz ref={meshRef} nodes={MESH_NODES} selfId="self" height={420} />)
    const radarPing = getMeshElement('.mesh-radar-ping')

    await waitFor(() => expect(radarPing.style.opacity).toBe('0'))
    expect(radarPing.style.transform).toBe('scale(1)')

    await act(async () => {
      expect(meshRef.current?.playTraffic('self', 'lemony')).toBe(true)
    })

    expect(getMeshPackets()).toHaveLength(0)
  })

  it('keeps repeated MeshViz traffic instances independent on the same transition', async () => {
    const meshRef = createRef<MeshVizHandle>()
    const sourceNodeColor = MESH_VIZ_DOT_COLOR_SCHEMES.dark[0].nodeColors[3]

    render(<MeshViz ref={meshRef} nodes={MESH_NODES} selfId="self" height={420} />)

    await act(async () => {
      expect(meshRef.current?.playTraffic('self', 'lemony')).toBe(true)
      expect(meshRef.current?.playTraffic('self', 'lemony')).toBe(true)
    })

    const packets = getMeshPackets()

    expect(packets).toHaveLength(2)
    expect(packets.every((packet) => packet.style.opacity === '0.92')).toBe(true)
    expect(packets.every((packet) => packet.style.transition.includes('opacity'))).toBe(true)
    expect(packets.every((packet) => packet.style.background.includes(sourceNodeColor))).toBe(true)
  })

  it('does not create MeshViz traffic packets for invalid transitions', async () => {
    const meshRef = createRef<MeshVizHandle>()

    render(<MeshViz ref={meshRef} nodes={MESH_NODES} selfId="self" height={420} />)

    await act(async () => {
      expect(meshRef.current?.playTraffic('self', 'self')).toBe(false)
      expect(meshRef.current?.playTraffic('self', 'missing-node')).toBe(false)
    })

    expect(getMeshPackets()).toHaveLength(0)
  })

  it('repositions in-flight MeshViz traffic packets when the canvas resizes', async () => {
    const meshRef = createRef<MeshVizHandle>()

    render(<MeshViz ref={meshRef} nodes={MESH_NODES} selfId="self" height={420} />)

    await act(async () => {
      expect(meshRef.current?.playTraffic('self', 'lemony')).toBe(true)
    })

    const packet = getMeshPackets()[0]

    if (!packet) {
      throw new Error('Expected an in-flight mesh packet')
    }

    const initialTransform = packet.style.transform

    setMeshCanvasSize(1000, 500)
    await act(async () => {
      triggerMeshResize()
    })

    expect(packet.style.transform).not.toBe(initialTransform)
  })

  it('keeps MeshViz utility controls functional without blocking empty canvas space', async () => {
    const user = userEvent.setup()
    const requestFullscreen = vi.fn(() => Promise.resolve())
    HTMLElement.prototype.requestFullscreen = requestFullscreen

    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    await user.click(screen.getByRole('button', { name: /fullscreen/i }))
    expect(requestFullscreen).toHaveBeenCalledTimes(1)

    const devControlGroup = screen.getByRole('group', { name: /mesh debug controls/i })
    const debugMenuButton = screen.getByRole('button', { name: /^debug$/i })

    expect(devControlGroup).toHaveClass('pointer-events-none')
    expect(debugMenuButton).toHaveClass('pointer-events-auto')

    await user.click(debugMenuButton)
    expect(screen.getByRole('button', { name: /random traffic/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /self traffic/i })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: /debug node actions/i })).toBeInTheDocument()
    const addNodesTrigger = screen.getByRole('button', { name: /^add nodes$/i })
    const disabledRemoveNodesTrigger = screen.getByRole('button', { name: /^remove nodes$/i })

    expect(addNodesTrigger).toHaveAttribute('aria-haspopup', 'menu')
    expect(addNodesTrigger).toHaveAttribute('aria-expanded', 'false')
    expect(disabledRemoveNodesTrigger).toHaveAttribute('aria-haspopup', 'menu')
    expect(disabledRemoveNodesTrigger).toHaveAttribute('aria-expanded', 'false')
    expect(disabledRemoveNodesTrigger).toBeDisabled()
    expect(screen.queryByRole('menuitem', { name: /debug client/i })).not.toBeInTheDocument()

    await user.click(addNodesTrigger)
    expect(addNodesTrigger).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('menu', { name: /add debug nodes/i })).toHaveClass('absolute', 'left-full')
    expect(screen.getByRole('menuitem', { name: /debug client/i })).toHaveAttribute('aria-keyshortcuts', 'Control+1')
    expect(screen.getByRole('menuitem', { name: /debug worker/i })).toHaveAttribute('aria-keyshortcuts', 'Control+2')
    expect(screen.getByRole('menuitem', { name: /debug host/i })).toHaveAttribute('aria-keyshortcuts', 'Control+3')

    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))
    await user.click(debugMenuButton)
    const removeNodesTrigger = screen.getByRole('button', { name: /^remove nodes$/i })

    expect(removeNodesTrigger).toBeEnabled()
    expect(removeNodesTrigger).toHaveAttribute('aria-haspopup', 'menu')
    expect(removeNodesTrigger).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByRole('menuitem', { name: /debug client/i })).not.toBeInTheDocument()

    await user.click(removeNodesTrigger)
    expect(removeNodesTrigger).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('menu', { name: /remove debug nodes/i })).toHaveClass('absolute', 'left-full')
    expect(screen.getByRole('menuitem', { name: /debug client/i })).toHaveAttribute('aria-keyshortcuts', 'Shift+1')
    expect(screen.getByRole('menuitem', { name: /debug worker/i })).toHaveAttribute('aria-keyshortcuts', 'Shift+2')
    expect(screen.getByRole('menuitem', { name: /debug host/i })).toHaveAttribute('aria-keyshortcuts', 'Shift+3')
    expect(screen.getByRole('button', { name: /debug boundaries/i })).toHaveAttribute('aria-keyshortcuts', 'Control+B')
    expect(screen.getByRole('button', { name: /toggle grid style \(lines\)/i })).toHaveAttribute(
      'aria-keyshortcuts',
      'Control+G'
    )
    expect(screen.getByRole('group', { name: /dot theme options/i })).toHaveAttribute('aria-keyshortcuts', 'Control+C')
    expect(screen.getByRole('button', { name: /cycle dot theme/i })).toHaveAttribute('aria-keyshortcuts', 'Control+C')
    expect(screen.getByRole('button', { name: /dot theme 1: ash signal/i })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('Z')).toBeInTheDocument()
    expect(screen.getByText('X')).toBeInTheDocument()
    expect(screen.getByText('Ctrl+G')).toBeInTheDocument()
    expect(screen.getByText('Ctrl+C')).toBeInTheDocument()
    expect(screen.getByText('Shift+1')).toBeInTheDocument()
    expect(screen.getByText('Shift+2')).toBeInTheDocument()
    expect(screen.getByText('Shift+3')).toBeInTheDocument()
  })

  it('supports MeshViz debug hotkeys outside text editing targets', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expect(screen.queryByRole('button', { name: /view debug-/i })).not.toBeInTheDocument()

    await fireWindowKeyDownInAct({ key: '1', ctrlKey: true })
    expect(screen.getByText(/3 nodes \+ 1 debug/i)).toBeInTheDocument()

    await fireWindowKeyDownInAct({ key: '2', ctrlKey: true })
    expect(screen.getByText(/3 nodes \+ 2 debug/i)).toBeInTheDocument()

    await fireWindowKeyDownInAct({ key: '3', ctrlKey: true })
    expect(screen.getByText(/3 nodes \+ 3 debug/i)).toBeInTheDocument()

    await fireWindowKeyDownInAct({ key: '!', code: 'Digit1', shiftKey: true })
    expect(screen.getByText(/3 nodes \+ 2 debug/i)).toBeInTheDocument()

    await fireWindowKeyDownInAct({ key: '@', code: 'Digit2', shiftKey: true })
    expect(screen.getByText(/3 nodes \+ 1 debug/i)).toBeInTheDocument()

    await fireWindowKeyDownInAct({ key: '#', code: 'Digit3', shiftKey: true })
    expect(screen.queryByText(/3 nodes \+ \d debug/i)).not.toBeInTheDocument()

    await fireWindowKeyDownInAct({ key: 'b', ctrlKey: true })
    expect(screen.getByTestId('mesh-centered-bounds-box')).toBeInTheDocument()

    await fireWindowKeyDownInAct({ key: 'g', ctrlKey: true })
    expect(screen.queryByTestId('mesh-viz-line-grid')).not.toBeInTheDocument()
    expect(screen.getByTestId('mesh-viz-dot-grid')).toBeInTheDocument()

    const dotThemeEvent = new KeyboardEvent('keydown', { key: 'c', ctrlKey: true, bubbles: true, cancelable: true })
    await act(async () => {
      window.dispatchEvent(dotThemeEvent)
    })
    expect(dotThemeEvent.defaultPrevented).toBe(true)
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.74 0.12 190 / 12%)')

    const randomTrafficEvent = new KeyboardEvent('keydown', { key: 'z', bubbles: true, cancelable: true })
    await applyMeshVizInteraction(() => {
      window.dispatchEvent(randomTrafficEvent)
    })
    expect(randomTrafficEvent.defaultPrevented).toBe(true)

    const selfTrafficEvent = new KeyboardEvent('keydown', { key: 'x', bubbles: true, cancelable: true })
    await applyMeshVizInteraction(() => {
      window.dispatchEvent(selfTrafficEvent)
    })
    expect(selfTrafficEvent.defaultPrevented).toBe(true)

    const input = document.createElement('input')
    document.body.append(input)
    input.focus()
    await applyMeshVizInteraction(() => {
      fireEvent.keyDown(input, { key: '1', ctrlKey: true })
    })
    expect(screen.queryByText(/3 nodes \+ \d debug/i)).not.toBeInTheDocument()
    input.remove()
  })

  it('doubles MeshViz viewport and debug controls while the canvas is fullscreen', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const debugButton = screen.getByRole('button', { name: /^debug$/i })
    const zoomInButton = screen.getByRole('button', { name: /zoom in/i })
    const zoomOutButton = screen.getByRole('button', { name: /zoom out/i })
    const resetButton = screen.getByRole('button', { name: /reset view/i })

    expect(debugButton).toHaveClass('gap-1.5', 'px-2.5', 'py-1', 'text-[length:var(--density-type-annotation)]')
    expect(zoomInButton).toHaveClass('size-[26px]')
    expect(zoomOutButton).toHaveClass('size-[26px]')
    expect(resetButton).toHaveClass('size-[26px]')

    setFullscreenElement(canvas)
    fireEvent(document, new Event('fullscreenchange'))

    await waitFor(() =>
      expect(debugButton).toHaveClass('gap-3', 'px-5', 'py-2', 'text-[length:var(--density-type-caption)]')
    )
    await waitFor(() => expect(zoomInButton).toHaveClass('size-[52px]'))
    expect(zoomOutButton).toHaveClass('size-[52px]')
    expect(resetButton).toHaveClass('size-[52px]')

    setFullscreenElement(null)
    fireEvent(document, new Event('fullscreenchange'))

    await waitFor(() =>
      expect(debugButton).toHaveClass('gap-1.5', 'px-2.5', 'py-1', 'text-[length:var(--density-type-annotation)]')
    )
    await waitFor(() => expect(zoomInButton).toHaveClass('size-[26px]'))
  })
})

// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type {
  EntryAnimation,
  ExitAnimation,
  LineTransition,
  LineRevealAnimation,
  PendingLineTransition,
  RenderVariant,
  ScreenNode,
  UpdateTwinkle
} from '@/features/dashboard/components/topology/types'
import { buildLineMesh } from '@/features/dashboard/components/topology/render/line-mesh'
import {
  Harness,
  activeGl,
  createFakeWebGLContext,
  createMutableRef,
  createRenderNode,
  flushAnimationFrame,
  installTopologyCanvasEnvironment,
  mocks,
  resetTopologyCanvasEnvironment,
  setActiveGl,
  setSupportsStandardDerivatives
} from './useTopologyCanvas.test-harness'
import { RENDER_VARIANTS } from '@/features/dashboard/components/topology/theme/render-variants'

beforeAll(() => {
  installTopologyCanvasEnvironment()
})

beforeEach(() => {
  resetTopologyCanvasEnvironment()
})

afterEach(() => {
  cleanup()
})

describe('useTopologyCanvas light transition regressions', () => {
  it('uses the render variant line shader and applies line blend state separately', () => {
    const applyLineBlendMode = vi.fn()
    const applyPointBlendMode = vi.fn()
    const renderVariant: RenderVariant = {
      ...RENDER_VARIANTS.dark,
      lineFragmentShader: 'custom-line-fragment-shader',
      pointFragmentShader: 'custom-point-fragment-shader',
      applyLineBlendMode,
      applyPointBlendMode,
      buildLines: () => ({
        positions: new Float32Array([0, 0, 80, 80]),
        colors: new Float32Array([0.4, 0.6, 0.9, 0.28, 0.4, 0.6, 0.9, 0.12])
      })
    }

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={createMutableRef<LineTransition | null>(null)}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={createMutableRef<PendingLineTransition | null>(null)}
        renderNodes={[createRenderNode('peer', { x: 0.2, y: 0.3 })]}
        renderVariant={renderVariant}
        screenNodesRef={createMutableRef<ScreenNode[]>([])}
        seenNodeIdsRef={createMutableRef<Set<string>>(new Set(['peer']))}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />
    )

    expect(mocks.createProgramMock).toHaveBeenNthCalledWith(
      1,
      expect.anything(),
      'point-vertex-shader',
      'custom-point-fragment-shader'
    )
    expect(mocks.buildLineFragmentShaderSourceMock).toHaveBeenCalledWith('custom-line-fragment-shader', {
      useStandardDerivatives: false
    })
    expect(mocks.createProgramMock).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      'line-vertex-shader',
      'custom-line-fragment-shader'
    )
    const expectedMesh = buildLineMesh({
      positions: new Float32Array([0, 0, 80, 80]),
      colors: new Float32Array([0.4, 0.6, 0.9, 0.28, 0.4, 0.6, 0.9, 0.12]),
      lineWidthPx: renderVariant.lineWidthPx,
      devicePixelRatio: 1
    })
    expect(applyLineBlendMode).toHaveBeenCalledWith(activeGl)
    expect(applyPointBlendMode).toHaveBeenCalledWith(activeGl)
    expect(activeGl.drawArrays).toHaveBeenNthCalledWith(1, activeGl.TRIANGLES, 0, expectedMesh.positions.length / 2)
    expect(activeGl.drawArrays).toHaveBeenNthCalledWith(2, activeGl.POINTS, 0, 1)
    expect(activeGl.lineWidth).not.toHaveBeenCalled()
    expect(applyLineBlendMode.mock.invocationCallOrder[0]).toBeLessThan(activeGl.drawArrays.mock.invocationCallOrder[0])
    expect(applyPointBlendMode.mock.invocationCallOrder[0]).toBeGreaterThan(
      applyLineBlendMode.mock.invocationCallOrder[0]
    )
    expect(applyPointBlendMode.mock.invocationCallOrder[0]).toBeLessThan(
      activeGl.drawArrays.mock.invocationCallOrder[1]
    )
  })

  it('enables derivative-aware line shader source when the WebGL extension is available', () => {
    setSupportsStandardDerivatives(true)
    setActiveGl(createFakeWebGLContext())
    Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
      configurable: true,
      value: (contextType: string) => (contextType === 'webgl' ? activeGl : null)
    })

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={createMutableRef<LineTransition | null>(null)}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={createMutableRef<PendingLineTransition | null>(null)}
        renderNodes={[createRenderNode('peer', { x: 0.2, y: 0.3 })]}
        screenNodesRef={createMutableRef<ScreenNode[]>([])}
        seenNodeIdsRef={createMutableRef<Set<string>>(new Set(['peer']))}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />
    )

    expect(mocks.buildLineFragmentShaderSourceMock).toHaveBeenCalledWith(RENDER_VARIANTS.light.lineFragmentShader, {
      useStandardDerivatives: true
    })
    expect(mocks.createProgramMock).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      'line-vertex-shader',
      `derivatives:${RENDER_VARIANTS.light.lineFragmentShader}`
    )
  })

  it('keeps light line rendering unchanged when hover is active', () => {
    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>('peer')}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={createMutableRef<LineTransition | null>(null)}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={createMutableRef<PendingLineTransition | null>(null)}
        renderNodes={[
          createRenderNode('anchor', { x: 0.2, y: 0.2 }),
          createRenderNode('peer', { x: 0.8, y: 0.2 }),
          createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
          createRenderNode('stable-b', { x: 0.8, y: 0.8 })
        ]}
        screenNodesRef={createMutableRef<ScreenNode[]>([])}
        seenNodeIdsRef={createMutableRef<Set<string>>(new Set(['anchor', 'peer', 'stable-a', 'stable-b']))}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />
    )

    expect(mocks.buildProximityLinesMock).toHaveBeenCalledTimes(1)
    const baseCall = mocks.buildProximityLinesMock.mock.calls[0]?.[0]

    expect([...baseCall.highlightedNodeIds]).toEqual([])
    expect(baseCall.visiblePairKeys).toBeUndefined()
    expect(new Set(baseCall.screenNodes.map((node: { id: string }) => node.id))).toEqual(
      new Set(['anchor', 'peer', 'stable-a', 'stable-b'])
    )
    expect(activeGl.drawArrays).toHaveBeenCalledTimes(1)
    expect(activeGl.drawArrays).toHaveBeenCalledWith(activeGl.POINTS, 0, 4)
  })

  it('does not change line-builder screen-node sizes when hover changes in the normal light path', () => {
    const hoveredNodeIdRef = createMutableRef<string | null>(null)
    const renderNodes = [
      createRenderNode('anchor', { x: 0.2, y: 0.2, size: 16 }),
      createRenderNode('peer', { x: 0.8, y: 0.2, size: 14 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8, size: 10 }),
      createRenderNode('stable-b', { x: 0.8, y: 0.8, size: 11 })
    ]
    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef,
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef: createMutableRef<LineTransition | null>(null),
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef: createMutableRef<PendingLineTransition | null>(null),
      screenNodesRef: createMutableRef<ScreenNode[]>([]),
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(renderNodes.map((node) => node.id))),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1)
    }

    const { rerender } = render(<Harness {...refs} renderNodes={renderNodes} selfNodeId="peer" />)

    const baseCall = mocks.buildProximityLinesMock.mock.calls[0]?.[0]
    const baseIds = baseCall.screenNodes.map((node: { id: string }) => node.id)
    const baseSizes = new Map(baseCall.screenNodes.map((node: { id: string; size: number }) => [node.id, node.size]))
    expect(baseIds).toEqual(['anchor', 'peer', 'stable-b', 'stable-a'])

    mocks.buildProximityLinesMock.mockClear()
    hoveredNodeIdRef.current = 'peer'
    rerender(<Harness {...refs} renderNodes={renderNodes.map((node) => ({ ...node }))} selfNodeId="peer" />)
    flushAnimationFrame()

    expect(mocks.buildProximityLinesMock).not.toHaveBeenCalled()
    expect(refs.screenNodesRef.current.map((node) => node.id)).toEqual(['peer', 'anchor', 'stable-b', 'stable-a'])
    const hoveredScreenNode = refs.screenNodesRef.current.find((node) => node.id === 'peer')
    expect(hoveredScreenNode?.hitSize).toBeGreaterThan(hoveredScreenNode?.size ?? 0)
    const stableLineNodes = [...refs.screenNodesRef.current]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((node) => node.id)
    expect(stableLineNodes.slice().sort()).toEqual(baseIds.slice().sort())
    expect(new Map(refs.screenNodesRef.current.map((node) => [node.id, node.size]))).toEqual(baseSizes)
  })
})

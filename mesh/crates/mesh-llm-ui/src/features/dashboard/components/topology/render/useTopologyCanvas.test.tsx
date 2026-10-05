// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { ENTRY_ANIMATION_DURATION_MS, LINE_REVEAL_DURATION_MS } from '@/features/dashboard/components/topology/helpers'
import type {
  EntryAnimation,
  ExitAnimation,
  LineTransition,
  LineRevealAnimation,
  PendingLineTransition,
  UpdateTwinkle
} from '@/features/dashboard/components/topology/types'
import {
  Harness,
  clock,
  createMutableRef,
  createRenderNode,
  createScreenNode,
  flushAnimationFrame,
  installTopologyCanvasEnvironment,
  mocks,
  resetTopologyCanvasEnvironment
} from './useTopologyCanvas.test-harness'

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
  it('keeps snapshot lines stable during the join pre-reveal branch', () => {
    mocks.state.scenario = 'join'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('anchor', { x: 0.2, y: 0.2 })),
      createScreenNode(createRenderNode('peer', { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.8, y: 0.8 }))
    ]
    const currentRenderNodes = [
      createRenderNode('anchor', { x: 0.2, y: 0.2 }),
      createRenderNode('peer', { x: 0.8, y: 0.2 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.8, y: 0.8 }),
      createRenderNode('new-node', { x: 0.5, y: 0.5 })
    ]

    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(['new-node']),
      removedNodeIds: new Set()
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(['anchor', 'peer', 'stable-a', 'stable-b'])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1)
    }

    const { rerender } = render(<Harness {...refs} renderNodes={currentRenderNodes} selfNodeId="peer" />)

    clock.now = LINE_REVEAL_DURATION_MS + 1
    rerender(<Harness {...refs} renderNodes={currentRenderNodes.map((node) => ({ ...node }))} selfNodeId="peer" />)

    const finalCall = mocks.buildProximityLinesMock.mock.calls[mocks.buildProximityLinesMock.mock.calls.length - 1]?.[0]
    expect(finalCall.screenNodes.map((node: { id: string }) => node.id)).toEqual([
      'anchor',
      'peer',
      'stable-a',
      'stable-b'
    ])
    expect(finalCall.visiblePairKeys).toBeUndefined()
    const pairAlphaOverrides = finalCall.pairAlphaOverrides
    expect(pairAlphaOverrides).toBeInstanceOf(Map)
    if (!pairAlphaOverrides) {
      throw new Error('Expected pairAlphaOverrides during the join pre-reveal branch')
    }
    expect([...pairAlphaOverrides.keys()]).toEqual(['anchor::peer'])
  })

  it('does not build a hover-specific light line pass during the join pre-reveal branch', () => {
    mocks.state.scenario = 'join'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('anchor', { x: 0.2, y: 0.2 })),
      createScreenNode(createRenderNode('peer', { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.8, y: 0.8 }))
    ]
    const currentRenderNodes = [
      createRenderNode('anchor', { x: 0.2, y: 0.2 }),
      createRenderNode('peer', { x: 0.8, y: 0.2 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.8, y: 0.8 }),
      createRenderNode('new-node', { x: 0.5, y: 0.5 })
    ]

    const hoveredNodeIdRef = createMutableRef<string | null>(null)
    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(['new-node']),
      removedNodeIds: new Set()
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef,
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(['anchor', 'peer', 'stable-a', 'stable-b'])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1)
    }

    const { rerender } = render(<Harness {...refs} renderNodes={currentRenderNodes} selfNodeId="peer" />)

    mocks.buildProximityLinesMock.mockClear()
    hoveredNodeIdRef.current = 'peer'
    clock.now = LINE_REVEAL_DURATION_MS + 1
    rerender(<Harness {...refs} renderNodes={currentRenderNodes.map((node) => ({ ...node }))} selfNodeId="peer" />)
    flushAnimationFrame()

    const calls = mocks.buildProximityLinesMock.mock.calls.map((call) => call[0])
    const finalCall = calls[calls.length - 1]

    expect(calls).toHaveLength(2)
    expect(calls.map((call) => [...call.highlightedNodeIds])).toEqual([[], []])
    expect(calls.every((call) => call.visiblePairKeys === undefined)).toBe(true)
    expect(finalCall.pairAlphaOverrides).toBeInstanceOf(Map)
    if (!finalCall.pairAlphaOverrides) {
      throw new Error('Expected pairAlphaOverrides during the hover join pre-reveal branch')
    }
    expect([...finalCall.pairAlphaOverrides.keys()]).toEqual(['anchor::peer'])
    expect(new Set(finalCall.screenNodes.map((node: { id: string }) => node.id))).toEqual(
      new Set(['anchor', 'peer', 'stable-a', 'stable-b'])
    )
  })

  it('only fades the changed outgoing edge during the join pre-entry snapshot branch', () => {
    mocks.state.scenario = 'join'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('anchor', { x: 0.2, y: 0.2 })),
      createScreenNode(createRenderNode('peer', { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.8, y: 0.8 }))
    ]
    const currentRenderNodes = [
      createRenderNode('anchor', { x: 0.2, y: 0.2 }),
      createRenderNode('peer', { x: 0.8, y: 0.2 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.8, y: 0.8 }),
      createRenderNode('new-node', { x: 0.5, y: 0.5 })
    ]

    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(['new-node']),
      removedNodeIds: new Set()
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(['anchor', 'peer', 'stable-a', 'stable-b'])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1)
    }

    const { rerender } = render(<Harness {...refs} renderNodes={currentRenderNodes} selfNodeId="peer" />)

    mocks.buildProximityLinesMock.mockClear()
    clock.now = LINE_REVEAL_DURATION_MS - 1
    rerender(<Harness {...refs} renderNodes={currentRenderNodes.map((node) => ({ ...node }))} selfNodeId="peer" />)
    flushAnimationFrame()

    const finalCall = mocks.buildProximityLinesMock.mock.calls[mocks.buildProximityLinesMock.mock.calls.length - 1]?.[0]
    expect(finalCall.screenNodes.map((node: { id: string }) => node.id)).toEqual([
      'anchor',
      'peer',
      'stable-a',
      'stable-b'
    ])
    expect(finalCall.visiblePairKeys).toBeUndefined()

    const pairAlphaOverrides = finalCall.pairAlphaOverrides
    expect(pairAlphaOverrides).toBeInstanceOf(Map)
    if (!pairAlphaOverrides) {
      throw new Error('Expected pairAlphaOverrides during the join pre-entry snapshot branch')
    }
    expect([...pairAlphaOverrides.keys()]).toEqual(['anchor::peer'])
    expect(pairAlphaOverrides.get('anchor::peer')).toBeGreaterThan(0)
    expect(pairAlphaOverrides.get('anchor::peer')).toBeLessThan(1)
    expect(pairAlphaOverrides.has('stable-a::stable-b')).toBe(false)
  })

  it('keeps transition creation local when join churn spills through existing hubs', () => {
    mocks.state.scenario = 'join-spillover'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('anchor', { x: 0.15, y: 0.2 })),
      createScreenNode(createRenderNode('peer', { x: 0.4, y: 0.35 })),
      createScreenNode(createRenderNode('worker', { x: 0.65, y: 0.45 })),
      createScreenNode(createRenderNode('remote', { x: 0.8, y: 0.65 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.85, y: 0.82 }))
    ]
    const currentRenderNodes = [
      createRenderNode('anchor', { x: 0.15, y: 0.2 }),
      createRenderNode('peer', { x: 0.4, y: 0.35 }),
      createRenderNode('worker', { x: 0.65, y: 0.45 }),
      createRenderNode('remote', { x: 0.8, y: 0.65 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.85, y: 0.82 }),
      createRenderNode('new-node', { x: 0.28, y: 0.3 })
    ]

    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(['new-node']),
      removedNodeIds: new Set()
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={lineTransitionRef}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={pendingLineTransitionRef}
        renderNodes={currentRenderNodes}
        screenNodesRef={screenNodesRef}
        seenNodeIdsRef={createMutableRef<Set<string>>(
          new Set(['anchor', 'peer', 'worker', 'remote', 'stable-a', 'stable-b'])
        )}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />
    )

    expect(lineTransitionRef.current?.outgoingPairKeys).toEqual(new Set(['anchor::peer']))
    expect(lineTransitionRef.current?.incomingPairKeys).toEqual(new Set(['anchor::new-node', 'new-node::peer']))
    expect(lineTransitionRef.current?.stableVisiblePairKeys).toEqual(
      new Set(['peer::worker', 'worker::remote', 'peer::worker-2', 'worker::remote-2', 'stable-a::stable-b'])
    )
  })

  it('treats a same-pair reroute around a new blocker as an outgoing and incoming light-edge transition', () => {
    mocks.state.scenario = 'join-route-change'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('anchor', { x: 0.15, y: 0.2 })),
      createScreenNode(createRenderNode('peer', { x: 0.45, y: 0.35 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.85, y: 0.82 }))
    ]
    const currentRenderNodes = [
      createRenderNode('anchor', { x: 0.15, y: 0.2 }),
      createRenderNode('peer', { x: 0.45, y: 0.35 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.85, y: 0.82 }),
      createRenderNode('new-node', { x: 0.28, y: 0.3 })
    ]

    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(['new-node']),
      removedNodeIds: new Set()
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={lineTransitionRef}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={pendingLineTransitionRef}
        renderNodes={currentRenderNodes}
        screenNodesRef={screenNodesRef}
        seenNodeIdsRef={createMutableRef<Set<string>>(new Set(['anchor', 'peer', 'stable-a', 'stable-b']))}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />
    )

    expect(lineTransitionRef.current?.outgoingPairKeys).toEqual(new Set(['anchor::peer']))
    expect(lineTransitionRef.current?.incomingPairKeys).toEqual(new Set(['anchor::peer']))
    expect(lineTransitionRef.current?.stableVisiblePairKeys).toEqual(new Set(['stable-a::stable-b']))
  })

  it('keeps remote current-only spillover out of the no-outgoing join pre-reveal branch', () => {
    mocks.state.scenario = 'join-spillover-incoming-only'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('anchor', { x: 0.15, y: 0.2 })),
      createScreenNode(createRenderNode('peer', { x: 0.4, y: 0.35 })),
      createScreenNode(createRenderNode('worker', { x: 0.65, y: 0.45 })),
      createScreenNode(createRenderNode('remote', { x: 0.8, y: 0.65 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.85, y: 0.82 }))
    ]
    const currentRenderNodes = [
      createRenderNode('anchor', { x: 0.15, y: 0.2 }),
      createRenderNode('peer', { x: 0.4, y: 0.35 }),
      createRenderNode('worker', { x: 0.65, y: 0.45 }),
      createRenderNode('remote', { x: 0.8, y: 0.65 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.85, y: 0.82 }),
      createRenderNode('new-node', { x: 0.28, y: 0.3 })
    ]

    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(['new-node']),
      removedNodeIds: new Set()
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(
        new Set(['anchor', 'peer', 'worker', 'remote', 'stable-a', 'stable-b'])
      ),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1)
    }

    const { rerender } = render(<Harness {...refs} renderNodes={currentRenderNodes} selfNodeId="peer" />)

    mocks.buildProximityLinesMock.mockClear()
    clock.now = 30
    rerender(<Harness {...refs} renderNodes={currentRenderNodes.map((node) => ({ ...node }))} selfNodeId="peer" />)
    flushAnimationFrame()

    const finalCall = mocks.buildProximityLinesMock.mock.calls[mocks.buildProximityLinesMock.mock.calls.length - 1]?.[0]
    expect(lineTransitionRef.current?.outgoingPairKeys).toEqual(new Set())
    expect(lineTransitionRef.current?.incomingPairKeys).toEqual(new Set(['anchor::new-node', 'new-node::peer']))
    expect(finalCall.screenNodes.map((node: { id: string }) => node.id)).toEqual([
      'anchor',
      'peer',
      'worker',
      'remote',
      'stable-a',
      'stable-b'
    ])
    expect(finalCall.visiblePairKeys).toBeUndefined()
    const pairAlphaOverrides = finalCall.pairAlphaOverrides
    expect(pairAlphaOverrides).toBeInstanceOf(Map)
    if (!pairAlphaOverrides) {
      throw new Error('Expected pairAlphaOverrides during the no-outgoing join pre-reveal branch')
    }
    expect(pairAlphaOverrides.size).toBe(0)
  })

  it('keeps the previous snapshot during the leave move-settle window before revealing replacement edges', () => {
    mocks.state.scenario = 'leave'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('peer', { x: 0.25, y: 0.25 })),
      createScreenNode(createRenderNode('removed-node', { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.8, y: 0.8 }))
    ]
    const currentRenderNodes = [
      createRenderNode('peer', { x: 0.25, y: 0.25 }),
      createRenderNode('replacement', { x: 0.82, y: 0.26 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.8, y: 0.8 })
    ]

    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(),
      removedNodeIds: new Set(['removed-node'])
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(['peer', 'removed-node', 'stable-a', 'stable-b'])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1)
    }

    const { rerender } = render(<Harness {...refs} renderNodes={currentRenderNodes} selfNodeId="peer" />)

    mocks.buildProximityLinesMock.mockClear()
    clock.now = LINE_REVEAL_DURATION_MS + 1
    rerender(<Harness {...refs} renderNodes={currentRenderNodes.map((node) => ({ ...node }))} selfNodeId="peer" />)
    flushAnimationFrame()

    const finalCall = mocks.buildProximityLinesMock.mock.calls[mocks.buildProximityLinesMock.mock.calls.length - 1]?.[0]
    expect(finalCall.screenNodes.map((node: { id: string }) => node.id)).toEqual([
      'peer',
      'removed-node',
      'stable-a',
      'stable-b'
    ])
    expect(finalCall.visiblePairKeys).toBeUndefined()
    const pairAlphaOverrides = finalCall.pairAlphaOverrides
    expect(pairAlphaOverrides).toBeInstanceOf(Map)
    if (!pairAlphaOverrides) {
      throw new Error('Expected pairAlphaOverrides during the leave move-settle window')
    }
    expect([...pairAlphaOverrides.keys()]).toEqual(['peer::removed-node'])
    expect(pairAlphaOverrides.get('peer::removed-node')).toBe(0)
  })

  it('reveals the replacement edge only after the leave move-settle window completes', () => {
    mocks.state.scenario = 'leave'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('peer', { x: 0.25, y: 0.25 })),
      createScreenNode(createRenderNode('removed-node', { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.8, y: 0.8 }))
    ]
    const currentRenderNodes = [
      createRenderNode('peer', { x: 0.25, y: 0.25 }),
      createRenderNode('replacement', { x: 0.82, y: 0.26 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.8, y: 0.8 })
    ]

    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(),
      removedNodeIds: new Set(['removed-node'])
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(['peer', 'removed-node', 'stable-a', 'stable-b'])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1)
    }

    const { rerender } = render(<Harness {...refs} renderNodes={currentRenderNodes} selfNodeId="peer" />)

    mocks.buildProximityLinesMock.mockClear()
    clock.now = LINE_REVEAL_DURATION_MS + ENTRY_ANIMATION_DURATION_MS + 1
    rerender(<Harness {...refs} renderNodes={currentRenderNodes.map((node) => ({ ...node }))} selfNodeId="peer" />)
    flushAnimationFrame()

    const finalCall = mocks.buildProximityLinesMock.mock.calls[mocks.buildProximityLinesMock.mock.calls.length - 1]?.[0]
    expect(finalCall.visiblePairKeys).toEqual(new Set(['peer::replacement', 'stable-a::stable-b']))
    const pairAlphaOverrides = finalCall.pairAlphaOverrides
    expect(pairAlphaOverrides).toBeInstanceOf(Map)
    if (!pairAlphaOverrides) {
      throw new Error('Expected pairAlphaOverrides during the leave reveal branch')
    }
    expect([...pairAlphaOverrides.keys()]).toEqual(['peer::replacement'])
    expect(pairAlphaOverrides.get('peer::replacement')).toBeGreaterThan(0)
    expect(pairAlphaOverrides.get('peer::replacement')).toBeLessThanOrEqual(1)
  })

  it('keeps moved existing nodes line-hidden until they settle during the light leave reveal path', () => {
    mocks.state.scenario = 'leave'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('peer', { x: 0.25, y: 0.25 })),
      createScreenNode(createRenderNode('removed-node', { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.8, y: 0.8 }))
    ]
    const currentRenderNodes = [
      createRenderNode('peer', { x: 0.25, y: 0.25 }),
      createRenderNode('replacement', { x: 0.82, y: 0.26 }),
      createRenderNode('stable-a', { x: 0.52, y: 0.62 }),
      createRenderNode('stable-b', { x: 0.8, y: 0.8 })
    ]

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(
        new Map([
          ['peer', { x: 0.25 * 800, y: 0.25 * 600 }],
          ['stable-a', { x: 0.2 * 800, y: 0.8 * 600 }],
          ['stable-b', { x: 0.8 * 800, y: 0.8 * 600 }]
        ])
      ),
      lineTransitionRef: createMutableRef<LineTransition | null>(null),
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef: createMutableRef<PendingLineTransition | null>({
        addedNodeIds: new Set(),
        removedNodeIds: new Set(['removed-node'])
      }),
      screenNodesRef: createMutableRef(previousScreenNodes),
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(['peer', 'removed-node', 'stable-a', 'stable-b'])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1)
    }

    const { rerender } = render(<Harness {...refs} renderNodes={currentRenderNodes} selfNodeId="peer" />)

    mocks.buildProximityLinesMock.mockClear()
    clock.now = LINE_REVEAL_DURATION_MS + ENTRY_ANIMATION_DURATION_MS + 1
    rerender(<Harness {...refs} renderNodes={currentRenderNodes.map((node) => ({ ...node }))} selfNodeId="peer" />)
    flushAnimationFrame()

    const finalCall = mocks.buildProximityLinesMock.mock.calls[mocks.buildProximityLinesMock.mock.calls.length - 1]?.[0]
    const movedNode = finalCall.screenNodes.find((node: { id: string }) => node.id === 'stable-a')

    expect(finalCall.visiblePairKeys).toEqual(new Set(['peer::replacement']))
    expect(movedNode?.lineRevealProgress).toBe(0)
  })

  it('keeps removal transition creation local when current churn spills through replacement edges', () => {
    mocks.state.scenario = 'leave-spillover'

    const previousScreenNodes = [
      createScreenNode(createRenderNode('peer', { x: 0.25, y: 0.25 })),
      createScreenNode(createRenderNode('removed-node', { x: 0.52, y: 0.22 })),
      createScreenNode(createRenderNode('stable-a', { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode('stable-b', { x: 0.8, y: 0.8 }))
    ]
    const currentRenderNodes = [
      createRenderNode('peer', { x: 0.25, y: 0.25 }),
      createRenderNode('replacement', { x: 0.54, y: 0.24 }),
      createRenderNode('worker-2', { x: 0.68, y: 0.44 }),
      createRenderNode('remote-2', { x: 0.82, y: 0.62 }),
      createRenderNode('stable-a', { x: 0.2, y: 0.8 }),
      createRenderNode('stable-b', { x: 0.8, y: 0.8 })
    ]

    const screenNodesRef = createMutableRef(previousScreenNodes)
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(),
      removedNodeIds: new Set(['removed-node'])
    })
    const lineTransitionRef = createMutableRef<LineTransition | null>(null)

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={lineTransitionRef}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={pendingLineTransitionRef}
        renderNodes={currentRenderNodes}
        screenNodesRef={screenNodesRef}
        seenNodeIdsRef={createMutableRef<Set<string>>(new Set(['peer', 'removed-node', 'stable-a', 'stable-b']))}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />
    )

    expect(lineTransitionRef.current?.outgoingPairKeys).toEqual(new Set(['peer::removed-node']))
    expect(lineTransitionRef.current?.incomingPairKeys).toEqual(new Set(['peer::replacement']))
    expect(lineTransitionRef.current?.stableVisiblePairKeys).toEqual(
      new Set(['replacement::worker-2', 'worker-2::remote-2', 'stable-a::stable-b'])
    )
  })
})

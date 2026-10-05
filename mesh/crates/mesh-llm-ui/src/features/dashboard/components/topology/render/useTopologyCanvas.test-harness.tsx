// @vitest-environment jsdom
/* eslint-disable react-refresh/only-export-components */

import '@testing-library/jest-dom/vitest'
import { type MutableRefObject } from 'react'
import { vi } from 'vitest'

import { RENDER_VARIANTS } from '@/features/dashboard/components/topology/theme/render-variants'
import type {
  EntryAnimation,
  ExitAnimation,
  LineTransition,
  LineRevealAnimation,
  PendingLineTransition,
  RenderNode,
  RenderVariant,
  ScreenNode,
  UpdateTwinkle
} from '@/features/dashboard/components/topology/types'
import { useTopologyCanvas } from '@/features/dashboard/components/topology/render/useTopologyCanvas'

type ScenarioName =
  'join' | 'join-route-change' | 'join-spillover' | 'join-spillover-incoming-only' | 'leave' | 'leave-spillover'

const hoistedMocks = vi.hoisted(() => {
  const state = { scenario: 'join' as ScenarioName }

  const buildProximityLinesMock = vi.fn(
    (input: {
      screenNodes: Array<{
        id: string
        size: number
        hitSize: number
        lineRevealProgress: number
      }>
      highlightedNodeIds: Set<string>
      visiblePairKeys?: Set<string>
      pairAlphaOverrides?: Map<string, number>
    }) => {
      const ids = new Set(input.screenNodes.map((node) => node.id))

      let pairKeys: string[]
      let pairRouteSignatures: Map<string, string>
      if (input.visiblePairKeys) {
        pairKeys = [...input.visiblePairKeys]
      } else if (state.scenario === 'join') {
        pairKeys = ids.has('new-node') ? ['anchor::new-node', 'new-node::peer'] : ['anchor::peer', 'stable-a::stable-b']
      } else if (state.scenario === 'join-route-change') {
        pairKeys = ['anchor::peer', 'stable-a::stable-b']
      } else if (state.scenario === 'join-spillover') {
        pairKeys = ids.has('new-node')
          ? ['anchor::new-node', 'new-node::peer', 'peer::worker-2', 'worker::remote-2', 'stable-a::stable-b']
          : ['anchor::peer', 'peer::worker', 'worker::remote', 'stable-a::stable-b']
      } else if (state.scenario === 'join-spillover-incoming-only') {
        pairKeys = ids.has('new-node')
          ? ['anchor::new-node', 'new-node::peer', 'worker::remote-2', 'peer::worker-2', 'stable-a::stable-b']
          : ['stable-a::stable-b', 'worker::remote', 'peer::worker']
      } else if (state.scenario === 'leave') {
        pairKeys = ids.has('replacement') ? ['peer::replacement'] : ['peer::removed-node', 'stable-a::stable-b']
      } else {
        pairKeys = ids.has('replacement')
          ? ['peer::replacement', 'replacement::worker-2', 'worker-2::remote-2', 'stable-a::stable-b']
          : ['peer::removed-node', 'stable-a::stable-b']
      }

      // eslint-disable-next-line prefer-const -- assigned after conditional control flow branches determine pairKeys
      pairRouteSignatures = new Map(
        pairKeys.map((pairKey) => {
          if (state.scenario === 'join-route-change' && pairKey === 'anchor::peer') {
            const signature = ids.has('new-node')
              ? JSON.stringify({
                  mode: 'detour',
                  axis: 'y',
                  side: 'up',
                  blockerIds: ['new-node']
                })
              : JSON.stringify({ mode: 'straight', blockerIds: [] })
            return [pairKey, signature]
          }

          return [pairKey, JSON.stringify({ mode: 'straight', blockerIds: [] })]
        })
      )

      return {
        positions: new Float32Array(),
        colors: new Float32Array(),
        pairKeys,
        pairRouteSignatures
      }
    }
  )

  const createProgramMock = vi.fn(() => ({}))
  const buildLineFragmentShaderSourceMock = vi.fn(
    (fragmentSource: string, options: { useStandardDerivatives: boolean }) =>
      options.useStandardDerivatives ? `derivatives:${fragmentSource}` : fragmentSource
  )

  return {
    state,
    buildProximityLinesMock,
    createProgramMock,
    buildLineFragmentShaderSourceMock
  }
})

export const mocks = {
  state: hoistedMocks.state,
  buildProximityLinesMock: hoistedMocks.buildProximityLinesMock,
  createProgramMock: hoistedMocks.createProgramMock,
  buildLineFragmentShaderSourceMock: hoistedMocks.buildLineFragmentShaderSourceMock
}

vi.mock('@/features/dashboard/components/topology/render/line-builders', () => ({
  buildProximityLines: hoistedMocks.buildProximityLinesMock
}))

vi.mock('@/features/dashboard/components/topology/render/shaders', () => ({
  buildLineFragmentShaderSource: hoistedMocks.buildLineFragmentShaderSourceMock,
  createProgram: hoistedMocks.createProgramMock,
  DARK_LINE_FRAGMENT_SHADER: 'dark-line-fragment-shader',
  DARK_POINT_FRAGMENT_SHADER: 'dark-point-fragment-shader',
  LIGHT_LINE_FRAGMENT_SHADER: 'light-line-fragment-shader',
  LIGHT_POINT_FRAGMENT_SHADER: 'light-point-fragment-shader',
  LINE_VERTEX_SHADER: 'line-vertex-shader',
  POINT_VERTEX_SHADER: 'point-vertex-shader'
}))

class MockResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

export const clock = { now: 0 }
export const rafState = {
  nextId: 1,
  callbacks: new Map<number, FrameRequestCallback>()
}

export let supportsStandardDerivatives = false

export function setSupportsStandardDerivatives(value: boolean) {
  supportsStandardDerivatives = value
}

export let activeGl: ReturnType<typeof createFakeWebGLContext>

export function setActiveGl(next: ReturnType<typeof createFakeWebGLContext>) {
  activeGl = next
}

export function flushAnimationFrame(timestamp = clock.now) {
  const callbacks = [...rafState.callbacks.values()]
  rafState.callbacks.clear()
  for (const callback of callbacks) {
    callback(timestamp)
  }
}

export function createFakeWebGLContext() {
  const attributeLocations = new Map<string, number>([
    ['a_position', 0],
    ['a_size', 1],
    ['a_color', 2],
    ['a_pulse', 3],
    ['a_twinkle', 4],
    ['a_lineCoord', 5]
  ])

  const uniformLocation = {} as WebGLUniformLocation
  const buffer = {} as WebGLBuffer
  const program = {} as WebGLProgram

  return {
    ARRAY_BUFFER: 0x8892,
    BLEND: 0x0be2,
    COLOR_BUFFER_BIT: 0x4000,
    DYNAMIC_DRAW: 0x88e8,
    FLOAT: 0x1406,
    LINES: 0x0001,
    ONE: 1,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    POINTS: 0x0000,
    SRC_ALPHA: 0x0302,
    TRIANGLES: 0x0004,
    blendFunc: vi.fn(),
    blendFuncSeparate: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    bufferSubData: vi.fn(),
    clear: vi.fn(),
    clearColor: vi.fn(),
    createBuffer: vi.fn(() => buffer),
    deleteBuffer: vi.fn(),
    deleteProgram: vi.fn(),
    disableVertexAttribArray: vi.fn(),
    drawArrays: vi.fn(),
    enable: vi.fn(),
    enableVertexAttribArray: vi.fn(),
    getAttribLocation: vi.fn((_program: WebGLProgram, name: string) => attributeLocations.get(name) ?? 0),
    getExtension: vi.fn((name: string) =>
      name === 'OES_standard_derivatives' && supportsStandardDerivatives ? {} : null
    ),
    getUniformLocation: vi.fn(() => uniformLocation),
    lineWidth: vi.fn(),
    uniform1f: vi.fn(),
    uniform2f: vi.fn(),
    useProgram: vi.fn(),
    vertexAttribPointer: vi.fn(),
    viewport: vi.fn(),
    createProgram: vi.fn(() => program)
  }
}

export function createRenderNode(id: string, overrides: Partial<RenderNode> = {}): RenderNode {
  return {
    id,
    label: id,
    subtitle: id,
    role: 'Worker',
    latencyLabel: '0ms',
    vramLabel: '0 GB',
    modelLabel: '',
    gpuLabel: '',
    x: 0.1,
    y: 0.1,
    size: 10,
    color: [0.2, 0.3, 0.4, 1],
    lineColor: [0.2, 0.3, 0.4, 1],
    pulse: 0,
    selectedModelMatch: false,
    z: 0,
    ...overrides
  }
}

export function createScreenNode(node: RenderNode): ScreenNode {
  return {
    ...node,
    px: node.x * 800,
    py: node.y * 600,
    hitSize: node.size,
    lineRevealProgress: 1
  }
}

export function createMutableRef<T>(current: T): MutableRefObject<T> {
  return { current }
}

export function Harness({
  animationRef,
  canvasRef,
  exitAnimationRef,
  hostRef,
  hoveredNodeIdRef,
  lastScreenPositionsRef,
  lineTransitionRef,
  lineRevealRef,
  panRef,
  pendingLineTransitionRef,
  renderNodes,
  renderVariant = RENDER_VARIANTS.light,
  screenNodesRef,
  seenNodeIdsRef,
  selfNodeId,
  twinkleAnimationRef,
  zoomRef
}: {
  canvasRef: MutableRefObject<HTMLCanvasElement | null>
  hostRef: MutableRefObject<HTMLDivElement | null>
  screenNodesRef: MutableRefObject<ScreenNode[]>
  animationRef: MutableRefObject<Map<string, EntryAnimation>>
  lineRevealRef: MutableRefObject<Map<string, LineRevealAnimation>>
  exitAnimationRef: MutableRefObject<Map<string, ExitAnimation>>
  twinkleAnimationRef: MutableRefObject<Map<string, UpdateTwinkle>>
  pendingLineTransitionRef: MutableRefObject<PendingLineTransition | null>
  lineTransitionRef: MutableRefObject<LineTransition | null>
  lastScreenPositionsRef: MutableRefObject<Map<string, { x: number; y: number }>>
  seenNodeIdsRef: MutableRefObject<Set<string>>
  hoveredNodeIdRef: MutableRefObject<string | null>
  zoomRef: MutableRefObject<number>
  panRef: MutableRefObject<{ x: number; y: number }>
  renderNodes: RenderNode[]
  renderVariant?: RenderVariant
  selfNodeId?: string
}) {
  useTopologyCanvas({
    animationRef,
    canvasRef,
    exitAnimationRef,
    hostRef,
    hoveredNodeIdRef,
    lastScreenPositionsRef,
    lineTransitionRef,
    lineRevealRef,
    panRef,
    pendingLineTransitionRef,
    renderNodes,
    renderVariant,
    screenNodesRef,
    seenNodeIdsRef,
    selfNodeId,
    twinkleAnimationRef,
    zoomRef
  })

  return (
    <div ref={hostRef}>
      <canvas ref={canvasRef} />
    </div>
  )
}

export function installTopologyCanvasEnvironment() {
  Object.defineProperty(window, 'ResizeObserver', {
    configurable: true,
    writable: true,
    value: MockResizeObserver
  })

  Object.defineProperty(globalThis, 'WebGLRenderingContext', {
    configurable: true,
    writable: true,
    value: function WebGLRenderingContext() {}
  })

  Object.defineProperty(window, 'devicePixelRatio', {
    configurable: true,
    value: 1
  })

  Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({
      width: 800,
      height: 600,
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 800,
      bottom: 600,
      toJSON: () => ({})
    })
  })

  Object.defineProperty(window, 'requestAnimationFrame', {
    configurable: true,
    writable: true,
    value: (callback: FrameRequestCallback) => {
      const id = rafState.nextId
      rafState.nextId += 1
      rafState.callbacks.set(id, callback)
      return id
    }
  })

  Object.defineProperty(window, 'cancelAnimationFrame', {
    configurable: true,
    writable: true,
    value: (id: number) => {
      rafState.callbacks.delete(id)
    }
  })

  Object.defineProperty(performance, 'now', {
    configurable: true,
    value: () => clock.now
  })
}

export function resetTopologyCanvasEnvironment() {
  mocks.state.scenario = 'join'
  mocks.buildProximityLinesMock.mockClear()
  mocks.buildLineFragmentShaderSourceMock.mockClear()
  mocks.createProgramMock.mockClear()
  rafState.callbacks.clear()
  rafState.nextId = 1
  clock.now = 0
  supportsStandardDerivatives = false

  activeGl = createFakeWebGLContext()
  Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
    configurable: true,
    value: (contextType: string) => (contextType === 'webgl' ? activeGl : null)
  })
}

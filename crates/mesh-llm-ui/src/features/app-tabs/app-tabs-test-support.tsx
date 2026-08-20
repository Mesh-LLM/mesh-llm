/* eslint-disable react-refresh/only-export-components */
import { type ReactElement, type ReactNode } from 'react'
import { act, fireEvent, render as rtlRender, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, vi } from 'vitest'
import { AppProviders } from '@/app/providers/AppProviders'
import { APP_STORAGE_KEYS, MESH_NODES } from '@/features/app-tabs/data'
import type { MeshNode } from '@/features/app-tabs/types'
import { env } from '@/lib/env'
let resizeCallback: ResizeObserverCallback | undefined
let meshCanvasWidth = 800
let meshCanvasHeight = 420
let fullscreenElement: Element | null = null
const DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT = 20
const DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT = 7
const DEBUG_PLACEMENT_CLUSTER_PADDING_PERCENT = 24
const DEBUG_PLACEMENT_CLUSTER_GROWTH_PERCENT = 4

// Full-surface UI workflows can exceed Vitest's default timeout under CI worker contention.
vi.setConfig({ testTimeout: 15_000 })

function TestProviders({ children }: { children: ReactNode }) {
  return (
    <AppProviders initialDataMode="harness" persistDataMode={false}>
      {children}
    </AppProviders>
  )
}

function render(ui: ReactElement) {
  return rtlRender(ui, { wrapper: TestProviders })
}

class ControlledResizeObserver implements ResizeObserver {
  constructor(callback: ResizeObserverCallback) {
    resizeCallback = callback
  }

  observe(_target: Element, _options?: ResizeObserverOptions): void {}

  unobserve(_target: Element): void {}

  disconnect(): void {}
}

const controlledResizeObserver: ResizeObserver = {
  observe(): void {},
  unobserve(): void {},
  disconnect(): void {}
}

function createMatchMedia(matches: boolean) {
  return vi.fn((query: string): MediaQueryList => ({
    matches,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: () => false
  }))
}

function setMeshCanvasSize(width: number, height: number) {
  meshCanvasWidth = width
  meshCanvasHeight = height
}

export function triggerMeshResize() {
  if (resizeCallback) {
    resizeCallback([], controlledResizeObserver)
  }
}

export async function applyMeshVizInteraction(action: () => void) {
  await act(async () => {
    action()
  })
}

async function triggerMeshResizeInAct() {
  await applyMeshVizInteraction(triggerMeshResize)
}

async function fireWindowKeyDownInAct(init: KeyboardEventInit) {
  await applyMeshVizInteraction(() => {
    fireEvent.keyDown(window, init)
  })
}

function setFullscreenElement(element: Element | null) {
  fullscreenElement = element
}

function getMeshCanvas() {
  const canvasElement = document.querySelector('.mesh-canvas')

  if (!(canvasElement instanceof HTMLElement)) {
    throw new Error('Expected mesh canvas element')
  }

  return canvasElement
}

function getMeshElement(selector: string) {
  const element = document.querySelector(selector)

  if (!(element instanceof HTMLElement)) {
    throw new Error(`Expected ${selector} element`)
  }

  return element
}

function getMeshPackets() {
  return Array.from(document.querySelectorAll('.mesh-packet')).filter(
    (element): element is HTMLElement => element instanceof HTMLElement
  )
}

function getTomlSource() {
  return screen.getByRole('textbox', { name: /configuration toml source/i }) as HTMLTextAreaElement
}

function countTomlOccurrences(value: string) {
  return getTomlSource().value.split(value).length - 1
}

async function expectTomlOccurrences(user: ReturnType<typeof userEvent.setup>, value: string, expected: number) {
  await user.click(screen.getByRole('tab', { name: 'TOML Output' }))
  expect(countTomlOccurrences(value)).toBe(expected)
  await user.click(screen.getByRole('tab', { name: 'Model Deployment' }))
}

function getMeshLinkPairs() {
  return screen.getAllByTestId('mesh-link').map((element) => {
    if (element.tagName.toLowerCase() !== 'line') {
      throw new Error('Expected mesh link to be an SVG line')
    }

    const source = element.dataset.sourceNodeId
    const target = element.dataset.targetNodeId

    if (!source || !target) {
      throw new Error('Expected mesh link source and target data attributes')
    }

    return [source, target].sort() as [string, string]
  })
}

function linkKey(pair: [string, string]) {
  return pair.join('::')
}

function linkDegree(pairs: [string, string][], nodeId: string) {
  return pairs.filter((pair) => pair.includes(nodeId)).length
}

function nonClientLinkDegree(pairs: [string, string][], nodeId: string, clientIds: Set<string>) {
  return pairs.filter((pair) => pair.includes(nodeId) && pair.every((id) => !clientIds.has(id))).length
}

function getNodeButton(label: string) {
  const button = screen.getByRole('button', { name: label })

  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Expected ${label} to be a button`)
  }

  return button
}

function getMeshNodeLabel(nodeId: string) {
  const label = screen
    .getAllByTestId('mesh-node-label')
    .find((element) => element.getAttribute('data-node-id') === nodeId)

  if (!(label instanceof HTMLElement)) {
    throw new Error(`Expected mesh node label for ${nodeId}`)
  }

  return label
}

function getMeshNodeContextHighlight(nodeId: string) {
  const highlight = screen
    .getAllByTestId('mesh-node-context-highlight')
    .find((element) => element.getAttribute('data-node-id') === nodeId)

  if (!(highlight instanceof HTMLElement)) {
    throw new Error(`Expected mesh node context highlight for ${nodeId}`)
  }

  return highlight
}

function getMeshNodeCore(nodeId: string) {
  const core = screen
    .getAllByTestId('mesh-node-core')
    .find((element) => element.getAttribute('data-node-id') === nodeId)

  if (!(core instanceof HTMLElement)) {
    throw new Error(`Expected mesh node core for ${nodeId}`)
  }

  return core
}

function expectMeshNodeCoreFill(nodeId: string, color: string, mixPercent: '14%' | '18%') {
  const core = getMeshNodeCore(nodeId)

  expect(core.style.color).toBe(color)
  expect(core.style.backgroundColor).toBe(`color-mix(in oklab, currentColor ${mixPercent}, var(--color-panel-strong))`)
}

function getMeshNodeCoreOverlay(nodeId: string) {
  const overlay = screen
    .getAllByTestId('mesh-node-core-overlay')
    .find((element) => element.getAttribute('data-node-id') === nodeId)

  if (!(overlay instanceof HTMLElement)) {
    throw new Error(`Expected mesh node core overlay for ${nodeId}`)
  }

  return overlay
}

async function openDebugMenu(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /^debug$/i }))
}

async function openTrafficDebugMenu(user: ReturnType<typeof userEvent.setup>) {
  await openDebugMenu(user)
}

async function openAddDebugNodesMenu(user: ReturnType<typeof userEvent.setup>) {
  await openDebugMenu(user)
  const addNodesTrigger = screen.getByRole('button', { name: /^add nodes$/i })

  fireEvent.pointerEnter(addNodesTrigger)
  await screen.findByRole('menuitem', { name: /debug client/i })
}

async function openRemoveDebugNodesMenu(user: ReturnType<typeof userEvent.setup>) {
  await openDebugMenu(user)
  const removeNodesTrigger = screen.getByRole('button', { name: /^remove nodes$/i })

  fireEvent.pointerEnter(removeNodesTrigger)
  await screen.findByRole('menuitem', { name: /debug client/i })
}

function pixelValue(value: string) {
  return Number.parseFloat(value.replace('px', ''))
}

function debugNodeCoordinates() {
  return screen.getAllByRole('button', { name: /view debug-/i }).map((button) => {
    const x = Number(button.getAttribute('data-node-x'))
    const y = Number(button.getAttribute('data-node-y'))

    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new Error('Expected debug node to expose deterministic coordinates')
    }

    return { x, y }
  })
}

function nearestNodeDistance(point: Pick<MeshNode, 'x' | 'y'>, nodes: Array<Pick<MeshNode, 'x' | 'y'>>) {
  return Math.min(...nodes.map((node) => Math.hypot(point.x - node.x, point.y - node.y)))
}

function placementCentroid(nodes: Array<Pick<MeshNode, 'x' | 'y'>>) {
  return {
    x: nodes.reduce((sum, node) => sum + node.x, 0) / nodes.length,
    y: nodes.reduce((sum, node) => sum + node.y, 0) / nodes.length
  }
}

function distanceFrom(point: Pick<MeshNode, 'x' | 'y'>, origin: Pick<MeshNode, 'x' | 'y'>) {
  return Math.hypot(point.x - origin.x, point.y - origin.y)
}

function placementClusterRadius(nodes: Array<Pick<MeshNode, 'x' | 'y'>>, debugCount: number) {
  const centroid = placementCentroid(nodes)
  const baseRadius = Math.max(...nodes.map((node) => distanceFrom(node, centroid)))

  return Math.max(
    DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT * 2,
    baseRadius +
      DEBUG_PLACEMENT_CLUSTER_PADDING_PERCENT +
      Math.sqrt(debugCount + 1) * DEBUG_PLACEMENT_CLUSTER_GROWTH_PERCENT
  )
}

export const OVERSIZED_MESH_NODES = [
  ...MESH_NODES,
  { id: 'north-edge', label: 'NORTH EDGE', subLabel: 'TEST EDGE', x: 6, y: 2, status: 'online' as const },
  { id: 'south-edge', label: 'SOUTH EDGE', subLabel: 'TEST EDGE', x: 94, y: 98, status: 'online' as const }
]
export const EXPANDED_BOUNDARY_MESH_NODES = [
  ...MESH_NODES,
  { id: 'west-boundary', label: 'WEST EDGE', subLabel: 'TEST EDGE', x: -40, y: 50, status: 'online' as const },
  { id: 'east-boundary', label: 'EAST EDGE', subLabel: 'TEST EDGE', x: 180, y: 50, status: 'online' as const }
]
export const HUGE_BOUNDARY_MESH_NODES = [
  ...MESH_NODES,
  { id: 'far-west-boundary', label: 'FAR WEST', subLabel: 'TEST EDGE', x: -400, y: 50, status: 'online' as const },
  { id: 'far-east-boundary', label: 'FAR EAST', subLabel: 'TEST EDGE', x: 700, y: 50, status: 'online' as const }
]

beforeEach(() => {
  env.isDevelopment = true
  resizeCallback = undefined
  setMeshCanvasSize(800, 420)
  globalThis.ResizeObserver = ControlledResizeObserver
  window.matchMedia = createMatchMedia(false)

  Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
    configurable: true,
    get() {
      return this.classList.contains('mesh-canvas') ? meshCanvasWidth : 0
    }
  })
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
    configurable: true,
    get() {
      return this.classList.contains('mesh-canvas') ? meshCanvasHeight : 0
    }
  })

  HTMLElement.prototype.setPointerCapture = vi.fn()
  HTMLElement.prototype.releasePointerCapture = vi.fn()
  HTMLElement.prototype.hasPointerCapture = vi.fn(() => true)
  HTMLElement.prototype.requestFullscreen = vi.fn(() => Promise.resolve())
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
  document.documentElement.removeAttribute('data-theme')
  document.body.style.userSelect = ''
  window.localStorage.removeItem(APP_STORAGE_KEYS.featureFlagOverrides)
  setFullscreenElement(null)
  Object.defineProperty(document, 'fullscreenElement', {
    configurable: true,
    get: () => fullscreenElement
  })
})

function createMockDataTransfer() {
  const data = new Map<string, string>()

  return {
    dropEffect: 'none',
    effectAllowed: 'all',
    get types() {
      return Array.from(data.keys())
    },
    getData: vi.fn((type: string) => data.get(type) ?? ''),
    setData: vi.fn((type: string, value: string) => {
      data.set(type, value)
    }),
    setDragImage: vi.fn()
  }
}

export {
  createMatchMedia,
  createMockDataTransfer,
  countTomlOccurrences,
  DEBUG_PLACEMENT_CLUSTER_GROWTH_PERCENT,
  DEBUG_PLACEMENT_CLUSTER_PADDING_PERCENT,
  DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT,
  DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT,
  debugNodeCoordinates,
  distanceFrom,
  expectMeshNodeCoreFill,
  expectTomlOccurrences,
  fireWindowKeyDownInAct,
  getMeshCanvas,
  getMeshElement,
  getMeshLinkPairs,
  getMeshNodeContextHighlight,
  getMeshNodeCore,
  getMeshNodeCoreOverlay,
  getMeshNodeLabel,
  getMeshPackets,
  getTomlSource,
  getNodeButton,
  linkDegree,
  linkKey,
  nearestNodeDistance,
  nonClientLinkDegree,
  openAddDebugNodesMenu,
  openDebugMenu,
  openRemoveDebugNodesMenu,
  openTrafficDebugMenu,
  pixelValue,
  placementCentroid,
  placementClusterRadius,
  render,
  setFullscreenElement,
  setMeshCanvasSize,
  triggerMeshResizeInAct
}

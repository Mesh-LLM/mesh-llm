import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { animate, type JSAnimation } from 'animejs'
import { cn } from '@/lib/cn'
import { isDevelopmentMode } from '@/lib/env'
import { buildMeshLinks } from '@/features/network/lib/mesh-links'
import {
  MESH_VIZ_DOT_COLOR_SCHEMES,
  meshVizDotColorSchemeAtIndex,
  themeFromDocument
} from '@/features/network/lib/mesh-viz-dot-color-schemes'
import {
  calculateMaxZoomOut,
  calculateNodeBounds,
  centerScreenRect,
  clampViewportToPanBounds,
  DEFAULT_VIEWPORT,
  focusPointWithinNodeBounds,
  GRID_SIZE_PX,
  gridPatternTransform,
  nodeBoundsToScreenRect,
  nodeFitsInsideViewport,
  PAN_DEAD_ZONE_PX,
  pointToScreen,
  type LayerTransform,
  type Point,
  type Viewport,
  type WorldPoint,
  viewportsMatch
} from '@/features/network/lib/mesh-viewport'
import type { MeshNode, Peer, ResolvedTheme } from '@/features/app-tabs/types'
import type { MeshVizGridMode } from '@/features/network/components/MeshVizDebugControls'
import {
  NODE_LABEL_FADE_THRESHOLD,
  nodeVisuals,
  prefersReducedMotion,
  type DebugMeshNode
} from '@/features/network/components/MeshViz.helpers'
import { type MeshVizNodeLifecycle } from '@/features/network/components/MeshVizNode'
import { useMeshVizTraffic } from '@/features/network/components/useMeshVizTraffic'
import { MeshVizCanvas } from '@/features/network/components/MeshVizCanvas'
import { useMeshVizViewport } from '@/features/network/components/useMeshVizViewport'
import { useMeshVizDebugControls } from '@/features/network/components/useMeshVizDebugControls'
import {
  useMeshVizInteractions,
  type DragState,
  type PinchZoomState,
  type TouchPointState
} from '@/features/network/components/useMeshVizInteractions'
import {
  useMeshVizLifecycleAnimations,
  type LinkRestoreTimelineRecord,
  type MeshLifecycleTimelineRecord
} from '@/features/network/components/useMeshVizLifecycleAnimations'

type MeshVizProps = {
  nodes: MeshNode[]
  selfId: string
  meshId?: string
  selectedNodeId?: string
  hoveredNodeId?: string
  dimmedNodeIds?: Set<string>
  onPick?: (node: MeshNode) => void
  height?: number
  accent?: string
  compact?: boolean
  enableDebugShortcuts?: boolean
  animateTopology?: boolean
  onFullscreen?: () => void
  onReady?: () => void
  getNodePeer?: (node: MeshNode) => Peer | undefined
}
export type MeshVizHandle = {
  playTraffic: (sourceNodeId: string, targetNodeId: string) => boolean
}

const RADAR_PULSE_DURATION = 2000
const RADAR_PULSE_LOOP_DELAY = 1000
const RADAR_PULSE_EASE = 'linear'

export const MeshViz = forwardRef<MeshVizHandle, MeshVizProps>(function MeshViz(
  {
    nodes,
    selfId,
    meshId,
    selectedNodeId,
    hoveredNodeId: externalHoveredNodeId,
    dimmedNodeIds,
    height,
    compact = false,
    enableDebugShortcuts = false,
    animateTopology = true,
    onFullscreen,
    onReady,
    getNodePeer
  }: MeshVizProps,
  ref
) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const gridPatternRef = useRef<SVGPatternElement>(null)
  const gridPathRef = useRef<SVGPathElement>(null)
  const gridDotRef = useRef<SVGCircleElement>(null)
  const gridAccentDotRef = useRef<SVGCircleElement>(null)
  const gridTertiaryDotRef = useRef<SVGCircleElement>(null)
  const svgPanLayerRef = useRef<SVGGElement>(null)
  const nodeLayerRef = useRef<HTMLDivElement>(null)
  const labelLayerRef = useRef<HTMLDivElement>(null)
  const packetLayerRef = useRef<HTMLDivElement>(null)
  const panTransformFrameRef = useRef<number | null>(null)
  const liveLayerTransformRef = useRef<LayerTransform>({ x: 0, y: 0, scale: 1 })
  const renderedViewportRef = useRef<Viewport>(DEFAULT_VIEWPORT)
  const liveLayerBaseViewportRef = useRef<Viewport>(DEFAULT_VIEWPORT)
  const liveLayerTransformActiveRef = useRef(false)
  const wheelZoomCommitTimeoutRef = useRef<number | null>(null)
  // During drag, viewportRef is live while React state stays committed; this flag clears the transient layer
  // transform after the committed viewport has rendered so there is no visible snap-back frame.
  const pendingPanTransformResetRef = useRef(false)
  const viewportAnimationRef = useRef<JSAnimation | undefined>(undefined)
  const canvasSizeRef = useRef({ width: 0, height: 0 })
  const viewportRef = useRef<Viewport>(DEFAULT_VIEWPORT)
  const zoomFocusRef = useRef<WorldPoint | undefined>(undefined)
  const zoomAnchorRef = useRef<Point | undefined>(undefined)
  const debugNodeCounterRef = useRef(0)
  const fittedNodesSignatureRef = useRef('')
  const fittedNodeIdsRef = useRef<Set<string>>(new Set())
  const topologyNodeSnapshotsRef = useRef<Map<string, MeshNode>>(new Map())
  const hasTopologySnapshotRef = useRef(false)
  const nodeLifecycleTimeoutsRef = useRef<Map<string, number>>(new Map())
  const nodeLifecycleAnimationKeysRef = useRef<Set<string>>(new Set())
  const meshLifecycleTimelineRecordsRef = useRef<Map<number, MeshLifecycleTimelineRecord>>(new Map())
  const meshLifecycleTimelineIdRef = useRef(0)
  const previousLinkIdsRef = useRef<Set<string>>(new Set())
  const linkRestoreAnimationIdsRef = useRef<Set<string>>(new Set())
  const linkRestoreTimelineRecordsRef = useRef<Map<number, LinkRestoreTimelineRecord>>(new Map())
  const linkRestoreTimelineIdRef = useRef(0)
  const hasUserControlledViewportRef = useRef(false)
  const wasFullscreenRef = useRef(false)
  const onReadyRef = useRef(onReady)
  const hasCalledOnReadyRef = useRef(false)
  useEffect(() => {
    onReadyRef.current = onReady
  }, [onReady])
  const dragRef = useRef<DragState>({
    active: false,
    pointerId: null,
    originX: 0,
    originY: 0,
    panX: 0,
    panY: 0
  })
  const touchPointersRef = useRef<Map<number, TouchPointState>>(new Map())
  const pinchZoomRef = useRef<PinchZoomState>({
    active: false,
    initialDistance: 1,
    initialZoom: DEFAULT_VIEWPORT.zoom
  })
  const radarPingRef = useRef<HTMLSpanElement>(null)
  const [openNodeId, setOpenNodeId] = useState<string | undefined>()
  const [localHoveredNodeId, setLocalHoveredNodeId] = useState<string | undefined>()
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })
  const [viewport, setViewportState] = useState<Viewport>(DEFAULT_VIEWPORT)
  const [isPanning, setIsPanning] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showPanBounds, setShowPanBounds] = useState(false)
  const [gridMode, setGridMode] = useState<MeshVizGridMode>('line')
  const [dotColorSchemeIndex, setDotColorSchemeIndex] = useState(0)
  const [dotColorSchemeTheme, setDotColorSchemeTheme] = useState<ResolvedTheme>(() =>
    typeof document === 'undefined' ? 'dark' : themeFromDocument()
  )
  const [debugNodes, setDebugNodes] = useState<DebugMeshNode[]>([])
  const [exitingNodes, setExitingNodes] = useState<MeshNode[]>([])
  const [nodeLifecyclePhases, setNodeLifecyclePhases] = useState<Record<string, MeshVizNodeLifecycle>>({})
  const reduceMotion = prefersReducedMotion()
  const isDevelopment = isDevelopmentMode()
  const debugShortcutsEnabled = isDevelopment || enableDebugShortcuts
  const currentRenderNodes = useMemo(() => [...nodes, ...debugNodes], [debugNodes, nodes])
  const currentRenderNodeIds = useMemo(() => new Set(currentRenderNodes.map((node) => node.id)), [currentRenderNodes])
  const pendingExitingNodes = useMemo(() => {
    if (reduceMotion) {
      return []
    }

    return [...topologyNodeSnapshotsRef.current.entries()]
      .filter(([nodeId]) => !currentRenderNodeIds.has(nodeId))
      .filter(([nodeId]) => nodeLifecyclePhases[nodeId] !== undefined || nodeLifecycleTimeoutsRef.current.has(nodeId))
      .map(([, node]) => node)
  }, [currentRenderNodeIds, nodeLifecyclePhases, reduceMotion])
  const pendingExitingNodeIds = useMemo(
    () => new Set(pendingExitingNodes.map((node) => node.id)),
    [pendingExitingNodes]
  )
  const renderNodes = useMemo(
    () => [
      ...currentRenderNodes,
      ...exitingNodes.filter((node) => !currentRenderNodeIds.has(node.id)),
      ...pendingExitingNodes.filter(
        (node) => !currentRenderNodeIds.has(node.id) && !exitingNodes.some((exitingNode) => exitingNode.id === node.id)
      )
    ],
    [currentRenderNodeIds, currentRenderNodes, exitingNodes, pendingExitingNodes]
  )
  const meshSeed = useMemo(
    () =>
      meshId ??
      `${selfId}:${nodes
        .map((node) => node.id)
        .sort()
        .join('|')}`,
    [meshId, nodes, selfId]
  )
  const links = useMemo(() => buildMeshLinks(renderNodes, getNodePeer), [getNodePeer, renderNodes])
  const nodesFitSignature = useMemo(
    () => renderNodes.map((node) => `${node.id}:${node.x}:${node.y}`).join('|'),
    [renderNodes]
  )
  const linkCount = links.length
  const shouldFadeNodeLabels = renderNodes.length >= NODE_LABEL_FADE_THRESHOLD
  const hoveredNodeId = externalHoveredNodeId ?? localHoveredNodeId
  const safeCanvasWidth = Math.max(canvasSize.width, 1)
  const safeCanvasHeight = Math.max(canvasSize.height, 1)
  const gridSize = Math.max(18, GRID_SIZE_PX * viewport.zoom)
  const gridTransform = gridPatternTransform(viewport, gridSize)
  const dotColorSchemes = MESH_VIZ_DOT_COLOR_SCHEMES[dotColorSchemeTheme]
  const dotColorScheme = meshVizDotColorSchemeAtIndex(dotColorSchemeTheme, dotColorSchemeIndex)
  const nodeBounds = useMemo(
    () => calculateNodeBounds(currentRenderNodes, { width: safeCanvasWidth, height: safeCanvasHeight }),
    [currentRenderNodes, safeCanvasHeight, safeCanvasWidth]
  )
  const nodeBoundsRect = nodeBoundsToScreenRect(nodeBounds, viewport)
  const deadZoneRect = nodeBoundsRect
    ? {
        x: nodeBoundsRect.x - PAN_DEAD_ZONE_PX,
        y: nodeBoundsRect.y - PAN_DEAD_ZONE_PX,
        width: nodeBoundsRect.width + PAN_DEAD_ZONE_PX * 2,
        height: nodeBoundsRect.height + PAN_DEAD_ZONE_PX * 2
      }
    : undefined
  const centeredBoundsRect = nodeBoundsRect ? centerScreenRect(nodeBoundsRect) : undefined

  const {
    calculateFitViewport,
    clearViewportLayerTransform,
    fitNodes,
    scheduleViewportLayerTransform,
    setViewport,
    transitionViewportTo,
    zoomAroundPoint,
    zoomAtCenter
  } = useMeshVizViewport({
    canvasSizeRef,
    currentRenderNodes,
    dragRef,
    fittedNodesSignatureRef,
    gridAccentDotRef,
    gridDotRef,
    gridPathRef,
    gridPatternRef,
    gridTertiaryDotRef,
    hasUserControlledViewportRef,
    labelLayerRef,
    liveLayerBaseViewportRef,
    liveLayerTransformActiveRef,
    liveLayerTransformRef,
    nodeLayerRef,
    nodesFitSignature,
    packetLayerRef,
    panTransformFrameRef,
    pendingPanTransformResetRef,
    pinchZoomRef,
    reduceMotion,
    renderedViewportRef,
    setViewportState,
    svgPanLayerRef,
    viewportAnimationRef,
    viewportRef,
    wheelZoomCommitTimeoutRef,
    zoomAnchorRef,
    zoomFocusRef
  })

  const nodeLifecyclePhase = useCallback(
    (nodeId: string): MeshVizNodeLifecycle => {
      if (!animateTopology) {
        return 'present'
      }

      if (pendingExitingNodeIds.has(nodeId)) {
        return 'leaving'
      }

      if (nodeLifecyclePhases[nodeId]) {
        return nodeLifecyclePhases[nodeId]
      }

      if (!hasTopologySnapshotRef.current || topologyNodeSnapshotsRef.current.has(nodeId)) {
        return 'present'
      }

      return 'entering'
    },
    [animateTopology, nodeLifecyclePhases, pendingExitingNodeIds]
  )

  const linkLifecyclePhase = useCallback(
    (sourceNodeId: string, targetNodeId: string): MeshVizNodeLifecycle => {
      const sourcePhase = nodeLifecyclePhase(sourceNodeId)
      const targetPhase = nodeLifecyclePhase(targetNodeId)

      if (sourcePhase === 'leaving' || targetPhase === 'leaving') return 'leaving'
      if (sourcePhase === 'entering' || targetPhase === 'entering') return 'entering'
      return 'present'
    },
    [nodeLifecyclePhase]
  )
  const nodeColorForTraffic = useCallback(
    (node: MeshNode) =>
      nodeVisuals(node, getNodePeer?.(node), node.id === selfId, false, dotColorScheme.nodeColors).fill,
    [dotColorScheme.nodeColors, getNodePeer, selfId]
  )

  const updateCanvasSize = useCallback(() => {
    const canvasElement = canvasRef.current

    if (!canvasElement) {
      return
    }

    const nextSize = {
      width: canvasElement.clientWidth,
      height: canvasElement.clientHeight
    }
    canvasSizeRef.current = nextSize
    setCanvasSize((current) =>
      current.width === nextSize.width && current.height === nextSize.height ? current : nextSize
    )
  }, [])

  const { clearTrafficPackets, placeTrafficPackets, playRandomTraffic, playSelfTraffic, playTraffic } =
    useMeshVizTraffic({
      canvasRef,
      canvasSizeRef,
      links,
      liveLayerBaseViewportRef,
      liveLayerTransformActiveRef,
      nodeColorForTraffic,
      packetLayerRef,
      reduceMotion,
      renderNodes,
      selfId,
      updateCanvasSize,
      viewportRef
    })

  useMeshVizLifecycleAnimations({
    currentRenderNodes,
    currentRenderNodeIds,
    renderNodes,
    links,
    reduceMotion,
    nodeLifecyclePhase,
    linkLifecyclePhase,
    nodeLayerRef,
    svgPanLayerRef,
    topologyNodeSnapshotsRef,
    hasTopologySnapshotRef,
    nodeLifecycleTimeoutsRef,
    nodeLifecycleAnimationKeysRef,
    meshLifecycleTimelineRecordsRef,
    meshLifecycleTimelineIdRef,
    previousLinkIdsRef,
    linkRestoreAnimationIdsRef,
    linkRestoreTimelineRecordsRef,
    linkRestoreTimelineIdRef,
    setNodeLifecyclePhases,
    setExitingNodes,
    openNodeId,
    localHoveredNodeId,
    setOpenNodeId,
    setLocalHoveredNodeId
  })

  const { addDebugNode, cycleDotColorScheme, removeDebugNode, selectDotColorScheme } = useMeshVizDebugControls({
    debugNodeCounterRef,
    debugShortcutsEnabled,
    meshSeed,
    nodes,
    playRandomTraffic,
    playSelfTraffic,
    setDebugNodes,
    setDotColorSchemeIndex,
    setDotColorSchemeTheme,
    setGridMode,
    setShowPanBounds
  })

  useImperativeHandle(ref, () => ({ playTraffic }), [playTraffic])

  useEffect(() => {
    const canvasElement = canvasRef.current

    if (!canvasElement) {
      return undefined
    }

    updateCanvasSize()

    const resizeObserver =
      typeof ResizeObserver === 'undefined'
        ? undefined
        : new ResizeObserver(() => {
            updateCanvasSize()
            placeTrafficPackets()
          })

    resizeObserver?.observe(canvasElement)

    return () => {
      resizeObserver?.disconnect()
      clearTrafficPackets()
      viewportAnimationRef.current?.revert()
      viewportAnimationRef.current = undefined
    }
  }, [clearTrafficPackets, placeTrafficPackets, updateCanvasSize])

  useEffect(() => {
    const radarPingElement = radarPingRef.current

    if (!radarPingElement || reduceMotion) {
      if (radarPingElement) {
        radarPingElement.style.opacity = '0'
        radarPingElement.style.transform = 'scale(1)'
      }

      return undefined
    }

    radarPingElement.style.opacity = '0.6'
    radarPingElement.style.transform = 'scale(1)'

    const animation = animate(radarPingElement, {
      opacity: { from: 0.6, to: 0 },
      scale: { from: 1, to: 2.6 },
      duration: RADAR_PULSE_DURATION,
      ease: RADAR_PULSE_EASE,
      loop: true,
      loopDelay: RADAR_PULSE_LOOP_DELAY
    })

    return () => {
      animation.revert()
      radarPingElement.style.opacity = '0.6'
      radarPingElement.style.transform = 'scale(1)'
    }
  }, [reduceMotion])

  useEffect(() => {
    if (canvasSize.width <= 0 || canvasSize.height <= 0) {
      return
    }

    if (!hasCalledOnReadyRef.current) {
      hasCalledOnReadyRef.current = true
      onReadyRef.current?.()
    }

    const nodesChanged = fittedNodesSignatureRef.current !== nodesFitSignature
    const previousNodeIds = fittedNodeIdsRef.current
    const trackedPreviousNodes = previousNodeIds.size > 0
    const addedNodes = trackedPreviousNodes ? currentRenderNodes.filter((node) => !previousNodeIds.has(node.id)) : []
    const addedNodeOutsideViewport = addedNodes.some(
      (node) => !nodeFitsInsideViewport(node, canvasSize, viewportRef.current)
    )
    const trackCurrentTopology = () => {
      fittedNodesSignatureRef.current = nodesFitSignature
      fittedNodeIdsRef.current = new Set(currentRenderNodes.map((node) => node.id))
    }

    if (addedNodeOutsideViewport) {
      trackCurrentTopology()
      zoomFocusRef.current = undefined
      zoomAnchorRef.current = undefined
      hasUserControlledViewportRef.current = false
      transitionViewportTo(calculateFitViewport(canvasSize))
      return
    }

    if (hasUserControlledViewportRef.current) {
      if (nodesChanged) {
        trackCurrentTopology()
      }

      if (zoomFocusRef.current && !focusPointWithinNodeBounds(currentRenderNodes, canvasSize, zoomFocusRef.current)) {
        zoomFocusRef.current = undefined
        zoomAnchorRef.current = undefined
      }

      const clampedViewport = clampViewportToPanBounds(
        currentRenderNodes,
        canvasSize,
        viewportRef.current,
        zoomFocusRef.current,
        calculateMaxZoomOut(currentRenderNodes, canvasSize)
      )

      if (!viewportsMatch(clampedViewport, viewportRef.current)) {
        transitionViewportTo(clampedViewport)
      }

      return
    }

    trackCurrentTopology()

    const fitViewport = calculateFitViewport(canvasSize)

    if (trackedPreviousNodes && nodesChanged) {
      zoomFocusRef.current = undefined
      zoomAnchorRef.current = undefined
      transitionViewportTo(fitViewport)
      return
    }

    setViewport(fitViewport)
  }, [calculateFitViewport, canvasSize, currentRenderNodes, nodesFitSignature, setViewport, transitionViewportTo])

  useLayoutEffect(() => {
    renderedViewportRef.current = viewport

    if (pendingPanTransformResetRef.current && viewportsMatch(viewport, viewportRef.current)) {
      pendingPanTransformResetRef.current = false
      clearViewportLayerTransform()
    }

    placeTrafficPackets()
  }, [clearViewportLayerTransform, placeTrafficPackets, viewport])

  useEffect(
    () => () => {
      if (panTransformFrameRef.current !== null) {
        window.cancelAnimationFrame(panTransformFrameRef.current)
      }

      if (wheelZoomCommitTimeoutRef.current !== null) {
        window.clearTimeout(wheelZoomCommitTimeoutRef.current)
      }
    },
    []
  )

  useEffect(() => {
    if (!isPanning) {
      return undefined
    }

    const previousUserSelect = document.body.style.userSelect
    document.body.style.userSelect = 'none'

    return () => {
      document.body.style.userSelect = previousUserSelect
    }
  }, [isPanning])

  useEffect(() => {
    const syncFullscreenState = () => {
      const isCanvasFullscreen = document.fullscreenElement === canvasRef.current

      setIsFullscreen(isCanvasFullscreen)

      if (wasFullscreenRef.current && !isCanvasFullscreen) {
        updateCanvasSize()
        fitNodes()
      }

      wasFullscreenRef.current = isCanvasFullscreen
    }

    syncFullscreenState()
    document.addEventListener('fullscreenchange', syncFullscreenState)

    return () => {
      document.removeEventListener('fullscreenchange', syncFullscreenState)
    }
  }, [fitNodes, updateCanvasSize])

  const { handleCanvasPointerDown, handleCanvasPointerMove, handleFullscreen, stopPanning } = useMeshVizInteractions({
    canvasRef,
    canvasSizeRef,
    currentRenderNodes,
    viewportRef,
    viewportAnimationRef,
    zoomFocusRef,
    renderedViewportRef,
    liveLayerBaseViewportRef,
    liveLayerTransformActiveRef,
    wheelZoomCommitTimeoutRef,
    pendingPanTransformResetRef,
    hasUserControlledViewportRef,
    dragRef,
    touchPointersRef,
    pinchZoomRef,
    setIsPanning,
    setOpenNodeId,
    setLocalHoveredNodeId,
    setViewport,
    clearViewportLayerTransform,
    scheduleViewportLayerTransform,
    zoomAroundPoint,
    onFullscreen
  })

  const screenLinks = links.map((link) => ({
    ...link,
    sourcePoint: pointToScreen(link.source, safeCanvasWidth, safeCanvasHeight, viewport),
    targetPoint: pointToScreen(link.target, safeCanvasWidth, safeCanvasHeight, viewport),
    dimmed: dimmedNodeIds?.has(link.source.id) || dimmedNodeIds?.has(link.target.id) || false
  }))
  const maxZoomOut = calculateMaxZoomOut(currentRenderNodes, canvasSize)
  const maxZoomOutLabel = maxZoomOut.toFixed(2)
  const viewportControlClassName = cn(
    'ui-control grid place-items-center rounded-[var(--radius)] border',
    isFullscreen ? 'size-[52px]' : 'size-[26px]'
  )
  const viewportControlIconClassName = isFullscreen ? 'size-6' : 'size-3'

  return (
    <MeshVizCanvas
      canvasRef={canvasRef}
      gridPatternRef={gridPatternRef}
      gridPathRef={gridPathRef}
      gridDotRef={gridDotRef}
      gridAccentDotRef={gridAccentDotRef}
      gridTertiaryDotRef={gridTertiaryDotRef}
      svgPanLayerRef={svgPanLayerRef}
      nodeLayerRef={nodeLayerRef}
      labelLayerRef={labelLayerRef}
      packetLayerRef={packetLayerRef}
      radarPingRef={radarPingRef}
      safeCanvasWidth={safeCanvasWidth}
      safeCanvasHeight={safeCanvasHeight}
      gridSize={gridSize}
      gridTransform={gridTransform}
      gridMode={gridMode}
      dotColorScheme={dotColorScheme}
      screenLinks={screenLinks}
      isDevelopment={isDevelopment}
      showPanBounds={showPanBounds}
      nodeBoundsRect={nodeBoundsRect}
      deadZoneRect={deadZoneRect}
      centeredBoundsRect={centeredBoundsRect}
      isPanning={isPanning}
      isFullscreen={isFullscreen}
      height={height}
      compact={compact}
      nodes={nodes}
      debugNodes={debugNodes}
      linkCount={linkCount}
      maxZoomOutLabel={maxZoomOutLabel}
      renderNodes={renderNodes}
      dimmedNodeIds={dimmedNodeIds}
      selfId={selfId}
      selectedNodeId={selectedNodeId}
      openNodeId={openNodeId}
      hoveredNodeId={hoveredNodeId}
      shouldFadeNodeLabels={shouldFadeNodeLabels}
      reduceMotion={reduceMotion}
      viewport={viewport}
      nodeLifecyclePhase={nodeLifecyclePhase}
      linkLifecyclePhase={linkLifecyclePhase}
      getNodePeer={getNodePeer}
      dotColorSchemeIndex={dotColorSchemeIndex}
      dotColorSchemes={dotColorSchemes}
      onPointerDown={handleCanvasPointerDown}
      onPointerMove={handleCanvasPointerMove}
      onPointerUp={stopPanning}
      onPointerCancel={stopPanning}
      onFullscreen={handleFullscreen}
      onAddDebugNode={addDebugNode}
      onRemoveDebugNode={removeDebugNode}
      onDotColorSchemeChange={selectDotColorScheme}
      onDotColorSchemeNext={cycleDotColorScheme}
      onGridModeChange={setGridMode}
      onPlayRandomTraffic={playRandomTraffic}
      onPlaySelfTraffic={playSelfTraffic}
      onShowPanBoundsChange={setShowPanBounds}
      onZoomAtCenter={zoomAtCenter}
      onFitNodes={fitNodes}
      onNodeHoverStart={setLocalHoveredNodeId}
      onNodeHoverEnd={(nodeId) => {
        setLocalHoveredNodeId((current) => (current === nodeId ? undefined : current))
      }}
      onNodeToggleOpen={(nodeId) => {
        setOpenNodeId((current) => (current === nodeId ? undefined : nodeId))
      }}
      onNodeCloseOpen={() => setOpenNodeId(undefined)}
      viewportControlClassName={viewportControlClassName}
      viewportControlIconClassName={viewportControlIconClassName}
    />
  )
})

MeshViz.displayName = 'MeshViz'

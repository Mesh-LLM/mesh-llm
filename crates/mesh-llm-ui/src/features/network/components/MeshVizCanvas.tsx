import type { PointerEvent as ReactPointerEvent, RefObject } from 'react'
import { Maximize2, Minus, Plus, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/cn'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import type { MeshLink } from '@/features/network/lib/mesh-links'
import type { MeshVizDotColorScheme } from '@/features/network/lib/mesh-viz-dot-color-schemes'
import type { Point, Viewport } from '@/features/network/lib/mesh-viewport'
import { MeshVizDebugControls, type MeshVizGridMode } from '@/features/network/components/MeshVizDebugControls'
import { MeshVizNode, MeshVizNodeLabel, type MeshVizNodeLifecycle } from '@/features/network/components/MeshVizNode'
import type { DebugMeshNode, DebugNodeShortcut } from '@/features/network/components/MeshViz.helpers'

type ScreenRect = {
  x: number
  y: number
  width: number
  height: number
}

type ScreenLink = MeshLink & {
  sourcePoint: Point
  targetPoint: Point
  dimmed: boolean
}

export type MeshVizCanvasProps = {
  canvasRef: RefObject<HTMLDivElement | null>
  gridPatternRef: RefObject<SVGPatternElement | null>
  gridPathRef: RefObject<SVGPathElement | null>
  gridDotRef: RefObject<SVGCircleElement | null>
  gridAccentDotRef: RefObject<SVGCircleElement | null>
  gridTertiaryDotRef: RefObject<SVGCircleElement | null>
  svgPanLayerRef: RefObject<SVGGElement | null>
  nodeLayerRef: RefObject<HTMLDivElement | null>
  labelLayerRef: RefObject<HTMLDivElement | null>
  packetLayerRef: RefObject<HTMLDivElement | null>
  radarPingRef: RefObject<HTMLSpanElement | null>
  safeCanvasWidth: number
  safeCanvasHeight: number
  gridSize: number
  gridTransform: string
  gridMode: MeshVizGridMode
  dotColorScheme: MeshVizDotColorScheme
  screenLinks: ScreenLink[]
  isDevelopment: boolean
  showPanBounds: boolean
  nodeBoundsRect?: ScreenRect
  deadZoneRect?: ScreenRect
  centeredBoundsRect?: ScreenRect
  isPanning: boolean
  isFullscreen: boolean
  height?: number
  compact: boolean
  nodes: MeshNode[]
  debugNodes: DebugMeshNode[]
  linkCount: number
  maxZoomOutLabel: string
  renderNodes: MeshNode[]
  dimmedNodeIds?: Set<string>
  selfId: string
  selectedNodeId?: string
  openNodeId?: string
  hoveredNodeId?: string
  shouldFadeNodeLabels: boolean
  reduceMotion: boolean
  viewport: Viewport
  nodeLifecyclePhase: (nodeId: string) => MeshVizNodeLifecycle
  linkLifecyclePhase: (sourceNodeId: string, targetNodeId: string) => MeshVizNodeLifecycle
  getNodePeer?: (node: MeshNode) => Peer | undefined
  dotColorSchemeIndex: number
  dotColorSchemes: readonly MeshVizDotColorScheme[]
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerUp: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerCancel: (event: ReactPointerEvent<HTMLDivElement>) => void
  onFullscreen: () => void
  onAddDebugNode: (shortcut: DebugNodeShortcut) => void
  onRemoveDebugNode: (shortcut: DebugNodeShortcut) => void
  onDotColorSchemeChange: (index: number) => void
  onDotColorSchemeNext: () => void
  onGridModeChange: (mode: MeshVizGridMode) => void
  onPlayRandomTraffic: () => void
  onPlaySelfTraffic: () => void
  onShowPanBoundsChange: (show: boolean) => void
  onZoomAtCenter: (factor: number) => void
  onFitNodes: () => void
  onNodeHoverStart: (nodeId: string) => void
  onNodeHoverEnd: (nodeId: string) => void
  onNodeToggleOpen: (nodeId: string) => void
  onNodeCloseOpen: () => void
  viewportControlClassName: string
  viewportControlIconClassName: string
}

export function MeshVizCanvas({
  canvasRef,
  gridPatternRef,
  gridPathRef,
  gridDotRef,
  gridAccentDotRef,
  gridTertiaryDotRef,
  svgPanLayerRef,
  nodeLayerRef,
  labelLayerRef,
  packetLayerRef,
  radarPingRef,
  safeCanvasWidth,
  safeCanvasHeight,
  gridSize,
  gridTransform,
  gridMode,
  dotColorScheme,
  screenLinks,
  isDevelopment,
  showPanBounds,
  nodeBoundsRect,
  deadZoneRect,
  centeredBoundsRect,
  isPanning,
  isFullscreen,
  height,
  compact,
  nodes,
  debugNodes,
  linkCount,
  maxZoomOutLabel,
  renderNodes,
  dimmedNodeIds,
  selfId,
  selectedNodeId,
  openNodeId,
  hoveredNodeId,
  shouldFadeNodeLabels,
  reduceMotion,
  viewport,
  nodeLifecyclePhase,
  linkLifecyclePhase,
  getNodePeer,
  dotColorSchemeIndex,
  dotColorSchemes,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onPointerCancel,
  onFullscreen,
  onAddDebugNode,
  onRemoveDebugNode,
  onDotColorSchemeChange,
  onDotColorSchemeNext,
  onGridModeChange,
  onPlayRandomTraffic,
  onPlaySelfTraffic,
  onShowPanBoundsChange,
  onZoomAtCenter,
  onFitNodes,
  onNodeHoverStart,
  onNodeHoverEnd,
  onNodeToggleOpen,
  onNodeCloseOpen,
  viewportControlClassName,
  viewportControlIconClassName
}: MeshVizCanvasProps) {
  return (
    <section className="panel-shell flex h-full min-h-0 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <header className="flex shrink-0 items-center justify-between border-b border-border-soft px-4 py-3">
        <h2 className="type-panel-title">Mesh overview</h2>
        {!compact && (
          <button
            onClick={onFullscreen}
            type="button"
            className="ui-control inline-flex items-center gap-1.5 rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium"
          >
            <Maximize2 className="size-3" /> Fullscreen
          </button>
        )}
      </header>
      <div className="flex min-h-0 flex-1 p-3.5">
        <div
          ref={canvasRef}
          data-testid="mesh-canvas"
          className={cn(
            'relative w-full touch-none overflow-hidden rounded-[var(--radius-lg)] mesh-canvas',
            isPanning ? 'cursor-grabbing' : 'cursor-grab'
          )}
          style={{
            height: height ?? '100%',
            background:
              'radial-gradient(ellipse at 60% 40%, color-mix(in oklab, var(--color-accent) 10%, var(--color-panel-strong)) 0%, var(--color-panel-strong) 60%, var(--color-panel) 100%)'
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerCancel}
        >
          <div className="absolute left-3.5 top-3 z-10 flex flex-wrap items-center gap-2.5 font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.14em] text-muted-foreground">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-panel/90 px-2 py-px text-accent">
              <span className="size-[5px] rounded-full bg-current mesh-live-pulse" /> Live
            </span>
            <span>
              {nodes.length} nodes{debugNodes.length > 0 ? ` + ${debugNodes.length} debug` : ''} · {linkCount} links ·
              Nearest mesh
            </span>
          </div>

          {isDevelopment && (
            <div
              data-testid="mesh-max-zoom-label"
              className="pointer-events-none absolute right-3.5 top-3 z-10 rounded-full border border-border bg-panel/90 px-2 py-px font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.14em] text-muted-foreground"
            >
              Max Zoom: {maxZoomOutLabel}
            </div>
          )}

          <svg
            viewBox={`0 0 ${safeCanvasWidth} ${safeCanvasHeight}`}
            preserveAspectRatio="none"
            className="pointer-events-none absolute inset-0 h-full w-full overflow-hidden"
            role="img"
            aria-label="Nearest mesh topology"
          >
            <defs>
              <pattern
                ref={gridPatternRef}
                id="mesh-viz-grid"
                width={gridSize}
                height={gridSize}
                patternUnits="userSpaceOnUse"
                patternTransform={gridTransform}
              >
                {gridMode === 'line' ? (
                  <path
                    ref={gridPathRef}
                    data-testid="mesh-viz-line-grid"
                    d={`M ${gridSize} 0 L 0 0 0 ${gridSize}`}
                    fill="none"
                    stroke="color-mix(in oklab, var(--color-foreground) 7.2%, transparent)"
                    strokeWidth="1"
                  />
                ) : (
                  <>
                    <circle
                      ref={gridDotRef}
                      data-testid="mesh-viz-dot-grid"
                      cx="0"
                      cy="0"
                      r="1.35"
                      fill={dotColorScheme.colors[0]}
                    />
                    <circle
                      ref={gridAccentDotRef}
                      data-testid="mesh-viz-accent-dot-grid"
                      cx={gridSize / 2}
                      cy={gridSize / 2}
                      r="1.25"
                      fill={dotColorScheme.colors[1]}
                    />
                    <circle
                      ref={gridTertiaryDotRef}
                      data-testid="mesh-viz-tertiary-dot-grid"
                      cx="0"
                      cy={gridSize / 2}
                      r="0.85"
                      fill={dotColorScheme.colors[2]}
                    />
                  </>
                )}
              </pattern>
            </defs>
            <rect width={safeCanvasWidth} height={safeCanvasHeight} fill="url(#mesh-viz-grid)" />
            <g ref={svgPanLayerRef}>
              {screenLinks.map((link) => (
                <line
                  key={link.id}
                  className="mesh-link"
                  data-link-lifecycle={linkLifecyclePhase(link.source.id, link.target.id)}
                  data-mesh-link-id={link.id}
                  data-source-node-id={link.source.id}
                  data-target-node-id={link.target.id}
                  data-testid="mesh-link"
                  pathLength={1}
                  x1={link.sourcePoint.x}
                  y1={link.sourcePoint.y}
                  x2={link.targetPoint.x}
                  y2={link.targetPoint.y}
                  stroke="color-mix(in oklab, var(--color-accent) 48%, var(--color-border))"
                  strokeDasharray="0.0275 0.0275"
                  strokeLinecap="round"
                  strokeWidth="1"
                  opacity={link.dimmed ? '0.18' : '0.62'}
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              {isDevelopment && showPanBounds && nodeBoundsRect && deadZoneRect && centeredBoundsRect && (
                <g aria-label="Mesh pan bounds debug overlay">
                  <rect
                    data-testid="mesh-pan-dead-zone-box"
                    x={deadZoneRect.x}
                    y={deadZoneRect.y}
                    width={deadZoneRect.width}
                    height={deadZoneRect.height}
                    fill="color-mix(in oklab, var(--color-accent) 7%, transparent)"
                    stroke="color-mix(in oklab, var(--color-accent) 72%, transparent)"
                    strokeDasharray="8 6"
                    strokeWidth="1.2"
                    vectorEffect="non-scaling-stroke"
                  />
                  <rect
                    data-testid="mesh-node-bounds-box"
                    x={nodeBoundsRect.x}
                    y={nodeBoundsRect.y}
                    width={nodeBoundsRect.width}
                    height={nodeBoundsRect.height}
                    fill="none"
                    stroke="color-mix(in oklab, var(--color-good) 78%, transparent)"
                    strokeWidth="1.4"
                    vectorEffect="non-scaling-stroke"
                  />
                  <rect
                    data-testid="mesh-centered-bounds-box"
                    x={centeredBoundsRect.x}
                    y={centeredBoundsRect.y}
                    width={centeredBoundsRect.width}
                    height={centeredBoundsRect.height}
                    fill="none"
                    stroke="color-mix(in oklab, var(--color-warn) 82%, transparent)"
                    strokeDasharray="6 5"
                    strokeWidth="1.2"
                    vectorEffect="non-scaling-stroke"
                  />
                </g>
              )}
            </g>
          </svg>

          <div
            ref={packetLayerRef}
            className="pointer-events-none absolute inset-0 z-[5]"
            data-testid="mesh-packet-layer"
            style={{ transformOrigin: '0 0', willChange: 'transform' }}
            aria-hidden="true"
          />

          <div
            ref={nodeLayerRef}
            className="absolute inset-0"
            style={{ transformOrigin: '0 0', willChange: 'transform' }}
          >
            {renderNodes.map((node) => {
              const peer = getNodePeer?.(node)
              const isDimmed = dimmedNodeIds?.has(node.id) ?? false

              return (
                <div key={node.id} style={{ opacity: isDimmed ? 0.24 : 1, transition: 'opacity 180ms ease' }}>
                  <MeshVizNode
                    node={node}
                    peer={peer}
                    selfId={selfId}
                    selectedNodeId={selectedNodeId}
                    openNodeId={openNodeId}
                    hoveredNodeId={hoveredNodeId}
                    shouldFadeNodeLabels={shouldFadeNodeLabels}
                    reduceMotion={reduceMotion}
                    canvasWidth={safeCanvasWidth}
                    canvasHeight={safeCanvasHeight}
                    viewport={viewport}
                    nodeColors={dotColorScheme.nodeColors}
                    lifecycle={nodeLifecyclePhase(node.id)}
                    radarPingRef={radarPingRef}
                    onHoverStart={onNodeHoverStart}
                    onHoverEnd={onNodeHoverEnd}
                    onToggleOpen={onNodeToggleOpen}
                    onCloseOpen={onNodeCloseOpen}
                  />
                </div>
              )
            })}
          </div>

          <div
            ref={labelLayerRef}
            className="pointer-events-none absolute inset-0 z-[40]"
            data-testid="mesh-node-label-layer"
            style={{ transformOrigin: '0 0', willChange: 'transform' }}
            aria-hidden="true"
          >
            {renderNodes.map((node) => {
              const peer = getNodePeer?.(node)
              const isDimmed = dimmedNodeIds?.has(node.id) ?? false

              return (
                <div key={node.id} style={{ opacity: isDimmed ? 0.24 : 1, transition: 'opacity 180ms ease' }}>
                  <MeshVizNodeLabel
                    node={node}
                    peer={peer}
                    selfId={selfId}
                    selectedNodeId={selectedNodeId}
                    openNodeId={openNodeId}
                    hoveredNodeId={hoveredNodeId}
                    shouldFadeNodeLabels={shouldFadeNodeLabels}
                    reduceMotion={reduceMotion}
                    canvasWidth={safeCanvasWidth}
                    canvasHeight={safeCanvasHeight}
                    viewport={viewport}
                    nodeColors={dotColorScheme.nodeColors}
                    lifecycle={nodeLifecyclePhase(node.id)}
                  />
                </div>
              )
            })}
          </div>

          {isDevelopment && (
            <MeshVizDebugControls
              debugNodeCount={debugNodes.length}
              dotColorSchemeIndex={dotColorSchemeIndex}
              dotColorSchemes={dotColorSchemes}
              gridMode={gridMode}
              isFullscreen={isFullscreen}
              onAddDebugNode={onAddDebugNode}
              onDotColorSchemeChange={onDotColorSchemeChange}
              onDotColorSchemeNext={onDotColorSchemeNext}
              onGridModeChange={onGridModeChange}
              onPlayRandomTraffic={onPlayRandomTraffic}
              onPlaySelfTraffic={onPlaySelfTraffic}
              onRemoveDebugNode={onRemoveDebugNode}
              onShowPanBoundsChange={onShowPanBoundsChange}
              showPanBounds={showPanBounds}
            />
          )}

          <div className="absolute bottom-3 right-3 flex flex-col gap-1.5">
            <button
              type="button"
              className={viewportControlClassName}
              aria-label="Zoom in"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => onZoomAtCenter(1.12)}
            >
              <Plus className={viewportControlIconClassName} />
            </button>
            <button
              type="button"
              className={viewportControlClassName}
              aria-label="Zoom out"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => onZoomAtCenter(0.88)}
            >
              <Minus className={viewportControlIconClassName} />
            </button>
            <button
              type="button"
              className={viewportControlClassName}
              aria-label="Reset view"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={onFitNodes}
            >
              <RotateCcw className={viewportControlIconClassName} />
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

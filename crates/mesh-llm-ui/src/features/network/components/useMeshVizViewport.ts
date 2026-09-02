import { useCallback, type MutableRefObject } from 'react'
import { animate, type JSAnimation } from 'animejs'
import {
  calculateMaxZoomOut,
  clamp,
  clampViewportToPanBounds,
  DEFAULT_VIEWPORT,
  FIT_PADDING_PX,
  GRID_SIZE_PX,
  gridPatternTransform,
  isIdentityLayerTransform,
  MAX_ZOOM,
  NODE_VISUAL_BOUNDS_PADDING_PX,
  type LayerTransform,
  type Point,
  type Viewport,
  type WorldPoint,
  viewportLayerTransform,
  viewportsMatch
} from '@/features/network/lib/mesh-viewport'
import type { MeshNode } from '@/features/app-tabs/types'

const VIEWPORT_RECLAMP_DURATION = 220
const VIEWPORT_RECLAMP_EASE = 'outExpo'
const WHEEL_ZOOM_COMMIT_DELAY_MS = 90

type MeshVizViewportArgs = {
  gridPatternRef: MutableRefObject<SVGPatternElement | null>
  gridPathRef: MutableRefObject<SVGPathElement | null>
  gridDotRef: MutableRefObject<SVGCircleElement | null>
  gridAccentDotRef: MutableRefObject<SVGCircleElement | null>
  gridTertiaryDotRef: MutableRefObject<SVGCircleElement | null>
  svgPanLayerRef: MutableRefObject<SVGGElement | null>
  nodeLayerRef: MutableRefObject<HTMLDivElement | null>
  labelLayerRef: MutableRefObject<HTMLDivElement | null>
  packetLayerRef: MutableRefObject<HTMLDivElement | null>
  panTransformFrameRef: MutableRefObject<number | null>
  liveLayerTransformRef: MutableRefObject<LayerTransform>
  renderedViewportRef: MutableRefObject<Viewport>
  liveLayerBaseViewportRef: MutableRefObject<Viewport>
  liveLayerTransformActiveRef: MutableRefObject<boolean>
  wheelZoomCommitTimeoutRef: MutableRefObject<number | null>
  pendingPanTransformResetRef: MutableRefObject<boolean>
  viewportAnimationRef: MutableRefObject<JSAnimation | undefined>
  canvasSizeRef: MutableRefObject<{ width: number; height: number }>
  viewportRef: MutableRefObject<Viewport>
  zoomFocusRef: MutableRefObject<WorldPoint | undefined>
  zoomAnchorRef: MutableRefObject<Point | undefined>
  fittedNodesSignatureRef: MutableRefObject<string>
  hasUserControlledViewportRef: MutableRefObject<boolean>
  dragRef: MutableRefObject<{ active: boolean }>
  pinchZoomRef: MutableRefObject<{ active: boolean }>
  currentRenderNodes: MeshNode[]
  nodesFitSignature: string
  reduceMotion: boolean
  setViewportState: (nextViewport: Viewport) => void
}

export function useMeshVizViewport({
  gridPatternRef,
  gridPathRef,
  gridDotRef,
  gridAccentDotRef,
  gridTertiaryDotRef,
  svgPanLayerRef,
  nodeLayerRef,
  labelLayerRef,
  packetLayerRef,
  panTransformFrameRef,
  liveLayerTransformRef,
  renderedViewportRef,
  liveLayerBaseViewportRef,
  liveLayerTransformActiveRef,
  wheelZoomCommitTimeoutRef,
  pendingPanTransformResetRef,
  viewportAnimationRef,
  canvasSizeRef,
  viewportRef,
  zoomFocusRef,
  zoomAnchorRef,
  fittedNodesSignatureRef,
  hasUserControlledViewportRef,
  dragRef,
  pinchZoomRef,
  currentRenderNodes,
  nodesFitSignature,
  reduceMotion,
  setViewportState
}: MeshVizViewportArgs) {
  const setViewport = useCallback(
    (nextViewport: Viewport, options?: { userControlled?: boolean }) => {
      const currentViewport = viewportRef.current

      if (options?.userControlled) {
        viewportAnimationRef.current?.revert()
        viewportAnimationRef.current = undefined
        hasUserControlledViewportRef.current = true
      }

      viewportRef.current = nextViewport

      if (!options?.userControlled && viewportsMatch(currentViewport, nextViewport)) {
        return
      }

      setViewportState(options?.userControlled ? { ...nextViewport } : nextViewport)
    },
    [hasUserControlledViewportRef, setViewportState, viewportAnimationRef, viewportRef]
  )

  const applyGridPattern = useCallback(
    (nextViewport: Viewport) => {
      const nextGridSize = Math.max(18, GRID_SIZE_PX * nextViewport.zoom)

      if (gridPatternRef.current) {
        gridPatternRef.current.setAttribute('width', `${nextGridSize}`)
        gridPatternRef.current.setAttribute('height', `${nextGridSize}`)
        gridPatternRef.current.setAttribute('patternTransform', gridPatternTransform(nextViewport, nextGridSize))
      }

      if (gridPathRef.current) {
        gridPathRef.current.setAttribute('d', `M ${nextGridSize} 0 L 0 0 0 ${nextGridSize}`)
      }

      if (gridDotRef.current) {
        gridDotRef.current.setAttribute('cx', '0')
        gridDotRef.current.setAttribute('cy', '0')
      }

      if (gridAccentDotRef.current) {
        const accentDotOffset = nextGridSize / 2

        gridAccentDotRef.current.setAttribute('cx', `${accentDotOffset}`)
        gridAccentDotRef.current.setAttribute('cy', `${accentDotOffset}`)
      }

      if (gridTertiaryDotRef.current) {
        gridTertiaryDotRef.current.setAttribute('cx', '0')
        gridTertiaryDotRef.current.setAttribute('cy', `${nextGridSize / 2}`)
      }
    },
    [gridAccentDotRef, gridDotRef, gridPathRef, gridPatternRef, gridTertiaryDotRef]
  )

  const applyLayerTransform = useCallback(
    (transform: LayerTransform) => {
      const isIdentity = isIdentityLayerTransform(transform)
      const htmlTransform = isIdentity
        ? ''
        : `translate3d(${transform.x}px, ${transform.y}px, 0) scale(${transform.scale})`
      const svgTransform = isIdentity ? '' : `translate(${transform.x} ${transform.y}) scale(${transform.scale})`

      if (svgPanLayerRef.current) {
        if (svgTransform) {
          svgPanLayerRef.current.setAttribute('transform', svgTransform)
        } else {
          svgPanLayerRef.current.removeAttribute('transform')
        }
      }

      if (nodeLayerRef.current) {
        nodeLayerRef.current.style.transform = htmlTransform
        nodeLayerRef.current.style.setProperty('--mesh-node-live-scale', isIdentity ? '1' : `${1 / transform.scale}`)
      }

      if (labelLayerRef.current) {
        labelLayerRef.current.style.transform = htmlTransform
        labelLayerRef.current.style.setProperty('--mesh-node-live-scale', isIdentity ? '1' : `${1 / transform.scale}`)
      }

      if (packetLayerRef.current) {
        packetLayerRef.current.style.transform = htmlTransform
      }

      applyGridPattern(viewportRef.current)
    },
    [applyGridPattern, labelLayerRef, nodeLayerRef, packetLayerRef, svgPanLayerRef, viewportRef]
  )

  const activateLiveLayerTransform = useCallback(() => {
    if (liveLayerTransformActiveRef.current) {
      return
    }

    liveLayerBaseViewportRef.current = renderedViewportRef.current
    liveLayerTransformActiveRef.current = true
  }, [liveLayerBaseViewportRef, liveLayerTransformActiveRef, renderedViewportRef])

  const scheduleViewportLayerTransform = useCallback(
    (nextViewport: Viewport) => {
      activateLiveLayerTransform()
      liveLayerTransformRef.current = viewportLayerTransform(liveLayerBaseViewportRef.current, nextViewport)

      if (panTransformFrameRef.current !== null) {
        return
      }

      panTransformFrameRef.current = window.requestAnimationFrame(() => {
        panTransformFrameRef.current = null
        applyLayerTransform(liveLayerTransformRef.current)
      })
    },
    [
      activateLiveLayerTransform,
      applyLayerTransform,
      liveLayerBaseViewportRef,
      liveLayerTransformRef,
      panTransformFrameRef
    ]
  )

  const clearViewportLayerTransform = useCallback(() => {
    liveLayerTransformActiveRef.current = false
    liveLayerBaseViewportRef.current = viewportRef.current
    liveLayerTransformRef.current = { x: 0, y: 0, scale: 1 }

    if (panTransformFrameRef.current !== null) {
      window.cancelAnimationFrame(panTransformFrameRef.current)
      panTransformFrameRef.current = null
    }

    applyLayerTransform(liveLayerTransformRef.current)
  }, [
    applyLayerTransform,
    liveLayerBaseViewportRef,
    liveLayerTransformActiveRef,
    liveLayerTransformRef,
    panTransformFrameRef,
    viewportRef
  ])

  const scheduleWheelZoomCommit = useCallback(() => {
    if (wheelZoomCommitTimeoutRef.current !== null) {
      window.clearTimeout(wheelZoomCommitTimeoutRef.current)
    }

    wheelZoomCommitTimeoutRef.current = window.setTimeout(() => {
      wheelZoomCommitTimeoutRef.current = null

      if (dragRef.current.active || pinchZoomRef.current.active) {
        return
      }

      pendingPanTransformResetRef.current = true
      setViewport(viewportRef.current, { userControlled: true })
    }, WHEEL_ZOOM_COMMIT_DELAY_MS)
  }, [dragRef, pendingPanTransformResetRef, pinchZoomRef, setViewport, viewportRef, wheelZoomCommitTimeoutRef])

  const flushLiveViewportTransform = useCallback(() => {
    if (wheelZoomCommitTimeoutRef.current !== null) {
      window.clearTimeout(wheelZoomCommitTimeoutRef.current)
      wheelZoomCommitTimeoutRef.current = null
    }

    if (!liveLayerTransformActiveRef.current) {
      return
    }

    pendingPanTransformResetRef.current = false
    clearViewportLayerTransform()
    viewportAnimationRef.current?.revert()
    viewportAnimationRef.current = undefined
    setViewportState(viewportRef.current)
  }, [
    clearViewportLayerTransform,
    liveLayerTransformActiveRef,
    pendingPanTransformResetRef,
    setViewportState,
    viewportAnimationRef,
    viewportRef,
    wheelZoomCommitTimeoutRef
  ])

  const transitionViewportTo = useCallback(
    (targetViewport: Viewport) => {
      const startViewport = viewportRef.current

      flushLiveViewportTransform()

      viewportAnimationRef.current?.revert()
      viewportAnimationRef.current = undefined

      if (viewportsMatch(startViewport, targetViewport)) {
        return
      }

      if (reduceMotion) {
        setViewport(targetViewport)
        return
      }

      const animatedViewport = { ...startViewport }

      viewportAnimationRef.current = animate(animatedViewport, {
        zoom: { from: startViewport.zoom, to: targetViewport.zoom },
        panX: { from: startViewport.panX, to: targetViewport.panX },
        panY: { from: startViewport.panY, to: targetViewport.panY },
        duration: VIEWPORT_RECLAMP_DURATION,
        ease: VIEWPORT_RECLAMP_EASE,
        loop: false,
        onUpdate: () => setViewport({ ...animatedViewport }),
        onComplete: () => {
          setViewport(targetViewport)
          viewportAnimationRef.current = undefined
        }
      })
    },
    [flushLiveViewportTransform, reduceMotion, setViewport, viewportAnimationRef, viewportRef]
  )

  const calculateFitViewport = useCallback(
    (size: { width: number; height: number }) => {
      if (currentRenderNodes.length === 0 || size.width <= 0 || size.height <= 0) {
        return DEFAULT_VIEWPORT
      }

      const minX = Math.min(...currentRenderNodes.map((node) => node.x))
      const maxX = Math.max(...currentRenderNodes.map((node) => node.x))
      const minY = Math.min(...currentRenderNodes.map((node) => node.y))
      const maxY = Math.max(...currentRenderNodes.map((node) => node.y))
      const availableWidth = Math.max(1, size.width - FIT_PADDING_PX * 2)
      const availableHeight = Math.max(1, size.height - FIT_PADDING_PX * 2)
      const worldWidth = Math.max(1, ((maxX - minX) / 100) * size.width)
      const worldHeight = Math.max(1, ((maxY - minY) / 100) * size.height)
      const minZoom = calculateMaxZoomOut(currentRenderNodes, size)
      const zoom = clamp(Math.min(availableWidth / worldWidth, availableHeight / worldHeight), minZoom, MAX_ZOOM)
      const centerX = ((minX + maxX) / 2 / 100) * size.width
      const centerY = ((minY + maxY) / 2 / 100) * size.height

      return {
        zoom,
        panX: size.width * 0.5 - centerX * zoom,
        panY: size.height * 0.5 - centerY * zoom
      }
    },
    [currentRenderNodes]
  )

  const fitNodes = useCallback(() => {
    viewportAnimationRef.current?.revert()
    viewportAnimationRef.current = undefined
    zoomFocusRef.current = undefined
    zoomAnchorRef.current = undefined
    flushLiveViewportTransform()
    hasUserControlledViewportRef.current = false
    fittedNodesSignatureRef.current = nodesFitSignature
    setViewport(calculateFitViewport(canvasSizeRef.current))
  }, [
    calculateFitViewport,
    canvasSizeRef,
    fittedNodesSignatureRef,
    flushLiveViewportTransform,
    hasUserControlledViewportRef,
    nodesFitSignature,
    setViewport,
    viewportAnimationRef,
    zoomAnchorRef,
    zoomFocusRef
  ])

  const zoomAroundPoint = useCallback(
    (nextZoom: number, anchorX: number, anchorY: number, options?: { live?: boolean }) => {
      const currentViewport = viewportRef.current
      const canvasSize = canvasSizeRef.current
      const minZoom = calculateMaxZoomOut(currentRenderNodes, canvasSize)
      const zoom = clamp(nextZoom, minZoom, MAX_ZOOM)
      const candidateFocus = {
        x: (anchorX - currentViewport.panX) / currentViewport.zoom,
        y: (anchorY - currentViewport.panY) / currentViewport.zoom
      }
      const currentFocus = zoomFocusRef.current
      const currentAnchor = zoomAnchorRef.current
      const sameZoomAnchor =
        currentAnchor &&
        Math.abs(currentAnchor.x - anchorX) <= NODE_VISUAL_BOUNDS_PADDING_PX &&
        Math.abs(currentAnchor.y - anchorY) <= NODE_VISUAL_BOUNDS_PADDING_PX
      const focusPoint = currentFocus && sameZoomAnchor ? currentFocus : candidateFocus

      zoomAnchorRef.current = { x: anchorX, y: anchorY }

      zoomFocusRef.current = focusPoint

      const nextViewport = clampViewportToPanBounds(
        currentRenderNodes,
        canvasSize,
        {
          zoom,
          panX: anchorX - focusPoint.x * zoom,
          panY: anchorY - focusPoint.y * zoom
        },
        focusPoint,
        minZoom
      )

      if (options?.live) {
        viewportAnimationRef.current?.revert()
        viewportAnimationRef.current = undefined
        hasUserControlledViewportRef.current = true
        viewportRef.current = nextViewport
        scheduleViewportLayerTransform(nextViewport)
        scheduleWheelZoomCommit()
        return
      }

      flushLiveViewportTransform()
      setViewport(nextViewport, { userControlled: true })
    },
    [
      currentRenderNodes,
      flushLiveViewportTransform,
      scheduleViewportLayerTransform,
      scheduleWheelZoomCommit,
      setViewport,
      canvasSizeRef,
      hasUserControlledViewportRef,
      viewportAnimationRef,
      viewportRef,
      zoomAnchorRef,
      zoomFocusRef
    ]
  )

  const zoomAtCenter = useCallback(
    (factor: number) => {
      zoomAroundPoint(
        viewportRef.current.zoom * factor,
        canvasSizeRef.current.width * 0.5,
        canvasSizeRef.current.height * 0.5
      )
    },
    [canvasSizeRef, viewportRef, zoomAroundPoint]
  )

  return {
    calculateFitViewport,
    clearViewportLayerTransform,
    fitNodes,
    flushLiveViewportTransform,
    scheduleViewportLayerTransform,
    setViewport,
    transitionViewportTo,
    zoomAroundPoint,
    zoomAtCenter
  }
}

import {
  useCallback,
  useEffect,
  type Dispatch,
  type MutableRefObject,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type SetStateAction
} from 'react'
import type { JSAnimation } from 'animejs'
import type { MeshNode } from '@/features/app-tabs/types'
import {
  calculateMaxZoomOut,
  clampViewportToPanBounds,
  type Point,
  type Viewport,
  type WorldPoint
} from '@/features/network/lib/mesh-viewport'

export type DragState = {
  active: boolean
  pointerId: number | null
  originX: number
  originY: number
  panX: number
  panY: number
}

export type TouchPointState = {
  pointerId: number
  clientX: number
  clientY: number
}

export type PinchZoomState = {
  active: boolean
  initialDistance: number
  initialZoom: number
}

type UseMeshVizInteractionsArgs = {
  canvasRef: RefObject<HTMLDivElement | null>
  canvasSizeRef: MutableRefObject<{ width: number; height: number }>
  currentRenderNodes: MeshNode[]
  viewportRef: MutableRefObject<Viewport>
  viewportAnimationRef: MutableRefObject<JSAnimation | undefined>
  zoomFocusRef: MutableRefObject<WorldPoint | undefined>
  renderedViewportRef: MutableRefObject<Viewport>
  liveLayerBaseViewportRef: MutableRefObject<Viewport>
  liveLayerTransformActiveRef: MutableRefObject<boolean>
  wheelZoomCommitTimeoutRef: MutableRefObject<number | null>
  pendingPanTransformResetRef: MutableRefObject<boolean>
  hasUserControlledViewportRef: MutableRefObject<boolean>
  dragRef: MutableRefObject<DragState>
  touchPointersRef: MutableRefObject<Map<number, TouchPointState>>
  pinchZoomRef: MutableRefObject<PinchZoomState>
  setIsPanning: Dispatch<SetStateAction<boolean>>
  setOpenNodeId: Dispatch<SetStateAction<string | undefined>>
  setLocalHoveredNodeId: Dispatch<SetStateAction<string | undefined>>
  setViewport: (nextViewport: Viewport, options?: { userControlled?: boolean }) => void
  clearViewportLayerTransform: () => void
  scheduleViewportLayerTransform: (nextViewport: Viewport) => void
  zoomAroundPoint: (nextZoom: number, anchorX: number, anchorY: number, options?: { live?: boolean }) => void
  onFullscreen?: () => void
}

const WHEEL_ZOOM_IN = 1.08
const WHEEL_ZOOM_OUT = 0.92

function touchPoints(touchPointersRef: MutableRefObject<Map<number, TouchPointState>>) {
  return Array.from(touchPointersRef.current.values())
}

function distanceBetweenTouchPoints(first: TouchPointState, second: TouchPointState) {
  return Math.hypot(second.clientX - first.clientX, second.clientY - first.clientY)
}

function midpointRelativeToCanvas(element: HTMLDivElement, first: TouchPointState, second: TouchPointState): Point {
  const rect = element.getBoundingClientRect()

  return {
    x: (first.clientX + second.clientX) * 0.5 - rect.left,
    y: (first.clientY + second.clientY) * 0.5 - rect.top
  }
}

export function useMeshVizInteractions({
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
}: UseMeshVizInteractionsArgs) {
  const beginPinchZoom = useCallback(() => {
    const [first, second] = touchPoints(touchPointersRef)

    if (!first || !second) {
      return
    }

    dragRef.current.active = false
    dragRef.current.pointerId = null
    setIsPanning(false)
    pinchZoomRef.current = {
      active: true,
      initialDistance: Math.max(1, distanceBetweenTouchPoints(first, second)),
      initialZoom: viewportRef.current.zoom
    }
  }, [dragRef, pinchZoomRef, setIsPanning, touchPointersRef, viewportRef])

  const handleCanvasPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return
      }

      if (event.target instanceof Element && event.target.closest('button, a, input, label')) {
        return
      }

      event.preventDefault()
      viewportAnimationRef.current?.revert()
      viewportAnimationRef.current = undefined
      hasUserControlledViewportRef.current = true
      pendingPanTransformResetRef.current = false
      if (wheelZoomCommitTimeoutRef.current !== null) {
        window.clearTimeout(wheelZoomCommitTimeoutRef.current)
        wheelZoomCommitTimeoutRef.current = null
      }

      if (!liveLayerTransformActiveRef.current) {
        clearViewportLayerTransform()
        liveLayerBaseViewportRef.current = renderedViewportRef.current
        liveLayerTransformActiveRef.current = true
      }

      event.currentTarget.setPointerCapture(event.pointerId)

      if (event.pointerType === 'touch') {
        touchPointersRef.current.set(event.pointerId, {
          pointerId: event.pointerId,
          clientX: event.clientX,
          clientY: event.clientY
        })

        if (touchPointersRef.current.size >= 2) {
          beginPinchZoom()
          setOpenNodeId(undefined)
          setLocalHoveredNodeId(undefined)
          return
        }
      }

      dragRef.current = {
        active: true,
        pointerId: event.pointerId,
        originX: event.clientX,
        originY: event.clientY,
        panX: viewportRef.current.panX,
        panY: viewportRef.current.panY
      }
      setIsPanning(true)
      setOpenNodeId(undefined)
      setLocalHoveredNodeId(undefined)
    },
    [
      beginPinchZoom,
      clearViewportLayerTransform,
      dragRef,
      hasUserControlledViewportRef,
      liveLayerBaseViewportRef,
      liveLayerTransformActiveRef,
      pendingPanTransformResetRef,
      renderedViewportRef,
      setIsPanning,
      setLocalHoveredNodeId,
      setOpenNodeId,
      touchPointersRef,
      viewportAnimationRef,
      viewportRef,
      wheelZoomCommitTimeoutRef
    ]
  )

  const handleCanvasPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.pointerType === 'touch' && touchPointersRef.current.has(event.pointerId)) {
        touchPointersRef.current.set(event.pointerId, {
          pointerId: event.pointerId,
          clientX: event.clientX,
          clientY: event.clientY
        })

        const [first, second] = touchPoints(touchPointersRef)

        if (first && second) {
          event.preventDefault()

          if (!pinchZoomRef.current.active) {
            beginPinchZoom()
          }

          const currentDistance = Math.max(1, distanceBetweenTouchPoints(first, second))
          const anchor = midpointRelativeToCanvas(event.currentTarget, first, second)
          const nextZoom = pinchZoomRef.current.initialZoom * (currentDistance / pinchZoomRef.current.initialDistance)

          zoomAroundPoint(nextZoom, anchor.x, anchor.y, { live: true })
          return
        }
      }

      const drag = dragRef.current

      if (!drag.active || drag.pointerId !== event.pointerId) {
        return
      }

      event.preventDefault()
      const canvasSize = canvasSizeRef.current
      const minZoom = calculateMaxZoomOut(currentRenderNodes, canvasSize)
      const nextViewport = clampViewportToPanBounds(
        currentRenderNodes,
        canvasSize,
        {
          zoom: viewportRef.current.zoom,
          panX: drag.panX + event.clientX - drag.originX,
          panY: drag.panY + event.clientY - drag.originY
        },
        zoomFocusRef.current,
        minZoom
      )

      viewportRef.current = nextViewport
      scheduleViewportLayerTransform(nextViewport)
    },
    [
      beginPinchZoom,
      canvasSizeRef,
      currentRenderNodes,
      dragRef,
      pinchZoomRef,
      scheduleViewportLayerTransform,
      touchPointersRef,
      viewportRef,
      zoomAroundPoint,
      zoomFocusRef
    ]
  )

  const stopPanning = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const shouldCommitViewport = dragRef.current.active && dragRef.current.pointerId === event.pointerId

      if (dragRef.current.pointerId === event.pointerId && event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }

      if (event.pointerType === 'touch') {
        touchPointersRef.current.delete(event.pointerId)

        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId)
        }
      }

      const shouldCommitPinchZoom = pinchZoomRef.current.active && touchPointersRef.current.size < 2

      dragRef.current.active = false
      dragRef.current.pointerId = null

      if (shouldCommitViewport || shouldCommitPinchZoom) {
        if (wheelZoomCommitTimeoutRef.current !== null) {
          window.clearTimeout(wheelZoomCommitTimeoutRef.current)
          wheelZoomCommitTimeoutRef.current = null
        }

        pinchZoomRef.current.active = false
        pendingPanTransformResetRef.current = true
        setViewport(viewportRef.current, { userControlled: true })
      }

      if (!shouldCommitPinchZoom && touchPointersRef.current.size < 2) {
        pinchZoomRef.current.active = false
      }

      setIsPanning(false)
    },
    [
      dragRef,
      pendingPanTransformResetRef,
      pinchZoomRef,
      setIsPanning,
      setViewport,
      touchPointersRef,
      viewportRef,
      wheelZoomCommitTimeoutRef
    ]
  )

  const handleCanvasWheel = useCallback(
    (event: WheelEvent) => {
      if (event.deltaY === 0) {
        return
      }

      if (event.cancelable) {
        event.preventDefault()
      }
      event.stopPropagation()

      const canvasElement = canvasRef.current
      if (!canvasElement) {
        return
      }

      const rect = canvasElement.getBoundingClientRect()
      const anchorX = event.clientX - rect.left
      const anchorY = event.clientY - rect.top
      const factor = event.deltaY > 0 ? WHEEL_ZOOM_OUT : WHEEL_ZOOM_IN

      zoomAroundPoint(viewportRef.current.zoom * factor, anchorX, anchorY, { live: true })
    },
    [canvasRef, viewportRef, zoomAroundPoint]
  )

  useEffect(() => {
    const canvasElement = canvasRef.current
    if (!canvasElement) {
      return undefined
    }

    canvasElement.addEventListener('wheel', handleCanvasWheel, { passive: false })

    return () => canvasElement.removeEventListener('wheel', handleCanvasWheel)
  }, [canvasRef, handleCanvasWheel])

  const handleFullscreen = useCallback(() => {
    if (onFullscreen) {
      onFullscreen()
      return
    }

    const canvasElement = canvasRef.current

    if (!canvasElement || typeof canvasElement.requestFullscreen !== 'function') {
      return
    }

    void canvasElement.requestFullscreen().catch((error: unknown) => {
      console.warn('Unable to enter mesh fullscreen mode', error)
    })
  }, [canvasRef, onFullscreen])

  return {
    handleCanvasPointerDown,
    handleCanvasPointerMove,
    handleFullscreen,
    stopPanning
  }
}

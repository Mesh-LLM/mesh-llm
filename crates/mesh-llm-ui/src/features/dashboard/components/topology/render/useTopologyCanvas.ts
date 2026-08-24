import { useEffect, useRef, type MutableRefObject } from 'react'

import {
  createTopologyFrameBuilder,
  type TopologyFrameData
} from '@/features/dashboard/components/topology/render/topology-frame-builder'
import {
  buildLineFragmentShaderSource,
  createProgram,
  LINE_VERTEX_SHADER,
  POINT_VERTEX_SHADER
} from '@/features/dashboard/components/topology/render/shaders'
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

type UseTopologyCanvasArgs = {
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
  renderVariant: RenderVariant
  selfNodeId?: string
  hopDelayMs?: number
}

export function useTopologyCanvas({
  canvasRef,
  hostRef,
  screenNodesRef,
  animationRef,
  lineRevealRef,
  exitAnimationRef,
  twinkleAnimationRef,
  pendingLineTransitionRef,
  lineTransitionRef,
  lastScreenPositionsRef,
  seenNodeIdsRef,
  hoveredNodeIdRef,
  zoomRef,
  panRef,
  renderNodes,
  renderVariant,
  selfNodeId,
  hopDelayMs
}: UseTopologyCanvasArgs) {
  const lineScreenNodesRef = useRef<ScreenNode[]>([])
  const renderNodesRef = useRef(renderNodes)

  useEffect(() => {
    renderNodesRef.current = renderNodes
  }, [renderNodes])

  useEffect(() => {
    const canvas = canvasRef.current
    const host = hostRef.current
    if (!canvas || !host) return
    if (typeof WebGLRenderingContext === 'undefined') return

    const gl = canvas.getContext('webgl', {
      alpha: true,
      antialias: false,
      premultipliedAlpha: true
    })
    if (!gl) return

    const supportsLineDerivatives = Boolean(gl.getExtension('OES_standard_derivatives'))
    const lineFragmentShader = buildLineFragmentShaderSource(renderVariant.lineFragmentShader, {
      useStandardDerivatives: supportsLineDerivatives
    })

    const pointProgram = createProgram(gl, POINT_VERTEX_SHADER, renderVariant.pointFragmentShader)
    const lineProgram = createProgram(gl, LINE_VERTEX_SHADER, lineFragmentShader)
    if (!pointProgram || !lineProgram) return

    const pointPositionLocation = gl.getAttribLocation(pointProgram, 'a_position')
    const pointSizeLocation = gl.getAttribLocation(pointProgram, 'a_size')
    const pointColorLocation = gl.getAttribLocation(pointProgram, 'a_color')
    const pointPulseLocation = gl.getAttribLocation(pointProgram, 'a_pulse')
    const pointTwinkleLocation = gl.getAttribLocation(pointProgram, 'a_twinkle')
    const pointResolutionLocation = gl.getUniformLocation(pointProgram, 'u_resolution')
    const pointTimeLocation = gl.getUniformLocation(pointProgram, 'u_time')

    const linePositionLocation = gl.getAttribLocation(lineProgram, 'a_position')
    const lineColorLocation = gl.getAttribLocation(lineProgram, 'a_color')
    const lineCoordLocation = gl.getAttribLocation(lineProgram, 'a_lineCoord')
    const lineResolutionLocation = gl.getUniformLocation(lineProgram, 'u_resolution')

    const pointPositionBuffer = gl.createBuffer()
    const pointSizeBuffer = gl.createBuffer()
    const pointColorBuffer = gl.createBuffer()
    const pointPulseBuffer = gl.createBuffer()
    const pointTwinkleBuffer = gl.createBuffer()
    const linePositionBuffer = gl.createBuffer()
    const lineColorBuffer = gl.createBuffer()
    const lineCoordBuffer = gl.createBuffer()
    if (
      !pointPositionBuffer ||
      !pointSizeBuffer ||
      !pointColorBuffer ||
      !pointPulseBuffer ||
      !pointTwinkleBuffer ||
      !linePositionBuffer ||
      !lineColorBuffer ||
      !lineCoordBuffer
    ) {
      return
    }

    const lineAttributeLocations = [linePositionLocation, lineColorLocation, lineCoordLocation]
    const pointAttributeLocations = [
      pointPositionLocation,
      pointSizeLocation,
      pointColorLocation,
      pointPulseLocation,
      pointTwinkleLocation
    ]
    const disableVertexAttributes = (locations: number[]) => {
      for (const location of locations) {
        if (location >= 0) {
          gl.disableVertexAttribArray(location)
        }
      }
    }

    let frame = 0
    let animationFrame = 0
    let width = 0
    let height = 0
    let cssWidth = 0
    let cssHeight = 0
    let devicePixelRatio = 1
    let _resizeDirty = false

    const MAX_CANVAS_PIXELS = 8_000_000
    const resize = () => {
      const rect = host.getBoundingClientRect()
      cssWidth = Math.max(1, rect.width)
      cssHeight = Math.max(1, rect.height)
      // Cap DPR so physical pixel count stays within budget — prevents
      // fill-rate bottleneck on large/fullscreen canvases.
      const rawDpr = window.devicePixelRatio || 1
      const maxDpr = Math.sqrt(MAX_CANVAS_PIXELS / (cssWidth * cssHeight))
      devicePixelRatio = Math.min(rawDpr, Math.max(1, maxDpr))
      width = Math.max(1, Math.round(cssWidth * devicePixelRatio))
      height = Math.max(1, Math.round(cssHeight * devicePixelRatio))
      canvas.width = width
      canvas.height = height
      canvas.style.width = `${rect.width}px`
      canvas.style.height = `${rect.height}px`
      gl.viewport(0, 0, width, height)
      _resizeDirty = true
    }

    let _glPointPosCap = 0
    let _glPointSizeCap = 0
    let _glPointColorCap = 0
    let _glPointPulseCap = 0
    let _glPointTwinkleCap = 0
    let _glLinePosCap = 0
    let _glLineColorCap = 0
    let _glLineCoordCap = 0

    const uploadToGLBuffer = (buffer: WebGLBuffer, data: Float32Array, prevCap: number): number => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
      if (data.byteLength > prevCap) {
        const cap = Math.max(data.byteLength * 2, 512)
        gl.bufferData(gl.ARRAY_BUFFER, cap, gl.DYNAMIC_DRAW)
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, data)
        return cap
      }
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data)
      return prevCap
    }

    const frameBuilder = createTopologyFrameBuilder({
      animationRef,
      exitAnimationRef,
      hoveredNodeIdRef,
      hopDelayMs,
      lastScreenPositionsRef,
      lineRevealRef,
      lineScreenNodesRef,
      lineTransitionRef,
      panRef,
      pendingLineTransitionRef,
      renderNodesRef,
      renderVariant,
      screenNodesRef,
      seenNodeIdsRef,
      selfNodeId,
      twinkleAnimationRef,
      zoomRef
    })

    const drawLineMesh = (positions: Float32Array, colors: Float32Array, coords: Float32Array) => {
      if (positions.length === 0) {
        return
      }

      renderVariant.applyLineBlendMode(gl)
      gl['useProgram'](lineProgram)
      gl.uniform2f(lineResolutionLocation, width, height)
      disableVertexAttributes(pointAttributeLocations)

      _glLinePosCap = uploadToGLBuffer(linePositionBuffer, positions, _glLinePosCap)
      gl.enableVertexAttribArray(linePositionLocation)
      gl.vertexAttribPointer(linePositionLocation, 2, gl.FLOAT, false, 0, 0)

      _glLineColorCap = uploadToGLBuffer(lineColorBuffer, colors, _glLineColorCap)
      gl.enableVertexAttribArray(lineColorLocation)
      gl.vertexAttribPointer(lineColorLocation, 4, gl.FLOAT, false, 0, 0)

      _glLineCoordCap = uploadToGLBuffer(lineCoordBuffer, coords, _glLineCoordCap)
      gl.enableVertexAttribArray(lineCoordLocation)
      gl.vertexAttribPointer(lineCoordLocation, 2, gl.FLOAT, false, 0, 0)

      gl.drawArrays(gl.TRIANGLES, 0, positions.length / 2)
    }

    let _lastHoveredId: string | null = null
    let _lastZoom = NaN
    let _lastPanX = NaN
    let _lastPanY = NaN
    let _frameDataCache: TopologyFrameData | null = null

    const render = () => {
      frame += 1

      const hoveredId = hoveredNodeIdRef.current ?? null
      const curZoom = zoomRef.current
      const curPan = panRef.current
      const needsBuild =
        _frameDataCache === null ||
        _resizeDirty ||
        hoveredId !== _lastHoveredId ||
        curZoom !== _lastZoom ||
        curPan.x !== _lastPanX ||
        curPan.y !== _lastPanY ||
        animationRef.current.size > 0 ||
        exitAnimationRef.current.size > 0 ||
        twinkleAnimationRef.current.size > 0 ||
        lineRevealRef.current.size > 0 ||
        lineTransitionRef.current !== null ||
        pendingLineTransitionRef.current !== null

      _lastHoveredId = hoveredId
      _lastZoom = curZoom
      _lastPanX = curPan.x
      _lastPanY = curPan.y
      if (_resizeDirty) _resizeDirty = false

      if (needsBuild) {
        _frameDataCache = frameBuilder.buildFrameData({ cssWidth, cssHeight, devicePixelRatio })
      }
      const {
        pointPositions,
        pointSizes,
        pointColors,
        pointPulses,
        pointTwinkles,
        linePositions,
        lineColors,
        lineCoords
      } = _frameDataCache!

      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.enable(gl.BLEND)

      drawLineMesh(linePositions, lineColors, lineCoords)

      renderVariant.applyPointBlendMode(gl)
      gl['useProgram'](pointProgram)
      gl.uniform2f(pointResolutionLocation, width, height)
      gl.uniform1f(pointTimeLocation, frame / 60)
      disableVertexAttributes(lineAttributeLocations)

      _glPointPosCap = uploadToGLBuffer(pointPositionBuffer, pointPositions, _glPointPosCap)
      gl.enableVertexAttribArray(pointPositionLocation)
      gl.vertexAttribPointer(pointPositionLocation, 2, gl.FLOAT, false, 0, 0)

      _glPointSizeCap = uploadToGLBuffer(pointSizeBuffer, pointSizes, _glPointSizeCap)
      gl.enableVertexAttribArray(pointSizeLocation)
      gl.vertexAttribPointer(pointSizeLocation, 1, gl.FLOAT, false, 0, 0)

      _glPointColorCap = uploadToGLBuffer(pointColorBuffer, pointColors, _glPointColorCap)
      gl.enableVertexAttribArray(pointColorLocation)
      gl.vertexAttribPointer(pointColorLocation, 4, gl.FLOAT, false, 0, 0)

      _glPointPulseCap = uploadToGLBuffer(pointPulseBuffer, pointPulses, _glPointPulseCap)
      gl.enableVertexAttribArray(pointPulseLocation)
      gl.vertexAttribPointer(pointPulseLocation, 1, gl.FLOAT, false, 0, 0)

      _glPointTwinkleCap = uploadToGLBuffer(pointTwinkleBuffer, pointTwinkles, _glPointTwinkleCap)
      gl.enableVertexAttribArray(pointTwinkleLocation)
      gl.vertexAttribPointer(pointTwinkleLocation, 1, gl.FLOAT, false, 0, 0)

      gl.drawArrays(gl.POINTS, 0, pointPositions.length / 2)

      animationFrame = window.requestAnimationFrame(render)
    }

    resize()
    render()

    const observer = new ResizeObserver(() => resize())
    observer.observe(host)

    return () => {
      window.cancelAnimationFrame(animationFrame)
      observer.disconnect()
      disableVertexAttributes(lineAttributeLocations)
      disableVertexAttributes(pointAttributeLocations)
      gl.bindBuffer(gl.ARRAY_BUFFER, null)
      gl.deleteBuffer(pointPositionBuffer)
      gl.deleteBuffer(pointSizeBuffer)
      gl.deleteBuffer(pointColorBuffer)
      gl.deleteBuffer(pointPulseBuffer)
      gl.deleteBuffer(pointTwinkleBuffer)
      gl.deleteBuffer(linePositionBuffer)
      gl.deleteBuffer(lineColorBuffer)
      gl.deleteBuffer(lineCoordBuffer)
      gl.deleteProgram(pointProgram)
      gl.deleteProgram(lineProgram)
    }
  }, [
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
    renderVariant,
    screenNodesRef,
    seenNodeIdsRef,
    selfNodeId,
    hopDelayMs,
    twinkleAnimationRef,
    zoomRef
  ])
}
